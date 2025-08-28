import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os # 추가
import torchvision.utils as vutils # 추가
from math import sqrt, sin, cos, pi
from vd.utils.registry import ARCH_REGISTRY
from vd.archs.arch_util import DCNv2Pack
    
class DilatedBlock(nn.Module):
    def __init__(self, in_channel, d_list, num_feat, reduction=16):
        super(DilatedBlock, self).__init__()
        self.d_list = d_list
        
        self.first_conv = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1)
        
        self.conv_layers = nn.ModuleList()
        c = in_channel
        for i in range(len(d_list)):
            dense_conv = nn.Sequential(nn.Conv2d(in_channels=c, out_channels=num_feat, kernel_size=3, dilation=d_list[i],
                                   padding=d_list[i]),
                                   nn.ReLU(inplace=True)
            )
            self.conv_layers.append(dense_conv)
            c = c + num_feat
            
        # Channel Attention
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c, c // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // reduction, c, bias=False),
            nn.Sigmoid()
        )
        
        self.conv_post = nn.Conv2d(in_channels=c, out_channels=in_channel, kernel_size=1, padding=0)

    def forward(self, x):
        t = self.first_conv(x)
        
        for conv_layer in self.conv_layers:
            _t = conv_layer(t)
            t = torch.cat([_t, t], dim=1)
            
        # Channel Attention
        b, c, _, _ = t.size()
        y = self.global_pool(t).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        t = t * y
        
        t = self.conv_post(t)
        
        return t


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // 4, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channel // 4, out_channel, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class FeatureExtract(nn.Module):
    def __init__(self, num_feat=64):
        super(FeatureExtract, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, num_feat // 4, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(num_feat // 4, num_feat, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DualLvTemAttenBlock_v2(nn.Module):
    def __init__(self, channels, return_weights=False, temp_scale=1.0):
        super().__init__()
        self.return_weights = return_weights
        self.temp_scale = temp_scale  # sharpening 강도 조절용

        # Local Temporal Attention: 각 프레임 간 차이를 1채널로 압축
        self.local_temp_attn_conv = nn.ModuleList([ nn.Conv2d(channels, 1, kernel_size=3, padding=1) for _ in range(3) ])

        # 각 프레임의 spatial attention map 생성 (채널 수를 줄였다가 다시 늘리는 Bottleneck 구조)
        self.spatial_attn_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=1),  # 채널 축소
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, kernel_size=1),  # 채널 복원
            nn.Sigmoid()  # attention map 생성 (0~1)
        )

        # Refinement: Dilated Conv + Channel Attention
        self.dilated_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 8),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 8, channels),
            nn.Sigmoid(),
        )

    def forward(self, feats):  # feats: [B, 3, C, H, W]
        B, T, C, H, W = feats.size()
        assert T == 3, "T must be 3 for this implementation."

        # Local Temporal Attention: 프레임별 점수
        temp_scores = []
        for i in range(T):
            score = self.local_temp_attn_conv[i](feats[:, i])  # [B, 1, H, W]
            temp_scores.append(score)

        scores = torch.cat(temp_scores, dim=1)  # [B, 3, H, W]
        frame_weights = torch.softmax(scores * self.temp_scale, dim=1)  # [B, 3, H, W]

        # 각 프레임의 spatial attention과 곱해서 weighted sum
        weighted_feats = []
        for i in range(3):
            attn_map = self.spatial_attn_conv(feats[:, i])  # [B, C, H, W]
            weight = frame_weights[:, i:i+1]                # [B, 1, H, W]
            weighted = feats[:, i] * attn_map * weight      # [B, C, H, W]
            weighted_feats.append(weighted)

        fused = sum(weighted_feats)  # [B, C, H, W]

        # refinement (Dilated + Channel Attention)
        refined = self.dilated_conv(fused)    # [B, C, H, W]
        channel_weight = self.channel_attn(refined).view(B, C, 1, 1)
        refined = refined * channel_weight + fused  # 채널 어텐션 결과 곱하고 residual 연결

        if self.return_weights:
            return refined, frame_weights
        else:
            return refined
        

class PCDAlignment(nn.Module):
    def __init__(self, num_feat=64, deformable_groups=8):
        super(PCDAlignment, self).__init__()
        self.offset_conv1 = nn.ModuleDict()
        self.offset_conv2 = nn.ModuleDict()
        self.offset_conv3 = nn.ModuleDict()
        self.dcn_pack = nn.ModuleDict()
        self.feat_conv = nn.ModuleDict()
        # Pyramids
        for i in range(3, 0, -1):
            level = f'l{i}'
            self.offset_conv1[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
            if i == 3:
                self.offset_conv2[level] = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            else:
                self.offset_conv2[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
                self.offset_conv3[level] = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.dcn_pack[level] = DCNv2Pack(num_feat, num_feat, 3, padding=1, deformable_groups=deformable_groups)

            if i < 3:
                self.feat_conv[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)

        # Cascading dcn
        self.cas_offset_conv1 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.cas_offset_conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.cas_dcnpack = DCNv2Pack(num_feat, num_feat, 3, padding=1, deformable_groups=deformable_groups)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_feat_l, ref_feat_l):
        # Pyramids
        upsampled_offset, upsampled_feat = None, None
        for i in range(3, 0, -1):
            level = f'l{i}'
            offset = torch.cat([nbr_feat_l[i - 1], ref_feat_l[i - 1]], dim=1)
            offset = self.lrelu(self.offset_conv1[level](offset))
            if i == 3:
                offset = self.lrelu(self.offset_conv2[level](offset))
            else:
                offset = self.lrelu(self.offset_conv2[level](torch.cat([offset, upsampled_offset], dim=1)))
                offset = self.lrelu(self.offset_conv3[level](offset))

            feat = self.dcn_pack[level](nbr_feat_l[i - 1], offset)
            if i < 3:
                feat = self.feat_conv[level](torch.cat([feat, upsampled_feat], dim=1))
            if i > 1:
                feat = self.lrelu(feat)

            if i > 1:  # upsample offset and features
                # x2: when we upsample the offset, we should also enlarge
                # the magnitude.
                upsampled_offset = self.upsample(offset) * 2
                upsampled_feat = self.upsample(feat)

        # Cascading
        offset = torch.cat([feat, ref_feat_l[0]], dim=1)
        offset = self.lrelu(self.cas_offset_conv2(self.lrelu(self.cas_offset_conv1(offset))))
        feat = self.lrelu(self.cas_dcnpack(feat, offset))
        return feat


@ARCH_REGISTRY.register()
class DemoireNet(nn.Module):
    def __init__(self, num_feat=64):
        super(DemoireNet, self).__init__()
        # extract
        self.feat_extract = nn.Conv2d(3, num_feat // 4, 3, 1, 1)
        
        # demoireing
        self.down1 = nn.Conv2d(num_feat, num_feat // 4, 3, 1, 1)
        # self.demoire1_1 = DilatedBlock(in_channel=num_feat, d_list=(1, 2, 3, 2, 1), num_feat=num_feat)
        self.demoire1_1 = DilatedBlock(in_channel=num_feat, d_list=(1, 2, 3, 2, 1), num_feat=num_feat // 2)
        self.down2 = nn.Conv2d(num_feat, num_feat // 4, 3, 1, 1)
        # self.demoire2_1 = DilatedBlock(in_channel=num_feat, d_list=(1, 2, 3, 2, 1), num_feat=num_feat)
        self.demoire2_1 = DilatedBlock(in_channel=num_feat, d_list=(1, 2, 3, 2, 1), num_feat=num_feat // 2)
        self.down3 = nn.Conv2d(num_feat, num_feat // 4, 3, 1, 1)
        # self.demoire3_1 = DilatedBlock(in_channel=num_feat, d_list=(1, 2, 3, 2, 1), num_feat=num_feat)
        self.demoire3_1 = DilatedBlock(in_channel=num_feat, d_list=(1, 2, 3, 2, 1), num_feat=num_feat // 2)

        # refinement
        self.refine1 = ConvBlock(num_feat, 3)
        self.refine2 = ConvBlock(num_feat, 3)
        self.refine3 = ConvBlock(num_feat, 3)

        # others
        self.pus = nn.PixelUnshuffle(2)
        self.bilinear_down = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_intermediate=False):
        b, t, c, h, w = x.size()
        x = x.view(-1, c, h, w)
        feat_0 = self.pus(self.feat_extract(x))
        
        # demoireing
        feat_l1 = self.pus(self.down1(feat_0))       
        feat_l1 = self.demoire1_1(feat_l1)
        feat_l2 = self.pus(self.down2(feat_l1))
        feat_l2 = self.demoire2_1(feat_l2)
        feat_l3 = self.pus(self.down3(feat_l2))
        feat_l3 = self.demoire3_1(feat_l3)
        
        # refinement
        feat_l1 = self.refine1(feat_l1)
        feat_l2 = self.refine2(feat_l2)
        feat_l3 = self.refine3(feat_l3)

        return feat_l1, feat_l2, feat_l3

@ARCH_REGISTRY.register()
class TemporalNet(nn.Module):
    def __init__(self, num_feat=64):
        super(TemporalNet, self).__init__()

        # 세 스케일에 대해 feature 추출기 정의 (각기 독립적인 FeatureExtract 모듈 사용)
        self.feat_extract1 = FeatureExtract(num_feat)
        self.feat_extract2 = FeatureExtract(num_feat)
        self.feat_extract3 = FeatureExtract(num_feat)

        # Dual Level Temporal Attention Block
        self.dlta_l1 = DualLvTemAttenBlock_v2(num_feat, return_weights=True)
        self.dlta_l2 = DualLvTemAttenBlock_v2(num_feat, return_weights=True)
        self.dlta_l3 = DualLvTemAttenBlock_v2(num_feat, return_weights=True)

        # PCD Alignment
        self.pcd_align = PCDAlignment(num_feat, deformable_groups=8)
        self.fusion = nn.Conv2d(num_feat * 3, num_feat, 3, 1, 1)

        # Upsampling + refinement block (2x2 pixel shuffle 기반)
        self.up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.refine1 = nn.Conv2d(num_feat // 4, num_feat // 4, 3, 1, 1)
        self.up2 = nn.Conv2d(num_feat // 4, num_feat, 3, 1, 1)
        self.refine2 = nn.Conv2d(num_feat // 4, num_feat // 4, 3, 1, 1)
        
        # 최종 RGB 이미지 출력
        self.out_conv = nn.Conv2d(num_feat // 4, 3, 3, 1, 1)

        # 기타 유틸리티 모듈
        self.ps = nn.PixelShuffle(2)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # Attention weight 저장용 변수 초기화
        self.attn_weights = {}

    def forward(self, feat_l1, feat_l2, feat_l3):   # 입력: 각 스케일별로 3프레임 스택된 [B*T, C, H, W]
            
        # FeatureExtract로 추출
        feat_l1 = self.feat_extract1(feat_l1)
        feat_l2 = self.feat_extract2(feat_l2)
        feat_l3 = self.feat_extract3(feat_l3)

        # B, T=3 기준으로 다시 shape 조정 ([B*T, C, H, W] → [B, T, C, H, W])
        B3, C, H, W = feat_l1.shape
        B = B3 // 3
        T = 3

        feat_l1 = feat_l1.view(B, T, C, H, W)
        feat_l2 = feat_l2.view(B, T, C, H // 2, W // 2)
        feat_l3 = feat_l3.view(B, T, C, H // 4, W // 4)

        # DLTA 적용 (스케일별 attention-weighted feature 생성)
        feat_l1_ref, weight_l1 = self.dlta_l1(feat_l1)
        feat_l2_ref, weight_l2 = self.dlta_l2(feat_l2)
        feat_l3_ref, weight_l3 = self.dlta_l3(feat_l3)

        # attention weight 저장
        self.attn_weights['l1'] = weight_l1.detach().cpu()
        self.attn_weights['l2'] = weight_l2.detach().cpu()
        self.attn_weights['l3'] = weight_l3.detach().cpu()

        # 정렬 기준이 되는 reference feature 리스트
        ref_feat_l = [feat_l1_ref, feat_l2_ref, feat_l3_ref]

        # 각 프레임에 대해 PCD alignment 수행
        aligned_feat = []
        for i in range(T):  # 세 프레임 반복
            nbr_feat_l = [ feat_l1[:, i], feat_l2[:, i], feat_l3[:, i] ] # i번째 프레임의 3스케일 feature
            aligned_feat.append(self.pcd_align(nbr_feat_l, ref_feat_l)) # PCD 정렬

        aligned_feat = torch.stack(aligned_feat, dim=1)  # [B, T, C, H, W]
        aligned_feat = aligned_feat.view(B, -1, H, W)    # [B, T*C, H, W]
        aligned_feat = self.leaky_relu(self.fusion(aligned_feat))  # [B, C, H, W] 

        # Upsample + refinement (2단계 PixelShuffle)
        aligned_feat = self.ps(self.up1(aligned_feat))             # [B, C/4, H*2, W*2]
        aligned_feat = self.leaky_relu(self.refine1(aligned_feat))
        aligned_feat = self.ps(self.up2(aligned_feat))             # [B, C/4, H*4, W*4]
        aligned_feat = self.leaky_relu(self.refine2(aligned_feat))
        out = self.out_conv(aligned_feat)                          # [B, 3, H*4, W*4]

        return out