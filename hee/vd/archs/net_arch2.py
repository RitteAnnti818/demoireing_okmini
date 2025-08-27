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
    def __init__(self, in_channel, d_list, num_feat):
        super(DilatedBlock, self).__init__()
        self.d_list = d_list
        self.conv_layers = nn.ModuleList()

        self.pre_conv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )

        c = in_channel
        for d in d_list:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=c, out_channels=num_feat,
                              kernel_size=3, dilation=d, padding=d, bias=True),
                    nn.ReLU(inplace=True)
                )
            )
            c = c + num_feat

        self.conv_post = nn.Conv2d(in_channels=c, out_channels=in_channel, kernel_size=1, padding=0, bias=True)

        # channel attention
        mid = max(1, in_channel // 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channel, mid, 1, bias=True)
        self.conv2 = nn.Conv2d(mid, mid, 1, bias=True)
        self.conv3 = nn.Conv2d(mid, in_channel, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        t = self.pre_conv(x)

        for conv_layer in self.conv_layers:
            _t = conv_layer(t)
            t = torch.cat([_t, t], dim=1)

        t = self.conv_post(t)

        # Channel attention
        y = self.pool(t)
        y = self.relu(self.conv1(y))
        y = self.relu(self.conv2(y))
        y = self.sigmoid(self.conv3(y))
        t = t * y + x

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


class adaptive_implicit_trans(nn.Module):
    def __init__(self, category_num=1, arg=0):
        super(adaptive_implicit_trans, self).__init__()
        if arg.random_filter == 1:
            self.it_weights = nn.Parameter(torch.rand(64*category_num, 1, 1, 1), requires_grad=False)
        else:
            self.it_weights = nn.Parameter(torch.ones(64*category_num, 1, 1, 1), requires_grad=False)
        if arg.dc == 1:
            for i in range(0,category_num):
                self.it_weights[i*64] = 1
        self.it_weights.requires_grad = True
        self.register_buffer('kernel', self.initialize_kernel())
        self.category_num = category_num
        self.concat_mode = arg.concat_mode
        if self.concat_mode == 2:
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.fc1 = nn.Linear((self.category_num) * 64, 64)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(64, 64 * (self.category_num))
            self.sigmoid = nn.Sigmoid()
    def initialize_kernel(self):
        conv_shape = (64, 64, 1, 1)
        kernel = torch.zeros(conv_shape)
        r1 = sqrt(1.0 / 8)
        r2 = sqrt(2.0 / 8)
        for i in range(8):
            _u = 2 * i + 1
            for j in range(8):
                _v = 2 * j + 1
                index = i * 8 + j
                for u in range(8):
                    for v in range(8):
                        index2 = u * 8 + v
                        t = cos(_u * u * pi / 16) * cos(_v * v * pi / 16)
                        t = t * r1 if u == 0 else t * r2
                        t = t * r1 if v == 0 else t * r2
                        kernel[index2, index, 0, 0] = t
        return kernel

    def forward(self, x, backbone):
        N, C, H, W = x.size()
        x = x.reshape(1, N * C, H, W)
        weight_list = self.it_weights.reshape(1, self.category_num, 64)
        weight_list = weight_list.repeat(N,1,1)
        backbone_temp = backbone.unsqueeze(2)
        backbone_params = backbone_temp * weight_list
        if self.concat_mode == 0:
            backbone_params = torch.sum(backbone_params, dim=1)
            backbone_params = backbone_params.reshape(N * 64, 1, 1, 1)
            kernel = self.kernel.repeat(N,1,1,1) * backbone_params
            x = F.conv2d(input=x, weight=kernel, padding='same', groups=N)
            x = x.reshape(N, C, H, W)
        else:
            backbone_params = backbone_params.reshape(N * 64 * (self.category_num), 1, 1, 1)
            kernel = self.kernel.repeat(N * (self.category_num), 1,1,1) * backbone_params
            x = F.conv2d(input=x, weight=kernel, padding='same', groups=N)
            x = x.reshape(N, C * (self.category_num), H, W)
            if self.concat_mode == 2:
                x_f = self.gap(x)
                x_f = x_f.reshape(N, -1)
                x_f = self.sigmoid(self.fc2(self.relu1(self.fc1(x_f))))
                x_f = x_f.reshape(N, C * (self.category_num), 1, 1)
                x = (x_f + 1) * x
        return x
    

class conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1):
        super(conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=True, dilation=dilation_rate)

    def forward(self, x_input):
        out = self.conv(x_input)
        return out

    
class DCT(nn.Module):
    def __init__(self):
        super(DCT, self).__init__()
        self.register_buffer('kernel', self.initialize_kernel())

    def initialize_kernel(self):
        conv_shape = (64, 64, 1, 1)
        kernel = torch.zeros(conv_shape)
        r1 = sqrt(1.0 / 8)
        r2 = sqrt(2.0 / 8)

        for u in range(8):
            for v in range(8):
                index = u * 8 + v
                _u = 2 * u + 1
                _v = 2 * v + 1
                for i in range(8):
                    for j in range(8):
                        index2 = i * 8 + j
                        t = cos(_u * i * pi / 16) * cos(_v * j * pi / 16)
                        t = t * r1 if u == 0 else t * r2
                        t = t * r1 if v == 0 else t * r2
                        kernel[index, index2, 0, 0] = t
        return kernel

    def forward(self, x):
        return F.conv2d(x, self.kernel, padding='same', groups=1)


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
            if offset.shape[1] != self.offset_conv1[level].in_channels:
                offset = F.pad(offset, (0, 0, 0, 0, 0, self.offset_conv1[level].in_channels - offset.shape[1])) 
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
        self.demoire1_1 = DilatedBlock(in_channel=num_feat, d_list=(1, 2, 3, 2, 1), num_feat=num_feat)
        self.down2 = nn.Conv2d(num_feat, num_feat // 4, 3, 1, 1)
        self.demoire2_1 = DilatedBlock(in_channel=num_feat, d_list=(1, 2, 3, 2, 1), num_feat=num_feat)
        self.down3 = nn.Conv2d(num_feat, num_feat // 4, 3, 1, 1)
        self.demoire3_1 = DilatedBlock(in_channel=num_feat, d_list=(1, 2, 3, 2, 1), num_feat=num_feat)

        self.moire_sep1 = adaptive_implicit_trans(arg.concat_mode = 2, arg.dc = 1, arg.random_filter = 1)
        self.moire_sep2 = adaptive_implicit_trans(num_feat, d_list=(1,2,3,2,1), inter_num=num_feat//2)
        self.moire_sep3 = adaptive_implicit_trans(num_feat, d_list=(1,2,3,2,1), inter_num=num_feat//2)

        # gamma
        self.gamma = nn.Parameter(B, C, 1, 1)

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
    
    def remove_adjacent_frames(self, x):
        b, t, c, h, w = x.size()
        x = x[:, 1, :, :, :].view(b, c, h, w)  # remove adjacent frames
        return x

    def forward(self, x, return_intermediate=False):
        b, t, c, h, w = x.size()
        x = x.view(-1, c, h, w)
        feat_0 = self.pus(self.feat_extract(x))
        
        # demoireing
        feat_l1 = self.pus(self.down1(feat_0))       

        clean1, moire1, loss1 = self.moire_sep1(feat_l1)

        feat_l1 = self.demoire1_1(clean1)  

        feat_l2 = self.pus(self.down2(feat_l1))
        clean2, moire2, loss2 = self.moire_sep2(feat_l2)
        feat_l2 = self.demoire2_1(clean2)

        feat_l3 = self.pus(self.down3(feat_l2))
        clean3, moire3, loss3 = self.moire_sep3(feat_l3)
        feat_l3 = self.demoire3_1(clean3)
        
        # refinement
        feat_l1 = self.refine1(feat_l1)
        feat_l2 = self.refine2(feat_l2)
        feat_l3 = self.refine3(feat_l3)

        # loss
        total_loss = loss1 + loss2 + loss3

        _, _, H1, W1 = clean1.shape
        clean1 = self.remove_adjacent_frames(clean1.view(b, t, -1, H1, W1))
        moire1 = self.remove_adjacent_frames(moire1.view(b, t, -1, H1, W1))

        return feat_l1, feat_l2, feat_l3, clean1, moire1, total_loss


@ARCH_REGISTRY.register()        
class TemporalNet(nn.Module):
    def __init__(self, num_feat=64):
        super(TemporalNet, self).__init__()
        # feature extraction
        self.feat_extract1 = FeatureExtract(num_feat)
        self.feat_extract2 = FeatureExtract(num_feat)
        self.feat_extract3 = FeatureExtract(num_feat)

        # alignment
        self.pcd_align = PCDAlignment(num_feat, deformable_groups=8)
        self.fusion = nn.Conv2d(num_feat * 3, num_feat, 3, 1, 1)

        # refinement
        self.up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.refine1 = nn.Conv2d(num_feat // 4, num_feat // 4, 3, 1, 1)
        self.up2 = nn.Conv2d(num_feat // 4, num_feat, 3, 1, 1)
        self.refine2 = nn.Conv2d(num_feat // 4, num_feat // 4, 3, 1, 1)
        self.out_conv = nn.Conv2d(num_feat // 4, 3, 3, 1, 1)

        # others
        self.ps = nn.PixelShuffle(2)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, feat_l1, feat_l2, feat_l3):
        # feature extraction
        feat_l1 = self.feat_extract1(feat_l1)
        feat_l2 = self.feat_extract2(feat_l2)
        feat_l3 = self.feat_extract3(feat_l3)

        # alignment
        # n, c, h, w = feat_l1.size()
        # aligned_feat = aligned_feat.view(b, -1, h, w)
        B3, C, H, W = feat_l1.shape
        B = B3 // 3
        t = 3   # number of temporal frames
        feat_l1 = feat_l1.view(B, t, C, H, W)
        feat_l2 = feat_l2.view(B, t, C, H // 2, W // 2)
        feat_l3 = feat_l3.view(B, t, C, H // 4, W // 4)

        ref_feat_l = [feat_l1[:, 1, :, :, :].clone(), feat_l2[:, 1, :, :, :].clone(), feat_l3[:, 1, :, :, :].clone()]

        aligned_feat = []
        for i in range(t):
            nbr_feat_l = [
                feat_l1[:, i], feat_l2[:, i], feat_l3[:, i]
            ]
            aligned_feat.append(self.pcd_align(nbr_feat_l, ref_feat_l))

        aligned_feat = torch.stack(aligned_feat, dim=1)
        # aligned_feat = aligned_feat.view(n // 3, -1, h, w)
        aligned_feat = aligned_feat.view(B, -1, H, W)
        aligned_feat = self.leaky_relu(self.fusion(aligned_feat))

        # refinement
        aligned_feat = self.ps(self.up1(aligned_feat))
        aligned_feat = self.leaky_relu(self.refine1(aligned_feat))
        aligned_feat = self.ps(self.up2(aligned_feat))
        aligned_feat = self.leaky_relu(self.refine2(aligned_feat))
        out = self.out_conv(aligned_feat)

        return out