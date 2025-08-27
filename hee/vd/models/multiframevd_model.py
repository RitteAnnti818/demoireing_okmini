from matplotlib import scale
import torch
import os.path as osp
import time
from tqdm import tqdm
from collections import OrderedDict
from vd.utils.registry import MODEL_REGISTRY
from vd.archs import build_network
from vd.models import BaseModel
from vd.metrics import calculate_metric, calculate_psnr_pt, calculate_ssim_pt, calculate_vd_psnr, calculate_vd_ssim
from vd.utils import get_root_logger
from vd.losses import build_loss
from vd.data.data_util import tensor2numpy, imwrite_gt
from deepspeed.profiling.flops_profiler import get_model_profile
from deepspeed.accelerator import get_accelerator
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os        

"""
def save_feature_channels(diff, save_dir, prefix='diff', param=False, energy=False):
    os.makedirs(save_dir, exist_ok=True)
    B, C, H, W = diff.shape
    assert B == 1, "지원되는 배치 크기는 1입니다."

    diff = diff[0]  # shape: (C, H, W)
    if param:
        assert H == 1 and W == 1, "param=True일 때는 shape이 (B, C, 1, 1)이어야 합니다."
        # (C, 1, 1) → (C,) → reshape (8, 8)
        param_map = diff.view(C).detach().cpu().numpy().reshape(8, 8)
        if energy:
            vmin, vmax = param_map.min(), param_map.max()
        else:
            vmin, vmax = -1, 1
        np.save(os.path.join(save_dir, f"{prefix}_param.npy"), param_map)
        
        fig, ax = plt.subplots()
        im = ax.imshow(param_map, cmap='jet', interpolation='nearest', vmin=vmin, vmax=vmax)
        ax.axis('off')

        plt.savefig(os.path.join(save_dir, f"{prefix}_param.png"), bbox_inches='tight', pad_inches=0)
        plt.close()

    else:
        for c in range(C):
            feat = diff[c]  # (H, W)

            # 정규화 (0~1)
            feat_min = feat.min()
            feat_max = feat.max()
            norm = (feat - feat_min) / (feat_max - feat_min + 1e-6)

            # Tensor → PIL 이미지로 변환
            img = TF.to_pil_image(norm.cpu())

            # 저장
            img.save(os.path.join(save_dir, f"{prefix}_ch{c:02d}.png"))"""


@MODEL_REGISTRY.register()
class MultiFrameVDModel(BaseModel):
    def __init__(self, opt):
        super(MultiFrameVDModel, self).__init__(opt)

        # define networks
        self.net_d = build_network(opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)

        self.net_t = build_network(opt['network_t'])
        self.net_t = self.model_to_device(self.net_t)
        with get_accelerator().device(0):
            flops_d, macs_d, params_d = get_model_profile(model=self.net_d, # model
                                    input_shape=(1, 3, 3, 720, 1280), # input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
                                    args=None, # list of positional arguments to the model.
                                    kwargs=None, # dictionary of keyword arguments to the model.
                                    print_profile=True, # prints the model graph with the measured profile attached to each module
                                    detailed=True, # print the detailed profile
                                    module_depth=-1, # depth into the nested modules, with -1 being the inner most modules
                                    top_modules=1, # the number of top modules to print aggregated profile
                                    warm_up=10, # the number of warm-ups before measuring the time of each module
                                    as_string=True, # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
                                    output_file=None, # path to the output file. If None, the profiler prints to stdout.
                                    ignore_modules=None) # the list of modules to ignore in the profiling
            
            flops_t, macs_t, params_t = get_model_profile(model=self.net_t, # model
                                    input_shape=None, # input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
                                    args=[torch.zeros((3, 3, 180, 320), device=self.device),
                                            torch.zeros((3, 3, 90, 160), device=self.device),
                                            torch.zeros((3, 3, 45, 80), device=self.device)], # list of positional arguments to the model.
                                    kwargs=None, # dictionary of keyword arguments to the model.
                                    print_profile=True, # prints the model graph with the measured profile attached to each module
                                    detailed=True, # print the detailed profile
                                    module_depth=-1, # depth into the nested modules, with -1 being the inner most modules
                                    top_modules=1, # the number of top modules to print aggregated profile
                                    warm_up=10, # the number of warm-ups before measuring the time of each module
                                    as_string=True, # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
                                    output_file=None, # path to the output file. If None, the profiler prints to stdout.
                                    ignore_modules=None)
            logger = get_root_logger()
            logger.info(f'DemoireNet FLOPs: {flops_d}, MACs: {macs_d}, Params: {params_d}')
            logger.info(f'TemporalNet FLOPs: {flops_t}, MACs: {macs_t}, Params: {params_t}')
            logger.info(f'Total FLOPs: {flops_d + flops_t}, Total MACs: {macs_d + macs_t}, Total Params: {params_d + params_t}')

        # load pretrained models
        # DemoireNet
        load_path_d = self.opt['path'].get('pretrain_network_d', None)
        if load_path_d is not None:
            param_key_d = self.opt['path'].get('param_key_d', 'params')
            self.load_network(self.net_d, load_path_d, self.opt['path'].get('strict_load_d', True), param_key_d)
        
        # TemporalNet
        load_path_t = self.opt['path'].get('pretrain_network_t', None)
        if load_path_t is not None:
            param_key_t = self.opt['path'].get('param_key_t', 'params')
            self.load_network(self.net_t, load_path_t, self.opt['path'].get('strict_load_t', True), param_key_t)
        
        if self.is_train:
            self.init_training_settings()

    def model_d_ema(self, decay=0.999):
        net_d = self.get_bare_model(self.net_d)
        net_d_params = dict(net_d.named_parameters())
        net_d_ema_params = dict(self.net_d_ema.named_parameters())
        for k in net_d_ema_params.keys():
            net_d_ema_params[k].data.mul_(decay).add_(net_d_params[k].data, alpha=1 - decay)

    def model_t_ema(self, decay=0.999):
        net_t = self.get_bare_model(self.net_t)
        net_t_params = dict(net_t.named_parameters())
        net_t_ema_params = dict(self.net_t_ema.named_parameters())
        for k in net_t_ema_params.keys():
            net_t_ema_params[k].data.mul_(decay).add_(net_t_params[k].data, alpha=1 - decay)    

    def init_training_settings(self):
        self.net_d.train()
        self.net_t.train()
        train_opt = self.opt['train']
        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')

        # ========== net_d EMA ==========
        self.net_d_ema = build_network(self.opt['network_d']).to(self.device)
        load_path_d = self.opt['path'].get('pretrain_network_d', None)
        if load_path_d is not None:
            self.load_network(
                self.net_d_ema,
                load_path_d,
                self.opt['path'].get('strict_load_d', True),
                'params_ema_d'
            )
        else:
            self.model_d_ema(0)  # copy net_d weights
        self.net_d_ema.eval()
        self.net_t_ema = build_network(self.opt['network_t']).to(self.device)
        load_path_t = self.opt['path'].get('pretrain_network_t', None)
        if load_path_t is not None:
            self.load_network(
                self.net_t_ema,
                load_path_t,
                self.opt['path'].get('strict_load_t', True),
                'params_ema_t'
            )
        else:
            self.model_t_ema(0)  # copy net_t weights
        self.net_t_ema.eval()

        # define losses
        # pixel loss
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None
            
        # perceptual loss
        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'gts' in data:
            self.gts = data['gts'].to(self.device)

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_d.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        for k, v in self.net_t.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        # 동일 lr 및 최적화 전략을 채택하기에 net_d 기준으로 설정
        optim_type = train_opt['optim_d'].pop('type')
        self.optimizer_d = self.get_optimizer(optim_type, optim_params, **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)

    def optimize_parameters(self, current_iter):
        self.optimizer_d.zero_grad()
        self.l1, self.l2, self.l3, self.clean, self.moire, _ = self.net_d(self.lq)
        self.output = self.net_t(self.l1, self.l2, self.l3)
        n, c, h, w = self.l1.size()
        self.l1 = self.l1.view(-1, 3, c, h, w)
        self.l2 = self.l2.view(-1, 3, c, h // 2, w // 2)
        self.l3 = self.l3.view(-1, 3, c, h // 4, w // 4)
        self.l1 = self.l1[:, 1, :, :, :].clone().squeeze(1)
        self.l2 = self.l2[:, 1, :, :, :].clone().squeeze(1)
        self.l3 = self.l3[:, 1, :, :, :].clone().squeeze(1)
        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            self.gt_l1 = F.interpolate(self.gt, scale_factor=0.25, mode='bilinear', align_corners=False)
            self.gt_l2 = F.interpolate(self.gt_l1, scale_factor=0.5, mode='bilinear', align_corners=False)
            self.gt_l3 = F.interpolate(self.gt_l2, scale_factor=0.5, mode='bilinear', align_corners=False)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
            l_pix_l1 = self.cri_pix(self.l1, self.gt_l1)
            l_pix_l2 = self.cri_pix(self.l2, self.gt_l2)
            l_pix_l3 = self.cri_pix(self.l3, self.gt_l3)
            l_pix_d = l_pix_l1 + l_pix_l2 + l_pix_l3
            loss_dict['l_pix_l1'] = l_pix_l1
            loss_dict['l_pix_l2'] = l_pix_l2
            loss_dict['l_pix_l3'] = l_pix_l3
            l_total += l_pix_d
            loss_dict['l_pix_d'] = l_pix_d
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style
        l_total.backward()
        self.optimizer_d.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)
        if self.ema_decay > 0:
            self.model_d_ema(decay=self.ema_decay)
            self.model_t_ema(decay=self.ema_decay)
        
        # ===== Attention 출력 저장 (for 디버깅) =====
        if hasattr(self.net_t, 'aligned_feat_list'):
            self.aligned_feat_list = self.net_t.aligned_feat_list
        if hasattr(self.net_t, 'attn_weights'):
            self.attn_weights = self.net_t.attn_weights

    def test(self):
        scale = self.opt.get('scale', 1)
        _, _, _, self.h_old, self.w_old = self.lq.size()

        if hasattr(self, 'net_ema'):
            self.net_d_ema.eval()
            self.net_t_ema.eval()
            with torch.no_grad():
                # self.l1, self.l2, self.l3 = self.net_d_ema(self.lq)
                self.l1, self.l2, self.l3, self.clean, self.moire, _ = self.net_d_ema(self.lq)
                self.output = self.net_t_ema(self.l1, self.l2, self.l3)
                self.output = self.output[:, :, :self.h_old * scale, :self.w_old * scale]
        else:
            self.net_d.eval()
            self.net_t.eval()
            with torch.no_grad():
                # self.l1, self.l2, self.l3 = self.net_d(self.lq)
                self.l1, self.l2, self.l3, self.clean, self.moire, _ = self.net_d(self.lq)
                self.output = self.net_t(self.l1, self.l2, self.l3)
                # self.output = self.net(self.lq)
                self.output = self.output[:, :, :self.h_old * scale, :self.w_old * scale]
            self.net_d.train()
            self.net_t.train()
        n, c, h, w = self.l1.size()
        self.l1 = self.l1.view(-1, 3, c, h, w)
        self.l1 = self.l1[:, 1, :, :, :].clone().squeeze(1)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)
        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
                self.metric_results['PSNR_l1'] = 0
                self.metric_results['SSIM_l1'] = 0
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        time_inf_total = 0.
        for idx, val_data in enumerate(dataloader):
            idx_str = f"{idx:04d}"
            img_name = val_data['key'][0]
            self.feed_data(val_data)
            st = time.time()
            self.test()
            st1 = time.time() - st
            time_inf_total += st1
            visuals = self.get_current_visuals()
            if self.opt['is_train']:
                sr_img_tensors = self.output.detach()
                metric_data['img'] = sr_img_tensors

                if 'gt' in visuals:
                    gt_img_tensors = self.gt.detach()
                    gt_l1_tensors = F.interpolate(gt_img_tensors, scale_factor=0.25, mode='bilinear', align_corners=False)
                    metric_data['img2'] = gt_img_tensors
                    metric_data['img2_l1'] = gt_l1_tensors
                    del self.gt
            else:
                sr_img = tensor2numpy(visuals['result'])
                metric_data['img'] = sr_img                
                # save clean and moire images
                metric_data['clean'] = tensor2numpy(self.clean.detach().cpu())
                metric_data['moire'] = tensor2numpy(self.moire.detach().cpu())
                save_dir_clean = osp.join(self.opt['path']['visualization'], dataset_name, 'clean', img_name)
                save_dir_moire = osp.join(self.opt['path']['visualization'], dataset_name, 'moire', img_name)
                os.makedirs(save_dir_clean, exist_ok=True)
                os.makedirs(save_dir_moire, exist_ok=True)
                imwrite_gt(metric_data['clean'], osp.join(save_dir_clean, f'{img_name}.png'))
                imwrite_gt(metric_data['moire'], osp.join(save_dir_moire, f'{img_name}.png'))
                """""
                save_dir_clean = osp.join(self.opt['path']['visualization'], dataset_name, 'clean', img_name)
                save_dir_moire = osp.join(self.opt['path']['visualization'], dataset_name, 'moire', img_name)
                os.makedirs(save_dir_clean, exist_ok=True)
                os.makedirs(save_dir_moire, exist_ok=True)
                save_feature_channels(self.clean, save_dir_clean, prefix='clean')
                save_feature_channels(self.moire, save_dir_moire, prefix='moire')
                """
                if 'gt' in visuals:
                    gt_img = visuals['gt']
                    gt_l1_img = F.interpolate(gt_img, scale_factor=0.25, mode='bilinear', align_corners=False)
                    gt_img = tensor2numpy(gt_img)
                    gt_l1_img = tensor2numpy(gt_l1_img)
                    metric_data['img2'] = gt_img
                    metric_data['img2_l1'] = gt_l1_img
                    del self.gt

                    """save_dir_clean = os.path.join(self.opt['path']['results_root'], 'ablation', img_name, 'clean')
                    save_dir_moire = os.path.join(self.opt['path']['results_root'], 'ablation', img_name, 'moire')
                    os.makedirs(save_dir_clean, exist_ok=True)
                    os.makedirs(save_dir_moire, exist_ok=True)

                    scale = self.opt.get('scale', 1)
                    target_size = (self.h_old * scale, self.w_old * scale)

                    for b in range(self.clean.size(0)):
                        clean_up = F.interpolate(self.clean[b].unsqueeze(0), size=target_size, mode='bilinear', align_corners=False)
                        moire_up = F.interpolate(self.moire[b].unsqueeze(0), size=target_size, mode='bilinear', align_corners=False)

                        clean_np = tensor2numpy(clean_up[0])  # (H, W, 3)
                        moire_np = tensor2numpy(moire_up[0])

                        clean_path = os.path.join(save_dir_clean, f"{img_name}_b{b:02d}.png")
                        moire_path = os.path.join(save_dir_moire, f"{img_name}_b{b:02d}.png")

                        imwrite_gt(clean_np, clean_path)
                        imwrite_gt(moire_np, moire_path)"""


            # tentative for out of GPU memory
            del self.lq
            del self.output
            del self.clean
            del self.moire
            torch.cuda.empty_cache()
            if save_img:
                if self.opt['is_train']:
                    pass
                else:
                    save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}.png')
                    sr_img_l1 = tensor2numpy(self.l1.detach().cpu())
                imwrite_gt(sr_img, save_img_path)
            """if save_img:
                if not self.opt['is_train']:
                    save_dir = osp.join(self.opt['path']['visualization'], dataset_name)
                    os.makedirs(save_dir, exist_ok=True)

                    # result (output) 저장
                    sr_img = tensor2numpy(self.output.detach().cpu())
                    save_img_path = osp.join(save_dir, f'{img_name}_output.png')
                    imwrite_gt(sr_img, save_img_path)

                    # clean 저장
                    if hasattr(self, 'clean'):
                        clean_img = tensor2numpy(self.clean.detach().cpu())
                        clean_path = osp.join(save_dir, f'{img_name}_clean.png')
                        imwrite_gt(clean_img, clean_path)

                    # moire 저장
                    if hasattr(self, 'moire'):
                        moire_img = tensor2numpy(self.moire.detach().cpu())
                        moire_path = osp.join(save_dir, f'{img_name}_moire.png')
                        imwrite_gt(moire_img, moire_path)

                    # l1 저장
                    sr_img_l1 = tensor2numpy(self.l1.detach().cpu())
                    l1_path = osp.join(save_dir, f'{img_name}_l1.png')
                    imwrite_gt(sr_img_l1, l1_path)"""

            """if with_metrics:
                if self.opt['is_train']:
                    self.metric_results['SSIM_l1'] += calculate_ssim_pt(self.l1.detach(),
                                                                           metric_data['img2_l1'],
                                                                           0).detach().cpu().numpy().sum()
                    self.metric_results['PSNR_l1'] += calculate_psnr_pt(self.l1.detach(),
                                                                           metric_data['img2_l1'],
                                                                           0).detach().cpu().numpy().sum()
                else:
                    self.metric_results['SSIM_l1'] += calculate_vd_ssim(sr_img_l1,
                                                                           metric_data['img2_l1']).sum()
                    self.metric_results['PSNR_l1'] += calculate_vd_psnr(sr_img_l1,
                                                                           metric_data['img2_l1']).sum()
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    if self.opt['is_train']:
                        self.metric_results[name] += calculate_metric(metric_data, opt_).detach().cpu().numpy().sum()
                    else:
                        self.metric_results[name] += calculate_metric(metric_data, opt_)"""
            if with_metrics:
                if self.opt['is_train']:
                    self.metric_results['SSIM_l1'] += calculate_ssim_pt(self.l1.detach(),
                                                                         metric_data['img2_l1'],
                                                                         0).detach().cpu().numpy().sum()
                    self.metric_results['PSNR_l1'] += calculate_psnr_pt(self.l1.detach(),
                                                                         metric_data['img2_l1'],
                                                                         0).detach().cpu().numpy().sum()
                else:
                    self.metric_results['SSIM_l1'] += calculate_vd_ssim(sr_img_l1,
                                                                         metric_data['img2_l1']).sum()
                    self.metric_results['PSNR_l1'] += calculate_vd_psnr(sr_img_l1,
                                                                         metric_data['img2_l1']).sum()
            
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    # 중복 키 방지용 필터링
                    allowed_keys = {'type', 'device', 'crop_border'}
                    opt_filtered = {k: v for k, v in opt_.items() if k in allowed_keys}
            
                    metric_inputs = {
                        'img': metric_data.get('pred') or metric_data.get('img'),
                        'img2': metric_data.get('gt') or metric_data.get('img2')
                    }
            
                    if self.opt['is_train']:
                        self.metric_results[name] += calculate_metric(metric_inputs, opt_filtered).detach().cpu().numpy().sum()
                    else:
                        self.metric_results[name] += calculate_metric(metric_inputs, opt_filtered)


            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()
        time_avg = time_inf_total / (idx + 1)
        logger = get_root_logger()
        logger.info('average test time: %.3f, total time: %.3f' % (time_avg, time_inf_total))
        if with_metrics:
            for metric in self.metric_results.keys():
                if self.opt['is_train']:
                    self.metric_results[metric] /= 2580
                else:
                    self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_d_ema'):
            self.save_network([self.net_d, self.net_d_ema], 'net_d', current_iter, param_key=['params', 'params_ema_d'])
        else:
            self.save_network(self.net_d, 'net_d', current_iter)
        if hasattr(self, 'net_t_ema'):
            self.save_network([self.net_t, self.net_t_ema], 'net_t', current_iter, param_key=['params', 'params_ema_t'])
        else:
            self.save_network(self.net_t, 'net_t', current_iter)
        self.save_training_state(epoch, current_iter)