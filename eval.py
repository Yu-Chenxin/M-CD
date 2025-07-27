import os
import cv2
import argparse
import numpy as np

import torch
import torch.nn as nn

from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from utils.visualize import print_iou, show_img
from engine.evaluator import Evaluator
from engine.logger import get_logger
from utils.metric import hist_info, compute_score
from dataloader.changeDataset import ChangeDataset
from models.builder import EncoderDecoder as segmodel
from dataloader.dataloader import ValPre
from PIL import Image

logger = get_logger()

class SegEvaluator(Evaluator):
    def func_per_iteration(self, data, device, config):
        As = data['A']
        Bs = data['B']
        label = data['gt']
        name = data['fn']
        pred = self.sliding_eval_rgbX(As, Bs, config.eval_crop_size, config.eval_stride_rate, device)
        hist_tmp, labeled_tmp, correct_tmp = hist_info(config.num_classes, pred, label)
        results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp, 'correct': correct_tmp}

        if self.save_path is not None:
            ensure_dir(self.save_path)

            fn = name + '.png'

            result_img = Image.fromarray(pred.astype(np.uint8)*255, mode='P')
            # 将 PIL 图像转换为 OpenCV 处理的 NumPy 数组
            pred_np = np.array(result_img)

            # 降噪处理：使用中值滤波
            pred_denoised = cv2.medianBlur(pred_np, ksize=3)  # ksize 是滤波核大小，一般为 3 或 5

            # 在二值掩码中查找轮廓
            contours, _ = cv2.findContours(pred_denoised, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 如果 Bs 是 PIL 图像，则将其转换为 NumPy 数组
            IMG = Bs
            if isinstance(IMG, Image.Image):
                Bs_np = np.array(IMG)
            else:
                Bs_np = IMG

            # 在 Bs 图像上绘制轮廓
            cv2.drawContours(Bs_np, contours, -1, (0, 0, 255), thickness=2)  # Red color with thickness 2

            # 保存带有轮廓的 Bs 图像
            bs_with_contours_path ="./Bs_with_contours"
            ensure_dir(bs_with_contours_path)
            cv2.imwrite(os.path.join(bs_with_contours_path, fn), Bs_np)


            # save raw result
            colormap = np.zeros((256, 1, 3), dtype=np.uint8)
            for i in range(256):
                colormap[i, 0, 0] = i  # Blue channel
                colormap[i, 0, 1] = i  # Green channel
                colormap[i, 0, 2] = i  # Red channel

            pred_fix = cv2.applyColorMap((pred * 255).astype(np.uint8), colormap)#cv2.COLORMAP_JET
            cv2.imwrite(os.path.join(self.save_path, fn), pred_fix)
            logger.info('Save the image ' + fn)

        if self.show_image:
            colors = self.dataset.get_class_colors()
            if 'img' in data:
                img = data['img']
            else:
                if isinstance(Bs, Image.Image):
                    img = np.array(Bs)
                else:
                    img = Bs
            image = img
            clean = np.zeros(label.shape)
            comp_img = show_img(colors, config.background, image, clean,
                                label,
                                pred)
            cv2.imshow('comp_image', comp_img)
            cv2.waitKey(0)

        return results_dict

    def compute_metric(self, results):
        hist = np.zeros((self.config.num_classes, self.config.num_classes))
        correct = 0
        labeled = 0
        count = 0
        for d in results:
            hist += d['hist']
            correct += d['correct']
            labeled += d['labeled']
            count += 1

        iou, recall, precision, mean_IoU, _, freq_IoU, mean_pixel_acc, pixel_acc = compute_score(hist, correct, labeled)
        result_line = print_iou(iou, recall, precision, freq_IoU, mean_pixel_acc, pixel_acc,
                                self.dataset.class_names, show_no_back=False)
        return result_line, mean_IoU

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='last', type=str)
    parser.add_argument('-d', '--devices', default='0', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--show_image', '-s', default=False,
                        action='store_true')
    parser.add_argument('--save_path', '-p', default=None)
    parser.add_argument('--dataset_name', '-n', default='mfnet', type=str)
    parser.add_argument('--split', '-c', default='test', type=str)

    args = parser.parse_args()
    all_dev = parse_devices(args.devices)
    
    dataset_name = args.dataset_name
    if dataset_name == 'levir':
        from configs.config_levir import config
    elif dataset_name == 'dsifn':
        from configs.config_dsifn import config
    elif dataset_name == 'whu':
        from configs.config_whu import config
    elif dataset_name == 'cdd':
        from configs.config_cdd import config
    elif dataset_name == 'sysu':
        from configs.config_sysu import config
    else:
        raise ValueError('Not a valid dataset name')

    network = segmodel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
    flops = network.flops()
    print("Gflops of the network: ", flops/(10**9))
    print("number of paramters: ", sum(p.numel() if p.requires_grad==True else 0 for p in network.parameters()))
    # 1/0
    data_setting = {'root': config.root_folder,
                    'A_format': config.A_format,
                    'B_format': config.B_format,
                    'gt_format': config.gt_format,
                    'class_names': config.class_names}
    val_pre = ValPre()
    dataset = ChangeDataset(data_setting, args.split, val_pre)
 
    with torch.no_grad():
        segmentor = SegEvaluator(dataset, config.num_classes, config.norm_mean,
                                 config.norm_std, network,
                                 config.eval_scale_array, config.eval_flip,
                                 all_dev, args.verbose, args.save_path,
                                 args.show_image, config)
        _, mean_IoU = segmentor.run_eval(config.checkpoint_dir, args.epochs, config.val_log_file,
                      config.link_val_log_file)

    #visualize erf

    # with torch.enable_grad():
    #     segmentor = SegEvaluator(dataset, config.num_classes, config.norm_mean,
    #                                 config.norm_std, network,
    #                                 config.eval_scale_array, config.eval_flip,
    #                                 all_dev, args.verbose, args.save_path,
    #                                 args.show_image, config)
        
    #     segmentor.get_erf(config.checkpoint_dir, args.epochs)