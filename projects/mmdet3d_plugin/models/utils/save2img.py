import copy
from typing import Dict, List, Optional, Sequence
import os
import random as prandom
import string
from pathlib import Path
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import cv2

from einops import rearrange
import torch
from torch import Tensor

from mmengine import mkdir_or_exist
from mmengine.dist import is_main_process
from mmengine.visualization.utils import check_type, tensor2ndarray
from mmdet.models.utils import multi_apply
from mmdet3d.structures import Det3DDataSample

from projects.mmdet3d_plugin.core.visualizer.pts_vis import get_visualize_sample
from projects.mmdet3d_plugin.core.visualizer.image_vis import save_img_single

__all__ = ['PointCloudImgSaver']


def get_colors_dict(num_colors: int = 15, thickness: int = 2):
    colors = np.multiply([
        plt.cm.get_cmap('gist_ncar', num_colors * 2)((i * 7 + 5) % num_colors * 2)[:3] \
        for i in range(num_colors * 2)], 255).astype(np.uint8).tolist()
    colors = [i[::-1] for i in colors]
    colors_dict = {i: {'color': color, 'thickness': thickness} for i, color in enumerate(colors)}

    return colors_dict


class PointCloudImgSaver(object):
    def __init__(self, data_preprocessor: Dict, plot_examples: int = 50,
                 plot_range: List = [-50, -50, 50, 50], draw_gt: bool = True, draw_pred: bool = True,
                 img_save_order: Optional[List[List[str]]] = None, img_save_index: Optional[List[List[int]]] = None,
                 pred_score_thr: float = 0.3, save_dir: str = './pts_img_vis',
                 transpose: bool = False, save_only_master: bool = False):
        if img_save_order is None:
            img_save_order = [['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT'],
                               ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']]
        self.img_save_order = img_save_order
        self.img_save_index = img_save_index
        self.transpose = transpose
        # BEV : default: x to right, y to up
        #       transpose: x to up, y to left

        # pred_config: for show scores, have two keys: pred_score_font_type, pred_score_font_size
        self.pred_config = dict(pred_score_font_type=cv2.FONT_HERSHEY_DUPLEX, pred_score_font_size=1)

        if data_preprocessor is not None:
            self.img_process = True
            self._channel_conversion = data_preprocessor.get('rgb_to_bgr', False) or \
                                       data_preprocessor.get('bgr_to_rgb', False)
            mean = data_preprocessor.get('mean', None)
            std = data_preprocessor.get('std', None)
            if mean is not None:
                self._enable_normalize = True
                self.mean = np.array(mean).reshape(-1, 1, 1)
                self.std = np.array(std).reshape(-1, 1, 1)
            else:
                self._enable_normalize = False
        else:
            self.img_process = False

        self.plot_examples = plot_examples
        self.render_count = 0

        self.draw_gt = draw_gt
        self.draw_pred = draw_pred
        self.pred_score_thr = pred_score_thr

        self.save_dir = save_dir
        mkdir_or_exist(self.save_dir)

        self.save_only_master = save_only_master

        self.vis_config = get_colors_dict()

        self.plot_range = plot_range
        self.frame = 0

    def single_img_process(self, img):
        if not self.img_process:
            return img

        if self._enable_normalize:
            new_img = img * self.std + self.mean  # rgb to opencv(bgr)

        if self._channel_conversion:
            new_img = new_img[:, [2, 1, 0], ...]  # rgb to opencv(bgr)

        new_img = rearrange(new_img, 'n c h w -> n h w c')  # (n c h w) to (n h w c)
        return new_img


    def add_data_sample_batch(self, batch_inputs_dict: Dict[List, torch.Tensor],
             batch_data_samples: List[Det3DDataSample], save_batch: int = -1):
        if self.render_count >= self.plot_examples or (self.save_only_master and not is_main_process()):
            return

        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        radar_points = batch_inputs_dict.get('radar_points', None)

        if save_batch == -1 or save_batch > len(batch_inputs_dict):
            index_list = [i for i in range(len(batch_data_samples))]
        else:
            index_list = [save_batch]

        data_input_list = []
        for i in index_list:
            data_input = defaultdict(lambda: None)
            data_sample = batch_data_samples[i]
            data_input['input_metas'] = data_sample.metainfo

            if imgs is not None:
                data_input['imgs'] = self.single_img_process(np.copy(tensor2ndarray(imgs[i])))
            if points is not None:
                data_input['points'] = np.copy(tensor2ndarray(points[i]))

            if self.draw_gt and 'gt_instances_3d' in data_sample and \
                    hasattr(data_sample.gt_instances_3d, 'bboxes_3d'):
                data_input['gt_bboxes_3d'] = data_sample.gt_instances_3d.bboxes_3d
                data_input['gt_labels_3d'] = tensor2ndarray(data_sample.gt_instances_3d.labels_3d)
            if self.draw_pred and 'pred_instances_3d' in data_sample and \
                    hasattr(data_sample.pred_instances_3d, 'bboxes_3d'):
                pred_instances_3d = data_sample.pred_instances_3d
                pred_instances_3d = pred_instances_3d[pred_instances_3d.scores_3d > self.pred_score_thr]
                data_input['pred_bboxes_3d'] = pred_instances_3d.bboxes_3d
                data_input['pred_labels_3d'] = tensor2ndarray(pred_instances_3d.labels_3d)
                data_input['pred_scores_3d'] = tensor2ndarray(pred_instances_3d.scores_3d)

            data_input_list.append(data_input)

        multi_apply(self.save, data_input_list)


    def save(self, data_input: defaultdict) -> bool:
        print(f"Rendering samples {self.render_count + 1}")
        input_metas = data_input.pop('input_metas')
        save_name_img, save_name_pts_img = self.get_save_name(input_metas)

        get_save_imgs_func = partial(self.get_save_imgs, input_metas=input_metas, vis_config=self.vis_config,
                                plot_range=self.plot_range, img_save_index=self.img_save_index,
                                pred_config=self.pred_config, transpose=self.transpose)

        gt_data_3d = defaultdict(lambda: None)
        pred_data_3d = defaultdict(lambda: None)

        gt_bboxes_3d = data_input.pop('gt_bboxes_3d', None)
        gt_labels_3d = data_input.pop('gt_labels_3d', None)
        pred_bboxes_3d = data_input.pop('pred_bboxes_3d', None)
        pred_labels_3d = data_input.pop('pred_labels_3d', None)
        pred_scores_3d = data_input.pop('pred_scores_3d', None)

        if gt_bboxes_3d is not None:
            gt_data_3d = get_save_imgs_func(data_input, gt_bboxes_3d, gt_labels_3d)
        if pred_bboxes_3d is not None:
            pred_data_3d = get_save_imgs_func(data_input, pred_bboxes_3d, pred_labels_3d, pred_scores_3d)
        if gt_bboxes_3d is None and pred_bboxes_3d is None:
            pred_data_3d = get_save_imgs_func(data_input)

        self.cv2_save(gt_data_3d, pred_data_3d, save_name_img, save_name_pts_img)

        self.render_count += 1
        return (True,)


    def get_save_name(self, input_metas):
        save_name_img = ''
        save_name_pts_img = ''
        self.frame += 1
        if 'filename' in input_metas:
            if isinstance(input_metas['filename'], list):
                filename_list = input_metas['filename']
            else:
                filename_list = [input_metas['filename']]

            if self.img_save_index is None:
                img_save_index = [[None for _ in row] for row in self.img_save_order]
                for i, row in enumerate(self.img_save_order):
                    for j, cam in enumerate(row):
                        index = [i for i, filename in enumerate(filename_list) if cam in filename][0]
                        img_save_index[i][j] = index
                self.img_save_index = img_save_index

            img_name = Path(filename_list[0]).stem.split('__')[0] + '.png'
            save_name_img = os.path.join(self.save_dir, img_name)
        else:
            img_name = f"{self.frame}.png"
            save_name_img = os.path.join(self.save_dir, img_name)
        if 'lidar_points' in input_metas:
            pts_img_name = Path(input_metas['lidar_points']['lidar_path']).stem.split('.')[0] + '.png'
            save_name_pts_img = os.path.join(self.save_dir, pts_img_name)
        else:
            save_name_pts_img = save_name_img.replace('.png', '__LIDAR_TOP.png')

        return save_name_img, save_name_pts_img


    @staticmethod
    def get_save_imgs(data_input, bboxes_3d=None, labels_3d=None, scores_3d=None, input_metas=None,
                      vis_config=None, plot_range=None, img_save_index=None, pred_config=None, transpose=False):
        data_3d = defaultdict(lambda: None)
        data_3d['pts_img'] = get_visualize_sample(data_input['points'], bboxes_3d, labels_3d, plot_range, vis_config)
        if transpose:
            data_3d['pts_img'] = np.rot90(data_3d['pts_img'], k=1)
        if data_input['imgs'] is not None:
            lidar2imgkey = 'lidar2img'
            if (not 'lidar2img' in input_metas ) and ("lidar2cam" in input_metas):
                lidar2imgkey = 'lidar2cam'
            img_plot_list = multi_apply(save_img_single, data_input['imgs'], input_metas[lidar2imgkey],
                                        boxes_3d=bboxes_3d, labels=labels_3d, scores=scores_3d,
                                        vis_config=vis_config, pred_config=pred_config)[0]
            data_3d['img'] = concat_img(img_plot_list, img_save_index)
        return data_3d

    @staticmethod
    def cv2_save(gt_data_3d, pred_data_3d, save_name_img, save_name_pts_img):
        drawn_img_3d = None
        drawn_pts_img = None

        if gt_data_3d['img'] is not None and pred_data_3d['img'] is not None:
            blank_image = np.zeros((50, gt_data_3d['img'].shape[1], 3), dtype=np.uint8)
            drawn_img_3d = np.concatenate((gt_data_3d['img'], blank_image, pred_data_3d['img']), axis=0)
        elif gt_data_3d['img'] is not None:
            drawn_img_3d = gt_data_3d['img']
        elif pred_data_3d['img'] is not None:
            drawn_img_3d = pred_data_3d['img']

        if gt_data_3d['pts_img'] is not None and pred_data_3d['pts_img'] is not None:
            blank_image = np.zeros((gt_data_3d['pts_img'].shape[0], 50, 3), dtype=np.uint8)
            drawn_pts_img = np.concatenate((gt_data_3d['pts_img'], blank_image, pred_data_3d['pts_img']), axis=1)
            pad_width = drawn_pts_img.shape[1] - pred_data_3d['pts_img'].shape[1]
            pred_img_pad = np.pad(pred_data_3d['pts_img'], ((0, 0), (0, pad_width), (0, 0)), mode='constant')
            drawn_pts_img = np.vstack([drawn_pts_img, pred_img_pad])
        elif gt_data_3d['pts_img'] is not None:
            drawn_pts_img = gt_data_3d['pts_img']
        elif pred_data_3d['pts_img'] is not None:
            drawn_pts_img = pred_data_3d['pts_img']

        if drawn_img_3d is not None:
            cv2.imwrite(save_name_img, drawn_img_3d)
        if drawn_pts_img is not None:
            cv2.imwrite(save_name_pts_img, drawn_pts_img)


def concat_img(img_plot_list, img_save_index):
    if img_save_index is not None:
        row1_list = [img_plot_list[i] for i in img_save_index[0]]
        row1 = np.concatenate(row1_list, axis=1)
        if len(img_save_index) == 2:
            row2_list = [np.flip(img_plot_list[i], axis=1) for i in img_save_index[1]]
            # row2_list = [img_plot_list[i] for i in img_save_index[1]]
            row2 = np.concatenate(row2_list, axis=1)
            result = np.concatenate([row1, row2], axis=0)
        else:
            result = row1
    else:
        result = np.concatenate(img_plot_list, axis=1)

    return result