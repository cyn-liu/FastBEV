import os
import copy
from collections import defaultdict
from typing import List, Dict
import numpy as np
from einops import rearrange, einsum

from mmengine import mkdir_or_exist
from mmengine.dist import is_main_process
from mmcv.transforms import BaseTransform
from mmdet3d.registry import TRANSFORMS
from mmdet3d.datasets.transforms import Pack3DDetInputs

from projects.mmdet3d_plugin.models.utils.save2img import PointCloudImgSaver

__all__ = ['CustomPack3DDetInputs', 'AddSupplementInfo', 'PointCloudImgVis']

@TRANSFORMS.register_module()
class CustomPack3DDetInputs(Pack3DDetInputs):
    INPUTS_KEYS = ['points', 'img', 'radars']

    def __init__(
        self,
        keys: tuple,
        meta_keys: tuple = ('sample_idx', 'scene_token', 'sample_token', 'prev_idx', 'next_idx',
                            'ori_shape', 'img_shape', 'pad_shape', 'lidar2img', 'cam2img', 'filename',
                            'box_mode_3d', 'box_type_3d', 'lidar2cam', 'lidar2radar', 'lidar_points',
                            'can_bus', 'lidar2ego', 'ego2global',),
        meta_key_extend: tuple = ('img_aug_matrix', 'lidar_aug_matrix'),) -> None:
        temp = list(meta_keys)
        temp.extend(list(meta_key_extend))
        meta_extend_keys = tuple(temp)
        super().__init__(keys=keys, meta_keys=meta_extend_keys)

@TRANSFORMS.register_module()
class AddSupplementInfo(BaseTransform):
    def __init__(self):
        pass

    def transform(self, input_dict: dict) -> dict:
        # lidar2cam: List[4*4], cam2img: List[3*3]
        if 'lidar2cam' not in input_dict:
            return input_dict

        lidar2imgs = []
        imgs2lidar = []
        for lidar2cam, cam2img in zip(input_dict['lidar2cam'], input_dict['cam2img']):
            viewpad = np.zeros((4, 4))
            viewpad[:cam2img.shape[-2], :cam2img.shape[-1]] = cam2img
            viewpad[3, 3] = 1
            lidar2img_rt = einsum(viewpad, lidar2cam, 'a b, b c -> a c')
            img2lidar_rt = np.linalg.inv(lidar2img_rt)

            lidar2img = lidar2img_rt.astype(np.float32)
            img2lidar = img2lidar_rt.astype(np.float32)
            lidar2imgs.append(lidar2img)
            imgs2lidar.append(img2lidar)

        input_dict['lidar2img'] = lidar2imgs
        input_dict['img2lidar'] = imgs2lidar

        return input_dict


@TRANSFORMS.register_module()
class PointCloudImgVis(BaseTransform):
    def __init__(self, cfg):
        self.saver = PointCloudImgSaver(data_preprocessor=None, **cfg)

    def transform(self, input_dict: dict) -> dict:
        self.add_data_sample(input_dict)
        return input_dict

    def add_data_sample(self, input_dict: dict):
        if self.saver.render_count >= self.saver.plot_examples or \
                (self.saver.save_only_master and not is_main_process()):
            return

        input_dict_clone = copy.deepcopy(input_dict)
        data_input = defaultdict(lambda: None)
        data_input['imgs'] = input_dict_clone.pop('img', None)
        data_input['points'] = input_dict_clone.pop('points', None)
        data_input['gt_bboxes_3d'] = input_dict_clone.get('gt_bboxes_3d', None)
        data_input['gt_labels_3d'] = input_dict_clone.get('gt_labels_3d', None)
        data_input['input_metas'] = input_dict_clone

        if 'lidar2img' not in data_input['input_metas'] and data_input['imgs'] is not None:
            lidar2img_list = []
            input_metas = data_input['input_metas']
            if isinstance(input_metas['filename'], list):
                lidar2cam_list = input_metas['lidar2cam']
                cam2img_list = input_metas['cam2img']
            else:
                lidar2cam_list = [input_metas['lidar2cam']]
                cam2img_list = [input_metas['cam2img']]

            for lidar2cam, intrinsic in zip(lidar2cam_list, cam2img_list):
                # lidar2cam: [4, 4], intrinsic: [3, 3]
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_list.append(viewpad @ lidar2cam)
            input_metas['lidar2img'] = lidar2img_list
            data_input['input_metas'] = input_metas

        self.saver.save(data_input)