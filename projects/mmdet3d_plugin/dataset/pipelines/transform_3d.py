from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
import cv2

import numpy as np
from numpy import random
import torch

import mmcv
from mmcv.transforms import BaseTransform
from mmcv.transforms import Pad as MMCV_Pad
from mmcv.transforms import RandomFlip as MMCV_RandomFlip
from mmcv.transforms.utils import cache_randomness
from mmcv.transforms import Resize as MMCV_Resize
from mmdet.structures.bbox import HorizontalBoxes, autocast_box_type
from mmdet3d.registry import TRANSFORMS
from mmdet3d.datasets.transforms.transforms_3d import (PhotoMetricDistortion3D, ObjectSample, GlobalRotScaleTrans,
                                                       RandomFlip3D, MultiViewWrapper)

from tools.utils.debug_print import DEBUG_print


__all__ = ['CustomPhotoMetricDistortion3D', 'UnifiedObjectSample', 'RandomDropData3D',
           'RandomAugBEV', 'RandomAugOneImage', 'CustomMultiViewWrapper']

Number = Union[int, float]

@TRANSFORMS.register_module()
class CustomPhotoMetricDistortion3D(PhotoMetricDistortion3D):

    def __init__(self,
                 brightness_delta: int = 32,
                 contrast_range: Sequence[Number] = (0.5, 1.5),
                 saturation_range: Sequence[Number] = (0.5, 1.5),
                 hue_delta: int = 18) -> None:
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    @cache_randomness
    def _random_flags(self) -> Sequence[Number]:
        mode = random.randint(2)
        brightness_flag = self.brightness_delta == 0
        contrast_flag = self.contrast_lower == self.contrast_upper == 1
        saturation_flag = self.saturation_lower == self.saturation_upper == 1
        hue_flag = self.hue_delta == 0
        swap_flag = random.randint(2)
        delta_value = random.uniform(-self.brightness_delta,
                                     self.brightness_delta)
        alpha_value = random.uniform(self.contrast_lower, self.contrast_upper)
        saturation_value = random.uniform(self.saturation_lower,
                                          self.saturation_upper)
        hue_value = random.uniform(-self.hue_delta, self.hue_delta)
        swap_value = random.permutation(3)

        return (mode, brightness_flag, contrast_flag, saturation_flag,
                hue_flag, swap_flag, delta_value, alpha_value,
                saturation_value, hue_value, swap_value)


@TRANSFORMS.register_module()
class UnifiedObjectSample(ObjectSample):
    """Sample GT objects to the data.

    Args:
        db_sampler (dict): Config dict of the database sampler.
        sample_2d (bool): Whether to also paste 2D image patch to the images
            This should be true when applying multi-modality cut-and-paste.
            Defaults to False.
    """

    def __init__(self,
                 sample_method: str = 'depth',
                 modify_points: bool = False,
                 mixup_rate: Optional[float] = -1,
                 **kwargs):
        super().__init__(**kwargs)

        self.sample_method = sample_method
        self.modify_points = modify_points
        self.mixup_rate = mixup_rate

    def transform(self, input_dict: dict) -> dict:
        """Transform function to sample ground truth objects to the data.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after object sampling augmentation,
            'points', 'gt_bboxes_3d', 'gt_labels_3d' keys are updated
            in the result dict.
        """
        if self.disabled:
            return input_dict

        with_points = 'points' in input_dict
        with_img = 'img' in input_dict and self.sample_2d

        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']

        if self.use_ground_plane:
            ground_plane = input_dict.get('plane', None)
            assert ground_plane is not None, '`use_ground_plane` is True ' \
                                             'but find plane is None'
        else:
            ground_plane = None
        # change to float for blending operation

        # Assume for now 3D & 2D bboxes are the same
        sampled_dict = self.db_sampler.sample_all(
            gt_bboxes_3d.tensor.numpy(),
            gt_labels_3d,
            ground_plane=ground_plane,
            with_points=with_points,
            with_img=with_img)

        if sampled_dict is not None:
            sampled_gt_bboxes_3d = sampled_dict['gt_bboxes_3d']
            sampled_gt_labels = sampled_dict['gt_labels_3d']

            gt_labels_3d = np.concatenate([gt_labels_3d, sampled_gt_labels], axis=0)
            gt_bboxes_3d = gt_bboxes_3d.new_box(
                np.concatenate([gt_bboxes_3d.tensor.numpy(), sampled_gt_bboxes_3d]))

            if with_points:
                points = input_dict['points']
                sampled_points = sampled_dict['points']
                sampled_points_idx = sampled_dict["points_idx"]
                points = self.remove_points_in_boxes(points, sampled_gt_bboxes_3d)
                points_idx = -1 * np.ones(len(points), dtype=np.int)
                # check the points dimension
                # points = points.cat([sampled_points, points])
                points = points.cat([points, sampled_points])
                points_idx = np.concatenate([points_idx, sampled_points_idx], axis=0)
                points_for_img = points.tensor.numpy()
            else:
                points_for_img = None
                points_idx = None
            if with_img:
                imgs = input_dict['img']

                lidar2img = []
                for cam in range(len(input_dict['lidar2cam'])):
                    lidar2cam = np.array(input_dict['lidar2cam'][cam])  # [4, 4]
                    intrinsic = np.array(input_dict['cam2img'][cam])  # [3, 3]
                    viewpad = np.eye(4)
                    viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                    lidar2img.append(viewpad @ lidar2cam)

                sampled_img = sampled_dict['images']
                sampled_num = len(sampled_gt_bboxes_3d)
                imgs, points_keep = self.unified_sample(imgs, lidar2img,
                                                        points_for_img,
                                                        points_idx, gt_bboxes_3d.corners.numpy(),
                                                        sampled_img, sampled_num)

                input_dict['img'] = imgs

                if self.modify_points and with_points:
                    points = points[points_keep]

        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d.astype(np.long)
        if with_points:
            input_dict['points'] = points
        return input_dict

    def unified_sample(self, imgs, lidar2img, points, points_idx, bboxes_3d, sampled_img, sampled_num):
        flag = points is not None

        # for boxes (include raw data and dbsample data)
        bboxes_3d = np.concatenate([bboxes_3d, np.ones_like(bboxes_3d[..., :1])], -1)
        is_raw = np.ones(len(bboxes_3d))
        is_raw[-sampled_num:] = 0
        is_raw = is_raw.astype(bool)
        raw_num = len(is_raw) - sampled_num
        if flag:
            # for point cloud
            points_3d = points[:, :4].copy()
            points_3d[:, -1] = 1
            points_keep = np.ones(len(points_3d)).astype(np.bool)
        else:
            points_keep = None
        new_imgs = imgs

        assert len(imgs) == len(lidar2img) and len(sampled_img) == sampled_num
        for _idx, (_img, _lidar2img) in enumerate(zip(imgs, lidar2img)):
            coord_img = bboxes_3d @ _lidar2img.T
            coord_img[..., :2] /= coord_img[..., 2, None]
            depth = coord_img[..., 2]
            img_mask = (depth > 0).all(axis=-1)
            img_count = img_mask.nonzero()[0]
            if img_mask.sum() == 0:
                continue
            depth = depth.mean(1)[img_mask]
            coord_img = coord_img[..., :2][img_mask]
            minxy = np.min(coord_img, axis=-2)
            maxxy = np.max(coord_img, axis=-2)
            bbox = np.concatenate([minxy, maxxy], axis=-1).astype(int)
            bbox[:, 0::2] = np.clip(bbox[:, 0::2], a_min=0, a_max=_img.shape[1] - 1)
            bbox[:, 1::2] = np.clip(bbox[:, 1::2], a_min=0, a_max=_img.shape[0] - 1)
            img_mask = ((bbox[:, 2:] - bbox[:, :2]) > 1).all(axis=-1)
            if img_mask.sum() == 0:
                continue
            depth = depth[img_mask]
            if 'depth' in self.sample_method:
                paste_order = depth.argsort()
                paste_order = paste_order[::-1]
            else:
                paste_order = np.arange(len(depth), dtype=np.int64)
            img_count = img_count[img_mask][paste_order]
            bbox = bbox[img_mask][paste_order]

            paste_mask = -255 * np.ones(_img.shape[:2], dtype=np.int)
            fg_mask = np.zeros(_img.shape[:2], dtype=np.int)
            # first crop image from raw image
            raw_img = []
            for _count, _box in zip(img_count, bbox):
                if is_raw[_count]:
                    raw_img.append(_img[_box[1]:_box[3], _box[0]:_box[2]])

            # then stitch the crops to raw image
            for _count, _box in zip(img_count, bbox):
                if is_raw[_count]:
                    if self.mixup_rate < 0:
                        _img[_box[1]:_box[3], _box[0]:_box[2]] = raw_img.pop(0)
                    else:
                        _img[_box[1]:_box[3], _box[0]:_box[2]] = \
                            _img[_box[1]:_box[3], _box[0]:_box[2]] * (1 - self.mixup_rate) + raw_img.pop(
                                0) * self.mixup_rate
                    fg_mask[_box[1]:_box[3], _box[0]:_box[2]] = 1
                else:
                    img_crop = sampled_img[_count - raw_num]
                    if len(img_crop) == 0: continue
                    img_crop = cv2.resize(img_crop, tuple(_box[[2, 3]] - _box[[0, 1]]))
                    if self.mixup_rate < 0:
                        _img[_box[1]:_box[3], _box[0]:_box[2]] = img_crop
                    else:
                        _img[_box[1]:_box[3], _box[0]:_box[2]] = \
                            _img[_box[1]:_box[3], _box[0]:_box[2]] * (1 - self.mixup_rate) + img_crop * self.mixup_rate

                paste_mask[_box[1]:_box[3], _box[0]:_box[2]] = _count

            new_imgs[_idx] = _img

            # calculate modify mask
            if self.modify_points and flag:
                points_img = points_3d @ _lidar2img.T
                points_img[:, :2] /= points_img[:, 2, None]
                depth = points_img[:, 2]
                img_mask = depth > 0
                if img_mask.sum() == 0:
                    continue
                img_mask = (points_img[:, 0] > 0) & (points_img[:, 0] < _img.shape[1]) & \
                           (points_img[:, 1] > 0) & (points_img[:, 1] < _img.shape[0]) & img_mask
                points_img = points_img[img_mask].astype(int)
                new_mask = paste_mask[points_img[:, 1], points_img[:, 0]] == (points_idx[img_mask] + raw_num)
                raw_fg = (fg_mask == 1) & (paste_mask >= 0) & (paste_mask < raw_num)
                raw_bg = (fg_mask == 0) & (paste_mask < 0)
                raw_mask = raw_fg[points_img[:, 1], points_img[:, 0]] | raw_bg[points_img[:, 1], points_img[:, 0]]
                keep_mask = new_mask | raw_mask
                points_keep[img_mask] = points_keep[img_mask] & keep_mask

        return new_imgs, points_keep

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(db_sampler={self.db_sampler},'
        repr_str += f' sample_2d={self.sample_2d},'
        repr_str += f' use_ground_plane={self.use_ground_plane})'
        repr_str += f' data_root={self.sampler_cfg.data_root},'
        repr_str += f' info_path={self.sampler_cfg.info_path},'
        repr_str += f' rate={self.sampler_cfg.rate},'
        repr_str += f' prepare={self.sampler_cfg.prepare},'
        repr_str += f' classes={self.sampler_cfg.classes},'
        repr_str += f' sample_groups={self.sampler_cfg.sample_groups}'
        return repr_str


@TRANSFORMS.register_module()
class RandomDropData3D(BaseTransform):

    def __init__(self,
                 dropout_config = dict(
                     points_drop_rate=0.0,
                     imgs_drop_rate=[0.0])):
        self.dropout_config = dropout_config

    def transform(self, input_dict):
        if 'points' in input_dict:
            points_drop_rate = self.dropout_config.get('points_drop_rate', 0.0)
            points_drop = True if random.rand() < points_drop_rate else False
        else:
            points_drop = False
        if 'img' in input_dict:
            img_nums = len(input_dict['img'])
            imgs_drop_rate = self.dropout_config.get('imgs_drop_rate', [0.0] * img_nums)
            if len(imgs_drop_rate) < img_nums:
                imgs_drop_rate += [0.0] * (img_nums - len(imgs_drop_rate))

            imgs_keep = []
            for i, img_drop_ratio in enumerate(imgs_drop_rate):
                keep = 0.0 if random.rand() < img_drop_ratio else 1.0
                imgs_keep.append(keep)
        else:
            imgs_keep = None

        if points_drop:
            # if drop lidar data, will keep all img data
            input_dict['points'].tensor = input_dict['points'].tensor * 0.0
        elif not points_drop and imgs_keep is not None:
            input_dict['img'] = [imgs_keep[img] * item for img, item in enumerate(input_dict['img'])]

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class RandomAugBEV(BaseTransform):

    def __init__(self, data_config=None, is_train=True):
        """
        data_config should have keys:
            rot_range: list(2)
            scale_ratio_range: tuple(2)  anti-clockwise
            translation_std: tuple(3)
            rand_flip: bool
            flip_ratio: float
            random_flip_direction: bool
            flip_direction: str ['horizontal', 'vertical']
            # In LIDAR coordinates, it flips the y (horizontal) or x (vertical) axis.
        """
        self.data_config = data_config
        if 'rot_range' not in data_config:
            self.data_config['rot_range'] = (0.0, 0.0)
        if 'scale_ratio_range' not in data_config:
            self.data_config['scale_ratio_range'] = (1., 1.)
        if 'translation_std' not in data_config:
            self.data_config['translation_std'] = (0, 0, 0)
        if 'rand_flip' not in data_config:
            self.data_config['rand_flip'] = False
        if 'flip_ratio' not in data_config:
            self.data_config['flip_ratio'] = 0.0
        if  data_config.get('random_flip_direction', False):
            assert 'flip_direction' not in data_config, f"Random flip direction not supported set flip_direction"
        else:
            self.data_config['random_flip_direction'] = False
        if 'flip_direction' in data_config:
            assert not data_config.get('random_flip_direction', False), f"have set flip_direction, not support random flip direction"
            assert data_config['flip_direction'] in ['horizontal', 'vertical']
        else:
            self.data_config['flip_direction'] = 'vertical'

        if 'test_rotate' not in data_config:
            self.data_config['test_rotate'] = 0.0
        if 'test_scale' not in data_config:
            self.data_config['test_scale'] = 1.0
        if 'test_flip' not in data_config:
            self.data_config['test_flip'] = False
        if 'test_flip_direction' not in data_config:
            self.data_config['test_flip_direction'] = 'vertical'

        self.shift_height = self.data_config.get('shift_height', False)
        self.is_train = is_train

    def sample_augmentation(self):
        if self.is_train:
            translation_std = np.array(self.data_config['translation_std'], dtype=np.float32)
            trans_factor = random.normal(scale=translation_std, size=3).T

            rotation = random.uniform(*self.data_config['rot_range'])
            scale_factor = random.uniform(*self.data_config['scale_ratio_range'])

            flip = self.data_config['rand_flip'] and random.rand() < self.data_config['flip_ratio']
            if self.data_config['random_flip_direction']:
                flip_direction = random.choice(['horizontal', 'vertical'])
            else:
                flip_direction = self.data_config['flip_direction']
        else:
            trans_factor = np.array([0, 0, 0], dtype=np.float32).T
            rotation = self.data_config['test_rotate']
            scale_factor = self.data_config['test_scale']
            flip = self.data_config['test_flip']
            flip_direction = self.data_config['test_flip_direction']

        return trans_factor, rotation, scale_factor, flip, flip_direction

    @staticmethod
    def _trans_bbox_points(input_dict: dict, trans_factor: float = 0.) -> None:
        input_dict['pcd_trans'] = trans_factor
        if 'gt_bboxes_3d' in input_dict and len(input_dict['gt_bboxes_3d'].tensor) != 0:
            input_dict['gt_bboxes_3d'].translate(trans_factor)
        if 'points' in input_dict:
            input_dict['points'].translate(trans_factor)
        if 'radar_points' in input_dict:
            input_dict['radar_points'].translate(trans_factor)

    @staticmethod
    def _rot_bbox_points(input_dict: dict, rotation: float=0.) -> None:
        rot_sin = torch.sin(torch.tensor(rotation))
        rot_cos = torch.cos(torch.tensor(rotation))
        ones = torch.ones_like(rot_cos)
        zeros = torch.zeros_like(rot_cos)
        rot_mat_T = torch.stack([
            torch.stack([rot_cos, rot_sin, zeros]),
            torch.stack([-rot_sin, rot_cos, zeros]),
            torch.stack([zeros, zeros, ones])
        ])
        input_dict['pcd_rotation'] = rot_mat_T
        input_dict['pcd_rotation_angle'] = rotation

        # Attention: rotate function is rotate the axis around z
        if 'gt_bboxes_3d' in input_dict and len(input_dict['gt_bboxes_3d'].tensor) != 0:
            input_dict['gt_bboxes_3d'].rotate(rotation)
        if 'points' in input_dict:
            input_dict['points'].rotate(rotation)
        if 'radar_points' in input_dict:
            input_dict['radar_points'].rotate(rotation)


    @staticmethod
    def _scale_bbox_points(input_dict: dict, scale_factor: float = 0.0, shift_height: bool = False) -> None:
        input_dict['pcd_scale_factor'] = scale_factor
        if 'gt_bboxes_3d' in input_dict and len(input_dict['gt_bboxes_3d'].tensor) != 0:
            input_dict['gt_bboxes_3d'].scale(scale_factor)
        if 'points' in input_dict:
            points = input_dict['points']
            points.scale(scale_factor)
            if shift_height:
                assert 'height' in points.attribute_dims.keys(), \
                    'setting shift_height=True but points have no height attribute'
                points.tensor[:, points.attribute_dims['height']] *= scale_factor
            input_dict['points'] = points
        if 'radar_points' in input_dict:
            input_dict['radar_points'].scale(scale_factor)


    @staticmethod
    def _flip_bbox_points(input_dict: dict, flip: bool = False, direction: str = 'horizontal') -> None:
        assert direction in ['horizontal', 'vertical']
        if flip:
            if 'gt_bboxes_3d' in input_dict and len(input_dict['gt_bboxes_3d'].tensor) != 0:
                input_dict['gt_bboxes_3d'].flip(direction)
            if 'points' in input_dict:
                input_dict['points'].flip(direction)
            if 'radar_points' in input_dict:
                input_dict['radar_points'].flip(direction)


    def transform(self, input_dict: dict) -> dict:
        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        trans_factor, rotation, scale_factor, flip, flip_direction = self.sample_augmentation()

        self._rot_bbox_points(input_dict, rotation)
        self._scale_bbox_points(input_dict, scale_factor, self.shift_height)
        self._trans_bbox_points(input_dict, trans_factor)
        self._flip_bbox_points(input_dict, flip, flip_direction)
        flag = 'HF' if flip_direction == 'horizontal' else 'VF'
        input_dict['transformation_3d_flow'].extend(['R', 'S', 'T'])
        if flip:
            input_dict['transformation_3d_flow'].extend([flag])

        if 'lidar2cam' in input_dict:
            transform_mat = self.get_transform_mat(trans_factor=trans_factor,
                                                   rot_matrix=input_dict['pcd_rotation'],
                                                   scale_factor=scale_factor,
                                                   flip=flip, flip_direction=flip_direction)
            transform_mat_inv = np.linalg.inv(transform_mat)
            input_dict['lidar2cam'] = input_dict['lidar2cam'] @ transform_mat_inv

        return input_dict

    @staticmethod
    def get_transform_mat(trans_factor, rot_matrix, scale_factor, flip, flip_direction) -> dict:
        if isinstance(rot_matrix, torch.Tensor):
            rot_matrix = rot_matrix.numpy()

        trans_mat = np.eye(4)
        trans_mat[:3, -1] = trans_factor
        rot_mat = np.eye(4)
        rot_mat[:rot_matrix.shape[0], :rot_matrix.shape[1]] = rot_matrix
        rot_mat[0, 1], rot_mat[1, 0] = -rot_mat[0, 1], -rot_mat[1, 0]
        scale_mat = np.array(
            [
                [scale_factor, 0, 0, 0],
                [0, scale_factor, 0, 0],
                [0, 0, scale_factor, 0],
                [0, 0, 0, 1],
            ])
        flip_mat = np.eye(4)
        # In LIDAR coordinates, it flips the y (horizontal) or x (vertical) axis.
        if flip and flip_direction == 'horizontal':
            flip_mat[1, 1] = -1
        if flip and flip_direction == 'vertical':
            flip_mat[0, 0] = -1

        transform_mat = np.eye(4)
        transform_mat = rot_mat @ transform_mat
        transform_mat = scale_mat @ transform_mat
        transform_mat = trans_mat @ transform_mat
        transform_mat = flip_mat @ transform_mat
        return transform_mat


@TRANSFORMS.register_module()
class RandomAugOneImage(BaseTransform):
    """Random resize, Crop one image, but not change labels
    """

    def __init__(self, data_config=None, is_train=True,):
        """
        data_config should have keys:
        # train-aug
            final_size: tuple(2)
            resize_range: tuple(2)
            crop_range: tuple(2)
            rot_range: tuple(2)
            rand_flip: bool
            flip_ratio: float
            pad: tuple(4)
            pad_color: tuple(3)
        # test-aug
            test_final_size: tuple(2)
            test_resize: float
            test_rotate: float
            test_flip: bool
        """
        self.data_config = data_config
        if 'resize_range' not in data_config:
            self.data_config['resize_range'] = (0.0, 0.0)
        if 'crop_range' not in data_config:
            self.data_config['crop_range'] = (0.0, 0.0)
        if 'rot_range' not in data_config:
            self.data_config['rot_range'] = (0.0, 0.0)
        if 'rand_flip' not in data_config:
            self.data_config['rand_flip'] = False
        if 'flip_ratio' not in data_config:
            self.data_config['flip_ratio'] = 0.0
        if 'pad' not in data_config:
            self.data_config['pad'] = (0.0, 0.0, 0.0, 0.0)
            self.data_config['pad_color'] = (0.0, 0.0, 0.0)

        if 'test_resize' not in data_config:
            self.data_config['test_resize'] = 0.0
        if 'test_rotate' not in data_config:
            self.data_config['test_rotate'] = 0.0
        if 'test_flip' not in data_config:
            self.data_config['test_flip'] = False

        self.is_train = is_train

    def sample_augmentation(self, H, W):
        if self.is_train:
            fH, fW = self.data_config['final_size']
            resize = float(fW) / float(W)
            resize += random.uniform(*self.data_config['resize_range'])
            resize_dims = (int(W * resize), int(H * resize))

            newW, newH  = resize_dims
            crop_h_start = (newH - fH) // 2
            crop_w_start = (newW - fW) // 2
            crop_h_start += int(random.uniform(*self.data_config['crop_range']) * fH)
            crop_w_start += int(random.uniform(*self.data_config['crop_range']) * fW)

            crop = (crop_w_start, crop_h_start, crop_w_start + fW, crop_h_start + fH)
            flip = self.data_config['rand_flip'] and random.rand() < self.data_config['flip_ratio']
            rotate = random.uniform(*self.data_config['rot_range'])
        else:
            fH, fW = self.data_config['test_final_size']
            resize = float(fW) / float(W)
            resize += self.data_config.get('test_resize', 0.0)
            resize_dims = (int(W * resize), int(H * resize))

            newW, newH = resize_dims
            crop_h_start = (newH - fH) // 2
            crop_w_start = (newW - fW) // 2
            crop = (crop_w_start, crop_h_start, crop_w_start + fW, crop_h_start + fH)

            flip = self.data_config['test_flip']
            rotate = self.data_config['test_rotate']

        pad_data = self.data_config['pad']
        pad_color = self.data_config['pad_color']
        pad = (pad_data, pad_color)

        return resize, resize_dims, crop, flip, rotate, pad

    def transform(self, input_dict: dict) -> dict:
        """pad images, masks, semantic segmentation maps.
        Args:
            input_dict (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        if 'img' not in input_dict:
            return input_dict
        # TODO: 在input_dict添加变换后的图像信息，如图像大小等
        resize, resize_dims, crop, flip, rotate, pad = self.sample_augmentation(
                                                            H=input_dict['img'].shape[0],
                                                            W=input_dict['img'].shape[1],)
        resize = input_dict['resize'] if 'resize' in input_dict else resize
        resize_dims = input_dict['resize_dims'] if 'resize_dims' in input_dict else resize_dims
        crop = input_dict['crop'] if 'crop' in input_dict else crop
        flip = input_dict['flip'] if 'flip' in input_dict else flip
        rotate = input_dict['rotate'] if 'rotate' in input_dict else rotate
        pad = input_dict['pad'] if 'pad' in input_dict else pad

        input_dict['img'], post_rot, post_tran = self.img_transform(
                                                input_dict['img'], torch.eye(2), torch.zeros(2),
                                                resize=resize,
                                                resize_dims=resize_dims,
                                                crop=crop,
                                                flip=flip,
                                                rotate=rotate,
                                                pad=pad)
        if 'cam2img' in input_dict:
            input_dict['cam2img'] = self.rts2proj(input_dict['cam2img'], post_rot, post_tran)

        input_dict['resize'] = resize
        input_dict['resize_dims'] = resize_dims
        input_dict['crop'] = crop
        input_dict['flip'] = flip
        input_dict['rotate'] = rotate
        input_dict['pad'] = pad

        return input_dict

    @staticmethod
    def rts2proj(intrinsic, post_rot=None, post_tran=None):
        # post_rot: 3*3, post_tran: 3
        if post_rot is None and post_tran is None:
            return intrinsic

        viewpad = np.eye(4)
        assert post_tran is not None, [post_rot, post_tran]
        viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = post_rot @ intrinsic
        viewpad[:3, 2] += post_tran
        return viewpad[:3, :3].astype(np.float32)

    def img_transform(self, img, post_rot, post_tran,
                      resize, resize_dims, crop,
                      flip, rotate, pad):
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate, pad)

        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        top, right, bottom, left = pad[0]
        post_tran[0] = post_tran[0] + left  # left
        post_tran[1] = post_tran[1] + top  # top

        ret_post_rot, ret_post_tran = np.eye(3), np.zeros(3)
        ret_post_rot[:2, :2] = post_rot
        ret_post_tran[:2] = post_tran

        return img, ret_post_rot, ret_post_tran

    def img_transform_core(self, img, resize_dims, crop, flip, rotate, pad):
        resized_img = mmcv.imresize(img, resize_dims)

        img = self.crop_img(resized_img, crop)

        if flip:
            img = mmcv.imflip(img, 'horizontal')

        img = mmcv.imrotate(img, -rotate / np.pi * 180)

        if any(x != 0 for x in pad[0]):
            img = mmcv.impad(img, padding=pad[0], pad_val=pad[1])
        return img

    @staticmethod
    def crop_img(img, crop, pad_fill=0):
        resized_img = np.ones((crop[3] - crop[1], crop[2] - crop[0], 3)) * pad_fill

        hsize, wsize = crop[3] - crop[1], crop[2] - crop[0]
        dh, dw, sh, sw = crop[1], crop[0], 0, 0

        if dh < 0:
            sh = -dh
            hsize += dh
            dh = 0
        if dh + hsize > img.shape[0]:
            hsize = img.shape[0] - dh
        if dw < 0:
            sw = -dw
            wsize += dw
            dw = 0
        if dw + wsize > img.shape[1]:
            wsize = img.shape[1] - dw
        resized_img[sh: sh + hsize, sw: sw + wsize] = img[dh: dh + hsize, dw: dw + wsize]
        return resized_img

    @staticmethod
    def get_rot(h):
        return torch.Tensor([[np.cos(h), np.sin(h)],
                             [-np.sin(h), np.cos(h)]])


@TRANSFORMS.register_module()
class CustomMultiViewWrapper(MultiViewWrapper):
    def __init__(
        self,
        global_key: list = ['sample_idx'],
        **kwargs) -> None:
        super().__init__(**kwargs)
        self.global_key = global_key

    def transform(self, input_dict: dict) -> dict:
        # store the augmentation related keys for each image.
        for key in self.collected_keys:
            if key not in input_dict or \
                    not isinstance(input_dict[key], list):
                input_dict[key] = []
        prev_process_dict = {}
        for img_id in range(len(input_dict['img'])):
            process_dict = {}

            # override the process dict (e.g. scale in random scale,
            # crop_size in random crop, flip, flip_direction in
            # random flip)
            if img_id != 0 and self.override_aug_config:
                for key in self.randomness_keys:
                    if key in prev_process_dict:
                        process_dict[key] = prev_process_dict[key]

            for key in self.process_fields:
                if key in input_dict:
                    process_dict[key] = input_dict[key][img_id]
            for key in self.global_key:
                if key in input_dict:
                    process_dict[key] = input_dict[key]
            process_dict = self.transforms(process_dict)
            # store the randomness variable in transformation.
            prev_process_dict = process_dict

            # store the related results to results_dict
            for key in self.process_fields:
                if key in process_dict:
                    input_dict[key][img_id] = process_dict[key]
            # update the keys
            for key in self.collected_keys:
                if key in process_dict:
                    if len(input_dict[key]) == img_id + 1:
                        input_dict[key][img_id] = process_dict[key]
                    else:
                        input_dict[key].append(process_dict[key])

        for key in self.collected_keys:
            if len(input_dict[key]) == 0:
                input_dict.pop(key)
        return input_dict
