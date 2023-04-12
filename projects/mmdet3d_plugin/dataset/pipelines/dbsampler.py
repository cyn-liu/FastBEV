# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import numpy as np
from typing import List, Optional

import mmengine

from mmdet3d.structures.ops import box_np_ops
from mmdet3d.datasets.transforms import data_augment_utils
from mmdet3d.registry import TRANSFORMS
from mmdet3d.datasets.transforms.dbsampler import DataBaseSampler

__all__ = ['UnifiedDataBaseSampler']


@TRANSFORMS.register_module()
class UnifiedDataBaseSampler(DataBaseSampler):
    """Class for sampling data from the ground truth database.

    Args:
        info_path (str): Path of groundtruth database info.
        data_root (str): Path of groundtruth database.
        rate (float): Rate of actual sampled over maximum sampled number.
        prepare (dict): Name of preparation functions and the input value.
        sample_groups (dict): Sampled classes and numbers.
        classes (list[str]): List of classes. Default: None.
        img_loader(dict): Config of points loader. Default: None, dict(
            type='LoadImageFromFile')
        points_loader(dict): Config of points loader. Default: dict(
            type='LoadPointsFromFile', load_dim=4, use_dim=[0,1,2,3])
    """

    def __init__(self, img_loader=None, **kwargs) -> None:
        if img_loader is not None:
            self.img_loader = TRANSFORMS.build(img_loader)
        super().__init__(**kwargs)

    @property
    def with_img_loader(self):
        return hasattr(self, 'img_loader') and self.img_loader is not None

    def sample_all(self,
                   gt_bboxes: np.ndarray,
                   gt_labels: np.ndarray,
                   with_points: bool = True,
                   with_img: bool = False,
                   ground_plane: Optional[np.ndarray] = None) -> dict:
        """Sampling all categories of bboxes.

        Args:
            gt_bboxes (np.ndarray): Ground truth bounding boxes.
            gt_labels (np.ndarray): Ground truth labels of boxes.
            with_points (bool): Whether to use the points sample.
            with_img (bool): Whether to use the image sample.
            ground_plane (np.ndarray, optional): Ground plane information.
                Defaults to None.

        Returns:
            dict: Dict of sampled 'pseudo ground truths'.

                - gt_labels_3d (np.ndarray): ground truths labels \
                    of sampled objects.
                - gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): \
                    sampled ground truth 3D bounding boxes
                - points (np.ndarray): sampled points
                - group_ids (np.ndarray): ids of sampled ground truths
        """
        sampled_num_dict = {}
        sample_num_per_class = []

        for class_name, max_sample_num in zip(self.sample_classes,
                                              self.sample_max_nums):
            class_label = self.cat2label[class_name]
            # sampled_num = int(max_sample_num -
            #                   np.sum([n == class_name for n in gt_names]))
            sampled_num = int(max_sample_num -
                              np.sum([n == class_label for n in gt_labels]))
            sampled_num = np.round(self.rate * sampled_num).astype(np.int64)
            sampled_num_dict[class_name] = sampled_num
            sample_num_per_class.append(sampled_num)

        sampled = []
        sampled_gt_bboxes = []
        avoid_coll_boxes = gt_bboxes

        for class_name, sampled_num in zip(self.sample_classes,
                                           sample_num_per_class):
            if sampled_num > 0:
                sampled_cls = self.sample_class_v2(class_name, sampled_num,
                                                avoid_coll_boxes)

                sampled += sampled_cls
                if len(sampled_cls) > 0:
                    if len(sampled_cls) == 1:
                        sampled_gt_box = sampled_cls[0]['box3d_lidar'][
                            np.newaxis, ...]
                    else:
                        sampled_gt_box = np.stack(
                            [s['box3d_lidar'] for s in sampled_cls], axis=0)

                    sampled_gt_bboxes += [sampled_gt_box]
                    avoid_coll_boxes = np.concatenate(
                        [avoid_coll_boxes, sampled_gt_box], axis=0)

        ret = None
        if len(sampled) > 0:
            sampled_gt_bboxes = np.concatenate(sampled_gt_bboxes, axis=0)
            # center = sampled_gt_bboxes[:, 0:3]

            gt_labels = np.array([self.cat2label[s['name']] for s in sampled],
                                 dtype=np.long)

            # num_sampled = len(sampled)
            s_points_list = []
            s_idx_list = []
            s_imgs_list = []
            count = 0
            for info in sampled:
                if with_points:
                    file_path = os.path.join(
                        self.data_root,
                        info['path']) if self.data_root else info['path']
                    results = dict(lidar_points=dict(lidar_path=file_path))
                    s_points = self.points_loader(results)['points']
                    s_points.translate(info['box3d_lidar'][:3])

                    # change below
                    idx_points = count * np.ones(len(s_points), dtype=np.int)
                    s_idx_list.append(idx_points)

                    count += 1

                    s_points_list.append(s_points)
                    point_list = s_points_list[0].cat(s_points_list)
                    points_idx = np.concatenate(s_idx_list, axis=0)
                else:
                    point_list = []
                    points_idx = None

                # add
                if with_img and self.with_img_loader:
                    if len(info['image_path']) > 0:
                        img_path = os.path.join(
                            self.data_root,
                            info['image_path']) if self.data_root else info['image_path']
                        img = dict(img_path=img_path)
                        s_img = self.img_loader(img)['img']
                    else:
                        s_img = []
                    s_imgs_list.append(s_img)

            if ground_plane is not None:
                xyz = sampled_gt_bboxes[:, :3]
                dz = (ground_plane[:3][None, :] *
                      xyz).sum(-1) + ground_plane[3]
                sampled_gt_bboxes[:, 2] -= dz
                if with_points:
                    for i, s_points in enumerate(s_points_list):
                        s_points.tensor[:, 2].sub_(dz[i])

            ret = {
                'gt_labels_3d': gt_labels,
                'gt_bboxes_3d': sampled_gt_bboxes,
                'points': point_list,
                "points_idx": points_idx,
                'images': s_imgs_list,
                'group_ids':
                np.arange(gt_bboxes.shape[0],
                          gt_bboxes.shape[0] + len(sampled))
            }

        return ret

