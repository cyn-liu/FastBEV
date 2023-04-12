# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

import mmdet3d
from mmdet3d.structures.bbox_3d.utils import limit_period


def normalize_bbox(bboxes, pc_range):
    """9 -> 10 (rot -> rot_sine, rot_cos)"""
    # x, y, z, l, w, h, rot.sin(), rot.cos(), vx, vy
    x = bboxes[..., 0:1]
    y = bboxes[..., 1:2]
    z = bboxes[..., 2:3]
    length = bboxes[..., 3:4].log()
    w = bboxes[..., 4:5].log()
    h = bboxes[..., 5:6].log()

    rot = bboxes[..., 6:7]
    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8]
        vy = bboxes[..., 8:9]
        normalized_bboxes = torch.cat(
            (x, y, z, length, w, h, rot.sin(), rot.cos(), vx, vy), dim=-1)
    else:
        normalized_bboxes = torch.cat(
            (x, y, z, length, w, h, rot.sin(), rot.cos()), dim=-1)
    return normalized_bboxes


def denormalize_bbox(normalized_bboxes, pc_range):
    """10 -> 9 (rot_sine, rot_cos -> rot)"""
    # x, y, z, l, w, h, rot, vx, vy
    rot_sine = normalized_bboxes[..., 6:7]

    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    # center in the bev
    x = normalized_bboxes[..., 0:1]
    y = normalized_bboxes[..., 1:2]
    z = normalized_bboxes[..., 2:3]

    # size
    length = normalized_bboxes[..., 3:4]
    w = normalized_bboxes[..., 4:5]
    h = normalized_bboxes[..., 5:6]

    length = length.exp()
    w = w.exp()
    h = h.exp()
    if normalized_bboxes.size(-1) > 8:
        # velocity
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]
        denormalized_bboxes = torch.cat(
            [x, y, z, length, w, h, rot, vx, vy], dim=-1)
    else:
        denormalized_bboxes = torch.cat([x, y, z, length, w, h, rot], dim=-1)

    return denormalized_bboxes
