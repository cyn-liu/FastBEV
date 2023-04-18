import math
import os
import numpy as np
import warnings
from typing import Dict, List, Optional, Sequence, Union, Tuple
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import copy
from functools import partial
from einops import rearrange, repeat, einsum, pack, unpack

from mmdet3d.registry import MODELS
from mmdet3d.models.detectors import Base3DDetector
from mmdet3d.structures import Det3DDataSample

from projects.mmdet3d_plugin.models.detectors import CustomBaseDetector

from tools.utils.debug_print import DEBUG_print
from ..detectors.FastBEV import FastBEV

__all__ = ['FastBEVSeg']


# TODO


@MODELS.register_module()
class FastBEVSeg(FastBEV, ***SegHead***):
    def __init__(
            self,
            input_modality: Optional[Dict[str, bool]] = None,
            use_grid_mask: bool = False,
            save2img_cfg: Optional[Dict] = None,
            backbone: Optional[Dict] = None,
            neck: Optional[Dict] = None,
            neck_fuse: Optional[Dict] = None,
            neck_3d: Optional[Dict] = None,
            bbox_head: Optional[Dict] = None,
            bev_size: Union[Tuple, List] = [200, 200],
            point_cloud_range: Optional[List] = None,
            num_points_in_pillar: int = 8,
            train_cfg: Optional[Dict] = None,
            test_cfg: Optional[Dict] = None,
            init_cfg: Optional[Dict] = None,
            extrinsic_noise: float = 0.,
            multi_scale_id: Optional[List] = None,
            raw_net: bool = True,
            data_preprocessor: Optional[dict] = None,
            with_cp: bool = False,
    ):
        super().FastBEV.__init__(init_cfg=init_cfg, data_preprocessor=data_preprocessor,
                         input_modality=input_modality,
                         use_grid_mask=use_grid_mask,
                         save2img_cfg=save2img_cfg,
                         img_backbone=backbone, img_neck=neck,
                         pts_bbox_head=bbox_head,
                         train_cfg=train_cfg,
                         test_cfg=test_cfg)


    def extract_feat(self, img: Tensor):
        def _inner_forward(feat):
            out = self.img_neck(feat)
            return out

        batch_image, image_ps = pack([img], '* c h w')

        # backbone and neck
        x = self.img_backbone(batch_image)  # [(256, H//4), (512, H//8), (1024, H//16), (2048, H//32)]
        if self.with_cp and x.requires_grad:
            mlvl_feats = cp.checkpoint(_inner_forward, x)
        else:
            mlvl_feats = _inner_forward(x)
        mlvl_feats = list(mlvl_feats)

        mlvl_feats_ = []
        # fpn output fusion
        for msid in self.multi_scale_id:
            fuse_feats = [mlvl_feats[msid]] + [self.resize_func(mlvl_feats[i], scale_factor=self.upsample_factor[i]) \
                                               for i in range(msid + 1, len(mlvl_feats))]
            fuse_feats = torch.cat(fuse_feats, dim=1)
            fuse_feats = getattr(self, f'neck_fuse_{msid}')(fuse_feats)
            mlvl_feats_.append(fuse_feats)  # [(256, H//4), (512, H//8), (1024, H//16)]


        if not (hasattr(self, 'DEPLOY') and self.DEPLOY):
            mlvl_feats = [unpack(feats, image_ps, '* c h w')[0] for feats in mlvl_feats_]
            return mlvl_feats  # [bs, n_cams, c, h, w]
        else:
            return mlvl_feats_  # [bs*n_cams, c, h, w]

    def net_forward(self,
            batch_inputs_dict: Union[dict, List[dict]],
            batch_data_samples: List[Det3DDataSample],
            **kwargs):
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        img = batch_inputs_dict.get('imgs', None)
        batch_size, n_cams, C, H, W = map(int, img.shape)
        device = img.device
        intrinsics_size = [batch_size, n_cams, 4, 4]
        lidar2camera, camera_intrinsics = get_extrinsic_intrinsic(batch_input_metas, intrinsics_size, device)
        torch.set_printoptions(sci_mode=False)

        mlvl_feats = self.extract_feat(img)
        volume = self.project_func(mlvl_feats, H, W, lidar2camera, camera_intrinsics)
        bev_feat = self.neck_3d(volume)
        return bev_feat

