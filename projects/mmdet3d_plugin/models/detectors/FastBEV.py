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

__all__ = ['FastBEV', 'FastBEVTRT']


@MODELS.register_module()
class FastBEV(CustomBaseDetector):
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
        super().__init__(init_cfg=init_cfg, data_preprocessor=data_preprocessor,
                         input_modality=input_modality,
                         use_grid_mask=use_grid_mask,
                         save2img_cfg=save2img_cfg,
                         img_backbone=backbone, img_neck=neck,
                         pts_bbox_head=bbox_head,
                         train_cfg=train_cfg,
                         test_cfg=test_cfg)
        self.neck_3d = MODELS.build(neck_3d)

        assert isinstance(neck_fuse['in_channels'], list), f"neck_fuse['in_channels'] must be a list"
        assert isinstance(neck_fuse['out_channels'], list), f"neck_fuse['out_channels'] must be a list"
        for i, (in_channels, out_channels) in enumerate(zip(neck_fuse['in_channels'], neck_fuse['out_channels'])):
            self.add_module(
                f'neck_fuse_{i}',
                nn.Conv2d(in_channels, out_channels, 3, 1, 1))

        self.multi_scale_id = multi_scale_id if multi_scale_id is not None else [0]

        if isinstance(bev_size, int):
            bev_size = [bev_size] * 2
        elif isinstance(bev_size, tuple):
            bev_size = list(bev_size)[:2]
        self.bev_size = bev_size
        self.num_points_in_pillar = num_points_in_pillar
        self.num_voxels = bev_size + [num_points_in_pillar]
        self.point_cloud_range = point_cloud_range
        self.voxel_size = [(point_cloud_range[3] - point_cloud_range[0]) / bev_size[0],
                           (point_cloud_range[4] - point_cloud_range[1]) / bev_size[1],
                           (point_cloud_range[5] - point_cloud_range[2]) / num_points_in_pillar]

        # test time extrinsic noise
        self.extrinsic_noise = extrinsic_noise

        self.project_func = self.raw_project if raw_net else self.changed_project

        self.resize_func = partial(resize, mode="bilinear", align_corners=False)
        # backbone scale factor
        self.img_scale_factor = [2**(i + 2) for i in self.img_backbone.out_indices]  # [4, 8, 16, 32]
        self.upsample_factor = [2**i for i in self.img_backbone.out_indices]  # [1, 2, 4, 8]

        # volume scale factor
        self.down_stride = [1, 4 / 3, 2]

        # checkpoint
        self.with_cp = with_cp

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

    def raw_project(self, mlvl_feats: List[Tensor], H: int, W: int,
                    lidar2camera: Tensor, camera_intrinsics: Tensor):
        """mlvl_feats: list([bs, n_cams, c, h, w])"""
        mlvl_volumes = []
        for lvl, lvl_feat in enumerate(mlvl_feats):
            volume_batch = []

            for batch_id, batch_feat in enumerate(lvl_feat):
                height = math.ceil(H / self.img_scale_factor[lvl])
                width = math.ceil(W / self.img_scale_factor[lvl])

                projection = compute_projection(
                    lidar2camera[batch_id], camera_intrinsics[batch_id], self.img_scale_factor[lvl],
                    noise=self.extrinsic_noise).to(batch_feat.device)

                n_voxels_ = torch.tensor(self.num_voxels) / torch.tensor([self.down_stride[lvl], self.down_stride[lvl], 1])
                voxel_size_ = torch.tensor(self.voxel_size) * torch.tensor([self.down_stride[lvl], self.down_stride[lvl], 1])

                points = get_points(  # [3, vx, vy, vz]
                    n_voxels=n_voxels_,
                    voxel_size=voxel_size_,
                    origin=torch.tensor(
                        (np.array(self.point_cloud_range[:3]) + np.array(self.point_cloud_range[3:])) / 2.),
                ).to(batch_feat.device).to(torch.float32)

                volume = backproject_inplace_raw(
                    batch_feat[:, :, :height, :width], points, projection)  # [c, vx, vy, vz]

                volume_batch.append(volume)
            mlvl_volumes.append(torch.stack(volume_batch))  # list([bs, c, vx, vy, vz])

        if len(mlvl_volumes) == 1:
            volume = mlvl_volumes[0]
        else:
            for i, mlvl_volume in enumerate(mlvl_volumes):
                mlvl_volume =  rearrange(mlvl_volume, 'bs c vx vy vz -> bs (c vz) vx vy')

                # upsampling to top level
                if i != 0:
                    mlvl_volume = self.resize_func(mlvl_volume, scale_factor=self.down_stride[i])

                mlvl_volumes[i] = mlvl_volume.unsqueeze(-1)  # [bs, c*vz, vx', vy'] -> [bs, c*vz, vx, vy, 1]
            volume = torch.cat(mlvl_volumes, dim=1)  # [bs, z1*c1+z2*c2+..., vx, vy, 1]

        return volume

    def changed_project(self, mlvl_feats: List[Tensor], H: int, W: int, lidar2camera: Tensor, camera_intrinsics: Tensor):
        device = mlvl_feats[0].device
        points = get_points(n_voxels=torch.tensor(self.num_voxels),
                            voxel_size=torch.tensor(self.voxel_size),
                            origin=torch.tensor(
                                (np.array(self.point_cloud_range[:3]) + np.array(self.point_cloud_range[3:])) / 2.),
        ).to(device).to(torch.float32)  # [3, num_voxels[0], num_voxels[1], num_voxels[2]]

        volume = backproject_inplace(mlvl_feats,
                                     points,
                                     lidar2camera,
                                     camera_intrinsics,
                                     [H, W],
                                     extrinsic_noise=self.extrinsic_noise)  # [bs, c, vx, vy, vz]
        return volume


class ProjectPlugin(torch.autograd.Function):
    output_size = None

    @staticmethod
    def set_output_size(output_size):
        ProjectPlugin.output_size = tuple(output_size)

    @staticmethod
    def symbolic(g, input, projection):
        return g.op("Plugin", input, projection, name_s='Project2Dto3D', info_s='')

    @staticmethod
    def forward(ctx, input: Tensor, projection: Tensor):
        # input: [bs*n_cams, h, w, c]
        # projection: [bs*n_cams, 3, 4]
        # output: [bev_h, bev_w, z, c']
        return torch.ones(ProjectPlugin.output_size)


@MODELS.register_module()
class FastBEVTRT(FastBEV):
    def __init__(self, **kwargs):
        bgr_to_rgb = kwargs['data_preprocessor'].get('rgb_to_bgr', False)
        rgb_to_bgr = kwargs['data_preprocessor'].get('rgb_to_bgr', False)
        self.channel_conversion = rgb_to_bgr or bgr_to_rgb
        super().__init__(**kwargs)
        self.neck_3d_single_input = int(self.neck_3d.fuse.in_channels // self.num_points_in_pillar)
        self.plugin = ProjectPlugin()
        self.plugin.set_output_size((*self.bev_size, self.num_points_in_pillar, self.neck_3d_single_input))
        self.lidar2cam = None
        self.cam2img = None
        self.projection = None
        self.param = None

    def set_cfg(self, cfg: Dict, device: str = 'cuda', dtype: str = torch.float32):
        for k, v in cfg.items():
            setattr(self, k, v)
        if self.lidar2cam is not None and not isinstance(self.lidar2cam, Tensor):
            lidar2cam = np.array(self.lidar2cam)
            self.lidar2cam = torch.from_numpy(lidar2cam).to(device).to(dtype)
        if self.cam2img is not None and not isinstance(self.cam2img, Tensor):
            cam2img = np.array(self.cam2img)
            self.cam2img = torch.from_numpy(cam2img).to(device).to(dtype)
        if self.params is not None:
            param_list = []
            for key, value_list in self.params.items():
                param_list += value_list
            self.param = torch.nn.Parameter(torch.tensor(param_list))  # 3+3+3+36 = 45


    def forward(self, img: Tensor):
        DEBUG_print(f"TRT forward")
        assert img.shape[0] == 1, f"batch size must be equal to 1, but got {img.shape[0]}"
        bs = 1
        if self.channel_conversion:
            img = img[..., [2, 1, 0]]  # convert bgr to rgb
        img = img.permute(0, 1, 4, 2, 3)
        mlvl_feats = self.extract_feat(img)

        if self.DEPLOY:
            # for TRT deploy: [bs*n_cams, h, w, c] -> [bs*n_cams, h, w, c]
            mlvl_feats = [feat.permute(0, 2, 3, 1) for feat in mlvl_feats]
            # TODO: hack: only use index 0
            volume = self.plugin.apply(mlvl_feats[0], self.param)
            volume = rearrange(volume, 'nx ny nz c -> 1 c nx ny nz')
        else:
            volume = self.project_func(mlvl_feats, *self.input_size, self.lidar2cam, self.cam2img)

        feature_bev = self.neck_3d(volume)  # only one

        cls_score, bbox_pred, dir_cls_preds = self.pts_bbox_head(feature_bev)
        num_anchors = [int(self.pts_bbox_head.num_anchors * feature_bev[i].shape[-2] * \
                           feature_bev[i].shape[-1]) for i in range(len(feature_bev))]
        num_classes = self.pts_bbox_head.num_classes
        cls_score = [item.permute(0, 2, 3, 1).reshape(bs, num_anchors[idx], num_classes) for idx, item in enumerate(cls_score)]
        bbox_pred = [item.permute(0, 2, 3, 1).reshape(bs, num_anchors[idx], -1) for idx, item in enumerate(bbox_pred)]
        dir_cls_preds = [item.permute(0, 2, 3, 1).reshape(bs, num_anchors[idx], 2) for idx, item in enumerate(dir_cls_preds)]
        if self.pts_bbox_head.use_sigmoid_cls:
            cls_score = [score.sigmoid() for score in cls_score]
        else:
            cls_score = [score.softmax(-1) for score in cls_score]

        result = torch.cat((cls_score[0], bbox_pred[0], dir_cls_preds[0]), dim=-1)
        DEBUG_print(f"final output shape: {result.shape}")

        return result

def get_extrinsic_intrinsic(batch_input_metas: Dict, result_size: Union[Tuple, List], device: str = 'cpu'):
    input_size, row, col = result_size[:-2], *result_size[-2:]

    lidar2camera = [item['lidar2cam'] for item in batch_input_metas]
    lidar2camera = torch.tensor(np.array(lidar2camera), dtype=torch.float32).to(device)  # [bs, n_cams, 4, 4]
    camera_intrinsics_3x3 = [item['cam2img'] for item in batch_input_metas]
    camera_intrinsics_3x3 = torch.tensor(np.array(camera_intrinsics_3x3), dtype=torch.float32)
    camera_intrinsics = torch.zeros((*input_size, 4, 4))
    camera_intrinsics[..., :3, :3] = camera_intrinsics_3x3
    camera_intrinsics[..., 3, 3] = 1
    camera_intrinsics = camera_intrinsics.to(device)  # [bs, n_cams, 4, 4]
    return lidar2camera, camera_intrinsics[..., :row, :col]

@torch.no_grad()
def get_points(n_voxels: Tensor, voxel_size: Tensor, origin: Tensor):
    points = torch.stack(torch.meshgrid([torch.arange(n_voxels[0]),
                                         torch.arange(n_voxels[1]),
                                         torch.arange(n_voxels[2]),]))
    new_origin = origin - n_voxels / 2.0 * voxel_size
    points = points * voxel_size.view(-1, 1, 1, 1) + new_origin.view(-1, 1, 1, 1)
    # points: [3, n_voxels[0], n_voxels[1], n_voxels[2]]
    return points

# def compute_projection(extrinsics: Tensor, intrinsics: Tensor, stride: Union[float, int], noise: float = 0.):
#     # extrinsics: [bs, 6, 4, 4]
#     # intrinsics: [bs, 6, 4, 4]
#     bs = extrinsics.size(0)
#     extrinsics = extrinsics.reshape(-1, 4, 4)
#     intrinsics = intrinsics.reshape(-1, 4, 4)
#
#     intrinsic_extrinsic = compute_projection_single_batch(extrinsics, intrinsics, stride, noise=noise)
#     intrinsic_extrinsic = intrinsic_extrinsic.reshape(bs, -1, 4, 4)
#
#     return intrinsic_extrinsic

def compute_projection(extrinsics: Tensor, intrinsics: Tensor, stride: Union[float, int], noise: float = 0.):
    # extrinsics: [*, 4, 4]
    # intrinsics: [*, 4, 4]
    extrinsics, extrinsics_ps = pack([extrinsics], '* d1 d2')
    intrinsics, intrinsics_ps = pack([intrinsics], '* d1 d2')

    intrinsics[:, :2] /= stride
    noise = noise if noise > 0 else 0

    intrinsic_extrinsic = einsum(intrinsics, extrinsics, 'n d1 d2, n d2 d3 -> n d1 d3') + noise

    intrinsic_extrinsic = unpack(intrinsic_extrinsic, extrinsics_ps, '* d1 d2')[0]
    return intrinsic_extrinsic

def backproject_inplace(mlvl_feats: List[Tensor], points: Tensor, lidar2camera: Tensor, camera_intrinsics: Tensor,
                        img_shape: Union[List, Tuple], extrinsic_noise: float = 0.0):
    """
    function: 2d feature + predefined point cloud -> 3d volume
    Args:
        mlvl_feats: list  [bs, n_cams, c, h, w]
        points: [3, vx, vy, vz]
        lidar2camera: [bs, n_cams, 4, 4]
        camera_intrinsics: [bs, n_cams, 4, 4]
        img_shape:
        extrinsic_noise: float
    Return:
        volume (Tensor): [bs, c, vx, vy, vz]
    """
    bs, n_imgs, n_channels, _, _ = map(int, mlvl_feats[0].shape)
    n_x_voxels, n_y_voxels, n_z_voxels = map(int, points.shape[-3:])
    num_voxels = n_x_voxels * n_y_voxels * n_z_voxels
    device = mlvl_feats[0].device

    volume = torch.zeros((bs, n_channels, num_voxels), device=device).type_as(mlvl_feats[0])
    points = repeat(points, 'd x y z -> bs n_imgs d (x y z)', bs=bs, n_imgs=n_imgs)
    points = torch.cat((points, torch.ones_like(points[:, :, :1])), dim=2)

    for lvl, mlvl_feat in enumerate(mlvl_feats):
        height, width = mlvl_feat.shape[-2], mlvl_feat.shape[-1]
        stride_i = math.ceil(img_shape[-1] / mlvl_feat.shape[-1])

        projection = compute_projection(lidar2camera,
                                        camera_intrinsics,
                                        stride_i,
                                        noise=extrinsic_noise).to(device)
        points_2d_3 = einsum(projection, points, 'bs n_imgs d1 d2, bs n_imgs d2 n -> bs n_imgs d1 n')

        eps = 1e-5
        valid = (points_2d_3[:, :, 2] > eps)

        points_2d_3 = points_2d_3[:, :, 0:2] / torch.maximum(
                points_2d_3[:, :, 2:3], torch.ones_like(points_2d_3[:, :, 2:3]) * eps)
        x = points_2d_3[:, :, 0].round().long()  # [bs, n_imgs, num_voxels]
        y = points_2d_3[:, :, 1].round().long()

        valid = (valid & (x >= 0) & (y >= 0) & (x < width) & (y < height))  # [bs, n_imgs, num_voxels]
        camera_voxel_intersection = repeat(valid.sum(1, keepdim=True), 'bs n_voxel-> bs n_imgs n_voxel', n_imgs=n_imgs)
        cross_flag = camera_voxel_intersection > 1
        valid = (valid & ~cross_flag)
        cross_valid = (valid & cross_flag)

        # non-cross area
        for bs_index in range(bs):
            for cam in range(n_imgs):
                volume[bs_index, :, valid[bs_index, cam]] += mlvl_feat[bs_index, cam,
                                                                       :,
                                                                       y[bs_index, cam, valid[bs_index, cam]],
                                                                       x[bs_index, cam, valid[bs_index, cam]]]
        # cross area
        for bs_index in range(bs):
            for cam in range(n_imgs):
                volume[bs_index, :, cross_valid[bs_index, cam]] += mlvl_feat[bs_index, cam,
                                                                             :,
                                                                             y[bs_index, cam, cross_valid[bs_index, cam]],
                                                                             x[bs_index, cam, cross_valid[bs_index, cam]]] / camera_voxel_intersection[bs_index, cam, cross_valid[bs_index, cam]]

    volume = rearrange(volume, 'bs c (x y z) -> bs c x y z') / len(mlvl_feats)
    return volume

def backproject_inplace_raw(features: Tensor, points: Tensor, projection: Tensor):
    """
    function: 2d feature + predefined point cloud -> 3d volume
    Args:
        features: [bs, n_cams, c, h, w]
        points: [3, vx, vy, vz]
        projection: [n_cams, 4, 4]
    Return:
        volume (Tensor): [c, vx, vy, vz]
    """
    n_imgs, n_channels, height, width = map(int, features.shape)  # [n_imgs, c, h, w]
    n_x_voxels, n_y_voxels, n_z_voxels = map(int, points.shape[-3:])
    points = repeat(points, 'd x y z -> n_imgs d (x y z)', n_imgs=n_imgs)
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    points_2d_3 = einsum(projection, points, 'n_imgs d1 d2, n_imgs d2 n_points -> n_imgs d1 n_points')
    x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long()
    y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()
    z = points_2d_3[:, 2]
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)

    # Feature filling, only valid features are filled, and duplicate features are directly overwritten
    volume = torch.zeros((n_channels, points.shape[-1]), device=features.device).type_as(features)
    for i in range(n_imgs):
        volume[:, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]]

    volume = rearrange(volume, 'c (x y z) -> c x y z', x=n_x_voxels, y=n_y_voxels, z=n_z_voxels)
    return volume


def resize(input_img,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input_img.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > input_w:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input_img, size, scale_factor, mode, align_corners)