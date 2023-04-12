from typing import Dict, List, Optional, Sequence, Any, Union, Tuple
from collections import defaultdict

import numpy as np
from einops import rearrange, repeat, einsum, pack, unpack
import torch
from torch import Tensor

from mmengine.logging import print_log
import mmdet3d
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes, limit_period

from projects.mmdet3d_plugin.models.structures import InputDataClass
from projects.mmdet3d_plugin.models.utils import GridMask, PointCloudImgSaver


__all__ = ['CustomBaseDetector']


@MODELS.register_module()
class CustomBaseDetector(MVXTwoStageDetector):
    """
    Args:
        pts_voxel_encoder (dict, optional): Point voxelization
            encoder layer. Defaults to None.
        pts_middle_encoder (dict, optional): Middle encoder layer
            of points cloud modality. Defaults to None.
        pts_fusion_layer (dict, optional): Fusion layer.
            Defaults to None.
        img_backbone (dict, optional): Backbone of extracting
            images feature. Defaults to None.
        pts_backbone (dict, optional): Backbone of extracting
            points features. Defaults to None.
        img_neck (dict, optional): Neck of extracting
            image features. Defaults to None.
        pts_neck (dict, optional): Neck of extracting
            points features. Defaults to None.
        pts_bbox_head (dict, optional): Bboxes head of
            point cloud modality. Defaults to None.
        img_roi_head (dict, optional): RoI head of image
            modality. Defaults to None.
        img_rpn_head (dict, optional): RPN head of image
            modality. Defaults to None.
        train_cfg (dict, optional): Train config of model.
            Defaults to None.
        test_cfg (dict, optional): Train config of model.
            Defaults to None.
        init_cfg (dict, optional): Initialize config of
            model. Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`Det3DDataPreprocessor`. Defaults to None.
    """

    def __init__(self,
                 input_modality: Optional[Dict[str, bool]] = None,
                 use_grid_mask: bool = False,
                 img_backbone: Optional[Dict] = None,
                 img_neck: Optional[Dict] = None,
                 img_neck_fuser: Optional[Dict] = None,
                 pts_voxel_encoder: Optional[Dict] = None,
                 pts_middle_encoder: Optional[Dict] = None,
                 pts_backbone: Optional[Dict] = None,
                 pts_neck: Optional[Dict] = None,
                 pts_neck_fuser: Optional[Dict] = None,
                 radar_backbone: Optional[Dict] = None,
                 radar_neck: Optional[Dict] = None,
                 radar_neck_fuser: Optional[Dict] = None,
                 pts_fusion_layer: Optional[Dict] = None,
                 bev_backbone: Optional[Dict] = None,
                 bev_neck: Optional[Dict] = None,
                 pts_bbox_head: Optional[Dict] = None,
                 save2img_cfg: Optional[Dict] = None,
                 train_cfg: Optional[Dict] = None,
                 test_cfg: Optional[Dict] = None,
                 init_cfg: Optional[Dict] = None,
                 data_preprocessor: Optional[Dict] = None,
                 **kwargs):
        if input_modality is None:
            input_modality = dict(use_lidar=False,
                                  use_camera=True,
                                  use_radar=False,
                                  use_map=False,
                                  use_external=False)

        if save2img_cfg is not None and save2img_cfg['plot_examples'] > 0:
            # save2img_cfg have key: plot_examples, plot_range, save_dir, save_only_master
            self.saver = PointCloudImgSaver(data_preprocessor=data_preprocessor, **save2img_cfg)
            self.save_batch_gt = 0

        super(CustomBaseDetector, self).__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor,
            img_backbone=img_backbone, img_neck=img_neck,
            pts_voxel_encoder=pts_voxel_encoder, pts_middle_encoder=pts_middle_encoder,
            pts_backbone=pts_backbone, pts_neck=pts_neck, pts_fusion_layer=pts_fusion_layer,
            pts_bbox_head=pts_bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            **kwargs)

        self.use_grid_mask = use_grid_mask
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)

        self.use_lidar = input_modality['use_lidar']
        self.use_camera = input_modality['use_camera']
        self.use_radar = input_modality['use_radar']
        assert self.use_lidar or self.use_camera, f"must use at least one of 'lidar' or 'camera "

        # img branch add
        if img_neck_fuser is not None:
            self.img_neck_fuser = MODELS.build(img_neck_fuser)

        # pts branch add
        if pts_neck_fuser is not None:
            self.pts_neck_fuser = MODELS.build(pts_neck_fuser)

        # radar branch
        if self.use_radar:
            if radar_backbone is not None:
                self.radar_backbone = MODELS.build(radar_backbone)
            if radar_neck is not None:
                self.radar_neck = MODELS.build(radar_neck)
            if radar_neck_fuser is not None:
                self.radar_neck_fuser = MODELS.build(radar_neck_fuser)

        # BEV branch
        if bev_backbone is not None:
            self.bev_backbone = MODELS.build(bev_backbone)
        if bev_neck is not None:
            self.bev_neck = MODELS.build(bev_neck)

        img_branch = []
        pts_branch = []
        branch_freeze = []
        self.freeze_branch(branch_freeze)

    @property
    def with_img_neck_fuser(self):
        """bool: Whether the detector has a neck fuser in image branch."""
        return hasattr(self, 'img_neck_fuser') and self.img_neck_fuser is not None

    @property
    def with_img_depth_head(self):
        """bool: Whether the detector has a depth predict head in image branch."""
        return hasattr(self, 'img_depth_head') and self.img_depth_head is not None

    @property
    def with_cam_vtransform(self):
        """bool: Whether the detector has a cam_vtransform in image branch."""
        return hasattr(self, 'cam_vtransform') and self.cam_vtransform is not None

    @property
    def with_pts_neck_fuser(self):
        """bool: Whether the detector has a neck fuser in pts branch."""
        return hasattr(self, 'pts_neck_fuser') and self.pts_neck_fuser is not None

    @property
    def with_radar_backbone(self):
        """bool: Whether the detector has a radar backbone"""
        return hasattr(self, 'radar_backbone') and self.radar_backbone is not None

    @property
    def with_radar_neck(self):
        """bool: Whether the detector has a neck in radar branch."""
        return hasattr(self, 'radar_neck') and self.radar_neck is not None

    @property
    def with_radar_neck_fuser(self):
        """bool: Whether the detector has a neck fuser in radar branch."""
        return hasattr(self, 'radar_neck_fuser') and self.radar_neck_fuser is not None

    @property
    def with_bev_backbone(self):
        return hasattr(self, 'bev_backbone') and self.bev_backbone is not None

    @property
    def with_bev_neck(self):
        return hasattr(self, 'bev_neck') and self.bev_neck is not None

    def freeze_branch(self, branch_name=None):
        if isinstance(branch_name, str):
            branch_name = [branch_name]
        assert isinstance(branch_name, list)
        for name in branch_name:
            if hasattr(self, name):
                print_log(f"Frozen branch: {name}", 'current')
                for param in getattr(self, name).parameters():
                    param.requires_grad = False

    def extract_img_feat(self, img: Optional[Tensor] = None, batch_input_metas: Optional[List[dict]] = None) -> Optional[List]:
        """
        Extract features of images.

        Args:
            img (Tensor, optional): raw img dara with shape of [bs, num_cams, c, h, w]. Defaults to None.
            batch_input_metas: (List[dict], optional)
        Returns:
            List of Tensor ([bs, num_cams, C, H, W]) or [None]
        """
        if not self.use_camera or not self.with_img_backbone or img is None:
            return [None]

        batch_image, ps = pack([img], '* c h w')

        if self.use_grid_mask:
            batch_image = self.grid_mask(batch_image)

        mlvl_img_feats = self.img_backbone(batch_image)
        # [B, C, H, W] -> [(256, H//4), (512, H//8), (1024, H//16), (2048, H//32)]

        if self.with_img_neck:
            mlvl_img_feats = self.img_neck(mlvl_img_feats)

        if self.with_img_neck_fuser:
            mlvl_img_feats = self.img_neck_fuser(mlvl_img_feats)

        mlvl_img_feats = [unpack(img_feat, ps, '* c h w')[0] for img_feat in mlvl_img_feats]
        return mlvl_img_feats

    def extract_pts_feat(
            self,
            voxel_dict: Dict[str, Tensor],
            points: Optional[List[Tensor]] = None,
            img_feats: Optional[Sequence[Tensor]] = None,
            batch_input_metas: Optional[List[dict]] = None
    ) -> Sequence[Tensor]:
        """Extract features of points.

        Args:
            voxel_dict(Dict[str, Tensor]): Dict of voxelization infos.
            points (List[tensor], optional):  Point cloud of multiple inputs.
            img_feats (list[Tensor], tuple[tensor], optional): Features from
                image backbone.
            batch_input_metas (list[dict], optional): The meta information
                of multiple samples. Defaults to None.
        Returns:
            List of Tensor ([bs, C, H, W]) or [None]
        """
        if not self.use_lidar or not self.with_pts_backbone or voxel_dict is None:
            return [None]

        voxel_features = self.pts_voxel_encoder(voxel_dict['voxels'],
                                                voxel_dict['num_points'],
                                                voxel_dict['coors'],
                                                batch_input_metas)
        batch_size = voxel_dict['coors'][-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, voxel_dict['coors'], batch_size)
        mlvl_pts_feats = self.pts_backbone(x)
        if self.with_pts_neck:
            mlvl_pts_feats = self.pts_neck(mlvl_pts_feats)

        # List ([bs, C, H, W])
        if self.with_pts_neck_fuser:
            mlvl_pts_feats = self.pts_neck_fuser(mlvl_pts_feats)

        return mlvl_pts_feats  # List([bs, C, H, W])    y,x

    def extract_radar_feat(self, radars: Optional[Tensor] = None) -> Optional[List]:
        """
        Extract features of radars.

        Args:
            radars (Tensor, optional): raw radar dara with shape of [bs, n_points, n_dims]  .Defaults to None.
        Returns:
            List of Tensor ([bs, C, H, W]) or [None]
        """
        if not self.use_radar or not self.with_radar_backbone or radars is None:
            return [None]

        assert radars.dim() == 3, f"Radar points must be a tensor like [bs, n_points, n_dims], but get {radars.size()}"

        mlvl_radar_feats = self.radar_backbone(radars)

        if self.with_radar_neck:
            mlvl_radar_feats = self.radar_neck(mlvl_radar_feats)
        if self.with_radar_neck_fuser:
            mlvl_radar_feats = self.radar_neck_fuser(mlvl_radar_feats)

        return mlvl_radar_feats  # [B, C, H, W]


    def extract_feat(self,
                     batch_inputs_dict: dict,
                     batch_input_metas: List[dict]) -> tuple:
        """Extract features from images, points and radars.

        Args:
            batch_inputs_dict (dict): Dict of batch inputs. It
                contains
                - imgs (tensor): Image tensor with shape (B, N, C, H, W).
                - points (List[tensor]):  Point cloud of multiple inputs.
                - radar_points (List[tensor]): Radar Point cloud of multiple inputs.
            batch_input_metas (list[dict]): Meta information of multiple inputs
                in a batch.

        Returns:
             dict: key: mlvl_img_feats, mlvl_pts_feats, mlvl_rad_feats
        """
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        voxel_dict = batch_inputs_dict.get('voxels', None)
        radar_points = batch_inputs_dict.get('radar_points', None)

        mlvl_feats = dict()
        mlvl_feats['mlvl_img_feats'] = self.extract_img_feat(imgs, batch_input_metas)

        mlvl_feats['mlvl_pts_feats'] = self.extract_pts_feat(
            voxel_dict,
            points=points,
            batch_input_metas=batch_input_metas)

        mlvl_feats['mlvl_rad_feats'] = self.extract_radar_feat(radar_points)

        return mlvl_feats

    def net_forward(
            self,
            batch_inputs_dict: Union[dict, List[dict]],
            batch_data_samples: List[Det3DDataSample],
            **kwargs):
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        mlvl_feats = self.extract_feat(batch_inputs_dict, batch_input_metas)

        return mlvl_feats

    def loss(self, batch_inputs_dict: Dict[List, torch.Tensor],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        """
        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' and `imgs` keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - imgs (torch.Tensor): Tensor of batch images, has shape
                  (B, C, H ,W)
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, .

        Returns:
            dict[str, Tensor]: A dictionary of loss components.

        """
        # 'ori_shape':
        # 'pad_shape':
        if hasattr(self, 'saver'):
            self.saver.add_data_sample_batch(batch_inputs_dict, batch_data_samples, save_batch=self.save_batch_gt)

        mlvl_feats = self.net_forward(batch_inputs_dict, batch_data_samples, **kwargs)

        losses = dict()
        losses_pts = self.pts_bbox_head.loss(mlvl_feats, batch_data_samples, **kwargs)
        losses.update(losses_pts)
        return losses

    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        """Forward of testing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input sample. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
                (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - bbox_3d (:obj:`BaseInstance3DBoxes`): Prediction of bboxes,
                contains a tensor with shape (num_instances, 7).
        """
        mlvl_feats = self.net_forward(batch_inputs_dict, batch_data_samples, **kwargs)

        results_list_3d = self.pts_bbox_head.predict(mlvl_feats, batch_data_samples, **kwargs)

        detsamples = self.add_pred_to_datasample(batch_data_samples, data_instances_3d=results_list_3d)
        if hasattr(self, 'saver'):
            self.saver.add_data_sample_batch(batch_inputs_dict, detsamples, save_batch=0)

        return detsamples

    @staticmethod
    def get_lidar2img(batch_input_metas, list_type='torch', device='cuda'):
        """
        Args:
            batch_input_metas (list[dict]): Meta information of multiple inputs in a batch.
            device: str
            list_type: str, 'torch' or 'numpy'
        Returns:
            lidar2imgs: List of  [n_cams, 4, 4]
            imgs2lidar: List of  [n_cams, 4, 4]
        """
        batch_lidar2cam = [np.stack(input_meta['lidar2cam']) for input_meta in batch_input_metas]  # [6*4*4]
        batch_cam2img = [np.stack(input_meta['cam2img']) for input_meta in batch_input_metas]  # [6*3*3]

        lidar2imgs = []
        imgs2lidar = []
        for lidar2cam, cam2img in zip(batch_lidar2cam, batch_cam2img):
            viewpad = np.zeros((6, 4, 4))
            viewpad[:, :cam2img.shape[-2], :cam2img.shape[-1]] = cam2img
            viewpad[:, 3, 3] = 1
            lidar2img_rt = einsum(viewpad, lidar2cam, 'n a b, n b c -> n a c')
            img2lidar_rt = np.linalg.inv(lidar2img_rt)
            if list_type == 'torch':
                lidar2img = torch.from_numpy(lidar2img_rt).float().to(device)
                img2lidar = torch.from_numpy(img2lidar_rt).float().to(device)
            else:
                lidar2img = lidar2img_rt.astype(np.float32)
                img2lidar = img2lidar_rt.astype(np.float32)
            lidar2imgs.append(lidar2img)
            imgs2lidar.append(img2lidar)

        return lidar2imgs, imgs2lidar


    @staticmethod
    def LidarBox3dVersionTransform(gt_bboxes_3d):
        if int(mmdet3d.__version__[0]) >= 1:
            # Begin hack adaptation to mmdet3d v1.0 ####
            gt_bboxes_3d = gt_bboxes_3d[0].tensor

            gt_bboxes_3d[:, [3, 4]] = gt_bboxes_3d[:, [4, 3]]
            gt_bboxes_3d[:, 6] = -gt_bboxes_3d[:, 6] - np.pi / 2
            gt_bboxes_3d[:, 6] = limit_period(
                gt_bboxes_3d[:, 6], period=np.pi * 2)

            gt_bboxes_3d = LiDARInstance3DBoxes(gt_bboxes_3d, box_dim=9)
            gt_bboxes_3d = [gt_bboxes_3d]
        return gt_bboxes_3d
