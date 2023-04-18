from typing import Dict, List, Optional, Union, Tuple
from mmcv.cnn import ConvModule
import torch
from torch import Tensor
from torch import nn as nn
from torch.nn import functional as F
from mmdet3d.structures import Det3DDataSample
import torch.utils.checkpoint as cp
from mmdet3d.models.segmentors.base import Base3DSegmentor
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig

from mmdet3d.registry import MODELS

__all__ = ['CustomSeg3DHead']


@MODELS.register_module()
class CustomSeg3DHead(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 class_num,
                 num_layers=2,
                 norm_cfg=dict(type='BN2d'),
                 stride=2,
                 is_transpose=True,
                 with_cp=False,
                 loss_cls: Optional[Dict] = None) -> None:
        super().__init__()

        self.is_transpose = is_transpose
        self.with_cp = with_cp

        model = nn.ModuleList()
        model.append(ConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU', inplace=True)))
        for i in range(num_layers):
            model.append(ConvModule(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU', inplace=True)))
        model.append(ConvModule(
            in_channels=out_channels,
            out_channels=class_num,
            kernel_size=3,
            stride=stride,
            padding=1,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU', inplace=True)))
        self.model = nn.Sequential(*model)

    def _init_decode_head(self, decode_head: Optional[Dict]) -> None:
        """Initialize ``decode_head``"""
        self.decode_head = MODELS.build(decode_head)
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head: Optional[Dict]) -> None:
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)

    def _init_loss_regularization(self,
                                  loss_regularization: Optional[Dict]) -> None:
        """Initialize ``loss_regularization``"""
        if loss_regularization is not None:
            if isinstance(loss_regularization, list):
                self.loss_regularization = nn.ModuleList()
                for loss_cfg in loss_regularization:
                    self.loss_regularization.append(MODELS.build(loss_cfg))
            else:
                self.loss_regularization = MODELS.build(loss_regularization)

    def predict(self,
                mlvl_feats: Tensor,
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        """Simple test with single scene.

        Args:
            batch_inputs_dict (dict): Input sample dict which
                includes 'points' and 'imgs' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - imgs (torch.Tensor, optional): Image tensor has shape
                    (B, C, H, W).
            batch_data_samples (list[:obj:`Det3DDataSample`]): The det3d
                data samples. It usually includes information such
                as `metainfo` and `gt_pts_sem_seg`.
            rescale (bool): Whether transform to original number of points.
                Will be used for voxelization based segmentors.
                Defaults to True.

        Returns:
            list[dict]: The output prediction result with following keys:

                - semantic_mask (Tensor): Segmentation mask of shape [N].
        """
        # 3D segmentations requires per-point prediction, so it's impossible
        # to use down-sampling to get a batch of scenes with same num_points
        # therefore, we only support testing one scene every time
        seg_pred_list = self.model.forward(mlvl_feats)

        return self.postprocess_result(seg_pred_list, batch_data_samples)

    def _decode_head_forward_train(self, mlvl_feats: Tensor,
             batch_data_samples: List[Det3DDataSample]):
        loss = 0
        for _, feats in enumerate(mlvl_feats):
            seg_pred_list = self.model.forward(feats)
            for bs, (seg_pred, data_samples) in enumerate(zip(seg_pred_list, batch_data_samples)):
                gt_mask = torch.from_numpy(data_samples.gt_pts_seg['seg_map']).float().to(seg_pred.device)
                loss += F.binary_cross_entropy(
                    seg_pred.sigmoid(), gt_mask)
        return loss

    def loss(self, mlvl_feats: Tensor,
             batch_data_samples: List[Det3DDataSample]) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs_dict (dict): Input sample dict which
                includes 'points' and 'imgs' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - imgs (torch.Tensor, optional): Image tensor has shape
                  (B, C, H, W).
            batch_data_samples (list[:obj:`Det3DDataSample`]): The det3d
                data samples. It usually includes information such
                as `metainfo` and `gt_pts_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components.
        """

        # extract features using backbone
        loss_decode = self._decode_head_forward_train(mlvl_feats, batch_data_samples)
        losses = {
            'seg_loss': loss_decode
        }

        # if self.with_auxiliary_head:
        #     loss_aux = self._auxiliary_head_forward_train(
        #         mlvl_feats, batch_data_samples)
        #     losses.update(loss_aux)
        #
        # if self.with_regularization_loss:
        #     loss_regularize = self._loss_regularization_forward_train()
        #     losses.update(loss_regularize)

        return losses