import math
from typing import Dict, List, Optional, Union, Tuple

import torch
from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.structures import bbox_overlaps_nearest_3d
from mmdet3d.utils import InstanceList, OptInstanceList
from mmdet3d.models.dense_heads.train_mixins import get_direction_target
from mmdet3d.models.dense_heads.free_anchor3d_head import FreeAnchor3DHead
from torch.cuda.amp import autocast

__all__ = ['CustomFreeAnchor3DHead']
@MODELS.register_module()
class CustomFreeAnchor3DHead(FreeAnchor3DHead):

    def get_anchors_trt(self,device: str = 'cuda'):
        featmap_sizes = [[100,100]]
        anchor_list = self.prior_generator.grid_anchors(featmap_sizes, device=device)
        self.anchors = torch.cat(anchor_list)


    def decode(self, deltas):
        anchors = self.anchors
        cas, cts = [], []
        box_ndim = anchors.shape[-1]
        if box_ndim > 7:
            xa, ya, za, wa, la, ha, ra, *cas = torch.split(anchors, 1, dim=-1)
            xt, yt, zt, wt, lt, ht, rt, *cts = torch.split(deltas, 1, dim=-1)
        else:
            xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim=-1)
            xt, yt, zt, wt, lt, ht, rt = torch.split(deltas, 1, dim=-1)

        za = za + ha / 2
        diagonal = torch.sqrt(la**2 + wa**2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * ha + za

        lg = torch.exp(lt) * la
        wg = torch.exp(wt) * wa
        hg = torch.exp(ht) * ha
        rg = rt + ra
        zg = zg - hg / 2
        cgs = [t + a for t, a in zip(cts, cas)]
        return torch.cat([xg, yg, zg, wg, lg, hg, rg, *cgs], dim=-1)

    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            dir_cls_preds: List[Tensor],
            batch_gt_instances_3d: InstanceList,
            batch_input_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> Dict:
        """Calculate loss of FreeAnchor head.

        Args:
            cls_scores (list[torch.Tensor]): Classification scores of
                different samples.
            bbox_preds (list[torch.Tensor]): Box predictions of
                different samples
            dir_cls_preds (list[torch.Tensor]): Direction predictions of
                different samples
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instances. It usually includes ``bboxes_3d`` and
                ``labels_3d`` attributes.
            batch_input_metas (list[dict]): Contain pcd and img's meta info.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, torch.Tensor]: Loss items.

                - positive_bag_loss (torch.Tensor): Loss of positive samples.
                - negative_bag_loss (torch.Tensor): Loss of negative samples.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        anchor_list = self.get_anchors(featmap_sizes, batch_input_metas)
        mlvl_anchors = [torch.cat(anchor) for anchor in anchor_list]

        # concatenate each level
        cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(
                cls_score.size(0), -1, self.num_classes)
            for cls_score in cls_scores
        ]
        bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(
                bbox_pred.size(0), -1, self.box_code_size)
            for bbox_pred in bbox_preds
        ]
        dir_cls_preds = [
            dir_cls_pred.permute(0, 2, 3,
                                 1).reshape(dir_cls_pred.size(0), -1, 2)
            for dir_cls_pred in dir_cls_preds
        ]

        cls_scores = torch.cat(cls_scores, dim=1)
        bbox_preds = torch.cat(bbox_preds, dim=1)
        dir_cls_preds = torch.cat(dir_cls_preds, dim=1)

        cls_probs = torch.sigmoid(cls_scores)
        box_prob = []
        num_pos = 0
        positive_losses = []
        for _, (anchors, gt_instance_3d, cls_prob, bbox_pred,
                dir_cls_pred) in enumerate(
                    zip(mlvl_anchors, batch_gt_instances_3d, cls_probs,
                        bbox_preds, dir_cls_preds)):

            gt_bboxes = gt_instance_3d.bboxes_3d.tensor.to(anchors.device)
            gt_labels = gt_instance_3d.labels_3d.to(anchors.device)
            with torch.no_grad():
                pred_boxes = self.bbox_coder.decode(anchors, bbox_pred)  # [n_pred, 4]

                object_box_iou = bbox_overlaps_nearest_3d(gt_bboxes, pred_boxes)  # [n_gt, n_pred]

                t1 = self.bbox_thr
                t2 = object_box_iou.max(
                    dim=1, keepdim=True).values.clamp(min=t1 + 1e-6)
                object_box_prob = ((object_box_iou - t1) / (t2 - t1)).clamp(
                    min=0, max=1)  # [n_gt, n_pred]

                num_obj = gt_labels.size(0)
                indices = torch.stack(
                    [torch.arange(num_obj).type_as(gt_labels), gt_labels],
                    dim=0)  # [2, n_gt], first dimension: row indices, second dimension: col indices

                object_cls_box_prob = torch.sparse_coo_tensor(
                    indices, object_box_prob)  # shape: [n_gt, num_class, n_pred]
                # object_cls_box_prob.to_dense() to show the Tensor

                """
                from "start" to "end" implement:
                image_box_iou = torch.sparse.max(object_cls_box_prob, dim=0).t()
                """
                # start
                box_cls_prob = torch.sparse.sum(
                    object_cls_box_prob, dim=0).to_dense()  # [num_class, n_pred]

                indices = torch.nonzero(box_cls_prob, as_tuple=False).t_()
                if indices.numel() == 0:
                    image_box_prob = torch.zeros(
                        anchors.size(0),
                        self.num_classes).type_as(object_box_prob)
                else:
                    nonzero_box_prob = torch.where(
                        (gt_labels.unsqueeze(dim=-1) == indices[0]),
                        object_box_prob[:, indices[1]],
                        torch.tensor(
                            [0]).type_as(object_box_prob)).max(dim=0).values

                    image_box_prob = torch.sparse_coo_tensor(
                        indices.flip([0]),
                        nonzero_box_prob,
                        size=(anchors.size(0), self.num_classes)).to_dense()
                # end

                box_prob.append(image_box_prob)

            # construct bags for objects
            match_quality_matrix = bbox_overlaps_nearest_3d(gt_bboxes, anchors)
            _, matched = torch.topk(
                match_quality_matrix,
                self.pre_anchor_topk,
                dim=1,
                sorted=False)
            del match_quality_matrix

            # matched_cls_prob: P_{ij}^{cls}
            matched_cls_prob = torch.gather(
                cls_prob[matched], 2,
                gt_labels.view(-1, 1, 1).repeat(1, self.pre_anchor_topk,
                                                1)).squeeze(2)

            # matched_box_prob: P_{ij}^{loc}
            matched_anchors = anchors[matched]
            matched_object_targets = self.bbox_coder.encode(
                matched_anchors,
                gt_bboxes.unsqueeze(dim=1).expand_as(matched_anchors))

            # direction classification loss
            loss_dir = None
            if self.use_direction_classifier:
                # also calculate direction prob: P_{ij}^{dir}
                matched_dir_targets = get_direction_target(
                    matched_anchors,
                    matched_object_targets,
                    self.dir_offset,
                    self.dir_limit_offset,
                    one_hot=False)
                loss_dir = self.loss_dir(
                    dir_cls_pred[matched].transpose(-2, -1),
                    matched_dir_targets,
                    reduction_override='none')

            # generate bbox weights
            if self.diff_rad_by_sin:
                bbox_preds_clone = bbox_pred.clone().to(matched_object_targets.dtype)  # change here
                bbox_preds_clone[matched], matched_object_targets = \
                    self.add_sin_difference(
                        bbox_preds_clone[matched], matched_object_targets)
            bbox_weights = matched_anchors.new_ones(matched_anchors.size())
            # Use pop is not right, check performance
            code_weight = self.train_cfg.get('code_weight', None)
            if code_weight:
                bbox_weights = bbox_weights * bbox_weights.new_tensor(
                    code_weight)
            loss_bbox = self.loss_bbox(
                bbox_preds_clone[matched],
                matched_object_targets,
                bbox_weights,
                reduction_override='none').sum(-1)

            if loss_dir is not None:
                loss_bbox += loss_dir
            matched_box_prob = torch.exp(-loss_bbox)

            # positive_losses: {-log( Mean-max(P_{ij}^{cls} * P_{ij}^{loc}) )}
            num_pos += len(gt_bboxes)
            with autocast(enabled=False):   # change here
                positive_losses.append(
                    self.positive_bag_loss(matched_cls_prob, matched_box_prob))

        positive_loss = torch.cat(positive_losses).sum() / max(1, num_pos)

        # box_prob: P{a_{j} \in A_{+}}
        box_prob = torch.stack(box_prob, dim=0)

        # negative_loss:
        # \sum_{j}{ FL((1 - P{a_{j} \in A_{+}}) * (1 - P_{j}^{bg})) } / n||B||
        with autocast(enabled=False):    # change here
            negative_loss = self.negative_bag_loss(cls_prob, box_prob).sum() / max(
                1, num_pos * self.pre_anchor_topk)

        losses = {
            'pos_loss': positive_loss,
            'neg_loss': negative_loss
        }
        return losses

