from typing import List, Optional, Union, Dict

import torch
from torch import Tensor

from mmengine import ConfigDict
from mmengine.structures import InstanceData
from mmdet.models.task_modules import AssignResult, BaseAssigner
from mmdet3d.registry import TASK_UTILS
from projects.mmdet3d_plugin.models.utils.utils import normalize_bbox

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

__all__ =['CustomHungarianAssigner3D']

@TASK_UTILS.register_module()
class CustomHungarianAssigner3D(BaseAssigner):
    """Computes one-to-one matching between predictions and ground truth. This
    class computes an assignment between the targets and the predictions based
    on the costs. The costs are weighted sum of two components:
    classification cost, regression L1 cost. The targets don't include the
    no_object, so generally there are more predictions than targets. After the
    one-to-one matching, the un-matched are treated as backgrounds. Thus, each
    query prediction will be assigned with `0` or a positive integer indicating
    the ground truth index:
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt
    Args:
        match_costs (dict[:obj:`ConfigDict`]): Match cost configs.
        pc_range (list[float]):
    """

    def __init__(self,
                 match_costs: Dict[str, Dict],
                 pc_range=None) -> None:
        assert isinstance(match_costs, dict), 'match_costs must dict.'
        self.match_costs = dict()

        support_match_costs = ['cls_cost', 'reg_cost', 'iou_cost']
        for loss_name, match_cost in match_costs.items():
            assert loss_name in support_match_costs, f"only support {support_match_costs}"
            self.match_costs[loss_name] = TASK_UTILS.build(match_cost)
        self.pc_range = pc_range

    def assign(self,
               pred_instance_3d: InstanceData,
               gt_instances_3d: InstanceData,
               input_metas: Optional[dict] = None,
               **kwargs):
        assert isinstance(gt_instances_3d.labels_3d, Tensor)

        gt_labels = gt_instances_3d.labels_3d
        gt_bboxes = gt_instances_3d.bboxes_3d
        device = gt_labels.device

        cls_pred = pred_instance_3d.scores
        bbox_pred = pred_instance_3d.priors

        gt_bboxes = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
                      dim=1).to(gt_labels.device)

        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred.new_full((num_bboxes, ),
                                              -1,
                                              dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        # 2. compute the weighted costs
        # classification and bboxcost.
        cls_cost = self.match_costs['cls_cost'](cls_pred, gt_labels)
        # regression L1 cost
        normalized_gt_bboxes = normalize_bbox(gt_bboxes, self.pc_range)
        reg_cost = self.match_costs['reg_cost'](bbox_pred[:, :8], normalized_gt_bboxes[:, :8])

        # weighted sum of above two costs
        cost = cls_cost + reg_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        cost = torch.nan_to_num(cost, nan=100.0, posinf=100.0, neginf=-100.0)
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            bbox_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            bbox_pred.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)
