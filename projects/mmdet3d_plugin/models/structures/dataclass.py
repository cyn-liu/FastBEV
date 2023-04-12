from typing import Dict, List, Optional, Sequence
from dataclasses import dataclass

from torch import Tensor

from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes, limit_period

__all__ = ['InputDataClass']


@dataclass
class InputDataClass:
    imgs: Tensor = None
    voxel_dict: Dict[str, Tensor] = None
    points: Optional[List[Tensor]] = None
    radar_points: Optional[List[Tensor]] = None
    gt_bboxes_3d: List[LiDARInstance3DBoxes] = None
    gt_labels_3d: List[Tensor] = None
    gt_bboxes_ignore: List[Tensor] = None
    pred_bboxes_3d: List[LiDARInstance3DBoxes] = None
    pred_labels_3d: List[Tensor] = None
    pred_bboxes_ignore: List[Tensor] = None