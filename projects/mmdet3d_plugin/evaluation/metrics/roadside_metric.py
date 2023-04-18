
from mmdet3d.registry import METRICS
from mmdet3d.evaluation.metrics.nuscenes_metric import NuScenesMetric


@METRICS.register_module()
class RoadsideMetric(NuScenesMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)