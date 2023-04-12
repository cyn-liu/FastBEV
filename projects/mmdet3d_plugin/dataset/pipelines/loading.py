from mmdet3d.registry import TRANSFORMS
from mmdet3d.datasets.transforms import LoadAnnotations3D


__all__ = ['CustomLoadAnnotations3D']


@TRANSFORMS.register_module()
class CustomLoadAnnotations3D(LoadAnnotations3D):
    def _load_bboxes_3d(self, results: dict) -> dict:
        if 'ann_info' in results:
            results['gt_bboxes_3d'] = results['ann_info']['gt_bboxes_3d']
        else:
            results['gt_bboxes_3d'] = results['eval_ann_info']['gt_bboxes_3d']
        return results

    def _load_bboxes_depth(self, results: dict) -> dict:
        if 'ann_info' in results:
            results['depths'] = results['ann_info']['depths']
            results['centers_2d'] = results['ann_info']['centers_2d']
        else:
            results['depths'] = results['eval_ann_info']['depths']
            results['centers_2d'] = results['eval_ann_info']['centers_2d']
        return results

    def _load_labels_3d(self, results: dict) -> dict:
        if 'ann_info' in results:
            results['gt_labels_3d'] = results['ann_info']['gt_labels_3d']
        else:
            results['gt_labels_3d'] = results['eval_ann_info']['gt_labels_3d']
        return results

    def _load_attr_labels(self, results: dict) -> dict:
        if 'ann_info' in results:
            results['attr_labels'] = results['ann_info']['attr_labels']
        else:
            results['attr_labels'] = results['eval_ann_info']['attr_labels']
        return results

    def _load_bboxes(self, results: dict) -> None:
        if 'ann_info' in results:
            results['gt_bboxes'] = results['ann_info']['gt_bboxes']
        else:
            results['gt_bboxes'] = results['eval_ann_info']['gt_bboxes']

    def _load_labels(self, results: dict) -> None:
        if 'ann_info' in results:
            results['gt_bboxes_labels'] = results['ann_info']['gt_bboxes_labels']
        else:
            results['gt_bboxes_labels'] = results['eval_ann_info']['gt_bboxes_labels']
