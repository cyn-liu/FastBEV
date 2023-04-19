from typing import Callable, List, Union
import os
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from loguru import logger

from mmengine import mkdir_or_exist
from mmdet3d.registry import DATASETS
from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset
from mmdet3d.structures import LiDARInstance3DBoxes
import mmcv
import matplotlib.pyplot as plt

__all__ = ['MyNuscenesDataset']


@DATASETS.register_module()
class MyNuscenesDataset(NuScenesDataset):

    def __init__(self,  **kwargs) -> None:
        super().__init__( **kwargs)
        self.pred_score_thr = 0.1
        self.order_input = True
        self.fps = 25
        self.show_score = True
        self.save_img = False


    def vis(self, results, runner):
        worker_dir = runner.work_dir
        results =[result[0].cpu() for result in results]

        new_det_results = []
        for i in range(len(results)):
            box_type = type(results[i].pred_instances_3d.bboxes_3d)
            boxes_3d = results[i].pred_instances_3d.bboxes_3d.tensor
            boxes_3d = box_type(boxes_3d, box_dim=9, origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
            new_det_results.append(dict(
                boxes_3d=boxes_3d,
                scores_3d=results[i].pred_instances_3d.scores_3d,
                labels_3d=results[i].pred_instances_3d.labels_3d))

        self.show(new_det_results, worker_dir, thr=self.pred_score_thr, fps=self.fps)


    def show(self, results, out_dir='trash', thr=0.5, fps=3):
        assert out_dir is not None, 'Expect out_dir, got none.'
        video_out_dir = os.path.join(out_dir, 'video')
        mkdir_or_exist(video_out_dir)

        colors = get_colors()
        all_img_gt, all_img_pred, all_bev_gt, all_bev_pred, all_img_filename = [], [], [], [], []

        logger.info(f"converting images")
        for i, result in tqdm(enumerate(results), total=len(results)):
            info = self.get_data_info(i)
            gt_bboxes = self.get_ann_info(i)
            scale_fac = 10
            bev_pred_img = np.zeros((100 * scale_fac, 100 * scale_fac, 3))
            bev_gt_img = np.zeros((100 * scale_fac, 100 * scale_fac, 3))

            scores = result['scores_3d'].numpy()
            bev_box_pred = result['boxes_3d'].corners.cpu().numpy()[:, [0, 2, 6, 4]][..., :2][scores > thr]
            labels = result['labels_3d'].numpy()[scores > thr]
            assert bev_box_pred.shape[0] == labels.shape[0]
            for idx in range(len(labels)):
                bev_pred_img = draw_bev_bbox_corner(bev_pred_img, bev_box_pred[idx], colors[labels[idx]],
                                                         scale_fac)

            bev_gt_bboxes = gt_bboxes['gt_bboxes_3d'].corners.numpy()[:, [0, 2, 6, 4]][..., :2]
            labels_gt = gt_bboxes['gt_labels_3d']
            for idx in range(len(labels_gt)):
                bev_gt_img = draw_bev_bbox_corner(bev_gt_img, bev_gt_bboxes[idx], colors[labels_gt[idx]],
                                                       scale_fac)

            bev_pred_img = process_bev_res_in_front(bev_pred_img)
            bev_gt_img = process_bev_res_in_front(bev_gt_img)
            all_bev_gt.append(mmcv.imrescale(bev_gt_img, 0.5))
            all_bev_pred.append(mmcv.imrescale(bev_pred_img, 0.5))

            img_gt_list = []
            img_pred_list = []
            
            for img, img_info in info['images'].items():
                fildname_cur = Path(info['images'][img]['img_path']).stem
                img_pred = imread(img_info['img_path'])
                img_gt = imread(img_info['img_path'])

                puttext(img_pred, fildname_cur)
                puttext(img_gt, fildname_cur)

                extrinsic = img_info['lidar2cam']
                intrinsic = img_info['cam2img'][:3, :3]
                projection = intrinsic @ extrinsic[:3]
                if not len(result['scores_3d']):
                    pass
                else:
                    # draw pred
                    corners = result['boxes_3d'].corners.cpu().numpy()
                    scores = result['scores_3d'].numpy()
                    labels = result['labels_3d'].numpy()
                    for corner, score, label in zip(corners, scores, labels):
                        if score < thr:
                            continue

                        txt_dict = None
                        if self.show_score:
                            txt_dict = dict(text=str("{:.2f}".format(score)),
                                            fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                            fontScale=1,
                                            color=colors[label],
                                            thickness=1)
                        draw_corners(img_pred, corner, colors[label], projection, txt_dict)

                    gt_corners = gt_bboxes['gt_bboxes_3d'].corners.numpy()
                    gt_labels = gt_bboxes['gt_labels_3d']
                    for corner, label in zip(gt_corners, gt_labels):
                        draw_corners(img_gt, corner, colors[label], projection)

                img_gt_list.append(mmcv.imrescale(img_gt, 0.5))
                img_pred_list.append(mmcv.imrescale(img_pred, 0.5))

            tmp_img_up_pred = np.concatenate(sort_list(img_pred_list[0:3], sort=[1, 0, 2]), axis=0)
            all_img_pred.append(tmp_img_up_pred)
            tmp_img_up_gt = np.concatenate(sort_list(img_gt_list[0:3], sort=[1, 0, 2]), axis=0)
            all_img_gt.append(tmp_img_up_gt)
            all_img_filename.append(Path(info['images']['CAM_FRONT']['img_path']).stem)

        if self.order_input:
            all_bev_pred_sorted = all_bev_pred
            all_bev_gt_sorted = all_bev_gt
            all_img_pred_sorted = all_img_pred
            all_img_gt_sorted = all_img_gt
        else:
            sorted_data = sorted(enumerate(all_img_filename), key=lambda x: datetime.strptime(x[1].split("_")[0], '%Y-%m-%d-%H-%M-%S'))
            sorted_indices = [x[0] for x in sorted_data]

            all_bev_pred_sorted = [all_bev_pred[i] for i in sorted_indices]
            all_bev_gt_sorted = [all_bev_gt[i] for i in sorted_indices]
            all_img_pred_sorted = [all_img_pred[i] for i in sorted_indices]
            all_img_gt_sorted = [all_img_gt[i] for i in sorted_indices]

        merged_image_tmp = process_img(all_bev_pred_sorted[0], all_img_pred_sorted[0])
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        pred_video = cv2.VideoWriter(os.path.join(video_out_dir, 'video_pred.mp4'), fourcc, fps,
                                     merged_image_tmp.shape[:2][::-1])
        gt_video = cv2.VideoWriter(os.path.join(video_out_dir, 'video_gt.mp4'), fourcc, fps,
                                   merged_image_tmp.shape[:2][::-1])

        logger.info(f"Starting save")
        for i in tqdm(range(len(all_bev_pred))):
            if self.save_img:
                img_out_dir = os.path.join(out_dir, 'img')
                mkdir_or_exist(img_out_dir)
                cv2.imwrite(img_out_dir + '/' + str(i) + '_predbev.png', all_bev_pred_sorted[i])
                cv2.imwrite(img_out_dir + '/' + str(i) + '_predimg.png', all_img_pred_sorted[i])
                cv2.imwrite(img_out_dir + '/' + str(i) + '_gtbev.png', all_bev_gt_sorted[i])
                cv2.imwrite(img_out_dir + '/' + str(i) + '_gtimg.png', all_img_gt_sorted[i])
            pred_video.write(np.uint8(process_img(all_bev_pred_sorted[i], all_img_pred_sorted[i])))
            gt_video.write(np.uint8(process_img(all_bev_gt_sorted[i], all_img_gt_sorted[i])))

        pred_video.release()
        gt_video.release()
        logger.info(f"Finished saving video to {video_out_dir}")

def puttext(img, name, loc=(10, 60), font=cv2.FONT_HERSHEY_DUPLEX ,color=(248, 202, 105)):
    cv2.putText(img, name, loc, font, 2, color, 2)

def draw_corners(img, corners, color, projection, text_dict=None):
    corners_3d_4 = np.concatenate((corners, np.ones((8, 1))), axis=1)
    corners_2d_3 = corners_3d_4 @ projection.T
    z_mask = corners_2d_3[:, 2] > 0
    corners_2d = corners_2d_3[:, :2] / corners_2d_3[:, 2:]
    corners_2d = corners_2d.astype(np.int)

    if text_dict is not None and z_mask.any():
        txt_x = int((corners_2d[:, 0].min() + corners_2d[:, 0].max()) // 2)
        txt_y = int(corners_2d[:, 1].min()) - 5
        text_dict['org'] = (txt_x, txt_y)
        cv2.putText(img, **text_dict)

    for i, j in [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]:
        if z_mask[i] and z_mask[j]:
            img = cv2.line(
                img=img,
                pt1=tuple(corners_2d[i]),
                pt2=tuple(corners_2d[j]),
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA)
    # drax `X' in the front
    if z_mask[0] and z_mask[5]:
        img = cv2.line(
            img=img,
            pt1=tuple(corners_2d[0]),
            pt2=tuple(corners_2d[5]),
            color=color,
            thickness=2,
            lineType=cv2.LINE_AA)
    if z_mask[1] and z_mask[4]:
        img = cv2.line(
            img=img,
            pt1=tuple(corners_2d[1]),
            pt2=tuple(corners_2d[4]),
            color=color,
            thickness=2,
            lineType=cv2.LINE_AA)

def draw_bev_bbox_corner(img, box, color, scale_fac):
    box = box[:, None, :]  # [4,1,2]
    box = box + 50
    box = box * scale_fac
    box = np.int0(box)
    # box[:, :, 0] -= 500 # X轴减少50米
    # box_tmp = box
    # box_tmp[:, :, 0], box_tmp[:, :, 1] = box[:, :, 1], box[:, :, 0]
    img = cv2.polylines(img, [box], isClosed=True, color=color, thickness=2)
    return img

def get_colors():
    colors = np.multiply([
            plt.cm.get_cmap('gist_ncar', 37)((i * 7 + 5) % 37)[:3] for i in range(37)
        ], 255).astype(np.uint8).tolist()
    colors = [i[::-1] for i in colors]
    return colors


def sort_list(_list, sort):
    assert len(_list) == len(sort)
    new_list = []
    for s in sort:
        new_list.append(_list[s])
    return new_list

def imread(img_path):
    img = mmcv.imread(img_path)
    return img

def process_bev_res_in_front(bev):
    bev = np.flip(bev, axis=0)
    return bev

def process_img(bev, img):
    # bev = np.rot90(bev, k=1, axes=(0, 1))
    scale_factor_bev = 1080 / bev.shape[0]
    scale_factor_img = 1080 / img.shape[0]
    bev = cv2.resize(bev, (int(bev.shape[1] * scale_factor_bev), int(bev.shape[0] * scale_factor_bev)))
    img = cv2.resize(img, (int(img.shape[1] * scale_factor_img), int(img.shape[0] * scale_factor_img)))
    merged_image = np.concatenate((img, bev), axis=1)
    return merged_image
