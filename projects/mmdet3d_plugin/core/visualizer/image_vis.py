# Copyright (c) OpenMMLab. All rights reserved.
import copy
import cv2
from PIL import Image
from typing import Union, Dict, Tuple
from pathlib import Path
import os
from matplotlib import pyplot as plt

import numpy as np
import torch
from torch import Tensor

from mmcv.image import imwrite
from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes


def save_img_single(
        img: Union[np.ndarray, Tensor],
        lidar2img: Union[np.ndarray, Tensor],
        boxes_3d: LiDARInstance3DBoxes,
        labels: np.ndarray,
        vis_config: Dict,
        scores: np.ndarray = None,
        pred_config: Dict = None):
    # pred_config: for show scores, have two keys: pred_score_font_type, pred_score_font_size
    if isinstance(img, Tensor):
        img = img.cpu().numpy()
    if isinstance(lidar2img, Tensor):
        lidar2img = lidar2img.cpu().numpy()

    if boxes_3d is None:
        new_img = img
    else:
        assert isinstance(boxes_3d, LiDARInstance3DBoxes)
        new_img = draw_lidar_bbox3d_on_img(boxes_3d, labels, img, lidar2img, vis_config, scores, pred_config)
    return (new_img, )


def project_pts_on_img(points,
                       raw_img,
                       lidar2img_rt,
                       max_distance=70,
                       thickness=-1):
    """Project the 3D points cloud on 2D image.

    Args:
        points (numpy.array): 3D points cloud (x, y, z) to visualize.
        raw_img (numpy.array): The numpy array of image.
        lidar2img_rt (numpy.array, shape=[4, 4]): The projection matrix
            according to the camera intrinsic parameters.
        max_distance (float): the max distance of the points cloud.
            Default: 70.
        thickness (int, optional): The thickness of 2D points. Default: -1.
    """
    img = raw_img.copy()
    num_points = points.shape[0]
    pts_4d = np.concatenate([points[:, :3], np.ones((num_points, 1))], axis=-1)
    pts_2d = pts_4d @ lidar2img_rt.T

    # cam_points is Tensor of Nx4 whose last column is 1
    # transform camera coordinate to image coordinate
    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=99999)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]

    fov_inds = ((pts_2d[:, 0] < img.shape[1])
                & (pts_2d[:, 0] >= 0)
                & (pts_2d[:, 1] < img.shape[0])
                & (pts_2d[:, 1] >= 0))

    imgfov_pts_2d = pts_2d[fov_inds, :3]  # u, v, d

    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pts_2d[i, 2]
        color = cmap[np.clip(int(max_distance * 10 / depth), 0, 255), :]
        cv2.circle(
            img,
            center=(int(np.round(imgfov_pts_2d[i, 0])),
                    int(np.round(imgfov_pts_2d[i, 1]))),
            radius=1,
            color=tuple(color),
            thickness=thickness,
        )
    cv2.imshow('project_pts_img', img.astype(np.uint8))
    cv2.waitKey(100)


def plot_rect3d_on_img(img,
                       num_rects,
                       rect_corners,
                       labels,
                       vis_config,
                       scores=None,
                       pred_config=None):
    """Plot the boundary lines of 3D rectangular on 2D images.

    Args:
        img (numpy.array): The numpy array of image.
        num_rects (int): Number of 3D rectangulars.
        rect_corners (numpy.array): Coordinates of the corners of 3D
            rectangular. Should be in the shape of [num_rect, 8, 2].
        labels (numpy.array):  [num_rect]
        scores (numpy.array):  [num_rect]
        vis_config: key: label, value have two keys:
            - color (tuple[int]): The color to draw bboxes. Default: (0, 255, 0).
            - thickness (int, optional): The thickness of bboxes. Default: 1.
        pred_config: for show scores, have two keys: pred_score_font_type, pred_score_font_size
    """
    line_indices = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7),
                    (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))
    if pred_config is not None:
        fontFace = pred_config['pred_score_font_type']
        fontScale = pred_config['pred_score_font_size']
    else:
        fontFace = cv2.FONT_HERSHEY_DUPLEX
        fontScale = 1
    for i in range(num_rects):
        corners = rect_corners[i].astype(np.int)
        txt_x = int((corners[:, 0].min() + corners[:, 0].max()) // 2)
        txt_y = int(corners[:, 1].min()) - 5
        label = labels[i]
        color = vis_config[label]['color']
        thickness = vis_config[label]['thickness']
        if scores is not None:
            text = str("{:.2f}".format(scores[i]))
            if txt_x>0 and txt_y > 0:
                cv2.putText(img, text, (txt_x, txt_y), fontFace, fontScale, color, thickness)
        try:
            for start, end in line_indices:
                x_start = min(corners[start, 0], 10000)
                y_start = max(corners[start, 1], 0)
                x_end = min(corners[end, 0], 10000)
                y_end = max(corners[end, 1], 0)
                cv2.line(img, (x_start, y_start), (x_end, y_end), color, thickness,
                         cv2.LINE_AA)
        except:
            continue

    return img.astype(np.uint8)


def draw_lidar_bbox3d_on_img(bboxes3d: LiDARInstance3DBoxes,
                             labels: np.ndarray,
                             raw_img: np.ndarray,
                             lidar2img_rt: np.ndarray,
                             vis_config: Dict = None,
                             scores: np.ndarray = None,
                             pred_config=None):
    """Project the 3D bbox on 2D plane and draw on input image.

    Args:
        bboxes3d (:obj:`LiDARInstance3DBoxes`):
            3d bbox in lidar coordinate system to visualize.
        labels (numpy.array):
        raw_img (numpy.array): The numpy array of image.
        lidar2img_rt (numpy.array, shape=[4, 4]): The projection matrix
            according to the camera intrinsic parameters.
        vis_config: key: label, value have two keys:
            - color (tuple[int]): The color to draw bboxes. Default: (0, 255, 0).
            - thickness (int, optional): The thickness of bboxes. Default: 1.
        scores (numpy.array):
        pred_config: for show scores, have two keys: pred_score_font_type, pred_score_font_size
    """
    img = raw_img.copy()
    corners_3d = bboxes3d.corners.cpu().numpy()
    num_bbox = corners_3d.shape[0]
    pts_4d = np.concatenate(
        [corners_3d.reshape(-1, 3),
         np.ones((num_bbox * 8, 1))], axis=-1)
    lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
    if isinstance(lidar2img_rt, torch.Tensor):
        lidar2img_rt = lidar2img_rt.cpu().numpy()
    pts_2d = pts_4d @ lidar2img_rt.T

    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)

    return plot_rect3d_on_img(img, num_bbox, imgfov_pts_2d, labels, vis_config, scores, pred_config)


# TODO: remove third parameter in all functions here in favour of img_metas
def draw_depth_bbox3d_on_img(bboxes3d,
                             raw_img,
                             calibs,
                             img_metas,
                             color=(0, 255, 0),
                             thickness=1):
    """Project the 3D bbox on 2D plane and draw on input image.

    Args:
        bboxes3d (:obj:`DepthInstance3DBoxes`, shape=[M, 7]):
            3d bbox in depth coordinate system to visualize.
        raw_img (numpy.array): The numpy array of image.
        calibs (dict): Camera calibration information, Rt and K.
        img_metas (dict): Used in coordinates transformation.
        color (tuple[int]): The color to draw bboxes. Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    from mmdet3d.structures.bbox_3d import points_cam2img
    from mmdet3d.models import apply_3d_transformation

    img = raw_img.copy()
    img_metas = copy.deepcopy(img_metas)
    corners_3d = bboxes3d.corners
    num_bbox = corners_3d.shape[0]
    points_3d = corners_3d.reshape(-1, 3)

    # first reverse the data transformations
    xyz_depth = apply_3d_transformation(
        points_3d, 'DEPTH', img_metas, reverse=True)

    # project to 2d to get image coords (uv)
    uv_origin = points_cam2img(xyz_depth,
                               xyz_depth.new_tensor(img_metas['depth2img']))
    uv_origin = (uv_origin - 1).round()
    imgfov_pts_2d = uv_origin[..., :2].reshape(num_bbox, 8, 2).numpy()

    return plot_rect3d_on_img(img, num_bbox, imgfov_pts_2d, color, thickness)


def draw_camera_bbox3d_on_img(bboxes3d,
                              raw_img,
                              cam2img,
                              img_metas,
                              color=(0, 255, 0),
                              thickness=1):
    """Project the 3D bbox on 2D plane and draw on input image.

    Args:
        bboxes3d (:obj:`CameraInstance3DBoxes`, shape=[M, 7]):
            3d bbox in camera coordinate system to visualize.
        raw_img (numpy.array): The numpy array of image.
        cam2img (dict): Camera intrinsic matrix,
            denoted as `K` in depth bbox coordinate system.
        img_metas (dict): Useless here.
        color (tuple[int]): The color to draw bboxes. Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    from mmdet3d.structures.bbox_3d import points_cam2img

    img = raw_img.copy()
    cam2img = copy.deepcopy(cam2img)
    corners_3d = bboxes3d.corners
    num_bbox = corners_3d.shape[0]
    points_3d = corners_3d.reshape(-1, 3)
    if not isinstance(cam2img, torch.Tensor):
        cam2img = torch.from_numpy(np.array(cam2img))
    cam2img = cam2img.reshape(3, 3).float().cpu()

    # project to 2d to get image coords (uv)
    uv_origin = points_cam2img(points_3d, cam2img)
    uv_origin = (uv_origin - 1).round()
    imgfov_pts_2d = uv_origin[..., :2].reshape(num_bbox, 8, 2).numpy()

    return plot_rect3d_on_img(img, num_bbox, imgfov_pts_2d, color, thickness)


def puttext(img, name, loc=(30, 60), font=cv2.FONT_HERSHEY_DUPLEX, fontScale=2, color=(248, 202, 105), thickness=2):
    try:
        cv2.putText(img, name, loc, font, fontScale, color, thickness)
    except:
        img = Image.fromarray(img)
        img = np.array(img)
        cv2.putText(img, name, loc, font, fontScale, color, thickness)