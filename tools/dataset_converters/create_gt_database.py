# Copyright (c) OpenMMLab. All rights reserved.
import pickle
from os import path as osp
import sys

import numpy as np
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO

import mmcv
import mmengine
from mmcv.ops import roi_align
from mmdet.evaluation import bbox_overlaps
from mmengine import track_iter_progress
from mmdet3d.registry import DATASETS
from mmdet3d.structures.ops import box_np_ops as box_np_ops

from tools.utils.logger_utils import get_root_logger


logger = get_root_logger()

def _poly2mask(mask_ann, img_h, img_w):
    if isinstance(mask_ann, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        rle = maskUtils.merge(rles)
    elif isinstance(mask_ann['counts'], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
    else:
        # rle
        rle = mask_ann
    mask = maskUtils.decode(rle)
    return mask


def _parse_coco_ann_info(ann_info):
    gt_bboxes = []
    gt_labels = []
    gt_bboxes_ignore = []
    gt_masks_ann = []

    for i, ann in enumerate(ann_info):
        if ann.get('ignore', False):
            continue
        x1, y1, w, h = ann['bbox']
        if ann['area'] <= 0:
            continue
        bbox = [x1, y1, x1 + w, y1 + h]
        if ann.get('iscrowd', False):
            gt_bboxes_ignore.append(bbox)
        else:
            gt_bboxes.append(bbox)
            gt_masks_ann.append(ann['segmentation'])

    if gt_bboxes:
        gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
        gt_labels = np.array(gt_labels, dtype=np.int64)
    else:
        gt_bboxes = np.zeros((0, 4), dtype=np.float32)
        gt_labels = np.array([], dtype=np.int64)

    if gt_bboxes_ignore:
        gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
    else:
        gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

    ann = dict(
        bboxes=gt_bboxes, bboxes_ignore=gt_bboxes_ignore, masks=gt_masks_ann)

    return ann


def crop_image_patch_v2(pos_proposals, pos_assigned_gt_inds, gt_masks):
    import torch
    from torch.nn.modules.utils import _pair
    device = pos_proposals.device
    num_pos = pos_proposals.size(0)
    fake_inds = (
        torch.arange(num_pos,
                     device=device).to(dtype=pos_proposals.dtype)[:, None])
    rois = torch.cat([fake_inds, pos_proposals], dim=1)  # Nx5
    mask_size = _pair(28)
    rois = rois.to(device=device)
    gt_masks_th = (
        torch.from_numpy(gt_masks).to(device).index_select(
            0, pos_assigned_gt_inds).to(dtype=rois.dtype))
    # Use RoIAlign could apparently accelerate the training (~0.1s/iter)
    targets = (
        roi_align(gt_masks_th, rois, mask_size[::-1], 1.0, 0, True).squeeze(1))
    return targets


def crop_image_patch(pos_proposals, gt_masks, pos_assigned_gt_inds, org_img):
    num_pos = pos_proposals.shape[0]
    masks = []
    img_patches = []
    for i in range(num_pos):
        gt_mask = gt_masks[pos_assigned_gt_inds[i]]
        bbox = pos_proposals[i, :].astype(np.int32)
        x1, y1, x2, y2 = bbox
        w = np.maximum(x2 - x1 + 1, 1)
        h = np.maximum(y2 - y1 + 1, 1)

        mask_patch = gt_mask[y1:y1 + h, x1:x1 + w]
        masked_img = gt_mask[..., None] * org_img
        img_patch = masked_img[y1:y1 + h, x1:x1 + w]

        img_patches.append(img_patch)
        masks.append(mask_patch)
    return img_patches, masks


def create_groundtruth_database(dataset_class_name,
                                data_path,
                                info_prefix,
                                info_path=None,
                                mask_anno_path=None,
                                used_classes=None,
                                database_save_path=None,
                                db_info_save_path=None,
                                relative_path=True,
                                add_rgb=False,
                                lidar_only=False,
                                bev_only=False,
                                coors_range=None,
                                with_mask=False):
    """Given the raw data, generate the ground truth database.

    Args:
        dataset_class_name (str): Name of the input dataset.
        data_path (str): Path of the data.
        info_prefix (str): Prefix of the info file.
        info_path (str, optional): Path of the info file.
            Default: None.
        mask_anno_path (str, optional): Path of the mask_anno.
            Default: None.
        used_classes (list[str], optional): Classes have been used.
            Default: None.
        database_save_path (str, optional): Path to save database.
            Default: None.
        db_info_save_path (str, optional): Path to save db_info.
            Default: None.
        relative_path (bool, optional): Whether to use relative path.
            Default: True.
        with_mask (bool, optional): Whether to use mask.
            Default: False.
    """
    logger.info(f'Create GT Database of {dataset_class_name}')
    logger.info(f'info_path: {info_path}')

    dataset_cfg = dict(type=dataset_class_name, data_root=data_path, ann_file=info_path)

    if dataset_class_name == 'NuScenesDataset':
        dataset_cfg.update(use_valid_flag=True,
                            modality=dict(use_lidar=True, use_camera=True),
                            data_prefix=dict(
                                pts='samples/LIDAR_TOP',
                                sweeps='sweeps/LIDAR_TOP',
                                CAM_FRONT='samples/CAM_FRONT',
                                CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
                                CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
                                CAM_BACK='samples/CAM_BACK',
                                CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
                                CAM_BACK_LEFT='samples/CAM_BACK_LEFT'),
                            pipeline=[
                                # dict(type='LoadMultiViewImageFromFiles'),
                                dict(
                                    type="LoadPointsFromFile",
                                    coord_type="LIDAR",
                                    load_dim=5,
                                    use_dim=5,),
                                dict(
                                    type="LoadPointsFromMultiSweeps",
                                    sweeps_num=10,
                                    use_dim=[0, 1, 2, 3, 4],
                                    pad_empty_sweeps=True,
                                    remove_close=True,),
                                dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True),
                            ],)
    else:
        logger.info(f"not supported {dataset_class_name}")
        return
    dataset = DATASETS.build(dataset_cfg)

    if database_save_path is None:
        database_save_path = osp.join(info_path.rsplit('/', 1)[0], f"{info_prefix}_gt_database")
    else:
        database_save_path = osp.join(database_save_path, f"{info_prefix}_gt_database")
    logger.info(f"database_save_path: {database_save_path}")
    if db_info_save_path is None:
        db_info_save_path = osp.join(info_path.rsplit('/', 1)[0], f"{info_prefix}_dbinfos_train.pkl")
    database_pts_path = osp.join(database_save_path, 'pts_dir')
    logger.info(f"pts database_save_path: {database_pts_path}")
    database_img_path = osp.join(database_save_path, 'img_dir')
    logger.info(f"img database_save_path: {database_img_path}")
    mmengine.mkdir_or_exist(database_save_path)
    mmengine.mkdir_or_exist(database_pts_path)
    mmengine.mkdir_or_exist(database_img_path)

    all_db_infos = dict()
    if with_mask:
        database_mask_img_path = osp.join(database_save_path, 'mask_img_dir')
        logger.info(f"img_with_mask database_save_path: {database_mask_img_path}")

        coco = COCO(osp.join(data_path, mask_anno_path))
        imgIds = coco.getImgIds()
        file2id = dict()
        for i in imgIds:
            info = coco.loadImgs([i])[0]
            file2id.update({info['file_name']: i})

    group_counter = 0
    for j in track_iter_progress(list(range(len(dataset)))):
        data_info = dataset.get_data_info(j)
        example = dataset.pipeline(data_info)
        annos = example['ann_info']
        image_idx = example['sample_idx']
        points = example['points'].tensor.numpy()
        gt_boxes_3d = annos['gt_bboxes_3d'].tensor.numpy()
        names = [dataset.metainfo['classes'][i] for i in annos['gt_labels_3d']]
        group_dict = dict()
        if 'group_ids' in annos:
            group_ids = annos['group_ids']
        else:
            group_ids = np.arange(gt_boxes_3d.shape[0], dtype=np.int64)
        difficulty = np.zeros(gt_boxes_3d.shape[0], dtype=np.int32)
        if 'difficulty' in annos:
            difficulty = annos['difficulty']

        num_obj = gt_boxes_3d.shape[0]
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes_3d)
        
        # load multi-view image
        input_img = {}
        input_info = {}
        for _cam, cam_info in example['images'].items():
            filename = cam_info['img_path']
            _img = mmcv.imread(filename, 'unchanged')
            input_img[_cam] = _img

            # obtain lidar to image transformation matrix
            lidar2cam = np.array(cam_info['lidar2cam'])
            intrinsic = np.array(cam_info['cam2img'])
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img = (viewpad @ lidar2cam)

            input_info[_cam]={
                'lidar2img': lidar2img,
                'lidar2cam': lidar2cam,
                'cam_intrinsic': viewpad}
        
        if with_mask:
            # prepare masks
            gt_boxes = annos['gt_bboxes']
            img_path = osp.split(example['img_info']['filename'])[-1]
            if img_path not in file2id.keys():
                logger.info(f"skip image {img_path} for empty mask")
                continue
            img_id = file2id[img_path]
            kins_annIds = coco.getAnnIds(imgIds=img_id)
            kins_raw_info = coco.loadAnns(kins_annIds)
            kins_ann_info = _parse_coco_ann_info(kins_raw_info)
            h, w = annos['img_shape'][:2]
            gt_masks = [_poly2mask(mask, h, w) for mask in kins_ann_info['masks']]
            # get mask inds based on iou mapping
            bbox_iou = bbox_overlaps(kins_ann_info['bboxes'], gt_boxes)
            mask_inds = bbox_iou.argmax(axis=0)
            valid_inds = (bbox_iou.max(axis=0) > 0.5)

            # mask the image
            # use more precise crop when it is ready
            # object_img_patches = np.ascontiguousarray(
            #     np.stack(object_img_patches, axis=0).transpose(0, 3, 1, 2))
            # crop image patches using roi_align
            # object_img_patches = crop_image_patch_v2(
            #     torch.Tensor(gt_boxes),
            #     torch.Tensor(mask_inds).long(), object_img_patches)
            object_img_patches, object_masks = crop_image_patch(
                gt_boxes, gt_masks, mask_inds, annos['img'])

        for i in range(num_obj):
            filename = f'{image_idx}_{names[i]}_{i}.bin'
            img_filename = f'{image_idx}_{names[i]}_{i}.png'
            abs_filepath = osp.join(database_pts_path, filename)
            abs_img_filepath = osp.join(database_img_path, img_filename)
            rel_filepath = osp.join(f"{info_prefix}_gt_database", 'pts_dir', filename)
            rel_img_filepath = osp.join(f'{info_prefix}_gt_database', 'img_dir', img_filename)

            # save point clouds and image patches for each object
            gt_points = points[point_indices[:, i]]
            gt_points[:, :3] -= gt_boxes_3d[i, :3]

            if with_mask:
                if object_masks[i].sum() == 0 or not valid_inds[i]:
                    # Skip object for empty or invalid mask
                    continue
                img_patch_path = osp.join(database_mask_img_path, filename + ".png")
                mask_patch_path = osp.join(database_mask_img_path, filename + ".mask.png")
                mmcv.imwrite(object_img_patches[i], img_patch_path)
                mmcv.imwrite(object_masks[i], mask_patch_path)

            with open(abs_filepath, 'w') as f:
                gt_points.tofile(f)

            img_crop, crop_key, crop_depth = find_img_crop(annos['gt_bboxes_3d'][i].corners.numpy(), 
                                                           input_img, 
                                                           input_info,  
                                                           points[point_indices[:, i]])
            if img_crop is not None:
                mmcv.imwrite(img_crop, abs_img_filepath)

            if (used_classes is None) or names[i] in used_classes:
                db_info = {"name": names[i],
                           "path": rel_filepath,
                           "image_idx": image_idx,
                           "gt_idx": i,
                           "box3d_lidar": gt_boxes_3d[i],
                           "num_points_in_gt": gt_points.shape[0],
                           "difficulty": difficulty[i],
                           'image_path': rel_img_filepath if img_crop is not None else '',
                           'image_crop_key': crop_key if img_crop is not None else '',
                           'image_crop_depth': crop_depth,
                           }
                local_group_id = group_ids[i]
                # if local_group_id >= 0:
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1
                db_info['group_id'] = group_dict[local_group_id]
                if 'score' in annos:
                    db_info['score'] = annos['score'][i]
                if with_mask:
                    db_info.update({"box2d_camera": gt_boxes[i]})
                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]

    for k, v in all_db_infos.items():
        logger.info(f"load {len(v)} {k} database infos")

    with open(db_info_save_path, 'wb') as f:
        logger.info(f"dump all_db_infos to: {db_info_save_path}")
        pickle.dump(all_db_infos, f)


def find_img_crop(gt_boxes_3d, input_img, input_info,  points):
    coord_3d = np.concatenate([gt_boxes_3d, np.ones_like(gt_boxes_3d[..., :1])], -1)
    coord_3d = coord_3d.squeeze(0)
    max_crop, crop_key = None, None
    crop_area, crop_depth = 0, 0

    for _key in input_img:
        lidar2img = np.array(input_info[_key]['lidar2img'])
        coord_img = coord_3d @ lidar2img.T
        coord_img[:,:2] /= coord_img[:,2,None]
        image_shape = input_img[_key].shape
        if (coord_img[2] <= 0).any():
            continue
        
        avg_depth = coord_img[:,2].mean()
        minxy = np.min(coord_img[:,:2], axis=-2)
        maxxy = np.max(coord_img[:,:2], axis=-2)
        bbox = np.concatenate([minxy, maxxy], axis=-1)
        bbox[0::2] = np.clip(bbox[0::2], a_min=0, a_max=image_shape[1]-1)
        bbox[1::2] = np.clip(bbox[1::2], a_min=0, a_max=image_shape[0]-1)
        bbox = bbox.astype(int)
        if ((bbox[2:]-bbox[:2]) <= 10).any():
            continue

        img_crop = input_img[_key][bbox[1]:bbox[3],bbox[0]:bbox[2]]
        if img_crop.shape[0] * img_crop.shape[1] > crop_area:
            max_crop = img_crop
            crop_key = _key
            crop_depth = avg_depth
    
    return max_crop, crop_key, crop_depth