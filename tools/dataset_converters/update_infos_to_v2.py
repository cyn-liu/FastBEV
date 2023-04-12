# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import time
from os import path as osp
from pathlib import Path
from pyquaternion import Quaternion
from typing import Callable, List, Union, Optional

import mmengine
import numpy as np
from nuscenes.nuscenes import NuScenes

from mmdet3d.datasets.convert_utils import (convert_annos,
                                            get_kitti_style_2d_boxes,
                                            get_nuscenes_2d_boxes)
# from mmdet3d.datasets.utils import convert_quaternion_to_matrix
from mmdet3d.structures import points_cam2img

def convert_ndarry_type(narray: Union[List[List], np.ndarray],
                        type: Optional[str] = None,
                        return_list: bool = False) -> Union[List[List], np.ndarray]:
    if isinstance(narray, list):
        narray = np.array(narray)
    assert type in ['float32',] if type is not None else True

    if type == 'float32':
        narray = narray.astype(np.float32)
    return narray if not return_list else narray.tolist()


def convert_quaternion_to_matrix(quaternion: List,
                                 translation: List = None,
                                 return_list: str = 'ndarry') -> List:
    """Compute a transform matrix by given quaternion and translation
    vector."""
    result = np.eye(4)
    result[:3, :3] = Quaternion(quaternion).rotation_matrix
    if translation is not None:
        result[:3, 3] = np.array(translation)
    return convert_ndarry_type(result, type='float32', return_list=False)


def get_empty_instance():
    """Empty annotation for single instance."""
    instance = dict(
        # (list[float], required): list of 4 numbers representing
        # the bounding box of the instance, in (x1, y1, x2, y2) order.
        bbox=None,
        # (int, required): an integer in the range
        # [0, num_categories-1] representing the category label.
        bbox_label=None,
        #  (list[float], optional): list of 7 (or 9) numbers representing
        #  the 3D bounding box of the instance,
        #  in [x, y, z, w, h, l, yaw]
        #  (or [x, y, z, w, h, l, yaw, vx, vy]) order.
        bbox_3d=None,
        # (bool, optional): Whether to use the
        # 3D bounding box during training.
        bbox_3d_isvalid=None,
        # (int, optional): 3D category label
        # (typically the same as label).
        bbox_label_3d=None,
        # (float, optional): Projected center depth of the
        # 3D bounding box compared to the image plane.
        depth=None,
        #  (list[float], optional): Projected
        #  2D center of the 3D bounding box.
        center_2d=None,
        # (int, optional): Attribute labels
        # (fine-grained labels such as stopping, moving, ignore, crowd).
        attr_label=None,
        # (int, optional): The number of LiDAR
        # points in the 3D bounding box.
        num_lidar_pts=None,
        # (int, optional): The number of Radar
        # points in the 3D bounding box.
        num_radar_pts=None,
        # (int, optional): Difficulty level of
        # detecting the 3D bounding box.
        difficulty=None,
        unaligned_bbox_3d=None)
    return instance


def get_empty_multicamera_instances(camera_types):

    cam_instance = dict()
    for cam_type in camera_types:
        cam_instance[cam_type] = None
    return cam_instance


def get_empty_lidar_points():
    lidar_points = dict(
        # (int, optional) : Number of features for each point.
        num_pts_feats=None,
        # (str, optional): Path of LiDAR data file.
        lidar_path=None,
        # (list[list[float]], optional): Transformation matrix
        # from lidar to ego-vehicle
        # with shape [4, 4].
        # (Referenced camera coordinate system is ego in KITTI.)
        lidar2ego=None,
    )
    return lidar_points


def get_empty_radar_points():
    radar_points = dict(
        # (int, optional) : Number of features for each point.
        num_pts_feats=None,
        # (str, optional): Path of RADAR data file.
        radar_path=None,
        # Transformation matrix from lidar to
        # ego-vehicle with shape [4, 4].
        # (Referenced camera coordinate system is ego in KITTI.)
        radar2ego=None,
    )
    return radar_points


def get_empty_img_info():
    img_info = dict(
        # (str, required): the path to the image file.
        img_path=None,
        # (int) The height of the image.
        height=None,
        # (int) The width of the image.
        width=None,
        # (str, optional): Path of the depth map file
        depth_map=None,
        # (list[list[float]], optional) : Transformation
        # matrix from camera to image with
        # shape [3, 3], [3, 4] or [4, 4].
        cam2img=None,
        # (list[list[float]]): Transformation matrix from lidar
        # or depth to image with shape [4, 4].
        lidar2img=None,
        # (list[list[float]], optional) : Transformation
        # matrix from camera to ego-vehicle
        # with shape [4, 4].
        cam2ego=None)
    return img_info


def get_single_image_sweep(camera_types):
    single_image_sweep = dict(
        # (float, optional) : Timestamp of the current frame.
        timestamp=None,
        # (list[list[float]], optional) : Transformation matrix
        # from ego-vehicle to the global
        ego2global=None)
    # (dict): Information of images captured by multiple cameras
    images = dict()
    for cam_type in camera_types:
        images[cam_type] = get_empty_img_info()
    single_image_sweep['images'] = images
    return single_image_sweep


def get_single_lidar_sweep():
    single_lidar_sweep = dict(
        # (float, optional) : Timestamp of the current frame.
        timestamp=None,
        # (list[list[float]], optional) : Transformation matrix
        # from ego-vehicle to the global
        ego2global=None,
        # (dict): Information of images captured by multiple cameras
        lidar_points=get_empty_lidar_points())
    return single_lidar_sweep


def get_empty_radar_info():
    radar_info = dict(
        # (str, required): the path to the radar file.
        radar_path=None,
        # (int, optional) : Number of features for each point.
        num_radar_feats=None,
        # (list[list[float]]): Transformation matrix from lidar
        # or depth to radar with shape [4, 4].
        lidar2radar=None,
        # (list[list[float]], optional) : Transformation
        # matrix from radar to ego-vehicle
        # with shape [4, 4].
        radar2ego=None)
    return radar_info


def get_single_radar_sweep(radar_types):
    single_radar_sweep = dict(
        # (float, optional) : Timestamp of the current frame.
        timestamp=None,
        # (list[list[float]], optional) : Transformation matrix
        # from ego-vehicle to the global
        ego2global=None,
        # (dict): Information of radars captured by multiple radars
        radar_points=get_empty_radar_points())
    radars = dict()
    for radar_type in radar_types:
        radars[radar_type] = get_empty_radar_info()
    single_radar_sweep['radars'] = radars
    return single_radar_sweep


def get_empty_standard_data_info(
        camera_types=['CAM0', 'CAM1', 'CAM2', 'CAM3', 'CAM4']):

    data_info = dict(
        # (str): Sample id of the frame.
        sample_idx=None,
        # (str, optional): '000010'
        token=None,
        **get_single_image_sweep(camera_types),
        # (dict, optional): dict contains information
        # of LiDAR point cloud frame.
        lidar_points=get_empty_lidar_points(),
        # (dict, optional) Each dict contains
        # information of Radar point cloud frame.
        radar_points=get_empty_radar_points(),
        # (list[dict], optional): Image sweeps data.
        image_sweeps=[],
        lidar_sweeps=[],
        radar_sweeps=dict(),  # my add
        instances=[],
        # (list[dict], optional): Required by object
        # detection, instance  to be ignored during training.
        instances_ignore=[],
        # (str, optional): Path of semantic labels for each point.
        pts_semantic_mask_path=None,
        # (str, optional): Path of instance labels for each point.
        pts_instance_mask_path=None)
    return data_info


def clear_instance_unused_keys(instance):
    keys = list(instance.keys())
    for k in keys:
        if instance[k] is None:
            del instance[k]
    return instance


def clear_data_info_unused_keys(data_info):
    keys = list(data_info.keys())
    empty_flag = True
    for key in keys:
        # we allow no annotations in datainfo
        if key in ['instances', 'cam_sync_instances', 'cam_instances']:
            empty_flag = False
            continue
        if isinstance(data_info[key], list):
            if len(data_info[key]) == 0:
                del data_info[key]
            else:
                empty_flag = False
        elif data_info[key] is None:
            del data_info[key]
        elif isinstance(data_info[key], dict):
            _, sub_empty_flag = clear_data_info_unused_keys(data_info[key])
            if sub_empty_flag is False:
                empty_flag = False
            else:
                # sub field is empty
                del data_info[key]
        else:
            empty_flag = False

    return data_info, empty_flag


def generate_nuscenes_camera_instances(info, nusc):

    # get bbox annotations for camera
    camera_types = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_FRONT_LEFT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT',
    ]

    empty_multicamera_instance = get_empty_multicamera_instances(camera_types)

    for cam in camera_types:
        cam_info = info['cams'][cam]
        # list[dict]
        ann_infos = get_nuscenes_2d_boxes(
            nusc,
            cam_info['sample_data_token'],
            visibilities=['', '1', '2', '3', '4'])
        empty_multicamera_instance[cam] = ann_infos

    return empty_multicamera_instance


def update_nuscenes_infos(data_root, pkl_path, out_dir):
    camera_types = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_FRONT_LEFT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT',
    ]
    
    radar_types = [
        'RADAR_FRONT', 
        'RADAR_FRONT_LEFT', 
        'RADAR_FRONT_RIGHT',  
        'RADAR_BACK_LEFT', 
        'RADAR_BACK_RIGHT'
    ]
    
    print(f'{pkl_path} will be modified. updated pkl will be saved to {out_dir}')
    if out_dir in pkl_path:
        print(f'Warning, you may overwriting '
              f'the original data {pkl_path}.')
    print(f'Reading from input file: {pkl_path}.')
    data_list = mmengine.load(pkl_path)
    METAINFO = {
        'classes':
        ('car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
         'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'),
    }
    nusc = NuScenes(
        version=data_list['metadata']['version'],
        dataroot=data_root,
        verbose=True)

    print('Start updating:')
    converted_list = []
    for i, ori_info_dict in enumerate(
            mmengine.track_iter_progress(data_list['infos'])):
        temp_data_info = get_empty_standard_data_info(
            camera_types=camera_types)
        temp_data_info['sample_idx'] = i
        temp_data_info['token'] = ori_info_dict['token']
        temp_data_info['ego2global'] = convert_quaternion_to_matrix(
            ori_info_dict['ego2global_rotation'],
            ori_info_dict['ego2global_translation'])  # ndarray(4, 4) or list
        temp_data_info['lidar_points']['num_pts_feats'] = ori_info_dict.get(
            'num_features', 5)
        temp_data_info['lidar_points']['lidar_path'] = Path(
            ori_info_dict['lidar_path']).name
        temp_data_info['lidar_points'][
            'lidar2ego'] = convert_quaternion_to_matrix(
                ori_info_dict['lidar2ego_rotation'],
                ori_info_dict['lidar2ego_translation'])  # ndarray(4, 4)
        # bc-breaking: Timestamp has divided 1e6 in pkl infos.
        temp_data_info['timestamp'] = ori_info_dict['timestamp'] / 1e6
        
        # my add
        temp_data_info['prev'] = ori_info_dict['prev']
        temp_data_info['next'] = ori_info_dict['next']
        temp_data_info['can_bus'] = convert_ndarry_type(ori_info_dict['can_bus'])  # ndarray(18,)
        temp_data_info['frame_idx'] = ori_info_dict['frame_idx']
        temp_data_info['scene_token'] = ori_info_dict['scene_token']
        
        for ori_sweep in ori_info_dict['sweeps']:
            temp_lidar_sweep = get_single_lidar_sweep()
            temp_lidar_sweep['lidar_points']['num_pts_feats'] = ori_sweep.get('num_features', 5)
            temp_lidar_sweep['lidar_points'][
                'lidar2ego'] = convert_quaternion_to_matrix(
                    ori_sweep['sensor2ego_rotation'],
                    ori_sweep['sensor2ego_translation'])  # ndarray(4, 4)
            temp_lidar_sweep['ego2global'] = convert_quaternion_to_matrix(
                ori_sweep['ego2global_rotation'],
                ori_sweep['ego2global_translation'])  # ndarray(4, 4)
            lidar2sensor = np.eye(4)
            rot = ori_sweep['sensor2lidar_rotation']
            trans = ori_sweep['sensor2lidar_translation']
            lidar2sensor[:3, :3] = rot.T
            lidar2sensor[:3, 3:4] = -1 * np.matmul(rot.T, trans.reshape(3, 1))
            temp_lidar_sweep['lidar_points'][
                'lidar2sensor'] = convert_ndarry_type(lidar2sensor, type='float32')  # ndarray(4, 4)
            temp_lidar_sweep['timestamp'] = ori_sweep['timestamp'] / 1e6
            file_path = ori_sweep['data_path']  # absolute path
            file_with_parent = osp.relpath(file_path, Path(file_path).parents[2])  # sweeps/LIDAR_TOP/001.bin
            temp_lidar_sweep['lidar_points']['lidar_path'] = file_with_parent
            temp_lidar_sweep['sample_data_token'] = ori_sweep[
                'sample_data_token']
            temp_data_info['lidar_sweeps'].append(temp_lidar_sweep)

        temp_data_info['images'] = {}
        for cam in ori_info_dict['cams']:
            empty_img_info = get_empty_img_info()
            empty_img_info['img_path'] = Path(
                ori_info_dict['cams'][cam]['data_path']).name
            empty_img_info['cam2img'] = convert_ndarry_type(ori_info_dict['cams'][cam][
                'cam_intrinsic'], type='float32')  # ndarray(3, 3)
            empty_img_info['sample_data_token'] = ori_info_dict['cams'][cam][
                'sample_data_token']
            # bc-breaking: Timestamp has divided 1e6 in pkl infos.
            empty_img_info[
                'timestamp'] = ori_info_dict['cams'][cam]['timestamp'] / 1e6
            empty_img_info['cam2ego'] = convert_quaternion_to_matrix(
                ori_info_dict['cams'][cam]['sensor2ego_rotation'],
                ori_info_dict['cams'][cam]['sensor2ego_translation'])  # ndarray(4, 4)
            lidar2sensor = np.eye(4)
            rot = ori_info_dict['cams'][cam]['sensor2lidar_rotation']
            trans = ori_info_dict['cams'][cam]['sensor2lidar_translation']
            lidar2sensor[:3, :3] = rot.T
            lidar2sensor[:3, 3:4] = -1 * np.matmul(rot.T, trans.reshape(3, 1))
            empty_img_info['lidar2cam'] = convert_ndarry_type(lidar2sensor, type='float32')  # ndarray(4, 4)
            temp_data_info['images'][cam] = empty_img_info
            
        # my add
        temp_data_info['radars'] = {}
        for radar in ori_info_dict['radars']:
            empty_radar_info = get_empty_radar_info()
            empty_radar_info['radar_path'] = Path(ori_info_dict['radars'][radar]['data_path']).name
            empty_radar_info['num_radar_feats'] = ori_info_dict['radars'][radar].get('num_features', 18)
            empty_radar_info['sample_data_token'] = ori_info_dict['radars'][radar][
                'sample_data_token']
            # bc-breaking: Timestamp has divided 1e6 in pkl infos.
            empty_radar_info[
                'timestamp'] = ori_info_dict['radars'][radar]['timestamp'] / 1e6
            empty_radar_info['radar2ego'] = convert_quaternion_to_matrix(
                ori_info_dict['radars'][radar]['sensor2ego_rotation'],
                ori_info_dict['radars'][radar]['sensor2ego_translation'])
            lidar2sensor = np.eye(4)
            rot = ori_info_dict['radars'][radar]['sensor2lidar_rotation']
            trans = ori_info_dict['radars'][radar]['sensor2lidar_translation']
            lidar2sensor[:3, :3] = rot.T
            lidar2sensor[:3, 3:4] = -1 * np.matmul(rot.T, trans.reshape(3, 1))
            empty_radar_info['lidar2radar'] = convert_ndarry_type(lidar2sensor, type='float32')
            temp_data_info['radars'][radar] = empty_radar_info
        
        for radar in ori_info_dict['radar_sweeps']:
            temp_data_info['radar_sweeps'][radar] = []
            for ori_sweep in ori_info_dict['radar_sweeps'][radar]:
                empty_radar_info = get_empty_radar_info()
                file_path = ori_sweep['data_path']  # absolute path
                file_with_parent = osp.relpath(file_path, Path(file_path).parents[2])  # sweeps/LIDAR_TOP/001.bin
                empty_radar_info['radar_path'] = file_with_parent
                empty_radar_info['num_radar_feats'] = ori_sweep.get('num_features', 18)
                empty_radar_info['sample_data_token'] = ori_sweep['sample_data_token']
                # bc-breaking: Timestamp has divided 1e6 in pkl infos.
                empty_radar_info['timestamp'] = ori_sweep['timestamp'] / 1e6
                empty_radar_info['radar2ego'] = convert_quaternion_to_matrix(
                    ori_sweep['sensor2ego_rotation'],
                    ori_sweep['sensor2ego_translation'])
                lidar2sensor = np.eye(4)
                rot = ori_sweep['sensor2lidar_rotation']
                trans = ori_sweep['sensor2lidar_translation']
                lidar2sensor[:3, :3] = rot.T
                lidar2sensor[:3, 3:4] = -1 * np.matmul(rot.T, trans.reshape(3, 1))
                empty_radar_info['lidar2radar'] = convert_ndarry_type(lidar2sensor, type='float32')
                temp_data_info['radar_sweeps'][radar].append(empty_radar_info)
        
        num_instances = ori_info_dict['gt_boxes'].shape[0] if 'gt_boxes' in ori_info_dict else 0
        ignore_class_name = set()
        for i in range(num_instances):
            empty_instance = get_empty_instance()
            empty_instance['bbox_3d'] = ori_info_dict['gt_boxes'][
                i, :].tolist()
            if ori_info_dict['gt_names'][i] in METAINFO['classes']:
                empty_instance['bbox_label'] = METAINFO['classes'].index(
                    ori_info_dict['gt_names'][i])
            else:
                ignore_class_name.add(ori_info_dict['gt_names'][i])
                empty_instance['bbox_label'] = -1
            empty_instance['bbox_label_3d'] = copy.deepcopy(
                empty_instance['bbox_label'])
            empty_instance['velocity'] = ori_info_dict['gt_velocity'][
                i, :].tolist()
            empty_instance['num_lidar_pts'] = ori_info_dict['num_lidar_pts'][i]
            empty_instance['num_radar_pts'] = ori_info_dict['num_radar_pts'][i]
            empty_instance['bbox_3d_isvalid'] = ori_info_dict['valid_flag'][i]
            empty_instance = clear_instance_unused_keys(empty_instance)
            temp_data_info['instances'].append(empty_instance)
        temp_data_info['cam_instances'] = generate_nuscenes_camera_instances(
            ori_info_dict, nusc)
        temp_data_info, _ = clear_data_info_unused_keys(temp_data_info)
        converted_list.append(temp_data_info)
    pkl_name = Path(pkl_path).name
    out_path = osp.join(out_dir, pkl_name)
    print(f'Writing to output file: {out_path}.')
    print(f'ignore classes: {ignore_class_name}')

    metainfo = dict()
    metainfo['categories'] = {k: i for i, k in enumerate(METAINFO['classes'])}
    if ignore_class_name:
        for ignore_class in ignore_class_name:
            metainfo['categories'][ignore_class] = -1
    metainfo['dataset'] = 'nuscenes'
    metainfo['version'] = data_list['metadata']['version']
    metainfo['info_version'] = '1.1'
    converted_data_info = dict(metainfo=metainfo, data_list=converted_list)

    mmengine.dump(converted_data_info, out_path, 'pkl')


def parse_args():
    parser = argparse.ArgumentParser(description='Arg parser for data coords '
                                     'update due to coords sys refactor.')
    parser.add_argument(
        '--dataset', type=str, default='kitti', help='name of dataset')
    parser.add_argument(
        '--pkl-path',
        type=str,
        default='./data/kitti/kitti_infos_train.pkl ',
        help='specify the root dir of dataset')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='converted_annotations',
        required=False,
        help='output direction of info pkl')
    args = parser.parse_args()
    return args


def update_pkl_infos(dataset, data_root, out_dir, pkl_path):
    if dataset.lower() == 'nuscenes':
        update_nuscenes_infos(data_root=data_root, pkl_path=pkl_path, out_dir=out_dir)
    else:
        raise NotImplementedError(f'Do not support convert {dataset} to v2.')


if __name__ == '__main__':
    args = parse_args()
    if args.out_dir is None:
        args.out_dir = args.root_dir
    update_pkl_infos(
        dataset=args.dataset, out_dir=args.out_dir, pkl_path=args.pkl_path)
