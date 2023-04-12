# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path

import mmcv
import numpy as np
from os import path as osp
import glob
import mmengine
from pathlib import Path
# import os
# from pyquaternion import Quaternion
# from shapely.geometry import MultiPoint, box
# from typing import List, Tuple, Union
# from mmdet3d.core.bbox.box_np_ops import points_cam2img
# from mmdet3d.datasets import NuScenesDataset
from datetime import datetime


CLASSES = [ "Pedestrian", "Car","MotorcyleRider", "Crane", "Motorcycle", "Bus", "BicycleRider", "Van", "Excavator", "TricycleRider","Truck"]
CLS_MAP = {cls: idx for idx, cls in enumerate(CLASSES)}
CAM_MAP = {"CAM_FRONT": "front",
           "CAM_LEFT": "left",
           "CAM_RIGHT": "right",}
# camera_types = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
#                 'CAM_BACK',  'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',]

export_show_pkl = True
test = False
show_pkl_len = 10

def create_roadside_pkl(label_dir=None):

    """Generate the train/val infos from the raw data.

    Args:
        nusc (:obj:`NuScenes`): Dataset class in the nuScenes dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool): Whether use the test mode. In the test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int): Max number of sweeps. Default: 10.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """

    data_dir = "/home/fuyu/zhangbin/datasets/roadside"
    label_dir = os.path.join(data_dir,"labels")
    label_files_front = glob.glob(label_dir + "/*_front.txt")

    metainfo = dict()
    metainfo['categories'] = {k: i for i, k in enumerate(CLASSES)}

    metainfo['dataset'] = 'roadside'
    metainfo['version'] = "v1.0.0"
    metainfo['info_version'] = 'v1.0.0'

    calibs = {"front": {"extrinsic":
                            np.array([0.00491626, -0.999262, -0.038084, 0.0681321, -0.427802, 0.0323217, -0.903295, 0.734028,
                             0.903859, 0.0207332, -0.427327, 0.150924, 0, 0, 0, 1]).reshape(4,4),
                        "intrinsic":
                            np.array([433.77155, 0.0, 630.76349, 0.0, 430.27045, 360.20135, 0.0, 0.0, 1.0]).reshape(3,3)},
              "left": {"extrinsic":
                            np.array(
                                [0.934122, -0.347548, -0.0814049, -2.23403, -0.152797, -0.183215, -0.971135,
                                 1.7682, 0.322602, 0.919597, -0.22425, 0.142537, 0.0, 0.0, 0.0, 1.0]).reshape(4, 4),
                        "intrinsic":
                            np.array([438.0170150785, 0.0, 640.5681684371, 0.0, 438.309381477, 357.37100860538, 0.0, 0.0, 1.0]).reshape(3,
                                                                                                                    3)},
              "right": {"extrinsic":
                            np.array(
                                [-0.917781, -0.387847, 0.0851653, 2.48184, -0.134267, 0.101265, -0.985757,
                                 1.83453, 0.373698, -0.916144, -0.145014, -0.438711, 0.0, 0.0, 0.0, 1.0]).reshape(4, 4),
                        "intrinsic":
                            np.array([439.5533468, 0.0, 642.093202381, 0.0, 441.628762688, 350.5719531952173, 0.0, 0.0, 1.0]).reshape(3,
                                                                                                                    3)},
              }
    if export_show_pkl:
        show_nusc_infos = [[] for i in range(show_pkl_len + 1)]
        sorted_data = sorted(enumerate(label_files_front), key=lambda x:
                (datetime.strptime(x[1].split("_")[0].split("/")[-1], '%Y-%m-%d-%H-%M-%S'),
                    int(x[1].split("_")[1].split(".")[0])))
        sorted_indices = [x[0] for x in sorted_data]

        label_files_front = [label_files_front[i] for i in sorted_indices]
    elif test:
        label_files_front = [label_files_front[0]] * 1000
    else:
        import random
        random.shuffle(label_files_front)
    import random
    random.shuffle(label_files_front)

    print(f"label_files_front: {len(label_files_front)}")
    split_val = int(len(label_files_front) * 0.9)
    print(label_files_front[0], label_files_front[1])

    train_nusc_infos = []
    val_nusc_infos = []
    for idx,label_file_front in enumerate(mmengine.track_iter_progress(label_files_front)):
        info_cameras = {}
        for CAM_TYPE,cam_type in CAM_MAP.items():

            label_file = label_file_front.replace("front", cam_type)
            image_file = label_file.replace("/labels/","/images/").replace(".txt",".png")

            img_path = Path(image_file).name
            cam2img = calibs[cam_type]["intrinsic"]
            lidar2cam = calibs[cam_type]["extrinsic"]

            info_cameras[CAM_TYPE] = dict(img_path=img_path,
                                          cam2img= cam2img,
                                          lidar2cam=lidar2cam)

        # process_3d
        label_file_3d = label_file_front.replace("/labels/","/labels_3d/").replace("_front.txt",".json")
        # print(label_file_3d)
        labels_3d = json.load(open(label_file_3d))


        instances = []

        for idx3d,label in enumerate(labels_3d):
            obj_id,obj_type,position,rotation,scale = label['obj_id'],label['obj_type'],label["psr"]["position"],label["psr"]["rotation"],label["psr"]["scale"]
            label_id = CLS_MAP[obj_type]
            pos = list(position.values())
            rot = list(rotation.values())[-1]
            scale = list(scale.values())
            bbox_3d = pos + scale + [rot]   # 速度 [0., 0.]  config code_size custom_values
            instance_info = dict(
                bbox_label=label_id,
                bbox_3d= bbox_3d,
                bbox_3d_isvalid=True,
                bbox_label_3d=label_id
            )
            instances.append(instance_info)
        train_nusc_infos.append(
            dict(
                sample_idx=idx, images=info_cameras, instances=instances))


        if export_show_pkl:
            show_nusc_infos[idx// (len(label_files_front)//(show_pkl_len))].append(
                dict(
                sample_idx=idx,  images=info_cameras,  instances=instances))
        if idx <= split_val:
            pass
            ## key: cam_instances 2D info
            # train_nusc_infos.append(
            #     dict(
            #     sample_idx=idx,  images=info_cameras,  instances=instances))
        else:
            val_nusc_infos.append(
                dict(
                sample_idx=idx,  images=info_cameras,  instances=instances))

    print('train sample: {}, val sample: {}'.format(
        len(train_nusc_infos), len(val_nusc_infos)))

    if export_show_pkl:
        for i in range(show_pkl_len+1):
            data = dict(data_list=show_nusc_infos[i], metainfo=metainfo)
            show_val_path = osp.join(data_dir,
                                     '{}_infos_show_{}.pkl'.format("roadside",i))
            mmengine.dump(data, show_val_path)
    else:
        data = dict(data_list=train_nusc_infos, metainfo=metainfo)
        info_path = osp.join(data_dir,
                             '{}_infos_train.pkl'.format("roadside"))
        mmengine.dump(data, info_path)

        data = dict(data_list=val_nusc_infos, metainfo=metainfo)
        info_val_path = osp.join(data_dir,
                                 '{}_infos_val.pkl'.format("roadside"))
        mmengine.dump(data, info_val_path)

    print("====over====")


create_roadside_pkl()