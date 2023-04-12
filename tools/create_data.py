import argparse
from os import path as osp
from datetime import datetime

import pretty_errors

from dataset_converters.create_gt_database import create_groundtruth_database
from dataset_converters import nuscenes_converter as nuscenes_converter
from dataset_converters.update_infos_to_v2 import update_pkl_infos

from tools.utils.logger_utils import logger


# 设置
pretty_errors.configure(
    separator_character = '*', # 用于创建标题行的字符，如果设置为None或者''则标题被禁用，之后的报错信息截图中可以理解该设置的意义。
    filename_display    = pretty_errors.FILENAME_FULL, # 如何显示文件名，可以是pretty_errors.FILENAME_COMPACT、pretty_errors.FILENAME_EXTENDED、pretty_errors.FILENAME_FULL
    line_number_first   = False, # 启用后，将首先显示行号，而不是文件名，二者会交换位置
    display_link        = True, # 启用后，错误下方会写入一个链接，VSCode允许您单击该链接。
    lines_before        = 5, # 显示发生异常的语句之前5行代码
    lines_after         = 3, # 显示发生异常的语句之后2行代码
    line_color          = pretty_errors.RED + '> ' + pretty_errors.default_config.line_color, # 转移序列设置导致异常的代码行的颜色
    code_color          = '  ' + pretty_errors.default_config.line_color, # 用于设置导致异常的代码行的颜色。
    truncate_code       = True, # 启用后，每行代码都将被截断显示
    display_locals      = True # 启用后，出现在顶部堆栈帧代码中的局部变量将与其值一起显示。
)


def nuscenes_data_prep(
                    root_path, 
                    can_bus_root_path, 
                    info_prefix, 
                    version, 
                    dataset_name, 
                    out_dir,
                    updated_out_dir,
                    database_save_path=None,
                    max_sweeps=10,
                    create_info=True,
                    creat_2d_anno=False,
                    update_info=True,
                    create_GT_database=True):
    if create_info:
        logger.info(f"Creating GT INFO")
        nuscenes_converter.create_nuscenes_infos(
                root_path, out_dir, can_bus_root_path, info_prefix, version=version, max_sweeps=max_sweeps)
    else:
        logger.info(f"skip create info!!!")

    # after update_pkl_infos, 2d annotation infos will be saved in .pkl(['data_list'][i]['cam_instances'])
    # if creat_2d_anno and version != 'v1.0-test':
    #     logger.info(f"Creating 2d GT annotations")
    #     info_train_path = osp.join(out_dir, f'{info_prefix}_infos_train.pkl')
    #     info_val_path = osp.join(out_dir, f'{info_prefix}_infos_val.pkl')
    #     nuscenes_converter.export_2d_annotation(
    #         root_path, info_train_path, version=version)
    #     nuscenes_converter.export_2d_annotation(
    #         root_path, info_val_path, version=version)
    # else:
    #     logger.info(f"skip create 2d GT annotations!!!")
        
    if update_info or create_info or creat_2d_anno:
        logger.info(f"Update pkl infos: {version}")
        if version == 'v1.0-test':
            info_test_path = osp.join(out_dir, f'{info_prefix}_infos_test.pkl')
            logger.info(f"Update pkl infos save path: {info_test_path}")
            update_pkl_infos('nuscenes', data_root=root_path, out_dir=updated_out_dir, pkl_path=info_test_path)
            return
        else:
            info_val_path = osp.join(out_dir, f'{info_prefix}_infos_val.pkl')
            logger.info(f"Update pkl infos save path: {info_val_path}")
            info_train_path = osp.join(out_dir, f'{info_prefix}_infos_train.pkl')
            logger.info(f"Update pkl infos save path: {info_train_path}")
            update_pkl_infos('nuscenes', data_root=root_path, out_dir=updated_out_dir, pkl_path=info_val_path)
            update_pkl_infos('nuscenes', data_root=root_path, out_dir=updated_out_dir, pkl_path=info_train_path)

    if create_GT_database and version != 'v1.0-test':
        logger.info(f"Creating GT base")
        create_groundtruth_database(dataset_name, root_path, info_prefix, 
                                    f"{updated_out_dir}/{info_prefix}_infos_train.pkl",
                                    database_save_path=database_save_path)
    else:
        logger.info(f"skip create groundtruth database!!!")


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='nuscenes', help='name of the dataset')
parser.add_argument('--root-path', type=str, default='./data/nuscenes', help='specify the root path of dataset')
parser.add_argument('--canbus', type=str, default='./data/nuscenes', help='specify the root path of nuScenes canbus')
parser.add_argument('--version', type=str, default='v1.0-trainval', required=False,
                                help='specify the dataset version, v1.0-trainval or v1.0-mini or v1.0-test')
parser.add_argument('--max-sweeps', type=int, default=10, required=False, help='specify sweeps of lidar per example')
parser.add_argument('--out-dir', type=str, default='./data', required=False, help='name of info pkl')
parser.add_argument('--updated-out-dir', type=str, default='./data', required=False, help='name of updated info pkl')
parser.add_argument('--db-save-path', type=str, default='./data', required=False, help='database_save_path')
parser.add_argument('--extra-tag', type=str, default='nuscenes')


args = parser.parse_args()

if __name__ == '__main__':
    # for pycharm debug
    import ctypes
    libgcc_s = ctypes.CDLL('libgcc_s.so.1')

    logger.info("-------------------------------Start-----------------------------------")
    from mmdet3d.utils import register_all_modules
    register_all_modules()
    
    # Set to spawn mode to avoid stuck when process dataset creating
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    
    s_time = datetime.now()

    if args.dataset == 'nuscenes':
        nuscenes_data_prep(root_path=args.root_path,
                           can_bus_root_path=args.canbus,
                           info_prefix=args.extra_tag,
                           version=args.version,
                           dataset_name='NuScenesDataset',
                           out_dir=args.out_dir,
                           updated_out_dir=args.updated_out_dir,
                           database_save_path=args.db_save_path,
                           max_sweeps=args.max_sweeps,
                           create_info=True,
                           update_info=True,
                           create_GT_database=True)
    else:
        logger.info(f"Not Supported dataset: {args.dataset}")

    e_time = datetime.now()
    run_time_s = int((e_time - s_time).seconds)
    hours = run_time_s // 3600
    minutes = run_time_s % 3600 // 60
    seconds = run_time_s % 60
    logger.info(f"all was done in {hours}h {minutes} m {seconds} s")
    logger.info("-------------------------------end-----------------------------------")
