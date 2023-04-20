# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
from pathlib import Path
import numpy as np

import pretty_errors

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmengine.runner.checkpoint import load_checkpoint
from mmengine import mkdir_or_exist

from mmdet3d.utils import register_all_modules, replace_ceph_backend


# for pycharm debug
import ctypes
import torch
from mmcv.cnn.utils import fuse_conv_bn

libgcc_s = ctypes.CDLL('libgcc_s.so.1')

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


# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet3D test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the onnx')
    parser.add_argument(
        '--ceph', action='store_true', help='Use ceph as data storage backend')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--task',
        type=str,
        choices=[
            'mono_det', 'multi-view_det', 'lidar_det', 'lidar_seg',
            'multi-modality_det'
        ],
        help='Determine the visualization method depending on the task.')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # Turn on visualization
        visualization_hook['draw'] = True
        if args.show:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = args.wait_time
        if args.show_dir:
            visualization_hook['test_out_dir'] = args.show_dir
        visualization_hook['vis_task'] = args.task
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')

    return cfg


def main():
    args = parse_args()

    # register all modules in mmdet3d into the registries
    # do not init the default scope here because it will be init in the runner
    register_all_modules(init_default_scope=False)

    # load config
    cfg = Config.fromfile(args.config)

    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmengine.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = os.path.relpath(cfg.plugin_dir)  # relpath, like "mmdet3d/plugin/"
                # Strip the file name, return the file path, and if it ends with /, return the input minus the last /
                if not plugin_dir.endswith('/'):
                    plugin_dir = plugin_dir + '/'
                _module_dir = os.path.dirname(plugin_dir)

                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(f"import module : {_module_path}")
                plg_lib = importlib.import_module(_module_path)
            else:
                raise ValueError(f"set 'plugin' but could not find plugin_dir")

    # TODO: We will unify the ceph support approach with other OpenMMLab repos
    if args.ceph:
        cfg = replace_ceph_backend(cfg)

    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join(cfg.get('work_dir_prefix', './work_dirs'),
                                osp.splitext(osp.basename(args.config))[0])
        cfg.work_dir = cfg.work_dir + '_onnx'
        cfg.work_dir = cfg.work_dir + cfg.get('work_dir_postfix', '')

    mkdir_or_exist(cfg.work_dir)
    os.system(f"cp -r ./projects ./{cfg.work_dir}/")

    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if cfg.optim_wrapper.type == 'AmpOptimWrapper':
        suffix = 'fp-amp'
    else:
        suffix = 'fp-32'

    bs, n_image, img_size = cfg.test_batch_size, cfg.n_cams, cfg.img_input_final_dim  # h, w
    input_size = [bs, n_image, *img_size, 3]  # bgr


    def __date():
        import datetime
        return datetime.datetime.now().strftime('%Y%m%d-%H%M')

    onnx_name = Path(cfg.load_from).parent.name + '—' + Path(cfg.load_from).stem + '_' + __date() + '.onnx'
    onnx_path = os.path.join(cfg.work_dir, onnx_name)
    cfg.model.type = cfg.model.type + 'TRT'

    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    model = runner.build_model(cfg.model)
    load_checkpoint(model, cfg.load_from)

    DEPLOY = False # False True
    img = test(input_size, use_fixed_value=False, fixed_value=2.2)

    # point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    # bev_size = [200, 200]
    # num_points_in_pillar = 6

    point_cloud_range = cfg["point_cloud_range"]
    bev_size = cfg["bev_size"]
    num_points_in_pillar = cfg["num_points_in_pillar"]
    n_voxels = bev_size + [num_points_in_pillar]

    voxel_size = [(point_cloud_range[3] - point_cloud_range[0]) / bev_size[0],
                       (point_cloud_range[4] - point_cloud_range[1]) / bev_size[1],
                       (point_cloud_range[5] - point_cloud_range[2]) / num_points_in_pillar]
    origin =[(point_cloud_range[i] + point_cloud_range[i+3])/2 for i in range(3) ]

    projection = get_projection("nuscenes")## roadside nuscenes

    default_cfg = dict(
        DEPLOY=DEPLOY,
        input_size=cfg.img_input_final_dim,
        params = dict(
            n_voxels=n_voxels,
            voxel_size = voxel_size,
            origin = origin,
            projection=projection,

        ),
        lidar2cam=[[
         [[ 0.0049, -0.9993, -0.0381,  0.0681],
          [-0.4278,  0.0323, -0.9033,  0.7340],
          [ 0.9039,  0.0207, -0.4273,  0.1509],
          [ 0.0000,  0.0000,  0.0000,  1.0000]],

         [[ 0.9341, -0.3475, -0.0814, -2.2340],
          [-0.1528, -0.1832, -0.9711,  1.7682],
          [ 0.3226,  0.9196, -0.2243,  0.1425],
          [ 0.0000,  0.0000,  0.0000,  1.0000]],

         [[-0.9178, -0.3878,  0.0852,  2.4818],
          [-0.1343,  0.1013, -0.9858,  1.8345],
          [ 0.3737, -0.9161, -0.1450, -0.4387],
          [ 0.0000,  0.0000,  0.0000,  1.0000]]]],
        cam2img=[[
         [[216.8858,   0.0000, 315.3817,   0.0000],
          [  0.0000, 215.1352, 192.1007,   0.0000],
          [  0.0000,   0.0000,   1.0000,   0.0000],
          [  0.0000,   0.0000,   0.0000,   1.0000]],

         [[219.0085,   0.0000, 320.2841,   0.0000],
          [  0.0000, 219.1547, 190.6855,   0.0000],
          [  0.0000,   0.0000,   1.0000,   0.0000],
          [  0.0000,   0.0000,   0.0000,   1.0000]],

         [[219.7767,   0.0000, 321.0466,   0.0000],
          [  0.0000, 220.8144, 187.2860,   0.0000],
          [  0.0000,   0.0000,   1.0000,   0.0000],
          [  0.0000,   0.0000,   0.0000,   1.0000]]]])

    model.set_cfg(default_cfg)

    model = fuse_conv_bn(model)
    if not DEPLOY:
        output = model(img)
        exit()

    with torch.no_grad():
        try:
            torch.onnx.export(
                model,
                (img),
                onnx_path,
                verbose=False,
                opset_version=12-int(DEPLOY),
                input_names=['input'],
                output_names=["output"],
                enable_onnx_checker=False
            )
        except torch.onnx.utils.ONNXCheckerError:
            print("check fail , it is nothing , dont worry ")
    print(f"ONNX saved in {onnx_path}")

def test(input_size, use_fixed_value=False, fixed_value=2.2):
    if use_fixed_value:
        input_img = torch.ones(input_size) * fixed_value
    else:
        from mmcv import imread
        bs, n, h, w, _= input_size
        mean = [123.675, 116.28, 103.53][: : -1]
        std = [58.395, 57.12, 57.375][: : -1]
        mean_tensor = torch.tensor(mean).view(-1, 1, 1)
        std_tensor = torch.tensor(std).view(-1, 1, 1)

        def get_aug_cfg(H, W, fH,  fW):
            resize = float(fW) / float(W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h_start = (newH - fH) // 2
            crop_w_start = (newW - fW) // 2
            crop = (crop_w_start, crop_h_start, crop_w_start + fW, crop_h_start + fH)
            pad_data = (0.0, 0.0, 0.0, 0.0)
            pad_color = (0.0, 0.0, 0.0)
            pad = (pad_data, pad_color)
            augment = dict(resize_dims=resize_dims, crop=crop, flip=False, rotate=0., pad=pad)
            return augment
        # # # ROADSIDE
        # img_name_list = [
        #     '/data/fuyu/yolo/roadside_yolo/images/2022-05-09-11-45-03_000346_front.png',
        #     '/data/fuyu/yolo/roadside_yolo/images/2022-05-09-11-45-03_000346_left.png',
        #     '/data/fuyu/yolo/roadside_yolo/images/2022-05-09-11-45-03_000346_right.png']
        # Nuscenes
        img_name_list = [
            "/data/fuyu/dataset/nuscenes/samples/CAM_FRONT/n015-2018-08-02-17-16-37+0800__CAM_FRONT__1533201471912460.jpg",
            "/data/fuyu/dataset/nuscenes/samples/CAM_FRONT_RIGHT/n015-2018-08-02-17-16-37+0800__CAM_FRONT_RIGHT__1533201471920339.jpg",
            "/data/fuyu/dataset/nuscenes/samples/CAM_FRONT_LEFT/n015-2018-08-02-17-16-37+0800__CAM_FRONT_LEFT__1533201471904844.jpg",
            "/data/fuyu/dataset/nuscenes/samples/CAM_BACK/n015-2018-08-02-17-16-37+0800__CAM_BACK__1533201471937525.jpg",
            "/data/fuyu/dataset/nuscenes/samples/CAM_BACK_LEFT/n015-2018-08-02-17-16-37+0800__CAM_BACK_LEFT__1533201471947423.jpg",
            "/data/fuyu/dataset/nuscenes/samples/CAM_BACK_RIGHT/n015-2018-08-02-17-16-37+0800__CAM_BACK_RIGHT__1533201471927893.jpg"]

        img_list = []
        for img_name in img_name_list:
            single_img = imread(img_name)  # h w c (bgr)
            height, width, _ = single_img.shape
            augment_cfg = get_aug_cfg(height, width, h, w)
            single_img = augment_image(single_img, **augment_cfg)
            single_img = single_img.transpose(2, 0, 1)  # h w c (bgr) -> c h w (bgr)
            single_img = torch.from_numpy(single_img)
            single_img = single_img.float()
            single_img = (single_img - mean_tensor) / std_tensor
            single_img = single_img.permute(1, 2, 0)  # c h w (bgr) -> h w c (bgr)
            img_list.append(single_img)
        input_img = torch.stack(img_list, dim=0).unsqueeze(0)  # bs n h w c (bgr)

    assert list(input_img.shape) == input_size, f"input_img.shape: {input_img.shape}, but input_size: {input_size}"
    return input_img

def augment_image(img, resize_dims, crop, flip, rotate, pad):
    from mmcv import imresize, imflip, imrotate, impad
    resized_img = imresize(img, resize_dims)

    img = crop_img(resized_img, crop)

    if flip:
        img = imflip(img, 'horizontal')

    img = imrotate(img, -rotate / np.pi * 180)

    if any(x != 0 for x in pad[0]):
        img = impad(img, padding=pad[0], pad_val=pad[1])
    return img

def crop_img(img, crop, pad_fill=0):
    resized_img = np.ones((crop[3] - crop[1], crop[2] - crop[0], 3)) * pad_fill

    hsize, wsize = crop[3] - crop[1], crop[2] - crop[0]
    dh, dw, sh, sw = crop[1], crop[0], 0, 0

    if dh < 0:
        sh = -dh
        hsize += dh
        dh = 0
    if dh + hsize > img.shape[0]:
        hsize = img.shape[0] - dh
    if dw < 0:
        sw = -dw
        wsize += dw
        dw = 0
    if dw + wsize > img.shape[1]:
        wsize = img.shape[1] - dw
    resized_img[sh: sh + hsize, sw: sw + wsize] = img[dh: dh + hsize, dw: dw + wsize]
    return resized_img

def get_projection(name ):
    projection = dict(
    nuscenes =
    [   138.9468,     90.3058,      2.6294,    -66.8441,
          0.7966,     39.1664,   -138.5949,    -73.1480,
         -0.0040,      0.9998,      0.0187,     -0.7629,
          0.0000,      0.0000,      0.0000,      1.0000,

       150.6518,    -66.5482,     -3.1942,    -34.7508,
         29.3922,     23.7879,   -138.4704,    -74.5002,
          0.8339,      0.5520,      0.0047,     -0.7533,
          0.0000,      0.0000,      0.0000,      1.0000,

         5.6111,    166.7817,      4.0304,    -94.9646,
        -28.5383,     23.5625,   -139.5369,    -72.5685,
         -0.8198,      0.5725,      0.0116,     -0.7393,
          0.0000,      0.0000,      0.0000,      1.0000,

       -89.4211,    -90.8019,     -1.5625,    -83.1825,
          0.7162,    -34.8258,    -89.2756,    -56.8892,
         -0.0045,     -1.0000,     -0.0076,     -0.9097,
          0.0000,      0.0000,      0.0000,      1.0000,

      -126.4499,    103.5008,      0.8873,    -71.1427,
        -32.0534,     -7.0667,   -139.2144,    -49.5560,
         -0.9482,     -0.3163,     -0.0293,     -0.4340,
          0.0000,      0.0000,      0.0000,      1.0000,

        33.5314,   -160.9706,     -6.6550,     -5.2437,
         34.4051,     -7.8389,   -139.1622,    -53.9746,
          0.9342,     -0.3562,     -0.0194,     -0.4277,
          0.0000,      0.0000,      0.0000,      1.0000],
      roadside=
      [71.53172, -52.54671, -35.75775, 15.59389,
                    20.39917, 2.73410, -69.10509, 46.72696,
                    0.90386, 0.02073, -0.42733, 0.15092,

                    76.97623, 54.60408, -22.41302, -110.90480,
                    7.00734, 33.80035, -63.89751, 103.67227,
                    0.32260, 0.91960, -0.22425, 0.14254,

                    -20.43310, -94.84116, -6.95973, 101.15097,
                    10.08508, -37.30505, -61.20711, 80.73156,
                    0.37370, -0.91614, -0.14501, -0.43871]

    )


    return projection[name]



if __name__ == '__main__':
    main()
