# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import pretty_errors

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmengine import mkdir_or_exist

from mmdet3d.utils import register_all_modules, replace_ceph_backend


# for pycharm debug
import ctypes
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
        help='the directory to save the file containing evaluation metrics')
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
        cfg.work_dir = cfg.work_dir + '_test'
        if 'pred_score_thr' in cfg:
            cfg.work_dir = cfg.work_dir + '_' + str(cfg.get('pred_score_thr'))
        cfg.work_dir = cfg.work_dir + cfg.get('work_dir_postfix', '')

    mkdir_or_exist(cfg.work_dir)
    os.system(f"cp -r ./projects ./{cfg.work_dir}/")

    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start testing
    runner.test()


if __name__ == '__main__':
    main()
