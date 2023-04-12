# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from pathlib import Path

from mmengine import Config, DictAction

from debug_print import DEBUG_printv2


def parse_args():
    parser = argparse.ArgumentParser(description='Print the whole config')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='arguments in dict')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    print(f'Config:\n{cfg.pretty_text}')

    file_name = Path(args.config).stem
    file = file_name + '_full.py'

    with open(file, "w") as file:
        file.write("{}\n".format(cfg.pretty_text))


if __name__ == '__main__':
    main()