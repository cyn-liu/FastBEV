#!/usr/bin/env bash

#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=4
NGPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)


export OMP_NUM_THREADS=8
export NCCL_DEBUG=info
export DEBUG=True
#export CUDA_LAUNCH_BLOCKING=1

#CONFIG="./projects/configs/msfusion/msbevfusion.py"
#CONFIG="./pointpillars_hv_fpn_head-free-anchor_sbn-all_8xb4-2x_nus-3d_full.py"
#CONFIG="./projects/configs/fastbev/fastbev.py"
#CONFIG="./projects/configs/roadside.py"
#CONFIG="./projects/configs/cmt/cmt.py"
CONFIG="./projects/configs/bevformer/bevformer.py"
TRAIN="./tools/custom_train.py"
PORT=${PORT:-25620}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun --nproc_per_node=$NGPUS --master_port=$PORT \
    $TRAIN $CONFIG --launcher pytorch ${@:1}

# 解释:
# "export OMP_NUM_THREADS=8" 是一条 Shell 命令，它将 OMP_NUM_THREADS 环境变量的值设为 8。
# OMP_NUM_THREADS 是一个用于控制并行计算的环境变量。它的值指定了当前的 OpenMP 程序应该使用多少个线程。
# 如果您正在使用 OpenMP 并行库来编写并行程序，您可以使用 OMP_NUM_THREADS 环境变量来控制程序使用的线程数。
# "export" 命令用于将变量设置为环境变量，这样它就可以在当前 Shell 会话中的所有程序中使用。

# "export NCCL_DEBUG=info" 将 NCCL_DEBUG 环境变量设置为 "info"。
# NCCL_DEBUG 是一个用于控制 NCCL（一种用于在多个 GPU 之间传输数据的库）调试信息的环境变量。
# 如果希望获取 NCCL 调试信息，可以将 NCCL_DEBUG 设置为 "info" 或 "verbose"。

# "PYTHONPATH=$(dirname $0)/..:$PYTHONPATH" 是一条 Shell 命令，它用于设置 PYTHONPATH 环境变量的值。
# PYTHONPATH 是一个用于控制 Python 搜索模块的路径的环境变量。它包含一系列目录的列表，Python 在导入模块时会搜索这些目录。
# 这条命令中，$0 指的是当前脚本的文件名。因此，$(dirname $0) 表示当前脚本所在的目录。然后，"/.." 表示当前目录的父目录。
# 最后，这条命令将新目录添加到 PYTHONPATH 的开头，然后将原来的 PYTHONPATH 值添加到末尾。
# 这意味着新目录会成为 Python 搜索模块的首选目录。

# ${@:1}: 会将当前 Shell 脚本的所有参数从第 1 个参数开始提取出来。