#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=7
#export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
NGPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)


export OMP_NUM_THREADS=8
export NCCL_DEBUG=info
export DEBUG=True

# fastbev
CONFIG="./projects/configs/roadside.py"
CHECKPOINT='/data/fuyu/output/mmbevdet/work_dirs/roadside_train_half_res_aug_20230405-2208/epoch_50.pth'

TEST="./tools/export.py"
PORT=${PORT:-29103}

PYTHONPATH="$(dirname $0)":$PYTHONPATH \
torchrun --nproc_per_node=$NGPUS --master_port=$PORT $TEST $CONFIG $CHECKPOINT --launcher pytorch ${@:1}
