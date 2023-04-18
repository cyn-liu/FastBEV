#!/usr/bin/env bash

#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=7
NGPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)


export OMP_NUM_THREADS=8
export NCCL_DEBUG=info
export DEBUG=True


CONFIG="./projects/configs/roadside.py"
CHECKPOINT='/home/xxxx/code/MSBEVFusion/work_dirs/roadside_train_half_res_aug_20230405-2208/epoch_50.pth'

TEST="./tools/custom_test.py"
PORT=${PORT:-29503}

PYTHONPATH="$(dirname $0)":$PYTHONPATH \
torchrun --nproc_per_node=$NGPUS --master_port=$PORT $TEST $CONFIG $CHECKPOINT --launcher pytorch ${@:1}
