#!/usr/bin/env bash

#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=6
NGPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)


export OMP_NUM_THREADS=8
export NCCL_DEBUG=info
export DEBUG=True

# ppt
#CONFIG="./ckpt/ppt_secfpn/pointpillars_hv_secfpn_sbn-all_8xb4-amp-2x_nus-3d_full.py"
#CHECKPOINT='./ckpt/ppt_secfpn/hv_pointpillars_secfpn_sbn-all_fp16_2x8_2x_nus-3d_20201020_222626-c3f0483e.pth'

# centerpoint
#CONFIG="./ckpt/centerpoint_secfpn/centerpoint_voxel0075_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d_full.py"
#CHECKPOINT='./ckpt/centerpoint_secfpn/centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_011659-04cb3a3b.pth'

# fastbev
CONFIG="./projects/configs/fastbev/fastbev-tiny.py"
CHECKPOINT='./work_dirs/fastbev-tiny_train_20230419-1302//epoch_1.pth'

#CONFIG="./projects/configs/roadside.py"
#CHECKPOINT='/home/fuyu/zhangbin/code/MSBEVFusion/work_dirs/roadside_train_half_res_aug_20230405-2208/epoch_50.pth'

TEST="./tools/custom_test.py"
PORT=${PORT:-29503}

PYTHONPATH="$(dirname $0)":$PYTHONPATH \
torchrun --nproc_per_node=$NGPUS --master_port=$PORT $TEST $CONFIG $CHECKPOINT --launcher pytorch ${@:1}
