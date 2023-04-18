# FastBEV

## Abstract
base: https://github.com/Sense-GVT/Fast-BEV ,

delete time sequence

update mm releated ,

add onnx export for tensorrt

# 用法

### 测试环境一

#### 本地

- cuda 10.2
- cudnn 8.4.0

#### 服务器

- cuda 11.7
- cudnn 8.4.0

#### 基础

- Python = 3.8
- [PyTorch](https://github.com/pytorch/pytorch) = 1.10.0
- [mmengine](http://github.com/open-mmlab/mmsegmentation) = 0.7.0
- [mmcv](https://github.com/open-mmlab/mmcv) = 2.0.0rc4 (>=2.0.0)
- [mmdetection](http://github.com/open-mmlab/mmdetection) = 3.0.0rc6 (>=3.0.0)
- [mmdetection3d](http://github.com/open-mmlab/mmdetection3d) = 1.1.0rc3 (>= 1.1.0)

### 测试环境二

#### 服务器

- cuda 11.7
- cudnn 8.4.0

#### 基础

- Python = 3.10
- [PyTorch](https://github.com/pytorch/pytorch) = 2.1.0
- [mmengine](http://github.com/open-mmlab/mmsegmentation) = 0.7.0
- [mmcv](https://github.com/open-mmlab/mmcv) = 2.0.0rc4 (>=2.0.0)
- [mmdetection](http://github.com/open-mmlab/mmdetection) = 3.0.0rc6 (>=3.0.0)
- [mmdetection3d](http://github.com/open-mmlab/mmdetection3d) = 1.1.0rc3 (>= 1.1.0)



# Getting Started

* [Installation](docs/install.md)
* [Prepare Dataset](docs/prepare_dataset.md)
* [Run and Eval](docs/getting_started.md)

### Evaluation

We also provide instructions for evaluating our pretrained models. Please download the checkpoints using the following script:

```bash
./tools/download_pretrained.sh
```

Then, you will be able to run:

```bash
torchpack dist-run -np 8 python tools/test.py [config file path] pretrained/[checkpoint name].pth --eval [evaluation type]
```

For example, if you want to evaluate the detection variant of BEVFusion, you can try:

```bash
torchpack dist-run -np 8 python tools/test.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml pretrained/bevfusion-det.pth --eval bbox
```

While for the segmentation variant of BEVFusion, this command will be helpful:

```bash
torchpack dist-run -np 8 python tools/test.py configs/nuscenes/seg/fusion-bev256d2-lss.yaml pretrained/bevfusion-seg.pth --eval map
```

### Training

We provide instructions to reproduce our results on nuScenes.

For example, if you want to train the camera-only variant for object detection, please run:

```bash
torchpack dist-run -np 8 python tools/train.py configs/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/default.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth
```

For camera-only BEV segmentation model, please run:

```bash
torchpack dist-run -np 8 python tools/train.py configs/nuscenes/seg/camera-bev256d2.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth
```

For LiDAR-only detector, please run:

```bash
torchpack dist-run -np 8 python tools/train.py configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml
```

For LiDAR-only BEV segmentation model, please run:

```bash
torchpack dist-run -np 8 python tools/train.py configs/nuscenes/seg/lidar-centerpoint-bev128.yaml
```

## Acknowledgements

BEVFusion is based on [mmdetection3d](https://github.com/open-mmlab/mmdetection3d). It is also greatly inspired by the following outstanding contributions to the open-source community: [LSS](https://github.com/nv-tlabs/lift-splat-shoot), [BEVDet](https://github.com/HuangJunjie2017/BEVDet), [TransFusion](https://github.com/XuyangBai/TransFusion), [CenterPoint](https://github.com/tianweiy/CenterPoint), [MVP](https://github.com/tianweiy/MVP), [FUTR3D](https://arxiv.org/abs/2203.10642), [CVT](https://github.com/bradyz/cross_view_transformers) and [DETR3D](https://github.com/WangYueFt/detr3d).

Please also check out related papers in the camera-only 3D perception community such as [BEVDet4D](https://arxiv.org/abs/2203.17054), [BEVerse](https://arxiv.org/abs/2205.09743), [BEVFormer](https://arxiv.org/abs/2203.17270), [M2BEV](https://arxiv.org/abs/2204.05088), [PETR](https://arxiv.org/abs/2203.05625) and [PETRv2](https://arxiv.org/abs/2206.01256), which might be interesting future extensions to BEVFusion.
