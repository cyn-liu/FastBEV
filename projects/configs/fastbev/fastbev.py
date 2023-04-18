_base_ = ['mmdet3d::_base_/default_runtime.py',
          'mmdet3d::_base_/schedules/cyclic-20e.py']

def __date():
    import datetime
    return datetime.datetime.now().strftime('%Y%m%d-%H%M')

work_dir_postfix = '_' + __date()

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

camera_types = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                'CAM_BACK',  'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',]

class_names=[ 'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
            'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']

metainfo = dict(classes=class_names)

input_modality = dict(use_lidar=False,
                      use_camera=True,
                      use_radar=False,
                      use_map=False,
                      use_can_bus=False,
                      use_external=False)

batch_size = 8
val_batch_size = 1
epochs = 24

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
bev_size = [200, 200]
num_points_in_pillar = 6

multi_scale_id=[0, 1, 2]  # 4x/8x/16x
raw_net=True
fuse_dim = len(multi_scale_id) if raw_net else 1

dataset_type = 'NuScenesDataset'
data_root = '/home/fuyu/zhangbin/datasets/msfusion/'

model = dict(
    type='FastBEV',
    raw_net=raw_net,
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[123.675, 116.28, 103.53],  # r g b
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='mmdet.FPN',
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        in_channels=[256, 512, 1024, 2048],
        out_channels=64,
        num_outs=4,
        relu_before_extra_convs=True),
    neck_fuse=dict(in_channels=[256, 192, 128], out_channels=[64, 64, 64]),
    neck_3d=dict(
        type='M2BevNeck',
        in_channels=256,
        out_channels=256,
        num_layers=6,
        stride=2,
        is_transpose=False,
        fuse=dict(in_channels=64*num_points_in_pillar*fuse_dim, out_channels=256),
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    bbox_head=dict(
        type='FreeAnchor3DHead',
        num_classes=10,
        in_channels=256,
        feat_channels=256,
        use_direction_classifier=True,
        pre_anchor_topk=25,
        bbox_thr=0.5,
        gamma=2.0,
        alpha=0.5,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[*point_cloud_range[:2], -1.8, *point_cloud_range[3:5], -1.8]],
            scales=[1],
            sizes=[[2.5981, 0.866, 1.0], [1.7321, 0.5774, 1.0],
                   [1.0, 1.0, 1.0], [0.4, 0.4, 1]],
            custom_values=[0, 0],
            rotations=[0, 1.57],
            reshape_out=True),
        assigner_per_size=False,
        diff_rad_by_sin=True,
        dir_offset=0.7854,  # pi/4
        dir_limit_offset=0,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=9),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.8),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    multi_scale_id=multi_scale_id,  # 4x
    bev_size=bev_size,
    point_cloud_range=point_cloud_range,
    num_points_in_pillar=num_points_in_pillar,
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='Max3DIoUAssigner',
            iou_calculator=dict(type='BboxOverlapsNearest3D'),
            pos_iou_thr=0.6,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        allowed_border=0,
        code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.25, 0.25],
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_pre=1000,
        nms_thr=0.2,
        score_thr=0.05,
        min_bbox_size=0,
        max_num=500))

data_prefix = dict(
    pts='samples/LIDAR_TOP',
    sweeps='sweeps/LIDAR_TOP',
    CAM_FRONT='samples/CAM_FRONT',
    CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
    CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
    CAM_BACK='samples/CAM_BACK',
    CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
    CAM_BACK_LEFT='samples/CAM_BACK_LEFT')

file_client_args = dict(backend='disk')

ida_aug_conf = {
        # train-aug
        'final_dim': (256, 704),
        'resize_lim': (-0.06, 0.11),
        'crop_lim': (-0.05, 0.05),
        'rot_lim': (-0.3925 * 3, 0.3925 * 3),
        'rand_flip': True,
        'flip_ratio': 0.5,
        # top, right, bottom, left
        'pad': (0, 0, 0, 0),
        'pad_color': (0, 0, 0),
        # test-aug
        'test_final_dim': (256, 704),
        'test_resize': 0.0,
        'test_rotate': 0.0,
        'test_flip': False,
    }

ida_aug_conf_without = {
        # train-aug
        'final_dim': (256, 704),
        # test-aug
        'test_final_dim': (256, 704),
    }

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', num_views=6),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='MultiViewWrapper',
         transforms=[
             dict(type='PhotoMetricDistortion3D'),
             dict(type='RandomAugOneImage', data_config=ida_aug_conf_without)],
         randomness_keys=['resize', 'resize_dims', 'crop', 'flip', 'pad', 'rotate', 'photometric_param'],
         collected_keys=['resize', 'resize_dims', 'crop', 'flip', 'pad', 'rotate', 'photometric_param'],
         override_aug_config=False,  # whether use the same aug config for multiview image
         process_fields=['img', 'cam2img', 'lidar2cam', 'filename']),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    # dict(type='PointCloudImgVis', plot_examples=20, plot_range=30, save_dir='./pts_img_vis/fastbev-' + __date()),
    dict(type='CustomPack3DDetInputs',
         keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', num_views=6),
    dict(type='CustomMultiViewWrapper',
         transforms=[
             dict(type='PhotoMetricDistortion3D'),
             dict(type='RandomAugOneImage', data_config=ida_aug_conf, is_train=False),],
         global_key=['sample_idx', 'gt_bboxes_3d'],
         randomness_keys=['resize', 'resize_dims', 'crop', 'flip', 'pad', 'rotate', 'photometric_param'],
         collected_keys=['resize', 'resize_dims', 'crop', 'flip', 'pad', 'rotate', 'photometric_param'],
         override_aug_config=False,  # whether use the same aug config for multiview image
         process_fields=['img', 'cam2img', 'lidar2cam', 'filename']),
    dict(type='Pack3DDetInputs', keys=['img'])
]
eval_pipeline = test_pipeline

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='nuscenes_infos_train.pkl',
            pipeline=train_pipeline,
            metainfo=metainfo,
            modality=input_modality,
            test_mode=False,
            data_prefix=data_prefix,
            use_valid_flag=True,
            # indices=200,  # for debug, too small may raise Error: float division by zero
            box_type_3d='LiDAR')))

test_dataloader = dict(
    batch_size=val_batch_size,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        data_prefix=data_prefix,
        test_mode=True,
        # indices=200,
        box_type_3d='LiDAR'))

val_dataloader = dict(
    batch_size=val_batch_size,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        test_mode=True,
        data_prefix=data_prefix,
        box_type_3d='LiDAR'))

val_evaluator = dict(
    type='NuScenesMetric',
    data_root=data_root,
    ann_file=data_root + 'nuscenes_infos_val.pkl',
    modality=input_modality,
    metric='bbox')
test_evaluator = val_evaluator

vis_backends = [dict(type='TensorboardVisBackend')]
visualizer = dict(type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

lr = 0.0002
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01,),
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}),
    clip_grad=dict(max_norm=35, norm_type=2))
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=8,
        eta_min=0.001,
        begin=0,
        end=8,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=epochs - 8,
        eta_min=1e-08,
        begin=8,
        end=epochs,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=8,
        eta_min=0.8947368421052632,
        begin=0,
        end=8,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=epochs - 8,
        eta_min=1,
        begin=8,
        end=epochs,
        by_epoch=True,
        convert_to_iter_based=True)
]

train_cfg = dict(_delete_=True, type='EpochBasedTrainLoop', max_epochs=epochs, val_interval=epochs)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

log_interval = 100
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=log_interval),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=5),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))

log_processor = dict(type='LogProcessor', window_size=log_interval, by_epoch=True)

# load_from = './ckpt/fastbev/fastbev_m5_epoch_20.pth'
resume = False