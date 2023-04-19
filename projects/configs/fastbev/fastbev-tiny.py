
_base_ = ['mmdet3d::_base_/default_runtime.py',
          'mmdet3d::_base_/schedules/cyclic-20e.py']

batch_size = 44
test_batch_size = 1
epochs = 50
num_workers = 4
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
bev_size = [200, 200]
num_points_in_pillar = 6

checkpoint_interval = 1
log_interval = 10
pred_score_thr = 0.4
show_pred_score = True
lr = 0.0002

n_cams = 6
img_input_final_dim = (256, 704)

# load_from = './work_dirs/fastbev-tiny_train_20230419-1302/epoch_3.pth'
load_from = None
resume = None


class_names=[ 'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
            'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']



dataset_type = 'MyNuscenesDataset'
data_root = '/data/fuyu/dataset/nuscenes'

metainfo = dict(classes=class_names)

input_modality = dict(use_lidar=False,
                      use_camera=True,
                      use_radar=False,
                      use_map=False,
                      use_can_bus=False,
                      use_external=False)


work_dir_postfix_name = "half_res_aug"

suffix = 'nuscenes_aug'
save2cfg = dict(
    plot_examples=10,
    plot_range=[*point_cloud_range[:2], *point_cloud_range[3:5]],
    draw_gt=True, draw_pred=True, pred_score_thr=pred_score_thr,
    transpose=True,
    save_only_master=True)



def __date():
    import datetime
    return datetime.datetime.now().strftime('%Y%m%d-%H%M')

save2img_cfg = dict(**save2cfg,
                    save_dir='./pts_img_vis/' + suffix + '-' + work_dir_postfix_name + '-' + __date())
save2img_pipline_cfg = dict(**save2cfg,
                            save_dir='./pts_img_vis_pipline/' + suffix + '-' + work_dir_postfix_name + '-' + __date())


plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'



work_dir_postfix = "_" + work_dir_postfix_name + '_' + __date()

multi_scale_id=[0]

model = dict(
    type='FastBEV',
    raw_net=True,
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        # BGR mean = [100.20131724 104.31293425 108.13566736], std = [63.01974606 56.10074886 60.47841278]
        mean=[123.675, 116.28, 103.53],  # r g b
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        # pad_size_divisor=32
    ),
    input_modality=input_modality,
    save2img_cfg=save2img_cfg,
    use_grid_mask=True,
    backbone=dict(
        type='mmdet.ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    neck=dict(
        type='mmdet.FPN',
        norm_cfg=dict(type='BN', requires_grad=True),
        in_channels=[64, 128, 256, 512],
        out_channels=64,
        num_outs=4,
        relu_before_extra_convs=True),
    neck_fuse=dict(in_channels=[256], out_channels=[64]),
    neck_3d=dict(
        type='M2BevNeck',
        in_channels=64*4,
        out_channels=192,
        num_layers=2,
        stride=2,
        is_transpose=True,
        fuse=dict(in_channels=64*num_points_in_pillar, out_channels=256),
        norm_cfg=dict(type='BN', requires_grad=True)),
    bbox_head=dict(
        type='CustomFreeAnchor3DHead',
        pre_anchor_topk=25,
        bbox_thr=0.5,
        gamma=2.0,
        alpha=0.5,
        num_classes=len(class_names),
        in_channels=192,
        feat_channels=192,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[*point_cloud_range[:2], -1.8, *point_cloud_range[3:5], -1.8]],
            sizes=[
                [2.5981,0.8660,  1.],  # 1.5/sqrt(3)
                [ 1.7321,0.5774, 1.],  # 1/sqrt(3)
                [1., 1., 1.],
                [0.4, 0.4, 1],
            ],
            custom_values=[0, 0], # 速度
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
            type='mmdet.CrossEntropyLoss', use_sigmoid=False, loss_weight=0.8)
    ),
    multi_scale_id=multi_scale_id,  # 4x
    bev_size=bev_size,
    point_cloud_range=point_cloud_range,
    num_points_in_pillar=num_points_in_pillar,
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
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
            debug=False)),
    test_cfg=dict(
        pts=dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_pre=1000,
            nms_thr=0.2,
            score_thr=0.05,
            min_bbox_size=0,
            max_num=500)))



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

ida_aug_conf = dict(
    distortion_cfg=dict(
            brightness_delta=32,
            contrast_range=(0.5, 1.5),
            saturation_range= (0.5, 1.5),
            hue_delta=0),
    # train-aug
    final_size=img_input_final_dim,
    resize_range=(-0.06, 0.11),
    crop_range=(-0.05, 0.05),
    rot_range=(-3.14159264 / 18, 3.14159264 / 18),
    rand_flip=True,
    flip_ratio=0.5,
    # top, right, bottom, left
    pad=(0, 0, 0, 0),
    pad_color=(0, 0, 0),
    # test-aug
    test_final_size=img_input_final_dim,
    test_resize=0.0,
    test_rotate=0.0,
    test_flip=False,
)

ida_aug_conf_without = dict(
    # train-aug
    final_size=img_input_final_dim,
    # test-aug
    test_final_size=img_input_final_dim
)

pts_aug_conf = dict(
    # train-aug
    rot_range=(-3.14159264 / 4, 3.14159264 / 4),  # (-pi/4, pi/4)
    scale_ratio_range=(0.95, 1.05),
    translation_std=(0, 0, 0),
    rand_flip=True,
    flip_ratio=0.5,
    random_flip_direction=True,
    # test-aug
    test_rotate=0.0,
    test_scale=1.0,
    test_flip=False,
    test_flip_direction='vertical',
)

pts_aug_conf_without = dict()

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', num_views=3),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='MultiViewWrapper',
         transforms=[
             dict(type='CustomPhotoMetricDistortion3D', **ida_aug_conf['distortion_cfg']),
             dict(type='RandomAugOneImage', data_config=ida_aug_conf)
         ],
         randomness_keys=['resize', 'resize_dims', 'crop', 'flip', 'pad', 'rotate', 'photometric_param'],
         collected_keys=['resize', 'resize_dims', 'crop', 'flip', 'pad', 'rotate', 'photometric_param'],
         override_aug_config=False,  # whether use the same aug config for multiview image
         process_fields=['img', 'cam2img', 'lidar2cam', 'filename']),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='AddSupplementInfo'),
    # dict(type='PointCloudImgVis', cfg=save2img_pipline_cfg),
    dict(type='CustomPack3DDetInputs', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', num_views=3),
    dict(type='CustomLoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='CustomMultiViewWrapper',
         transforms=[
             dict(type='CustomPhotoMetricDistortion3D', **ida_aug_conf['distortion_cfg']),
             dict(type='RandomAugOneImage', data_config=ida_aug_conf, is_train=False),],
         global_key=['sample_idx', 'gt_bboxes_3d'],
         randomness_keys=['resize', 'resize_dims', 'crop', 'flip', 'pad', 'rotate', 'photometric_param'],
         collected_keys=['resize', 'resize_dims', 'crop', 'flip', 'pad', 'rotate', 'photometric_param'],
         override_aug_config=False,  # whether use the same aug config for multiview image
         process_fields=['img', 'cam2img', 'lidar2cam', 'filename']),
    dict(type='AddSupplementInfo'),
    # dict(type='PointCloudImgVis', cfg=save2img_pipline_cfg),
    dict(type='CustomPack3DDetInputs', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CBGSDataset',dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='nuscenes_infos_train.pkl',
            pipeline=train_pipeline,
            metainfo=metainfo,
            modality=input_modality,
            test_mode=False,
            data_prefix=data_prefix,
            # indices=200,  # for debug, too small may raise Error: float division by zero
            box_type_3d='LiDAR'
            )
        ))

test_dataloader = dict(
    batch_size=test_batch_size,
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
        pred_score_thr=pred_score_thr,
        show_score=show_pred_score,
        test_mode=True,
        load_eval_anns=True,
        # indices=200,
        box_type_3d='LiDAR'))

test_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'nuscenes_infos_val.pkl',
    metric='bbox',
    format_only=True,
    default_cam_key = "CAM_FRONT",
    submission_prefix='results/kitti-3class/kitti_results')

vis_backends = [dict(type='TensorboardVisBackend')]
visualizer = dict(type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='float16',
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
val_cfg = None
# test_cfg = dict(type='TestLoop')

test_cfg = dict(type='CustomVisLoop', iter_nums=20,vis=True)


default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=log_interval),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=checkpoint_interval),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))

log_processor = dict(type='LogProcessor', window_size=log_interval, by_epoch=True)

runner_type = 'CustomRunner'