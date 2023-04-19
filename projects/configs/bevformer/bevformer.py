_base_ = ['mmdet3d::_base_/default_runtime.py',
          'mmdet3d::_base_/schedules/cyclic-20e.py']



camera_types = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                'CAM_BACK',  'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',]

class_names=['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
            'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']

metainfo = dict(classes=class_names)

input_modality = dict(use_lidar=True,
                      use_camera=True,
                      use_radar=False,
                      use_map=False,
                      use_can_bus=False,
                      use_external=False)

batch_size = 2
val_batch_size = 1
epochs = 20
log_interval = 100

load_from = None
resume = False

experiment_name = None  #  If not specified, timestamp will be used as ``experiment_name``
work_dir_postfix_name = ""

data_root = '/home/fuyu/zhangbin/datasets/ms-mini/'

work_dir_postfix = "_" + work_dir_postfix_name + '_' + __date()

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

def __date():
    import datetime
    return datetime.datetime.now().strftime('%Y%m%d-%H%M')

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# img branch
img_size = (900, 1600)
depth_range = [1e-3, (point_cloud_range[3] - point_cloud_range[0]) / 2]
head_input_channel = 256

# pts branch
pts_voxel_size = [0.075, 0.075, 0.2]  # pts_voxel_num = [1440, 1440, 40]
pts_voxel_num = [int((point_cloud_range[3] - point_cloud_range[0]) / pts_voxel_size[0]),
                 int((point_cloud_range[4] - point_cloud_range[1]) / pts_voxel_size[1]),
                 int((point_cloud_range[5] - point_cloud_range[2]) / pts_voxel_size[2])]

# bev branch
num_points_in_pillar = 4
bev_h = 200
bev_w = 200
embed_dims = 256  # query channels
_pos_dim_ = embed_dims // 2
_ffn_dim_ = embed_dims * 2  # FFN middle channels

transformer_batch_first = True
transformer_layer_nums = 3

suffix = 'bevformer'
save2cfg = dict(
    plot_examples=50,
    plot_range=[*point_cloud_range[:2], *point_cloud_range[3:5]],
    draw_gt=True, draw_pred=True, pred_score_thr=0.3,
    transpose=False,
    save_only_master=True)
save2img_cfg = dict(**save2cfg,
                    save_dir='./pts_img_vis/' + suffix + '-' + work_dir_postfix_name + '-' + __date())
save2img_pipline_cfg = dict(**save2cfg,
                            save_dir='./pts_img_vis_pipline/' + suffix + '-' + work_dir_postfix_name + '-' + __date())


model = dict(
    type='CustomBaseDetector',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=10,
            point_cloud_range=point_cloud_range,
            voxel_size=pts_voxel_size,
            max_voxels=(120000, 160000),),
        mean=[123.675, 116.28, 103.53],  # r g b
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    input_modality=input_modality,
    save2img_cfg=save2img_cfg,
    use_grid_mask=True,
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    img_neck=dict(
        type='mmdet.FPN',
        norm_cfg=dict(type='BN', requires_grad=True),
        in_channels=[512, 1024],
        out_channels=head_input_channel,
        add_extra_convs='on_output',
        num_outs=2,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='BEVFormerHead',
        bev_h=bev_h,
        bev_w=bev_w,
        point_cloud_range=point_cloud_range,
        num_query=500,
        num_classes=len(class_names),
        in_channels=head_input_channel,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type='BEVFormerTransformer',
            bev_h=bev_h,
            bev_w=bev_w,
            use_cams_embeds=True,
            embed_dims=embed_dims,
            encoder=dict(
                type='BaseTransformerEncoder',
                num_layers=transformer_layer_nums,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BaseTransformerEncoderLayer',
                    batch_first=transformer_batch_first,
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=embed_dims,
                            num_heads=8,
                            attn_drop=0.1,
                            dropout_layer=dict(type='Dropout', drop_prob=0.1)),
                       dict(
                           type='SpatialCrossAttention',
                           num_cams=len(camera_types),
                           deformable_attention=dict(
                               type='MSDeformableAttention3D',
                               embed_dims=embed_dims,
                               num_points=8,
                               num_levels=1))],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=embed_dims,
                        feedforward_channels=_ffn_dim_,
                        num_fcs=2,
                        ffn_drop=0.1,
                        act_cfg=dict(type='ReLU', inplace=True)),
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='BaseTransformerDecoder',
                num_layers=transformer_layer_nums,
                return_intermediate=True,
                transformerlayers=dict(
                    type='BaseTransformerDecoderLayer',
                    batch_first=transformer_batch_first,
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=embed_dims,
                            num_heads=8,
                            attn_drop=0.1,
                            dropout_layer=dict(type='Dropout', drop_prob=0.1),),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=embed_dims,
                            num_levels=1),],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=embed_dims,
                        feedforward_channels=_ffn_dim_,
                        num_fcs=2,
                        ffn_drop=0.1,
                        act_cfg=dict(type='ReLU', inplace=True)),
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')))),
        positional_encoding=dict(
            encoder=dict(
                type='mmdet.LearnedPositionalEncoding',
                num_feats=_pos_dim_,
                row_num_embed=bev_h,
                col_num_embed=bev_w)),
        code_size=10,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='mmdet.BalancedL1Loss', loss_weight=0.25, gamma=1),
        loss_iou=dict(type='mmdet.GIoULoss', loss_weight=0.)),
    train_cfg=dict(
        pts=dict(
            assigner=dict(
                type='CustomHungarianAssigner3D',
                match_costs=dict(
                    cls_cost=dict(type='FocalLossCost', weight=2.0),
                    reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                    iou_cost=dict(type='IoUCost', weight=0.0)  # Fake cost
                ),
                pc_range=point_cloud_range))),
    test_cfg=dict(
        pts=dict(
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_frame=500,
            score_threshold=0.6)))

dataset_type = 'NuScenesDataset'

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
db_sampler = dict(
    type='UnifiedDataBaseSampler',
    data_root=data_root,
    info_path=data_root + 'nuscenes_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            car=5,
            truck=5,
            bus=5,
            trailer=5,
            construction_vehicle=5,
            traffic_cone=5,
            barrier=5,
            motorcycle=5,
            bicycle=5,
            pedestrian=5)),
    classes=class_names,
    sample_groups=dict(
        car=2,
        truck=3,
        construction_vehicle=7,
        bus=4,
        trailer=6,
        barrier=2,
        motorcycle=6,
        bicycle=6,
        pedestrian=2,
        traffic_cone=2),
    img_loader=dict(type='LoadImageFromFile'),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4]))

ida_aug_conf = dict(
    distortion_cfg=dict(
            brightness_delta=32,
            contrast_range=(0.5, 1.5),
            saturation_range= (0.5, 1.5),
            hue_delta=0),
    # train-aug
    final_size=(320, 800),
    resize_range=(-0.06, 0.11),
    crop_range=(-0.05, 0.05),
    rot_range=(-3.14159264 / 18, 3.14159264 / 18),
    rand_flip=True,
    flip_ratio=0.5,
    # top, right, bottom, left
    pad=(0, 0, 0, 0),
    pad_color=(0, 0, 0),
    # test-aug
    test_final_size=(320, 800),
    test_resize=0.0,
    test_rotate=0.0,
    test_flip=False,
)

ida_aug_conf_without = dict(
    # train-aug
    final_dim=(320, 800),
    # test-aug
    test_final_dim=(320, 800)
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
    dict(type='LoadMultiViewImageFromFiles', num_views=6),
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4]),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='UnifiedObjectSample', db_sampler=db_sampler, sample_2d=True,
         modify_points=True, mixup_rate=-1),
    dict(type='MultiViewWrapper',
         transforms=[
             dict(type='CustomPhotoMetricDistortion3D'),
             dict(type='RandomAugOneImage', data_config=ida_aug_conf)],
         randomness_keys=['resize', 'resize_dims', 'crop', 'flip', 'pad', 'rotate', 'photometric_param'],
         collected_keys=['resize', 'resize_dims', 'crop', 'flip', 'pad', 'rotate', 'photometric_param'],
         override_aug_config=False,  # whether use the same aug config for multiview image
         process_fields=['img', 'cam2img', 'lidar2cam', 'filename']),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='RandomAugBEV', data_config=pts_aug_conf),
    dict(type='AddSupplementInfo'),
    # dict(type='PointCloudImgVis', cfg=save2img_pipline_cfg),
    dict(type='CustomPack3DDetInputs',
         keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'],
         # keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'],
         # keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'],
         meta_key_extend=['lidar2img', 'img2lidar'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', num_views=6),
    # dict(type='CustomMultiViewWrapper',
    #      transforms=[
    #          dict(type='CustomPhotoMetricDistortion3D'),
    #          dict(type='RandomAugOneImage', data_config=ida_aug_conf, is_train=False),],
    #      global_key=['sample_idx', 'gt_bboxes_3d'],
    #      randomness_keys=['resize', 'resize_dims', 'crop', 'flip', 'pad', 'rotate', 'photometric_param'],
    #      collected_keys=['resize', 'resize_dims', 'crop', 'flip', 'pad', 'rotate', 'photometric_param'],
    #      override_aug_config=False,  # whether use the same aug config for multiview image
    #      process_fields=['img', 'cam2img', 'lidar2cam', 'filename']),
    dict(type='AddSupplementInfo'),
    # dict(type='PointCloudImgVis', cfg=save2img_pipline_cfg),
    dict(type='CustomPack3DDetInputs',
         keys=['img'],
         meta_key_extend=['lidar2img', 'img2lidar'])
]
eval_pipeline = test_pipeline


train_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
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
        # indices=5000,  # each epoch iter nums, for debug
        box_type_3d='LiDAR'))

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

vis_backends = [dict(type='LocalVisBackend')]
# vis_backends = [dict(type='TensorboardVisBackend')]
# vis_backends = [dict(type='WandbVisBackend')]
visualizer = dict(type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

lr = 0.0001
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01, ),
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

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=log_interval),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=2),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))

log_processor = dict(type='LogProcessor', window_size=log_interval, by_epoch=True)

runner_type = 'CustomRunner'