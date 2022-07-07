model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowOnly',
        depth=50,
        pretrained=None,
        in_channels=17,
        base_channels=32,
        num_stages=3,
        out_indices=(2, ),
        stage_blocks=(4, 6, 3),
        conv1_stride_s=1,
        pool1_stride_s=1,
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 2),
        dilations=(1, 1, 1)),
    cls_head=dict(
        type='I3DHead',
        in_channels=512,
        num_classes=10,
        spatial_type='avg',
        dropout_ratio=0.5),
    train_cfg=dict(),
    test_cfg=dict(average_clips='prob'))
dataset_type = 'PoseDataset'
ann_file_train = 'data/ds_taichi/TEST3AUG4/NSNR/train_data.pkl'
ann_file_val = 'data/ds_taichi/TEST3AUG4/NSNR/test_data.pkl'
keypoints = [13, 14, 14, 15, 15, 44, 16, 45, 17, 47, 19, 5, 1, 6, 2, 7, 3]
left = [5, 6, 7, 44, 45, 47]
right = [1, 2, 3, 16, 17, 19]
train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1.0, allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(56, 56), keep_ratio=False),
    dict(
        type='Flip',
        flip_ratio=0.5,
        left_kp=[5, 6, 7, 44, 45, 47],
        right_kp=[1, 2, 3, 16, 17, 19]),
    dict(
        type='GenerateTaiChi17PoseTarget',
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1.0, allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='CenterCrop', crop_size=64),
    dict(
        type='GenerateTaiChi17PoseTarget',
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='UniformSampleFrames', clip_len=48, num_clips=10, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1.0, allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='CenterCrop', crop_size=64),
    dict(
        type='GenerateTaiChi17PoseTarget',
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False,
        double=True,
        left_kp=[5, 6, 7, 44, 45, 47],
        right_kp=[1, 2, 3, 16, 17, 19]),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=4,
    test_dataloader=dict(videos_per_gpu=4),
    train=dict(
        type='RepeatDataset',
        times=15,
        dataset=dict(
            type='PoseDataset',
            ann_file='data/ds_taichi/TEST3AUG4/NSNR/train_data.pkl',
            data_prefix='',
            pipeline=[
                dict(type='UniformSampleFrames', clip_len=48),
                dict(type='PoseDecode'),
                dict(type='PoseCompact', hw_ratio=1.0, allow_imgpad=True),
                dict(type='Resize', scale=(-1, 64)),
                dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
                dict(type='Resize', scale=(56, 56), keep_ratio=False),
                dict(
                    type='Flip',
                    flip_ratio=0.5,
                    left_kp=[5, 6, 7, 44, 45, 47],
                    right_kp=[1, 2, 3, 16, 17, 19]),
                dict(
                    type='GenerateTaiChi17PoseTarget',
                    sigma=0.6,
                    use_score=True,
                    with_kp=True,
                    with_limb=False),
                dict(type='FormatShape', input_format='NCTHW'),
                dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
                dict(type='ToTensor', keys=['imgs', 'label'])
            ])),
    val=dict(
        type='PoseDataset',
        ann_file='data/ds_taichi/TEST3AUG4/NSNR/test_data.pkl',
        data_prefix='',
        pipeline=[
            dict(
                type='UniformSampleFrames',
                clip_len=48,
                num_clips=1,
                test_mode=True),
            dict(type='PoseDecode'),
            dict(type='PoseCompact', hw_ratio=1.0, allow_imgpad=True),
            dict(type='Resize', scale=(-1, 64)),
            dict(type='CenterCrop', crop_size=64),
            dict(
                type='GenerateTaiChi17PoseTarget',
                sigma=0.6,
                use_score=True,
                with_kp=True,
                with_limb=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
        ]),
    test=dict(
        type='PoseDataset',
        ann_file='data/ds_taichi/TEST3AUG4/NSNR/test_data.pkl',
        data_prefix='',
        pipeline=[
            dict(
                type='UniformSampleFrames',
                clip_len=48,
                num_clips=10,
                test_mode=True),
            dict(type='PoseDecode'),
            dict(type='PoseCompact', hw_ratio=1.0, allow_imgpad=True),
            dict(type='Resize', scale=(-1, 64)),
            dict(type='CenterCrop', crop_size=64),
            dict(
                type='GenerateTaiChi17PoseTarget',
                sigma=0.6,
                use_score=True,
                with_kp=True,
                with_limb=False,
                double=True,
                left_kp=[5, 6, 7, 44, 45, 47],
                right_kp=[1, 2, 3, 16, 17, 19]),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
        ]))
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0003)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
lr_config = dict(policy='CosineAnnealing', by_epoch=False, min_lr=0)
total_epochs = 24
checkpoint_config = dict(interval=1)
workflow = [('train', 1)]
evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 5))
log_config = dict(interval=40, hooks=[dict(type='TextLoggerHook')])
output_config = dict(
    out='./model_pth/exp9/T3-fintune-with-pre-lrd01/test_result/results.pkl')
eval_config = dict(
    metric_out='./model_pth/exp9/T3-fintune-with-pre-lrd01/test_result',
    eval=[
        'top_k_accuracy', 'mean_class_accuracy', 'confusion_matrix',
        't_sne_vis'
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './model_pth/exp9/T3-fintune-with-pre-lrd01'
load_from = './model_pth/17part_ntu60_xsub_kp_0401/best_top1_acc_epoch_24.pth'
resume_from = None
find_unused_parameters = False
gpu_ids = range(0, 4)
omnisource = False
module_hooks = []
