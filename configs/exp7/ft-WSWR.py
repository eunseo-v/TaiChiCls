model = dict(
    type = 'Recognizer3D',
    backbone = dict(
        type = 'ResNet3dSlowOnly', # 直接设置了stem layer中的卷积核大小，t方向的步长
        depth = 50, # from {18, 34, 50, 101, 152}
        pretrained = None,
        in_channels = 5, # 采用5Part channel num of input features
        base_channels = 32, # stem layer output features channels
        num_stages = 3, # 选择论文里的放弃res2, 只有res3, 4, 5
        out_indices = (2, ),
        stage_blocks = (4, 6, 3), # 每一个res, block(也就是论文里中括号)的数量
        conv1_stride_s = 1, # Spatial stride of the first conv layer, stem layer中
        pool1_stride_s = 1, # Spatial stride of the first pooling layer, stem layer中
        inflate = (0, 1, 1), # Inflate Dims of each block.
        spatial_strides = (2, 2, 2), # Spatial strides of residual blocks of each stage，每一个res，空间长度都减半
        temporal_strides = (1, 1, 2), # Temporal strides of residual blocks of each stage，只有最后一个res，时间长度减半
        dilations = (1, 1, 1) # Dilation of each stage
        # frozen_stages = 3 # finetune条件下，不冻结主干网络的权重
    ),
    cls_head = dict(
        type = 'I3DHead',
        in_channels = 512, 
        num_classes = 10,
        spatial_type = 'avg',
        dropout_ratio = 0.5
    ),
    train_cfg = dict(),
    test_cfg = dict(
        average_clips = 'prob'
    ) # 测试时将一个sample的多个clip的结果取平均作为最终结果
)

dataset_type = 'PoseDataset'
ann_file_train = 'data/ds_taichi/TEST9AUG4/WSWR/train_data.pkl'
ann_file_val = 'data/ds_taichi/TEST9AUG4/WSWR/test_data.pkl'
left = [
    5, 6, 7, 8, 44, 45, 46, 47, 48, 49, 
    50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 
    60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 
    70, 71
]
right = [
    1, 2, 3, 4, 16, 17, 18, 19, 
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
    40, 41, 42, 43
]
train_pipeline = [
    dict(type = 'UniformSampleFrames', clip_len = 48),
    dict(type = 'PoseDecode'),
    dict(type = 'PoseCompact', hw_ratio = 1., allow_imgpad = True),
    dict(type = 'Resize', scale = (-1, 64)),
    dict(type = 'RandomResizedCrop', area_range=(0.56, 1.0)), # 随机比例裁剪
    dict(type = 'Resize', scale = (56, 56), keep_ratio = False),
    dict(type = 'Flip', flip_ratio = 0.5, left_kp = left, right_kp = right),
    dict(type = 'GenerateTaiChiPoseTarget', sigma =0.6, use_score = True, with_kp = True, with_limb = False), # 训练集的double一定要是False!
    dict(type = 'FormatShape', input_format = 'NCTHW'), # 一个样本，N指的是N_crops* N_clips
    dict(type = 'Collect', keys = ['imgs', 'label'], meta_keys = []),
    dict(type = 'ToTensor', keys = ['imgs', 'label'])
]
val_pipeline = [
    dict(type = 'UniformSampleFrames', clip_len = 48, num_clips = 1, test_mode = True),
    dict(type = 'PoseDecode'),
    dict(type = 'PoseCompact', hw_ratio = 1., allow_imgpad = True),
    dict(type = 'Resize', scale = (-1, 64)),
    dict(type = 'CenterCrop', crop_size = 64),
    dict(type = 'GenerateTaiChiPoseTarget', sigma=0.6, use_score = True, with_kp = True, with_limb = False),
    dict(type = 'FormatShape', input_format = 'NCTHW'),
    dict(type = 'Collect', keys = ['imgs', 'label'], meta_keys = []),
    dict(type = 'ToTensor', keys = ['imgs'])
]
test_pipeline = [
    dict(type = 'UniformSampleFrames', clip_len=48, num_clips=10, test_mode = True),
    dict(type = 'PoseDecode'),
    dict(type = 'PoseCompact', hw_ratio = 1., allow_imgpad = True),
    dict(type = 'Resize', scale=(-1, 64)),
    dict(type = 'CenterCrop', crop_size = 64),
    dict(type = 'GenerateTaiChiPoseTarget', sigma = 0.6, use_score = True, with_kp=True, with_limb=False, double=True, left_kp= left, right_kp=right),
    dict(type = 'FormatShape', input_format = 'NCTHW'),
    dict(type = 'Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type = 'ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=4,
    test_dataloader = dict(
        videos_per_gpu = 4
    ),
    train=dict(
        type='RepeatDataset',
        times=15,
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file_train,
            data_prefix="",
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix="",
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix="",
        pipeline=test_pipeline))
# optimizer
optimizer = dict(
    type='SGD', lr=0.01, momentum=0.9,
    weight_decay=0.0003)  # this lr is used for 4 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False, 
    min_lr=0)
total_epochs = 48
checkpoint_config = dict()
workflow = [('train', 1)]
evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 5))
log_config = dict(
    interval=40,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
'''
    output_config和eval_config是测试时使用的参数
'''
output_config = dict(
    out = './model_pth/exp7/ft-WSWR/test_result/results.pkl'
) # 保存测试集的各类分类概率和样本名 keys:'outputs','sample_names'
eval_config = dict(
    metric_out = './model_pth/exp7/ft-WSWR/test_result',
    eval = ['top_k_accuracy', 'mean_class_accuracy', 'confusion_matrix', 't_sne_vis'],
) # train中不写test_last test_best 在test.py中测试
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './model_pth/exp7/ft-WSWR'
load_from = './model_pth/5part_ntu60_xsub_kp_0331/epoch_24.pth'
resume_from = None
find_unused_parameters = False