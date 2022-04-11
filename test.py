from mmaction.datasets import build_dataset, build_dataloader


left=[1, 3, 5, 7, 9, 11, 13, 15]
right=[2, 4, 6, 8, 10, 12, 14, 16]
train_pipeline = [
    dict(type = 'UniformSampleFrames', clip_len = 48),
    dict(type = 'PoseDecode'),
    dict(type = 'PoseCompact', hw_ratio = 1., allow_imgpad = True),
    dict(type = 'Resize', scale = (-1, 64)),
    dict(type = 'RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type = 'Resize', scale = (56, 56), keep_ratio = False),
    dict(type = 'Flip', flip_ratio = 0.5, left_kp = left, right_kp = right),
    dict(type = 'GeneratePoseTarget', sigma =0.6, use_score = True, with_kp = True, with_limb = False, double = True),
    # dict(type = 'FormatShape', input_format = 'NCTHW'),
    # dict(type = 'Collect', keys = ['imgs', 'label'], meta_keys = []),
    # dict(type = 'ToTensor', keys = ['imgs', 'label'])
]

dataset_config = dict(
    type = 'PoseDataset',
    ann_file = './data/posec3d/ntu60_xsub_train.pkl',
    data_prefix = "",
    pipeline = train_pipeline
)

dataset = build_dataset(dataset_config)

    
pass