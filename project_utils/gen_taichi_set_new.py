import os
os.sys.path.append(
    os.path.abspath('.')[
        :os.path.abspath('.').find('myposec3d') + len('myposec3d')
    ]
)
import numpy as np
import random
import math
from mmcv import dump, load
from einops import rearrange

def random_train_test_split(length, test_ratio = 0.3, random_seed = 42):
    # 将indices分成训练集和测试集
    all_indices_list = list(range(length))
    offset = int(length* test_ratio)
    random.seed(random_seed) # 只一次有效
    random.shuffle(all_indices_list)
    test_indices = all_indices_list[:offset]
    train_indices = all_indices_list[offset:]
    return train_indices, test_indices

def train_sample_aug(key, repeat_idx, train_sample, train_aug, clip_len=64):
    '''
        单个原始的太极拳样本，扩展成train_aug个样本，并且每一个样本命名
        Args:
            key: str 'a1' - 'a10'
            repeat_idx: 该类的第几个样本
            train_sample: np.ndarray [72, 3, T]
            train_aug: int 增广的倍数
            clip_len: 增广后单个样本的长度
        Output: list
            [
                {
                    name: 'AXXXRXXXSXXX' Action Repetition Slice,都从1开始记
                    data: [V, C, T]
                } ... {} ... {}
            ]
    '''
    output = list() # 存储结果
    num_frames = train_sample.shape[-1]
    bids = np.array(
        [i* num_frames // clip_len for i in range(clip_len + 1)]
    ) # 样本的每一帧的选择区间 size clip_len+1
    bsize = np.diff(bids) # 区间的大小 size clip_len
    bst = bids[:clip_len] # 选择的帧数起始值
    all_inds = [] # 用于存储每一个样本由哪些帧组成
    for i in range(train_aug):
        offset = np.random.randint(bsize)
        all_inds.append(bst + offset)
    all_inds = np.stack(all_inds, axis = 0) # size [train_aug, clip_len]
    for i, frame_idx in enumerate(all_inds):
        name = 'A{}R{}S{}'.format(
            key[1:].zfill(3), str(repeat_idx+1).zfill(3),
            str(i+1).zfill(3)
        )
        data = train_sample[...,frame_idx]
        output.append(
            {
                'name': name,
                'data': data # [V, C, clip_len]
            }
        )
    return output

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians. ref: wikipedia: Euler–Rodrigues formula
    """
    if np.abs(axis).sum() < 1e-6 or np.abs(theta) < 1e-6:
        return np.eye(3)
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def taichi_shear_rot_aug(data, shear = False, rotate = False, rotate_angle=None, shear_angle_list = None):
    '''
        Args:
            data [T, V, C]
            rotate 绕y轴方向随机旋转±60°
            shear 平面延展
            rotate_angle 角度
            shear_angle_list [2, 3]
        Output:
            data [T, V, C]
    '''
    if rotate == True:
        matrix_y = rotation_matrix(
            axis = np.array([0, 1, 0]), # 只想让它绕着y轴随机转个角度
            theta = rotate_angle
        ) # 绕y轴旋转的旋转矩阵
        for i_f, frame in enumerate(data):
            # frame [V, C]
            if frame.sum() == 0: 
                continue
            # 按节点计算旋转后的坐标
            for i_j, joint in enumerate(frame):
                # joint [C]
                data[i_f, i_j] = np.dot(matrix_y, joint)
    if shear == True:
        R = np.array([[1,          shear_angle_list[0,0], shear_angle_list[1,0]],
                      [shear_angle_list[0,1], 1,          shear_angle_list[1,1]],
                      [shear_angle_list[0,2], shear_angle_list[1,2], 1        ]])
        # R = np.array([[1,          s1_list[0], s2_list[0]],
        #               [s1_list[1], 1,          s2_list[1]],
        #               [s1_list[2], s2_list[2], 1        ]])
        for i_f, frame in enumerate(data):
            # frame [V, C]
            # 按节点计算旋转后的坐标
            for i_j, joint in enumerate(frame):
                # joint [C]
                data[i_f, i_j] = np.dot(R, joint)
    return data

def test_sample_extraction(key, repeat_idx, test_sample):
    '''
        单个原始的太极拳样本，整理格式，并给样本命名
        Args:
            key: str 'a1' - 'a10'
            repeat_idx: 该类的第几个样本
            train_sample: np.ndarray [72, 3, T]
        Output: dict
            {
                name: 'AXXXRXXX' Action Repetition,都从1开始记
                data: [V, C, T]
            } ... {} ... {}
    '''
    name = 'A{}R{}'.format(
        key[1:].zfill(3), str(repeat_idx+1).zfill(3)
    )
    data = test_sample
    return {
        'name': name,
        'data': data
    }

def taichi_skeleton_annotation(skeleton, shear = False, rotate = False,
    rotate_angle=None, shear_angle_list = None,  train_mode = False
):
    '''
        单个太极拳样本的数据预处理和标注信息生成
        1. 所有的样本减去第一时刻根节点的(x,y,z)坐标，消除采集初始位置的影响
        2. 判断是否为训练集样本，执行shear和rotation操作
            rotation: 沿着y轴旋转随机角度(±60°=60/180*pi)，一个样本所有时刻旋转的角度相同
            shear: 按照crossclr论文中的r=0.5参数进行
        3. 进行坐标到像素值的变换，舍弃z轴的数值
            1) Perception坐标系到CV2坐标系x,y轴方向的变化
            2) 减去三个轴的最小值
            3) 乘以一个倍数，之前的数值是4，可以尝试更大的值
            4) 加上一个余量值，之前的数值为100
            5) 判断像素点的最大值是否在边界中(1920, 1080)
        Args:
            skeleton: dict{
                'name': AXXXRXXXSXXX
                'data': [V, C, clip_len]
            }
        Output:
            annotation: dict{
                'frame_dir': name,
                'label': int,
                'img_shape': (1080, 1920), # [h,w]
                'original_shape': (1080, 1920),
                'total_frames': int,
                'keypoint': np.ndarray [N=1, T, V, C=2],
                'keypoint_score': [N=1, T, V]
            }
    '''
    # input data [V, C, T]
    sample_name = skeleton['name']
    V, C, T = skeleton['data'].shape
    data = rearrange(
        skeleton['data'], 'v c t -> t v c'
    ) # [T, V, C]
    # step1. 减去第一时刻根节点index=0的坐标
    origin = data[0:1, 0:1, :].copy() # [1, 1, C]
    data = (data - origin) # [T, V, C]
    # step2. rotation和shear 操作
    if train_mode == True:
        data = taichi_shear_rot_aug(data, shear=shear, rotate=rotate, rotate_angle=rotate_angle, shear_angle_list = shear_angle_list)
    else: # 测试集保留原数据形式
        data = data
        # data [T, V, C]
    # step3. 舍弃z轴，坐标到像素值的变换
    joints = data[None, :] # [1, T, V, C]
    # 1) x, y轴方向变化
    joints[..., 0] = -joints[..., 0]
    joints[..., 1] = -joints[..., 1]
    # 2) 减去三个轴的最小值
    x_min = joints[..., 0].min()
    y_min = joints[..., 1].min()
    z_min = joints[..., 2].min()
    min_vec = np.array([x_min, y_min, z_min])
    joints = joints - min_vec
    # 3) 乘以一个倍数
    multiple_value = 5
    joints = joints* multiple_value
    # 4) 加上一个余量值
    margin_value = 100
    joints = joints + margin_value
    assert joints[...,0].max()<1440 # width max
    assert joints[...,1].max()<1920 # height max
    anno = dict()
    anno['keypoint'] = joints[..., :2] # [N=1, T, V, C=2]
    anno['keypoint_score'] = np.ones(
        shape = joints[...,0].shape,
        dtype = np.float32
    )
    anno['frame_dir'] = sample_name
    anno['img_shape'] = (1920, 1440) # [height, width]
    anno['original_shape'] = (1920, 1440) # [height, width]
    anno['total_frames'] = joints.shape[1]
    anno['label'] = int(sample_name.split('A')[1][:3]) - 1
    return anno    
    

def taichi_aug_method(random_seed, shear = False, rotate = False, rotate_angle_list = None, shear_angle_list = None):
    # 固定的random_seed防止收到样本的干扰
    taichi_raw_rel_path = 'data/taichi/action.npy'
    project_folder_path = os.path.abspath('.')[
        :os.path.abspath('.').find('myposec3d') + len('myposec3d')
    ]
    taichi_np_dict = np.load(
        os.path.join(
            project_folder_path,
            taichi_raw_rel_path
        ),
        allow_pickle=True, encoding = 'latin1'
    ).item()
    train_set_pkl = list() # 用于存储最终生成的样本
    test_set_pkl = list()
    train_aug_sample = list() # 用于存储第一步扩增后的样本
    test_aug_sample = list()
    for key in taichi_np_dict.keys(): # ['a1', ..., 'a20']
        sample_length = len(taichi_np_dict[key]) # 一类数据的样本数
        train_offset, test_offset = random_train_test_split(
            length = sample_length,
            test_ratio = 0.7,
            random_seed = random_seed
        ) # 用来测试数据预处理的作用，固定test_ratio和random_seed
        train_samples = [
            taichi_np_dict[key][train_offset_idx] for train_offset_idx in train_offset
        ] # 训练集的样本
        for (repeat_idx, train_sample) in zip(train_offset, train_samples):
            # 类--key'a1' 第几个动作--repeat_idx 动作样本--train_sample
            # 第一步：将单个样本扩增到train_aug个样本，并且命名(没有进行任何预处理操作)
            # 因为函数返回的是List，所以要用extend而不是append
            train_aug_sample.extend(
                train_sample_aug(
                    key = key, repeat_idx = repeat_idx,
                    train_sample = np.array(train_sample, dtype = np.float32),
                    train_aug = 4
                )
            ) # list train_aug_sample的每一个元素是一个dict{'name': 'AXXXRXXXSXXX', 'data': [V=72, C=3, T=64]}
        # 测试集不需要扩增，只需要记录即可
        test_samples = [
            taichi_np_dict[key][test_offset_idx] for test_offset_idx in test_offset
        ] # 测试集的样本
        for (repeat_idx, test_sample) in zip(test_offset, test_samples):
            # 函数返回的是dict，所以使用append
            test_aug_sample.append(
                test_sample_extraction(
                    key = key, repeat_idx = repeat_idx,
                    test_sample = np.array(test_sample, dtype=np.float32)
                )
            )
    
    # 第二步训练集的每一个样本进行shear或者rotation的变换，并进行坐标到像素的变换，生成annotation
    assert len(train_aug_sample) == len(rotate_angle_list)
    for idx, skeleton in enumerate(train_aug_sample):
        train_set_pkl.append(
            taichi_skeleton_annotation(
                skeleton = skeleton,
                shear=shear, rotate = rotate,
                rotate_angle = rotate_angle_list[idx], # 一个角度值
                shear_angle_list = shear_angle_list[idx], # [2, 3]
                train_mode=True
            )
        )
    # 测试集只进行坐标到像素的变换
    for skeleton in test_aug_sample:
        test_set_pkl.append(
            taichi_skeleton_annotation(
                skeleton=skeleton,
                train_mode= False
            )
        )
    # 第三步保存样本到文件
    if shear == True and rotate == True:
        folder_name = 'WSWR'
    elif shear == True and rotate == False:
        folder_name = 'WSNR'
    elif shear == False and rotate == True:
        folder_name = 'NSWR'
    elif shear == False and rotate == False:
        folder_name = 'NSNR' 
    else:
        raise ValueError('invalid shear or rotate')
    ds_taichi_folder_path = os.path.join(
        project_folder_path, 'data/ds_taichi/TEST7AUG4', folder_name
    )
    if not os.path.exists(ds_taichi_folder_path):
        os.makedirs(ds_taichi_folder_path)
    dump(
        train_set_pkl,
        os.path.join(
            ds_taichi_folder_path, 'train_data.pkl'
        )
    )
    dump(
        test_set_pkl,
        os.path.join(
            ds_taichi_folder_path, 'test_data.pkl'
        )
    )

if __name__ == '__main__':
    # 一共有1-10类，一类有20个动作，为了测试数据预处理的作用
    # 30%作为训练集，70%作为测试集，一个样本扩增倍数为4，TEST7AUG4
    # baseline 只减去最小值--NSNR
    # 包含Shear Augmentation--WSNR
    # 包含Rotation Augmentation--NSWR
    # Shear + Rotation--WSWR 
    # 要保证旋转的角度确定，就先将随机的角度存储下来，输入到函数中
    # 训练集数量20*10*0.3*4 = 240
    # rotate操作需要240个角度，shear需要6*240个角度

    train_samples = 240
    rotate_angle_list = []
    for i in range(train_samples):
        rotate_angle_list.append(random.uniform(-60/180*np.pi, 60/180*np.pi))
    rotate_angle_list = np.array(rotate_angle_list, dtype=np.float32) #240
    shear_angle_list = np.zeros((train_samples, 2, 3), dtype = np.float32) # [240, 2, 3]
    sap = 0.5
    for i in range(train_samples):
        shear_angle_list[i, :] = np.array([
            [random.uniform(-sap, sap), random.uniform(-sap, sap), random.uniform(-sap, sap)], 
            [random.uniform(-sap, sap), random.uniform(-sap, sap), random.uniform(-sap, sap)]
        ], dtype=np.float32)
    
    taichi_aug_method(
        random_seed=42, shear=True, rotate=True, 
        rotate_angle_list = rotate_angle_list,
        shear_angle_list = shear_angle_list
    ) # WSWR
    taichi_aug_method(
        random_seed=42, shear=True, rotate=False,
        rotate_angle_list = rotate_angle_list,
        shear_angle_list = shear_angle_list
    ) # WSNR
    taichi_aug_method(
        random_seed=42, shear=False, rotate=True,
        rotate_angle_list = rotate_angle_list,
        shear_angle_list = shear_angle_list
    ) # NSWR
    taichi_aug_method(
        random_seed=42, shear=False, rotate=False,
        rotate_angle_list = rotate_angle_list,
        shear_angle_list = shear_angle_list
    ) # NSNR