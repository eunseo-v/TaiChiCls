import os
os.sys.path.append(
    os.path.abspath('.')[
        :os.path.abspath('.').find('myposec3d') + len('myposec3d')
    ]
)
import matplotlib.pyplot as plt
import numpy as np
import math
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

def unit_vector(vector):
    """
        Returns the unit vetor of the vector
    """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    if np.abs(v1).sum() < 1e-6 or np.abs(v2).sum() < 1e-6:
        return 0
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


if __name__ == '__main__':
    project_folder_path = os.path.abspath('.')[
        :os.path.abspath('.').find('myposec3d') + len('myposec3d')
    ]
    taichi_raw_rel_path = 'data/taichi/action_mod.npy'
    taichi_np_dict = np.load(
        os.path.join(
            project_folder_path,
            taichi_raw_rel_path
        ),
        allow_pickle=True, encoding='latin1' 
    ).item()
    # 选择第一类的第一个样本
    sample = taichi_np_dict['a1'][0] # [72, 3, T]
    # 选择某一帧
    skeleton_xyz = sample[..., 500]
    x_min = skeleton_xyz[...,0].min()
    y_min = skeleton_xyz[...,1].min()
    z_min = skeleton_xyz[...,2].min()
    skeleton = skeleton_xyz - np.array([x_min, y_min, z_min], dtype=np.float32)
    # 绕着y轴旋转
    joint_rshoulder = skeleton[16,:]
    joint_lshoulder = skeleton[44,:]
    axis_x = np.cross(
        joint_rshoulder - joint_lshoulder,
        [1, 0, 0]
    )
    angle_x = angle_between(
        v1=joint_rshoulder - joint_lshoulder,
        v2 = [1, 0, 0]
    )
    matrix_x = rotation_matrix(
        axis = axis_x, 
        theta = angle_x
    )
    for i_j, joint in enumerate(skeleton):
        skeleton[i_j] = np.dot(matrix_x, joint)
    
    # 每一个通道关节点的颜色相同
    fig, ax = plt.subplots()
    # 画点
    keypoints_group = [
        [0, 9, 10, 11, 12, 13, 14, 15], 
        [
            44, 45, 46, 47, 48, 49, 
            50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 
            60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 
            70, 71
        ], 
        [
            16, 17, 18, 19, 
            20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
            30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
            40, 41, 42, 43
        ],
        [5, 6, 7, 8], 
        [1, 2, 3, 4]
    ]
    color = ["red","green","black","orange","purple"]
    for i,keypoints in enumerate(keypoints_group):
        ax.scatter(
            x = -skeleton[keypoints,0], y = skeleton[keypoints,1],
            s = 30, c = color[i]
        )    
    # 点连线
    limbs_group = [
        (
            (0, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (0, 1), (0, 5)
        ),
        (
            (12, 44), (44, 45), (45, 46), (46, 47),
            (47, 48), (47, 52), (47, 57), (47, 62), (47, 67),
            (48, 49), (49, 50), (50, 51),
            (52, 53), (53, 54), (54, 55), (55, 56),
            (57, 58), (58, 59), (59, 60), (60, 61),
            (62, 63), (63, 64), (64, 65), (65, 66),
            (67, 68), (68, 69), (69, 70), (70, 71)
        ),
        (
            (12, 16), (16, 17), (17, 18), (18, 19),
            (19, 20), (19, 24), (19, 29), (19, 34), (19, 39),
            (20, 21), (21, 22), (22, 23),
            (24, 25), (25, 26), (26, 27), (27, 28),
            (29, 30), (30, 31), (31, 32), (32, 33),
            (34, 35), (35, 36), (36, 37), (37, 38),
            (39, 40), (40, 41), (41, 42), (42, 43)
        ),
        (
            (5, 6), (6, 7), (7, 8)
        ),
        (
            (1, 2), (2, 3), (3, 4)
        )
    ]
    line_color = ['#DC143C', '#90EE90', '#778899', '#FFD700', '#DDA0DD']
    for i, limbs in enumerate(limbs_group):
        for (st_idx, end_idx) in limbs:
            ax.plot(
                [-skeleton[st_idx,0], -skeleton[end_idx,0]],
                [skeleton[st_idx,1], skeleton[end_idx,1]],
                c = line_color[i], 
                linewidth = '1', ls = 'dotted'
            )
    ax.set(aspect=0.7)
    ax.set_xticks([])
    ax.set_yticks([])
    [ax.spines[loc_axis].set_visible(False) for loc_axis in ['top','right','bottom','left']]
    ax.get_figure().savefig(
        os.path.join(
            project_folder_path,
            'fig/TaiChi-skeleton-uniform-2.png'
        )
    )