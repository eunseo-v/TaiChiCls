import os
os.sys.path.append(
    os.path.abspath('.')[
        :os.path.abspath('.').find('myposec3d') + len('myposec3d')
    ]
)
import matplotlib.pyplot as plt
import numpy as np
from mmcv import load

if __name__ == '__main__':
    project_folder_path = os.path.abspath('.')[
        :os.path.abspath('.').find('myposec3d') + len('myposec3d')
    ]
    ntu_rel_path = 'data/posec3d/ntu60_xsub_train.pkl'
    data = load(
        os.path.join(
            project_folder_path, ntu_rel_path
        )
    )[77]['keypoint'] # [1, T, V=17, C=2] 2
    skeleton = data[0,20] # [17, 2]
    fig, ax = plt.subplots()
    keypoints_group = [
        [0, 1, 2, 3, 4], [5, 7, 9], [6, 8, 10],
        [11, 13, 15], [12, 14, 16]
    ]
    color = ["red","green","black","orange","purple"]
    for i, keypoints in enumerate(keypoints_group):
        ax.scatter(
            x = skeleton[keypoints, 0], y = -skeleton[keypoints,1],
            s = 30, c = color[i]
        )
    limbs_group = [
        ((0, 1), (0, 2), (1, 3), (2, 4), (11, 12)),
        ((0, 5), (5, 7), (7, 9)),
        ((0, 6), (6, 8), (8, 10)),
        ((5, 11), (11, 13), (13, 15)),
        ((6, 12), (12, 14), (14, 16))
    ]
    line_color = ['#DC143C', '#90EE90', '#778899', '#FFD700', '#DDA0DD']
    for i, limbs in enumerate(limbs_group):
        for (st_idx, end_idx) in limbs:
            ax.plot(
                [skeleton[st_idx, 0], skeleton[end_idx, 0]],
                [-skeleton[st_idx,1], -skeleton[end_idx,1]],
                c = line_color[i],
                linewidth = '1', ls = 'dotted'
            )
    ax.set(aspect = 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    [ax.spines[loc_axis].set_visible(False) for loc_axis in ['top', 'right', 'bottom', 'left']]
    ax.get_figure().savefig(
        os.path.join(
            project_folder_path,
            'fig/NTU-skeleton-uniform-1.png'
        )
    )