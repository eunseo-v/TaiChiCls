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
    skeleton = data[0,70] # [17, 2]
    fig, ax = plt.subplots()
    ax.scatter(
        x = skeleton[:, 0], y = -skeleton[:, 1],
        s = 30,
        c = np.random.uniform(15, 512, len(skeleton[:, 0])),
        cmap = 'nipy_spectral'
    )
    limbs_group=(
        (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (5, 7),
        (7, 9), (0, 6), (6, 8), (8, 10), (5, 11), (11, 13),
        (13, 15), (6, 12), (12, 14), (14, 16), (11, 12)
    )
    for (st_idx, end_idx) in limbs_group:
        ax.plot(
            [skeleton[st_idx, 0], skeleton[end_idx, 0]],
            [-skeleton[st_idx, 1], -skeleton[end_idx, 1]],
            linewidth = '1', ls = 'dotted'
        )
    ax.set(aspect = 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    [ax.spines[loc_axis].set_visible(False) for loc_axis in ['top', 'right', 'bottom', 'left']]
    ax.get_figure().savefig(
        os.path.join(
            project_folder_path,
            'fig/NTU-skeleton-chaos-2.png'
        )
    )
    pass
