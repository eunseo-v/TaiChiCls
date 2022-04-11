import os
os.sys.path.append(
    os.path.abspath('.')[
        :os.path.abspath('.').find('myposec3d') + len('myposec3d')
    ]
)

from mmaction.datasets.pipelines import Compose
import cv2
import numpy as np
from mmcv import load
import moviepy.editor as mpy

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.6
FONTCOLOR = (255, 255, 255)
BGBLUE = (0, 119, 182)
THICKNESS = 1
LINETYPE = 1

keypoint_pipeline = [
    dict(type = 'PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    # dict(type='CenterCrop', crop_size=64),
    dict(type='GenerateTaiChiPoseTarget', sigma=0.6, use_score=True, with_kp=True, with_limb=False)
]

limb_pipeline = [
    dict(type = 'PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1.0, allow_imgpad=True), # 此时长宽比固定为1:1
    dict(type='Resize', scale=(-1, 64)),
    dict(type='CenterCrop', crop_size=64),
    dict(type='GenerateTaiChiPoseTarget', sigma=0.6, use_score=True, with_kp=False, with_limb=True)
]

def get_pseudo_heatmap(anno, flag = 'keypoint'):
    assert flag in ['keypoint', 'limb']
    pipeline = Compose(keypoint_pipeline if flag == 'keypoint' else limb_pipeline)
    return pipeline(anno)

def vis_heatmaps(heatmaps, channel = -1, ratio=8):
    # if channel is -1, draw all keypoints/limbs on the same map
    import matplotlib.cm as cm
    h, w, _ = heatmaps[0].shape
    newh, neww = int(h* ratio), int(w* ratio)

    if channel == -1:
        heatmaps = [np.max(x, axis = -1) for x in heatmaps]
    cmap = cm.viridis
    heatmaps = [(cmap(x)[..., :3] * 255).astype(np.uint8) for x in heatmaps]
    heatmaps = [cv2.resize(x, (neww, newh)) for x in heatmaps]
    return heatmaps

def add_label(frame, label, BGCOLOR=BGBLUE):
    threshold = 30
    def split_label(label):
        label = label.split()
        lines, cline = [], ''
        for word in label:
            if len(cline) + len(word) < threshold:
                cline = cline + ' ' + word
            else:
                lines.append(cline)
                cline = word
        if cline != '':
            lines += [cline]
        return lines
    
    if len(label) > 30:
        label = split_label(label)
    else:
        label = [label]
    label = ['Action: '] + label
    
    sizes = []
    for line in label:
        sizes.append(cv2.getTextSize(line, FONTFACE, FONTSCALE, THICKNESS)[0])
    box_width = max([x[0] for x in sizes]) + 10
    text_height = sizes[0][1]
    box_height = len(sizes) * (text_height + 6)
    
    cv2.rectangle(frame, (0, 0), (box_width, box_height), BGCOLOR, -1)
    for i, line in enumerate(label):
        location = (5, (text_height + 6) * i + text_height + 3)
        cv2.putText(frame, line, location, FONTFACE, FONTSCALE, FONTCOLOR, THICKNESS, LINETYPE)
    return frame

if __name__ == '__main__':
    project_folder_path = os.path.abspath('.')[
        :os.path.abspath('.').find('myposec3d') + len('myposec3d')
    ]
    aug_method= 'WSWR' # NSNR, NSWR, WSNR, WSWR
    data_path = os.path.join(
        project_folder_path, 'data/ds_taichi/TEST7AUG4/{}/train_data.pkl'.format(aug_method)
    )
    anno = load(data_path)[10] # 一个样本
    process_anno = get_pseudo_heatmap(anno, flag = 'limb')
    limb_mapvis = vis_heatmaps(process_anno['imgs'])
    limb_mapvis = [add_label(f, aug_method) for f in limb_mapvis]
    vid = mpy.ImageSequenceClip(limb_mapvis, fps = 24)
    vid.write_videofile(
        os.path.join(project_folder_path, 'test_model', '{}-{}.mp4'.format(aug_method, process_anno['frame_dir']))
    )
    pass
