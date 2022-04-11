'''
    关节点的可视化
'''
import os
import cv2
import os.path as osp
import decord
import numpy as np
import matplotlib.pyplot as plt
import urllib
import moviepy.editor as mpy
import random as rd
from mmpose.apis import vis_pose_result
from mmpose.models import TopDown
from mmcv import load, dump

ntu_test_ann = './wjsn_dance.pkl'
ntu_vid = './wjsn_dance.mp4'
FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.6
FONTCOLOR = (255, 255, 255)
BGBLUE = (0, 119, 182)
THICKNESS = 1
LINETYPE = 1

def add_label(frame, label, BGCOLOR=BGBLUE):
    '''
        视频显示中添加标签
    '''
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

def vis_skeleton(vid_path, anno, category_name=None, ratio=0.5):
    '''
        在视频中显示骨架信息
    '''
    vid = decord.VideoReader(vid_path)
    frames = [x.asnumpy() for x in vid]
    
    h, w, _ = frames[0].shape
    new_shape = (int(w * ratio), int(h * ratio))
    frames = [cv2.resize(f, new_shape) for f in frames]
    
    assert len(frames) == anno['total_frames']
    # The shape is N x T x K x 3
    kps = np.concatenate([anno['keypoint'], anno['keypoint_score'][..., None]], axis=-1)
    kps[..., :2] *= ratio
    # Convert to T x N x K x 3
    kps = kps.transpose([1, 0, 2, 3])
    vis_frames = []

    # we need an instance of TopDown model, so build a minimal one
    model = TopDown(backbone=dict(type='ShuffleNetV1'))

    for f, kp in zip(frames, kps):
        bbox = np.zeros([1, 4], dtype=np.float32)
        result = [dict(bbox=bbox, keypoints=k) for k in kp]
        vis_frame = vis_pose_result(model, f, result)
        
        if category_name is not None:
            vis_frame = add_label(vis_frame, category_name)
        
        vis_frames.append(vis_frame)
    return vis_frames


test_anno = load(ntu_test_ann)
ntu_categories = ['drink water', 'eat meal/snack', 'brushing teeth', 'brushing hair', 'drop', 'pickup', 
                  'throw', 'sitting down', 'standing up (from sitting position)', 'clapping', 'reading', 
                  'writing', 'tear up paper', 'wear jacket', 'take off jacket', 'wear a shoe', 
                  'take off a shoe', 'wear on glasses', 'take off glasses', 'put on a hat/cap', 
                  'take off a hat/cap', 'cheer up', 'hand waving', 'kicking something', 
                  'reach into pocket', 'hopping (one foot jumping)', 'jump up', 
                  'make a phone call/answer phone', 'playing with phone/tablet', 'typing on a keyboard', 
                  'pointing to something with finger', 'taking a selfie', 'check time (from watch)', 
                  'rub two hands together', 'nod head/bow', 'shake head', 'wipe face', 'salute', 
                  'put the palms together', 'cross hands in front (say stop)', 'sneeze/cough', 
                  'staggering', 'falling', 'touch head (headache)', 'touch chest (stomachache/heart pain)', 
                  'touch back (backache)', 'touch neck (neckache)', 'nausea or vomiting condition', 
                  'use a fan (with hand or paper)/feeling warm', 'punching/slapping other person', 
                  'kicking other person', 'pushing other person', 'pat on back of other person', 
                  'point finger at the other person', 'hugging other person', 
                  'giving something to other person', "touch other person's pocket", 'handshaking', 
                  'walking towards each other', 'walking apart from each other']

vis_frames = vis_skeleton(
    vid_path=ntu_vid, anno=test_anno,
    category_name='wjsn_dance'
)
vid=mpy.ImageSequenceClip(vis_frames, fps=24)
vid.write_videofile('wjsn_dance_pose_rgb.mp4')