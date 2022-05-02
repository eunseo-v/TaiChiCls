import os
os.sys.path.append(
    os.path.abspath('.')[
        :os.path.abspath('.').find('myposec3d') + len('myposec3d')
    ]
)
import torch
import numpy as np
from mmcv import load

def top_k_accuracy(scores, labels, topk=(1, )):
    """Calculate top k accuracy score.

    Args:
        scores (list[np.ndarray]): Prediction scores for each class.
        labels (list[int]): Ground truth labels.
        topk (tuple[int]): K value for top_k_accuracy. Default: (1, ).

    Returns:
        list[float]: Top k accuracy score for each k.
    """
    res = []
    labels = np.array(labels)[:, np.newaxis]
    for k in topk:
        max_k_preds = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
        match_array = np.logical_or.reduce(max_k_preds == labels, axis=1)
        topk_acc_score = match_array.sum() / match_array.shape[0]
        res.append(topk_acc_score)

    return res

if __name__ == '__main__':
    project_folder_path = os.path.abspath('.')[
        :os.path.abspath('.').find('myposec3d') + len('myposec3d')
    ]
    kp_path = os.path.join(
        project_folder_path,
        'model_pth/exp3/NSNR-linear5-T9A8/test_result/results.pkl'
    )
    lb_path = os.path.join(
        project_folder_path,
        'model_pth/exp6/NSNR-linear5-T9A8-limb/test_result/results.pkl'
    )
    kp_results = load(kp_path)
    lb_results = load(lb_path)
    labels = list() # 存测试集每一个样本的list
    fusion_scores = list()
    for name in kp_results['sample_names']:
        kp_idx = kp_results['sample_names'].index(name)
        lb_idx = lb_results['sample_names'].index(name)
        fusion_score = kp_results['outputs'][kp_idx] + lb_results['outputs'][lb_idx]
        fusion_scores.append(fusion_score)
        labels.append(
            int(name[1:4])-1
        )
    res = top_k_accuracy(scores=fusion_scores, labels=labels, topk=(1,))
    print(res)
    pass