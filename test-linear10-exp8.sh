CUDA_VISIBLE_DEVICES=3 python ./tools/test.py configs/exp8/linear10-T9A2.py ./model_pth/exp8/linear10-T9A2/best_top1_acc_epoch_41.pth
CUDA_VISIBLE_DEVICES=3 python ./tools/test.py configs/exp8/linear10-T9A4.py ./model_pth/exp8/linear10-T9A4/best_top1_acc_epoch_43.pth
CUDA_VISIBLE_DEVICES=3 python ./tools/test.py configs/exp8/linear10-T9A8.py ./model_pth/exp8/linear10-T9A8/best_top1_acc_epoch_45.pth
