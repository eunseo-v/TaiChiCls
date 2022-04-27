CUDA_VISIBLE_DEVICES=3 python ./tools/test.py configs/exp1/NSNR-linear5.py ./model_pth/exp1/NSNR-linear5/best_top1_acc_epoch_31.pth
CUDA_VISIBLE_DEVICES=3 python ./tools/test.py configs/exp1/NSWR-linear5.py ./model_pth/exp1/NSWR-linear5/best_top1_acc_epoch_45.pth
CUDA_VISIBLE_DEVICES=3 python ./tools/test.py configs/exp1/WSNR-linear5.py ./model_pth/exp1/WSNR-linear5/best_top1_acc_epoch_36.pth
CUDA_VISIBLE_DEVICES=3 python ./tools/test.py configs/exp1/WSWR-linear5.py ./model_pth/exp1/WSWR-linear5/best_top1_acc_epoch_32.pth
CUDA_VISIBLE_DEVICES=3 python ./tools/test.py configs/exp1/T9A4-NFNC.py ./model_pth/exp1/T9A4-NFNC/best_top1_acc_epoch_41.pth
CUDA_VISIBLE_DEVICES=3 python ./tools/test.py configs/exp1/T9A4-NFWC.py ./model_pth/exp1/T9A4-NFWC/best_top1_acc_epoch_44.pth
CUDA_VISIBLE_DEVICES=3 python ./tools/test.py configs/exp1/T9A4-WFNC.py ./model_pth/exp1/T9A4-WFNC/best_top1_acc_epoch_31.pth
CUDA_VISIBLE_DEVICES=3 python ./tools/test.py configs/exp1/T9A4-WFWC.py ./model_pth/exp1/T9A4-WFWC/best_top1_acc_epoch_31.pth
