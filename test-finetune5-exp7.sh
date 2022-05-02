CUDA_VISIBLE_DEVICES=3 python ./tools/test.py configs/exp7/ft-WSNR.py ./model_pth/exp7/ft-WSNR/best_top1_acc_epoch_19.pth
CUDA_VISIBLE_DEVICES=3 python ./tools/test.py configs/exp7/ft-NSWR.py ./model_pth/exp7/ft-NSWR/best_top1_acc_epoch_19.pth
CUDA_VISIBLE_DEVICES=3 python ./tools/test.py configs/exp7/ft-WSWR.py ./model_pth/exp7/ft-WSWR/best_top1_acc_epoch_30.pth

CUDA_VISIBLE_DEVICES=3 python ./tools/test.py configs/exp7/ft-NFNC.py ./model_pth/exp7/ft-NFNC/best_top1_acc_epoch_17.pth
CUDA_VISIBLE_DEVICES=3 python ./tools/test.py configs/exp7/ft-NFWC.py ./model_pth/exp7/ft-NFWC/best_top1_acc_epoch_19.pth
CUDA_VISIBLE_DEVICES=3 python ./tools/test.py configs/exp7/ft-WFNC.py ./model_pth/exp7/ft-WFNC/best_top1_acc_epoch_5.pth

CUDA_VISIBLE_DEVICES=3 python ./tools/test.py configs/exp7/ft-T9A2.py ./model_pth/exp7/ft-T9A2/best_top1_acc_epoch_9.pth
CUDA_VISIBLE_DEVICES=3 python ./tools/test.py configs/exp7/ft-T9A8.py ./model_pth/exp7/ft-T9A8/best_top1_acc_epoch_7.pth