CUDA_VISIBLE_DEVICES=7 python ./tools/test.py configs/exp0/NSNR-linear5-lr30.py ./model_pth/exp0/NSNR-linear5-lr30/best_top1_acc_epoch_24.pth
CUDA_VISIBLE_DEVICES=7 python ./tools/test.py configs/exp0/NSNR-linear5-lr3.py ./model_pth/exp0/NSNR-linear5-lr3/best_top1_acc_epoch_23.pth
CUDA_VISIBLE_DEVICES=7 python ./tools/test.py configs/exp0/NSNR-linear5-lr1.py ./model_pth/exp0/NSNR-linear5-lr1/best_top1_acc_epoch_20.pth
CUDA_VISIBLE_DEVICES=7 python ./tools/test.py configs/exp0/NSNR-linear5-lrd4.py ./model_pth/exp0/NSNR-linear5-lrd4/best_top1_acc_epoch_18.pth
CUDA_VISIBLE_DEVICES=7 python ./tools/test.py configs/exp0/NSNR-linear5-lrd1.py ./model_pth/exp0/NSNR-linear5-lrd1/best_top1_acc_epoch_16.pth
CUDA_VISIBLE_DEVICES=7 python ./tools/test.py configs/exp0/NSNR-linear5-lrd01.py ./model_pth/exp0/NSNR-linear5-lrd01/best_top1_acc_epoch_22.pth