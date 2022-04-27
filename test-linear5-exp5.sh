CUDA_VISIBLE_DEVICES=3 python ./tools/test.py configs/exp5/finetune-with-pre.py ./model_pth/exp5/fintune-with-pre/best_top1_acc_epoch_43.pth
CUDA_VISIBLE_DEVICES=3 python ./tools/test.py configs/exp5/finetune-with-pre-lrd1.py ./model_pth/exp5/fintune-with-pre-lrd1/best_top1_acc_epoch_28.pth
CUDA_VISIBLE_DEVICES=3 python ./tools/test.py configs/exp5/finetune-with-pre-lrd01.py ./model_pth/exp5/fintune-with-pre-lrd01/best_top1_acc_epoch_19.pth
CUDA_VISIBLE_DEVICES=3 python ./tools/test.py configs/exp5/finetune-without-pre.py ./model_pth/exp5/fintune-without-pre/best_top1_acc_epoch_44.pth
