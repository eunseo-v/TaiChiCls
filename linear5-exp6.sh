CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29502 ./tools/dist_train.sh configs/exp6/NSNR-linear5-T9A2.py 4 --validate --seed 42 --deterministic
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29502 ./tools/dist_train.sh configs/exp6/NSNR-linear5-T9A4.py 4 --validate --seed 42 --deterministic
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29502 ./tools/dist_train.sh configs/exp6/NSNR-linear5-T9A8.py 4 --validate --seed 42 --deterministic
