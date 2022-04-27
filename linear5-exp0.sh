CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh configs/exp0/NSNR-linear5-lr30.py 4 --validate --seed 42 --deterministic
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh configs/exp0/NSNR-linear5-lr3.py 4 --validate --seed 42 --deterministic
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh configs/exp0/NSNR-linear5-lr1.py 4 --validate --seed 42 --deterministic
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh configs/exp0/NSNR-linear5-lrd4.py 4 --validate --seed 42 --deterministic
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh configs/exp0/NSNR-linear5-lrd1.py 4 --validate --seed 42 --deterministic
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh configs/exp0/NSNR-linear5-lrd01.py 4 --validate --seed 42 --deterministic
