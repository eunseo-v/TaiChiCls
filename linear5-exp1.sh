CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh configs/exp1/NSNR-linear5.py 4 --validate --seed 42 --deterministic
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh configs/exp1/NSWR-linear5.py 4 --validate --seed 42 --deterministic
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh configs/exp1/WSNR-linear5.py 4 --validate --seed 42 --deterministic
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh configs/exp1/WSWR-linear5.py 4 --validate --seed 42 --deterministic
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh configs/exp1/T9A4-NFNC.py 4 --validate --seed 42 --deterministic
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh configs/exp1/T9A4-NFWC.py 4 --validate --seed 42 --deterministic
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh configs/exp1/T9A4-WFNC.py 4 --validate --seed 42 --deterministic
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh configs/exp1/T9A4-WFWC.py 4 --validate --seed 42 --deterministic
