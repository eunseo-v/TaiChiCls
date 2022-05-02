CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29502 ./tools/dist_train.sh configs/exp7/ft-WSNR.py 4 --validate --seed 42 --deterministic
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29502 ./tools/dist_train.sh configs/exp7/ft-NSWR.py 4 --validate --seed 42 --deterministic
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29502 ./tools/dist_train.sh configs/exp7/ft-WSWR.py 4 --validate --seed 43 --deterministic

CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29502 ./tools/dist_train.sh configs/exp7/ft-NFNC.py 4 --validate --seed 42 --deterministic
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29502 ./tools/dist_train.sh configs/exp7/ft-NFWC.py 4 --validate --seed 42 --deterministic
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29502 ./tools/dist_train.sh configs/exp7/ft-WFNC.py 4 --validate --seed 42 --deterministic

CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29502 ./tools/dist_train.sh configs/exp7/ft-T9A2.py 4 --validate --seed 42 --deterministic
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29502 ./tools/dist_train.sh configs/exp7/ft-T9A8.py 4 --validate --seed 42 --deterministic
