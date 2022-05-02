CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh configs/exp8/linear10-T9A2.py 4 --validate --seed 42 --deterministic
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh configs/exp8/linear10-T9A4.py 4 --validate --seed 42 --deterministic
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh configs/exp8/linear10-T9A8.py 4 --validate --seed 42 --deterministic
