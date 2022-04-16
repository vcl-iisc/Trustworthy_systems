# ::::::::::::::::::::: CIFAR10 :::::::::::::::::::::

python3 _code/ddb.py --dataset cifar10 --gpu 1 --model_name Target --load_model_pth _code/checkpoint/target.pth --save_pth KD_data/CIFAR_ddb_DeepFool.pth --batch_size 32

# python3 _code/ddb.py --dataset cifar10 --gpu 1 --model_name WideResNet --load_model_pth teacher_models/WRN_CIFAR10_clean.t7 --save_pth KD_data/CIFAR_WN_ddb.pth --batch_size 1

# CUDA_VISIBLE_DEVICES=1,2 python3 _code/FAS_cumm.py --dataset cifar10 --load_model_pth _code/checkpoint/target.pth --load_ddb_data KD_data/CIFAR_WN_ddb.pth --save_pth KD_data/CIFAR_fas_cumm.pth --batch_size 4096 --k 5