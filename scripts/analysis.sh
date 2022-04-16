## CIFAR10 (defaults)
# python3 _code/ddb.py
# CUDA_VISIBLE_DEVICES=0,1 python3 _code/FAS.py --dataset cifar10 --load_model_pth _code/checkpoint/target.pth --load_ddb_data savedir/CIFAR_ddb_step.pth --save_pth savedir/CIFAR_ddb_fas_cummul_invmask.pth --batch_size 4096 --k 5

# CUDA_VISIBLE_DEVICES=0,1 python3 _code/FAS_cumm.py --dataset cifar10 --load_model_pth _code/checkpoint/target.pth --load_ddb_data savedir/CIFAR_ddb_step.pth --save_pth savedir/CIFAR_ddb_fas_cummul_highfreq_remove.pth --batch_size 4096 --k 5

## TinyImageNet

# python3 _code/ddb.py --data _code/data/tiny-imagenet-200 --dataset tinyimagenet --gpu 0 --load_model_pth _code/checkpoint/tiny-imagenet-200/std_tinyimagenet_resnet_18_baseline/checkpoint.pth.tar --save_pth savedir/TIMGNET_ddb_step.pth
# CUDA_VISIBLE_DEVICES=2 python3 _code/FAS.py --dataset tinyimagenet --load_model_pth _code/checkpoint/tiny-imagenet-200/std_tinyimagenet_resnet_18_baseline/checkpoint.pth.tar --load_ddb_data savedir/TIMGNET_ddb_step_final.pth --save_pth savedir/TIMGNET_ddb_lip.pth

# CUDA_VISIBLE_DEVICES=0,1 python3 _code/FAS_cumm.py --dataset tinyimagenet --load_model_pth _code/checkpoint/tiny-imagenet-200/std_tinyimagenet_resnet_18_baseline/checkpoint.pth.tar --load_ddb_data savedir/TIMGNET_ddb_step_final.pth --save_pth savedir/TIMGNET_ddb_fas_cummul_highfreq_remove.pth --r_range 63 --batch_size 2048 --k 5

python3 _code/ddb.py --dataset cifar10 --gpu 1 --model_name Target --load_model_pth _code/checkpoint/target.pth --batch_size 32
CUDA_VISIBLE_DEVICES=0,1 python3 _code/FAS_cumm.py --dataset cifar10 --load_model_pth _code/checkpoint/target.pth --batch_size 4096 --k 5
