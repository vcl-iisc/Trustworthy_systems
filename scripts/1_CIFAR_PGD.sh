# CIFAR 10, PGD

## resnet18
# CUDA_VISIBLE_DEVICES=1 python3 _code/ddb.py --dataset cifar10 --model_name resnet18 --atk PGD --batch_size 32 --dry_run
# CUDA_VISIBLE_DEVICES=1 python3 _code/FAS_cumm.py --dataset cifar10 --model_name resnet18 --atk PGD --batch_size 4096 --k 5

# # # # ## googlenet
# CUDA_VISIBLE_DEVICES=1 python3 _code/ddb.py --dataset cifar10 --model_name googlenet --atk PGD --batch_size 32
# CUDA_VISIBLE_DEVICES=1 python3 _code/FAS_cumm.py --dataset cifar10 --model_name googlenet --atk PGD --batch_size 1024 --k 5

# # # # ## vgg11
# CUDA_VISIBLE_DEVICES=1 python3 _code/ddb.py --dataset cifar10 --model_name vgg11 --atk PGD --batch_size 32
# CUDA_VISIBLE_DEVICES=1 python3 _code/FAS_cumm.py --dataset cifar10 --model_name vgg11 --atk PGD --batch_size 4096 --k 5


# SVHN, PGD

# resnet18
# CUDA_VISIBLE_DEVICES=0 python3 _code/ddb.py --dataset svhn --model_name resnet18 --load_model_pth _code/checkpoint/svhn_resnet18_natural.pt --atk PGD --batch_size 32
# CUDA_VISIBLE_DEVICES=0 python3 _code/FAS_cumm.py --dataset svhn --model_name resnet18 --load_model_pth _code/checkpoint/svhn_resnet18_natural.pt --atk PGD --batch_size 4096 --k 5

# ## googlenet
# CUDA_VISIBLE_DEVICES=0 python3 _code/ddb.py --dataset svhn --model_name googlenet --load_model_pth _code/checkpoint/svhn_googlenet_natural.pt --atk PGD --batch_size 32
# CUDA_VISIBLE_DEVICES=0 python3 _code/FAS_cumm.py --dataset svhn --model_name googlenet --load_model_pth _code/checkpoint/svhn_googlenet_natural.pt --atk PGD --batch_size 1024 --k 5

# # vgg11
# CUDA_VISIBLE_DEVICES=0 python3 _code/ddb.py --dataset svhn --model_name vgg11 --load_model_pth _code/checkpoint/svhn_vgg11_natural.pt --atk PGD --batch_size 32
# CUDA_VISIBLE_DEVICES=0 python3 _code/FAS_cumm.py --dataset svhn --model_name vgg11 --load_model_pth _code/checkpoint/svhn_vgg11_natural.pt --atk PGD --batch_size 4096 --k 5

# :::::::::::::::::: TinyImageNet PGD ::::::::::::::::::

## resnet18
# CUDA_VISIBLE_DEVICES=2 python3 _code/ddb.py --dataset tinyimagenet --model_name resnet18 --load_model_pth _code/checkpoint/tiny-imagenet-200/std_tinyimagenet_resnet_18_baseline/checkpoint.pth.tar --atk PGD --batch_size 64
# CUDA_VISIBLE_DEVICES=2 python3 _code/FAS_cumm.py --dataset tinyimagenet --model_name resnet18 --load_model_pth _code/checkpoint/tiny-imagenet-200/std_tinyimagenet_resnet_18_baseline/checkpoint.pth.tar --atk PGD --batch_size 1024 --k 5

# CUDA_VISIBLE_DEVICES=2 python3 _code/ddb_2.py --dataset cifar10 --model_name resnet18 --load_model_pth _code/checkpoint/svhn_resnet18_natural.pt --atk AutoAttack --batch_size 32

# :::::::::::::::::: Robust cifar10 PGD ::::::::::::::::::

# ROBUST resnet18

# CUDA_VISIBLE_DEVICES=0 python3 _code/ddb.py --dataset cifar10 --model_name robust_resnet18 --load_model_pth _code/checkpoint/cifar10_resnet18_robust.pt --atk PGD --batch_size 64
# CUDA_VISIBLE_DEVICES=1 python3 _code/FAS_cumm.py --dataset cifar10 --model_name robust_resnet18 --load_model_pth _code/checkpoint/cifar10_resnet18_robust.pt --atk PGD --batch_size 1024 --k 5


# ROBUST resnet18 on test set

CUDA_VISIBLE_DEVICES=2 python3 _code/ddb.py --dataset cifar10 --model_name robust_resnet18_test --load_model_pth _code/checkpoint/cifar10_resnet18_robust.pt --atk DeepFool --batch_size 64
CUDA_VISIBLE_DEVICES=2 python3 _code/FAS_cumm.py --dataset cifar10 --model_name robust_resnet18_test --load_model_pth _code/checkpoint/cifar10_resnet18_robust.pt --atk DeepFool --batch_size 1024 --k 5