use_hf=0

atk='DeepFool'
cuda=1

# mod='mobilenet_ddb'
# mp='_code/checkpoint/MobileNetV2_ddbs_cdb_DeepFool_0_robust_resnet18_train_100_teacher_ResNet18_adv_False_0.8_8.0.t7'

# mod='mobilenet_trust'
# mp='_code/checkpoint/MobileNetV2_trust_more_DeepFool_0_robust_resnet18_train_100_teacher_ResNet18_adv_False_0.8_8.0.t7'



# # :::::::::::::::::: CIFAR 10, $atk ::::::::::::::::::

### resnet18
# CUDA_VISIBLE_DEVICES=$cuda python3 _code/ddb.py --dataset cifar10 --model_name resnet18 --atk $atk --batch_size 32 --split_use test --save_pth csv_data --auto_mod 0 --dry_run 0
# CUDA_VISIBLE_DEVICES=$cuda python3 _code/FAS_cumm.py --dataset cifar10 --model_name resnet18 --atk $atk --batch_size 128 --k 5 --split_use test --use_hf $use_hf --save_pth csv_data --auto_mod 0 --dry_run 0

# # ### googlenet
# # CUDA_VISIBLE_DEVICES=$cuda python3 _code/ddb.py --dataset cifar10 --model_name googlenet --atk $atk --batch_size 32 --split_use train
# # CUDA_VISIBLE_DEVICES=$cuda python3 _code/FAS_cumm.py --dataset cifar10 --model_name googlenet --atk $atk --batch_size 4096 --k 5 --split_use train --use_hf $use_hf

# # ### vgg11
# CUDA_VISIBLE_DEVICES=$cuda python3 _code/ddb.py --dataset cifar10 --model_name vgg11 --atk $atk --batch_size 32 --split_use test --save_pth csv_data --auto_mod 0
# CUDA_VISIBLE_DEVICES=$cuda python3 _code/FAS_cumm.py --dataset cifar10 --model_name vgg11 --atk $atk --batch_size 128 --k 5 --split_use test --use_hf $use_hf --save_pth csv_data --auto_mod 0

# # ## robust_resnet18

# CUDA_VISIBLE_DEVICES=$cuda python3 _code/ddb.py --dataset cifar10 --model_name robust_resnet18 --atk $atk --batch_size 32 --split_use test --dry_run 0 --save_pth csv_data --auto_mod 0
# CUDA_VISIBLE_DEVICES=$cuda python3 _code/FAS_cumm.py --dataset cifar10 --model_name robust_resnet18 --atk $atk --batch_size 128 --k 5 --split_use test --use_hf $use_hf --dry_run 0 --save_pth csv_data --auto_mod 0

# ## robust_wideresnet

# CUDA_VISIBLE_DEVICES=$cuda python3 _code/ddb.py --dataset cifar10 --model_name robust_wideresnet --atk $atk --batch_size 32 --split_use train
# CUDA_VISIBLE_DEVICES=$cuda python3 _code/FAS_cumm.py --dataset cifar10 --model_name robust_wideresnet --atk $atk --batch_size 4096 --k 5 --split_use train --use_hf $use_hf

## MobileNet

# CUDA_VISIBLE_DEVICES=$cuda python3 _code/ddb.py --dataset cifar10 --model_name $mod --load_model_pth $mp --atk $atk --batch_size 64 --split_use train --dry_run 0 --save_pth csv_data --auto_mod 0
# CUDA_VISIBLE_DEVICES=$cuda python3 _code/FAS_cumm.py --dataset cifar10 --model_name $mod --load_model_pth $mp --atk $atk --batch_size 128 --k 5 --split_use train --use_hf $use_hf --dry_run 0 --save_pth csv_data --auto_mod 0

mod='mobilenet_trust'
mp='_code/checkpoint/MobileNetV2_trust_more_DeepFool_0_robust_resnet18_train_100_teacher_ResNet18_adv_False_0.8_8.0.t7'

CUDA_VISIBLE_DEVICES=$cuda python3 _code/ddb.py --dataset cifar10 --model_name $mod --load_model_pth $mp --atk $atk --batch_size 64 --split_use train --dry_run 0 --save_pth csv_data --auto_mod 0
CUDA_VISIBLE_DEVICES=$cuda python3 _code/FAS_cumm.py --dataset cifar10 --model_name $mod --load_model_pth $mp --atk $atk --batch_size 128 --k 5 --split_use train --use_hf $use_hf --dry_run 0 --save_pth csv_data --auto_mod 0


# # :::::::::::::::::: CIFAR 100, $atk ::::::::::::::::::
# ## robust_resnet18

# CUDA_VISIBLE_DEVICES=$cuda python3 _code/ddb.py --dataset cifar100 --model_name robust_resnet18 --atk $atk --batch_size 128 --split_use test --dry_run 0
# CUDA_VISIBLE_DEVICES=$cuda python3 _code/FAS_cumm.py --dataset cifar100 --model_name robust_resnet18 --atk $atk --batch_size 4096 --k 5 --split_use test --use_hf $use_hf --dry_run 0

# # :::::::::::::::::: SVHN, $atk ::::::::::::::::::

# ### resnet18
# CUDA_VISIBLE_DEVICES=$cuda python3 _code/ddb.py --dataset svhn --model_name resnet18 --load_model_pth _code/checkpoint/svhn_resnet18_natural.pt --atk $atk --batch_size 32 --split_use train --dry_run 0
# CUDA_VISIBLE_DEVICES=$cuda python3 _code/FAS_cumm.py --dataset svhn --model_name resnet18 --load_model_pth _code/checkpoint/svhn_resnet18_natural.pt --atk $atk --batch_size 4096 --k 5 --split_use train --use_hf $use_hf --dry_run 0

# ### googlenet
# CUDA_VISIBLE_DEVICES=$cuda python3 _code/ddb.py --dataset svhn --model_name googlenet --load_model_pth _code/checkpoint/svhn_googlenet_natural.pt --atk $atk --batch_size 32 --split_use train  --dry_run 0
# CUDA_VISIBLE_DEVICES=$cuda python3 _code/FAS_cumm.py --dataset svhn --model_name googlenet --load_model_pth _code/checkpoint/svhn_googlenet_natural.pt --atk $atk --batch_size 4096 --k 5 --split_use train --use_hf $use_hf --dry_run 0

# ### vgg11
# CUDA_VISIBLE_DEVICES=$cuda python3 _code/ddb.py --dataset svhn --model_name vgg11 --load_model_pth _code/checkpoint/svhn_vgg11_natural.pt --atk $atk --batch_size 32 --split_use train  --dry_run 0
# CUDA_VISIBLE_DEVICES=$cuda python3 _code/FAS_cumm.py --dataset svhn --model_name vgg11 --load_model_pth _code/checkpoint/svhn_vgg11_natural.pt --atk $atk --batch_size 4096 --k 5 --split_use train --use_hf $use_hf  --dry_run 0


# :::::::::::::::::: TinyImageNet $atk (wrong below) ::::::::::::::::::

# CUDA_VISIBLE_DEVICES=$cuda python3 _code/ddb.py --dataset tinyimagenet --model_name resnet18 --load_model_pth _code/checkpoint/tiny-imagenet-200/std_tinyimagenet_resnet_18_baseline/checkpoint.pth.tar --atk $atk --batch_size 32 --split_use train --dry_run 0
# CUDA_VISIBLE_DEVICES=$cuda python3 _code/FAS_cumm.py --dataset tinyimagenet --model_name resnet18 --load_model_pth _code/checkpoint/tiny-imagenet-200/std_tinyimagenet_resnet_18_baseline/checkpoint.pth.tar --atk $atk --batch_size 1024 --k 5 --split_use train --use_hf $use_hf  --dry_run 0


## resnet18
# CUDA_VISIBLE_DEVICES=$cuda python3 _code/ddb.py --dataset tinyimagenet --model_name resnet18 --load_model_pth _code/checkpoint/tiny-imagenet-200/std_tinyimagenet_resnet_18_baseline/checkpoint.pth.tar --atk $atk --batch_size 64
# CUDA_VISIBLE_DEVICES=$cuda python3 _code/FAS_cumm.py --dataset tinyimagenet --model_name resnet18 --load_model_pth _code/checkpoint/tiny-imagenet-200/std_tinyimagenet_resnet_18_baseline/checkpoint.pth.tar --atk $atk --batch_size 1024 --k 5

# CUDA_VISIBLE_DEVICES=$cuda python3 _code/ddb_2.py --dataset cifar10 --model_name resnet18 --load_model_pth _code/checkpoint/svhn_resnet18_natural.pt --atk AutoAttack --batch_size 32


# python3 _code/trust_score_baseline.py --batch_size 32 --model_name robust_resnet18 --atk PGD
# python3 _code/trust_score_baseline.py --batch_size 32 --model_name robust_resnet18 --atk DeepFool
# python3 _code/trust_score_baseline.py --batch_size 32 --model_name resnet18 --atk PGD
# python3 _code/trust_score_baseline.py --batch_size 32 --model_name resnet18 --atk DeepFool