# :::::::::::::::::: TinyImageNet DeepFool ::::::::::::::::::

## resnet18
CUDA_VISIBLE_DEVICES=2 python3 _code/ddb_Timgnet.py --dataset tinyimagenet --model_name resnet18 --load_model_pth _code/checkpoint/tiny-imagenet-200/std_tinyimagenet_resnet_18_baseline/checkpoint.pth.tar --atk DeepFool --batch_size 1 --partdata 0
# CUDA_VISIBLE_DEVICES=1 python3 _code/FAS_cumm.py --dataset tinyimagenet --model_name resnet18 --load_model_pth _code/checkpoint/tiny-imagenet-200/std_tinyimagenet_resnet_18_baseline/checkpoint.pth.tar --atk DeepFool --batch_size 4096 --k 5

