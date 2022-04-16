# :::::::::::::::::: SVHN DeepFool ::::::::::::::::::

## resnet18
CUDA_VISIBLE_DEVICES=0 python3 _code/ddb.py --dataset svhn --model_name resnet18 --load_model_pth _code/checkpoint/svhn_resnet18_natural.pt --atk DeepFool --batch_size 32 && CUDA_VISIBLE_DEVICES=0 python3 _code/FAS_cumm.py --dataset svhn --model_name resnet18 --load_model_pth _code/checkpoint/svhn_resnet18_natural.pt --atk DeepFool --batch_size 4096 --k 5

## googlenet
CUDA_VISIBLE_DEVICES=1 python3 _code/ddb.py --dataset svhn --model_name googlenet --load_model_pth _code/checkpoint/svhn_googlenet_natural.pt --atk DeepFool --batch_size 32 && CUDA_VISIBLE_DEVICES=1 python3 _code/FAS_cumm.py --dataset svhn --model_name googlenet --load_model_pth _code/checkpoint/svhn_googlenet_natural.pt --atk DeepFool --batch_size 1024 --k 5

## vgg11
CUDA_VISIBLE_DEVICES=2 python3 _code/ddb.py --dataset svhn --model_name vgg11 --load_model_pth _code/checkpoint/svhn_vgg11_natural.pt --atk DeepFool --batch_size 32 && CUDA_VISIBLE_DEVICES=2 python3 _code/FAS_cumm.py --dataset svhn --model_name vgg11 --load_model_pth _code/checkpoint/svhn_vgg11_natural.pt --atk DeepFool --batch_size 4096 --k 5
