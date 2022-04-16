## svhn training

# CUDA_VISIBLE_DEVICES=2 python3 pretraining.py --dataset svhn --model_name resnet18 --batch_size 2048 --test-batch-size 2048 --epochs 30 --wandb 1
# CUDA_VISIBLE_DEVICES=2 python3 pretraining.py --dataset svhn --model_name googlenet --batch_size 256 --test-batch-size 1024 --epochs 30 --wandb 1
# CUDA_VISIBLE_DEVICES=2 python3 pretraining.py --dataset svhn --model_name vgg11 --batch_size 2048 --test-batch-size 1024 --epochs 30 --wandb 1

## tinyimagenet training

CUDA_VISIBLE_DEVICES=2 python3 pretraining.py --dataset tinyimagenet --model_name googlenet --batch_size 64 --test-batch-size 1024 --epochs 200 --lr 0.1 --wandb 1
CUDA_VISIBLE_DEVICES=0 python3 pretraining.py --dataset tinyimagenet --model_name resnet18 --batch_size 64 --test-batch-size 1024 --epochs 200 --lr 0.1 --wandb 1 
CUDA_VISIBLE_DEVICES=1 python3 pretraining.py --dataset tinyimagenet --model_name vgg11 --batch_size 64 --test-batch-size 1024 --epochs 200 --lr 0.1 --wandb 1

