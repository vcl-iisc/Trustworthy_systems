from ast import arg
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from attack import PGD, PGD_steps, DeepFool
import sys
sys.path.append('.')
from PyTorch_Src_train.cifar10_models.vgg import vgg11_bn
from PyTorch_Src_train.cifar10_models.resnet import resnet18
from PyTorch_Src_train.cifar10_models.googlenet import googlenet, GoogLeNet
# from advertorch.attacks import CarliniWagnerL2Attack
from custom_model.resnet import ResNet18_cifar, PreActResNet
import numpy as np
from models import Holdout, Target
from utils import imshow
import time,os,sys
from tqdm import tqdm
import argparse
from utils import setup_dataset_models_standard, ResNet18
import sys
sys.path.append('.')
from custom_model.wideresnet import WideResNet
import pandas as pd

class Normalize(nn.Module):
	def __init__(self, mean, std) :
		super(Normalize, self).__init__()
		self.register_buffer('mean', torch.Tensor(mean))
		self.register_buffer('std', torch.Tensor(std))
		
	def forward(self, input):
		# Broadcasting
		mean = self.mean.reshape(1, 3, 1, 1)
		std = self.std.reshape(1, 3, 1, 1)
		return (input - mean) / std

if __name__ == '__main__':
	start = time.time()
	parser = argparse.ArgumentParser(description='Perform Robustness Experiment')
	parser.add_argument('--data', type=str, default='data', help='location of the data corpus', required=False)
	parser.add_argument('--dataset',default='cifar10')
	parser.add_argument('--batch_size',help='Batch Size',default=32,type=int)
	parser.add_argument('--load_model_pth', default="_code/checkpoint/target.pth", type=str)
	parser.add_argument('--save_pth', default='csv_data', type=str)
	parser.add_argument('--model_name', default='robust_resnet18', type=str)
	parser.add_argument('--split_use', default='test', type=str)
	parser.add_argument('--atk', default='PGD', type=str)


	args = parser.parse_args()

	print(args)
	os.makedirs(os.path.join(args.save_pth, 'ddb_data',args.dataset), exist_ok=True)
	device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

	if args.dataset == 'cifar10':
		args.num_classes = 10
		if args.split_use == 'train':
			cifar10_train = dsets.CIFAR10(root=args.data, train=True, download=True, transform=transforms.ToTensor())
		if args.split_use == 'test':
			cifar10_train  = dsets.CIFAR10(root=args.data, train=False, download=True, transform=transforms.ToTensor())

		train_loader = torch.utils.data.DataLoader(cifar10_train, batch_size=args.batch_size, num_workers = 8, shuffle=False)
		# test_loader = torch.utils.data.DataLoader(cifar10_test, batch_size=args.batch_size, num_workers= 8, shuffle=False)

		# Adding a normalization layer for Resnet18.
		# We can't use torch.transforms because it supports only non-batch images.
		if args.model_name == 'Target':
			norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
			Target = Target().to(device)
			Target.load_state_dict(torch.load(args.load_model_pth))

			model = nn.Sequential(
				norm_layer,
				Target,
			).to(device)
		
		if args.model_name == 'vgg11':
			norm_layer = Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
			my_model = vgg11_bn(pretrained=True) # automatically picks up weights from PyTorch_Src_train/cifar10_models/state_dicts
			model = nn.Sequential(
				norm_layer,
				my_model).to(device) 
		
		if args.model_name == 'resnet18':
			norm_layer = Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
			my_model = resnet18(pretrained=True) # automatically picks up weights from PyTorch_Src_train/cifar10_models/state_dicts
			model = nn.Sequential(
				norm_layer,
				my_model).to(device) 
				
		if args.model_name == 'googlenet':
			norm_layer = Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
			my_model = googlenet(pretrained=True) # automatically picks up weights from PyTorch_Src_train/cifar10_models/state_dicts
			model = nn.Sequential(
				norm_layer,
				my_model).to(device) 

		if args.model_name == 'robust_resnet18':
			model = ResNet18_cifar(num_classes=10).to(device) # load robust trained weights
			args.load_model_pth ='_code/checkpoint/cifar10_resnet18_robust.pt'
			model.load_state_dict(torch.load(args.load_model_pth)['net'])

		if args.model_name == 'robust_wideresnet':
			model = WideResNet(num_classes=10)
			args.load_model_pth = '_code/checkpoint/cifar10/WRN_CIFAR10_adv.t7'
			model.load_state_dict(torch.load(args.load_model_pth)['net'])
			model.to(device)

	print(f'==> Model: {args.model_name} \t Dataset: {args.dataset} \t Attack {args.atk}')

	if args.atk == 'PGD':
		atk = PGD_steps(model, eps=8/255, alpha=2/225, random_start=True)
	if args.atk == 'DeepFool':
		atk = DeepFool(model, steps=20, overshoot=0.02)

	model = model.eval()
	total_flags = 0
	correct_flags = 0
	for sam, (images, labels) in enumerate(tqdm(train_loader)):
		images = images.to(device)
		labels = labels.to(device)

		# adv_images, steps_to_attack = atk(images, labels)
		# outputs = model(adv_images)
		outputs = model(images)
		pred_probs = F.softmax(outputs, dim=1)
		# print(pred_probs)
		cls_pred = torch.argmax(pred_probs, dim=1)
		for i in range(images.shape[0]):
			if pred_probs[i][cls_pred[i]]<0.5:
				total_flags += 1
				if cls_pred[i]==labels[i]:
					correct_flags += 1

		# print(cls_pred, labels)
		
		# exit(0)
	print(total_flags)
	print(correct_flags)