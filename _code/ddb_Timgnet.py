from ast import arg
import torch
import torchvision
import torch.nn as nn
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
	parser.add_argument('--dataset',help='cifar10 or tinyimagenet',default='cifar10')
	parser.add_argument('--batch_size',help='Batch Size',default=1,type=int)
	parser.add_argument('--load_model_pth', default="_code/checkpoint/target.pth", type=str)
	parser.add_argument('--save_pth', default='ddb_data/', type=str)
	parser.add_argument('--model_name', default='Target', type=str)
	parser.add_argument('--atk', default='DeepFool', type=str)
	parser.add_argument('--partdata', type=int)


	args = parser.parse_args()
	
	os.makedirs(args.save_pth + '/' + args.dataset, exist_ok=True)
	
	args.save_path = os.path.join(args.save_pth, (args.dataset + '/' +f'{args.model_name}_{args.atk}_{args.partdata}.pth'))
	device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

	if args.dataset == 'cifar10':
		args.num_classes = 10
		cifar10_train = dsets.CIFAR10(root=args.data, train=True, download=True, transform=transforms.ToTensor())
		cifar10_test  = dsets.CIFAR10(root=args.data, train=False, download=True, transform=transforms.ToTensor())

		train_loader = torch.utils.data.DataLoader(cifar10_train, batch_size=args.batch_size, num_workers = 8, shuffle=False)
		test_loader = torch.utils.data.DataLoader(cifar10_test, batch_size=args.batch_size, num_workers= 8, shuffle=False)


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
		
		if args.model_name == 'WideResNet':
			# norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
			model = WideResNet(num_classes=10)
			model.load_state_dict(torch.load(args.load_model_pth)['net'])
			# model = nn.Sequential(norm_layer,model)
			model.to(device)
			
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
	
	if args.dataset == 'tinyimagenet':
		args.num_classes = 200
		train_loader, val_loader, test_loader = setup_dataset_models_standard(args)
		print("Data Length: ", len(train_loader.dataset))
		if args.model_name == 'resnet18':
			model = ResNet18(num_classes=200).to(device)
			checkpoint = torch.load(args.load_model_pth)
			best_sa = checkpoint['best_sa']
			model.load_state_dict(checkpoint['state_dict'])

	if args.dataset == 'svhn':
		norm_layer = Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
		train_kwargs = {'batch_size': args.batch_size, 'shuffle': False}
		test_kwargs = {'shuffle': False}
		cuda_kwargs = {'num_workers': 8,
						'pin_memory': True,
						'shuffle': False}
						
		train_kwargs.update(cuda_kwargs)
		test_kwargs.update(cuda_kwargs)

		transform=transforms.Compose([
			transforms.Resize((32,32)),
			transforms.ToTensor(),
			])
            
		args.num_classes = 10
		dataset1 = torchvision.datasets.SVHN('data', download=True, transform=transform)
		dataset2 = torchvision.datasets.SVHN('data', transform=transform)
		train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
		if args.model_name == 'vgg11':
			my_model = vgg11_bn(num_classes=args.num_classes, pretrained=False) # automatically picks up weights from PyTorch_Src_train/cifar10_models/state_dicts
		
		if args.model_name == 'resnet18':
			my_model = resnet18(num_classes=args.num_classes, pretrained=False) # automatically picks up weights from PyTorch_Src_train/cifar10_models/state_dicts
				
		if args.model_name == 'googlenet':
			my_model = GoogLeNet(num_classes=args.num_classes) # automatically picks up weights from PyTorch_Src_train/cifar10_models/state_dicts
		
		
		model = nn.Sequential(
			norm_layer,
			my_model).to(device) 

		model.load_state_dict(torch.load(args.load_model_pth))
	
	print(f'==> Model: {args.model_name} \t Dataset: {args.dataset} \t Attack {args.atk}')
	model = model.eval()

	# %% [markdown]
	# ## 3. Adversarial Attack

	# %%
	correct = 0
	total = 0
	CIFAR_DDB = {}
	idx_list = []
	ddb_list = []
	label_list = []
	steps_to_attack_list = []
	pre_list = []
	attacked_samp = 0

	print('Time: ', time.time()-start)
	if args.atk == 'PGD':
		atk = PGD_steps(model, eps=8/255, alpha=2/225, random_start=True)
	if args.atk == 'DeepFool':
		atk = DeepFool(model, steps=20, overshoot=0.02)


	count = 0 
	start = time.time()
	start_test = True
	for sam, (images, labels) in enumerate(tqdm(train_loader)):
		images = images.to(device)
		labels = labels.to(device)

		start = time.time()
		adv_images, steps_to_attack = atk(images, labels)
		outputs = model(adv_images)
		_, pred = torch.max(outputs.data, 1)
	
		total += images.shape[0]
		# if steps_to_attack[0] < 20:
		#     attacked_samp += 1
		correct += (pred == labels).sum()

		ddb = torch.linalg.norm((images - adv_images), dim = (1,2,3))
		if start_test == True:
			ddb_list = ddb.cpu()
			label_list = labels.cpu()
			steps_to_attack_list = steps_to_attack
			pred_list = pred.cpu()
			start_test = False
		else:
			ddb_list = torch.cat((ddb_list, ddb.cpu()),0)
			label_list = torch.cat((label_list, labels.cpu()),0)
			steps_to_attack_list = steps_to_attack_list + steps_to_attack
			pred_list = torch.cat((pred_list, pred.cpu()),0)
		# print(steps_to_attack)
	print('Total elapsed time (sec): %.2f' % (time.time() - start))
	print('Robust accuracy: %.2f %%' % (100 * float(correct) / total))
	# print('Total attacked samples: %.2f' % (attacked_samp) )
	# print('Percetage attacked samples: %.2f' % (attacked_samp*100/total) )

	CIFAR_DDB['ddbs'], CIFAR_DDB['gt'] = ddb_list.tolist(), label_list.tolist()
	CIFAR_DDB['steps_to_atk'], CIFAR_DDB['predictions'] = steps_to_attack_list, pred_list.tolist()
	torch.save(CIFAR_DDB, args.save_path)
	print('Completed Saving DDB')