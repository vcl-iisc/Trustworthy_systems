import os
from re import L
import torch
import tqdm
import argparse
import numpy as np
import matplotlib.pyplot as plt
from models import Holdout, Target, MobileNetV2
import seaborn as sns
from utils import ResNet18
torch.manual_seed(0)
np.random.seed(0)
import torch.nn as nn
import torch.nn.functional as F
import tqdm, torchvision, torch
import torch.utils.data as data_utils
import torch.optim as optim
torch.set_printoptions(profile="full")
import warnings
warnings.filterwarnings("ignore")
import torchvision.transforms as transforms
import pandas as pd

from src.utils import load_data, load_model, AverageMeter, get_correct
from src.frequencyHelper import generateDataWithDifferentFrequencies_3Channel as freq_3t
from src.frequencyHelper import generateDataWithDifferentFrequencies_GrayScale as freq_t
from src.frequencyHelper import generateDataWithCummulativeFrequencies_3Channel as freq_cumm
from src.frequencyHelper import generateDataWithCummulativeFrequencies_3Channel_high as freq_cumm_high

import sys
sys.path.append('.')
from custom_model.resnet import ResNet18_cifar, PreActResNet
from PyTorch_Src_train.cifar10_models.vgg import vgg11_bn
from PyTorch_Src_train.cifar10_models.resnet import resnet18
from PyTorch_Src_train.cifar10_models.googlenet import googlenet, GoogLeNet
from utils import setup_dataset_models_standard, ResNet18
from custom_model.wideresnet import WideResNet
from torchattacks import AutoAttack,FGSM
import warnings
warnings.filterwarnings("ignore")

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
 

def freq_attr(loader, data_store, model, args, device):

	model.to(device)
	if args.auto_mod:
		print('Modifying inputs using auto_mod')
		mod_atk = FGSM(model, eps=8/255)

	model.eval()
	pbar = tqdm.tqdm(loader, unit="batch", leave=True, total=len(loader))
	radii = list(range(0, args.r_range+1))
	# scores = {cls_: {str(r): [] for r in radii} for cls_ in range(args.num_classes)}
	start=True
	model_preds=[]
	sam = 0
	# with torch.no_grad():
	for data, labels in pbar:
		flip_freq = np.zeros(data.shape[0], dtype=int)-1
		data = data.to(device)
		output_og = model(data)
		if args.auto_mod:
			data = mod_atk(data,labels).detach().clone().to(device)

		# print(output_og)
		conf_og, pred_og = torch.max(output_og, 1)

		start_in = True
		idxs=[]
		
		## For each radius
		for r in reversed(radii):
			## Check How the correct logit conf. has changed on some low-pass data
			if not args.use_hf:
				img_l = freq_cumm(data, r=r, device=device)
				output = model(img_l)

			else:
				img_h = freq_cumm_high(data, r=r, device=device)
				output = model(img_h)

			# data_l = img_l.to(device, dtype=torch.float)
			conf, pred = torch.max(output,1)
			# conf = torch.index_select(output, dim=1, index = pred_og)
			# conf = output[:, pred_og]
			# print(torch.diag(conf))
			for i in range(data.shape[0]):
				if i in idxs:
					continue
				else:
					if not args.use_hf:
						if pred[i]!=pred_og[i]:
							flip_freq[i] = r
							idxs.append(i)
					else:
						if pred[i]==pred_og[i]:
							flip_freq[i] = r
							idxs.append(r)
						
			# occ_score = (conf_og - torch.diag(conf))
			
			# if start_in==True:
			#     scores = occ_score.unsqueeze(0).T
			#     start_in = False
			# else:
			#     scores = torch.cat((scores, occ_score.unsqueeze(0).T), dim=1)
		if start == True:
			flip_frequencies = flip_freq.tolist()
			model_preds = pred_og.cpu().tolist()
			start = False
		else:
			flip_frequencies += flip_freq.tolist()
			model_preds += pred_og.cpu().tolist()
		
		if sam == 5 and args.dry_run:
			break
		sam += 1

			# FAS_score = torch.argmax(scores, axis=1)
			# FAS_score_k = torch.topk(scores, args.k, dim=1)[1]
			# FAS_score_k = FAS_score_k.float().mean(dim=1)
			# if start==True:
			#     FAS_scores=FAS_score
			#     FAS_scores_k = FAS_score_k
			#     start=False
			# else:
			#     FAS_scores = torch.cat((FAS_scores, FAS_score))
			#     FAS_scores_k = torch.cat((FAS_scores_k, FAS_score_k))
			# print(len(FAS_scores))
	# data_store['FAS_score_cummul'] = FAS_scores.cpu().numpy()
	# data_store['FAS_score_cummul_'+str(args.k)] = FAS_scores_k.cpu().numpy()
	data_store['Flipping_Freq'] = flip_frequencies
	data_store['Model_preds'] = model_preds
	data_store.to_csv(args.save_pth, index=False)
	print(f'Completed Frequency: Saved to {args.save_pth}')

	return data_store

def main(args):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	if args.dataset == 'tinyimagenet':
		args.num_classes = 200
		model = ResNet18(num_classes=args.num_classes).to(device)
		checkpoint = torch.load(args.load_model_pth)
		best_sa = checkpoint['best_sa']
		model.load_state_dict(checkpoint['state_dict'])

	if args.dataset == 'cifar10':   
		args.num_classes = 10
		
		if args.model_name == 'Target':
			norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
			Target = Target().to(device)
			Target.load_state_dict(torch.load(args.load_model_pth))

			model = nn.Sequential(
				norm_layer,
				Target,
			).to(device)
		print(args.model_name)
		if args.model_name == 'mobilenet_ddb':
			model = MobileNetV2(num_classes=10)
			model.load_state_dict(torch.load(args.load_model_pth, map_location='cpu')['net'])
			model.to(device)

		if args.model_name == 'mobilenet_trust':
			model = MobileNetV2(num_classes=10)
			model.load_state_dict(torch.load(args.load_model_pth, map_location='cpu')['net'])
			model.to(device)

		if args.model_name == 'mobilenet_random':
			model = MobileNetV2(num_classes=10)
			model.load_state_dict(torch.load(args.load_model_pth, map_location='cpu')['net'])
			model.to(device)
		if args.model_name == 'mobilenet_freq':
			model = MobileNetV2(num_classes=10)
			model.load_state_dict(torch.load(args.load_model_pth, map_location='cpu')['net'])
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

		if args.model_name == 'robust_wideresnet':
			# norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
			model = WideResNet(num_classes=10)
			args.load_model_pth = '_code/checkpoint/cifar10/WRN_CIFAR10_adv.t7'
			model.load_state_dict(torch.load(args.load_model_pth)['net'])
			# model = nn.Sequential(norm_layer,model)
			model.to(device)

		if args.model_name == 'robust_resnet18':
			model = ResNet18_cifar(num_classes=10).to(device) # load robust trained weights
			args.load_model_pth ='_code/checkpoint/cifar10_resnet18_robust.pt'
			model.load_state_dict(torch.load(args.load_model_pth)['net'])

	if args.dataset == 'svhn':   
		norm_layer = Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
		args.num_classes = 10
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

	if args.dataset == 'cifar100':
		args.num_classes = 100

		# train_loader = torch.utils.data.DataLoader(cifar100, batch_size=args.batch_size, num_workers = 8, shuffle=False)
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
			model = PreActResNet(num_classes=100).to(device) # load robust trained weights
			args.load_model_pth ='_code/checkpoint/cifar100/cifar100_linf_resnet18_ddpm.pt'
			model.load_state_dict(torch.load(args.load_model_pth))

			# model.load_state_dict(torch.load(args.load_model_pth)['net'])

		if args.model_name == 'robust_wideresnet':
			model = WideResNet(num_classes=10)
			args.load_model_pth = '_code/checkpoint/cifar10/WRN_CIFAR10_adv.t7'
			model.load_state_dict(torch.load(args.load_model_pth)['net'])
			model.to(device)
			
	model = nn.DataParallel(model)

	test_loader, data = load_ddb_data_fn(args)
	print(f'Data loaded from {args.load_ddb_data} and Model Loaded from {args.load_model_pth}')

	####################################################
	# if args.dataset == 'tinyimagenet':
	# 	data['ddbs'] = torch.stack(data['ddbs'])#[:1000]
	# 	# data['images'] = torch.stack(data['images'])#[:1000,:,:,:]
	# 	data['x'] = list(range(len(data['ddbs'])))
	print('Started FAS score...')
	data = freq_attr(test_loader, data, model, args, device)
	
	# args.batch_size = 128
	# test_loader, _ = load_ddb_data_fn(args)
	# print('Started Local Lip score...')

	# data = estimate_local_lip_v2( data, model, test_loader, perturb_steps=10, step_size=0.003, epsilon=0.01, device=device)

	print('-'*50)

def load_ddb_data_fn(args):
	data = pd.read_csv(args.load_ddb_data)
	# print(data['images'].shape)
	
	if args.dataset == 'cifar10':
		if args.split_use == 'train':
			tensor_dset = torchvision.datasets.CIFAR10(root='_code/data', train=True, download=True, transform=transforms.ToTensor())

		if args.split_use == 'test':
			tensor_dset = torchvision.datasets.CIFAR10(root='_code/data', train=False, download=True, transform=transforms.ToTensor())
		# train_loader = torch.utils.data.DataLoader(cifar10_train, batch_size=args.batch_size, num_workers = 8, shuffle=False)
		# test_loader = torch.utils.data.DataLoader(cifar10_test, batch_size=args.batch_size, num_workers= 8, shuffle=False)
		train_loader = torch.utils.data.DataLoader(tensor_dset, shuffle=False, batch_size=args.batch_size, num_workers=16, drop_last=False)

	if args.dataset == 'cifar100':
		args.num_classes = 100
		if args.split_use == 'train':
			cifar100 = torchvision.datasets.CIFAR100(root=args.data, train=True, download=True, transform=transforms.ToTensor())
		if args.split_use == 'test':
			cifar100  = torchvision.datasets.CIFAR100(root=args.data, train=False, download=True, transform=transforms.ToTensor())

		train_loader = torch.utils.data.DataLoader(cifar100, batch_size=args.batch_size, num_workers = 8, shuffle=False)

	if args.dataset == 'tinyimagenet':
		pass
		train_loader, val_loader, test_loader = setup_dataset_models_standard(args)
	#     tensor_dset = torch.utils.data.TensorDataset(torch.stack(data['images']), torch.tensor(data['ddbs']))
	if args.dataset == 'svhn':
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
		dataset1 = torchvision.datasets.SVHN('data', split= args.split_use,download=True, transform=transform)
		train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)

	return train_loader, data

def local_lip(model, x, xp, top_norm, reduction='mean'):
	model.eval()
	down = torch.flatten(x - xp, start_dim=1)
	if top_norm == "kl":
		criterion_kl = nn.KLDivLoss(reduction='none')
		top = criterion_kl(F.log_softmax(model(xp), dim=1),
						   F.softmax(model(x), dim=1))
		ret = torch.sum(top, dim=1) / torch.norm(down + 1e-6, dim=1)
	else:
		top = torch.flatten(model(x), start_dim=1) - torch.flatten(model(xp), start_dim=1)
		ret = torch.norm(top, dim=1, p=top_norm) / torch.norm(down + 1e-6, dim=1)

	if reduction == 'mean':
		return torch.mean(ret)
	elif reduction == 'sum':
		return torch.sum(ret)
	elif reduction == 'None':
		return ret
	else:
		raise ValueError(f"Not supported reduction: {reduction}")

def estimate_local_lip_v2(data, model,test_loader, top_norm=1,
		perturb_steps=10, step_size=0.003, epsilon=0.01,
		device="cuda"):
	model.eval()

	total_ep = 0.
	ret = []
	batch_loss = []
	for x, _ in tqdm.tqdm(test_loader):
		total_ep+=1
		# x = x[0]
		x = x.to(device)
		# generate adversarial example
		# if btm_norm in [1, 2, np.inf]:
		x_adv = x + 0.001 * torch.randn(x.shape).to(device)

		# Setup optimizers
		optimizer = optim.SGD([x_adv], lr=step_size)

		for i in range(perturb_steps):
			x_adv.requires_grad_(True)
			optimizer.zero_grad()
			with torch.enable_grad():
				loss = (-1) * local_lip(model, x, x_adv, top_norm)
			loss.backward()
			# renorming gradient
			eta = step_size * x_adv.grad.data.sign().detach()
			x_adv = x_adv.data.detach() + eta.detach()
			eta = torch.clamp(x_adv.data - x.data, -epsilon, epsilon)
			x_adv = x.data.detach() + eta.detach()
			x_adv = torch.clamp(x_adv, 0, 1.0)

		batch_loss += local_lip(model, x, x_adv, top_norm, reduction='None').detach().cpu().tolist()
		# if total_ep ==10:
		#     break
	data['lip'] = batch_loss
	# del data['images']
	torch.save(data, args.save_pth)
	print(f'Completed Lipschitz: Saved to {args.save_pth}')

	return data 
	
if __name__ == '__main__':

	## Add Arguments
	parser = argparse.ArgumentParser(description='Perform Robustness Experiment')

	parser.add_argument('--dataset',help='cifar10, tinyimagenet',default='cifar10')
	parser.add_argument('--data', type=str, default='data', help='location of the data corpus', required=False)
	parser.add_argument('--batch_size',help='Batch Size',default=4,type=int)
	parser.add_argument('--r_range', help='max radius range', default=31, type = int)
	parser.add_argument('--gpu', help='gpu-id', default='0', type=str)
	# parser.add_argument('--load_ddb_data', default='csv_data', type=str)
	parser.add_argument('--load_model_pth', default="_code/checkpoint/target.pth", type=str)
	parser.add_argument('--save_pth', default='csv_data', type=str)
	parser.add_argument('--k',help='top k',default=3,type=int)
	parser.add_argument('--model_name', default='Target', type=str)
	parser.add_argument('--atk', default='DeepFool', type=str)
	parser.add_argument('--split_use', default='test', type=str)
	parser.add_argument('--use_hf', default=0, type=int) # 1 --> Ruchit idea 0--> Original idea
	parser.add_argument('--dry_run', default=0, type=int)
	parser.add_argument('--auto_mod', default=0, type=int)
	parser.add_argument('--samples', default=100, type=int)

	args = parser.parse_args()
	# os.makedirs(args.save_pth + '/' + args.dataset, exist_ok=True)
	if args.dry_run:
		args.batch_size = 1
	os.makedirs(os.path.join(args.save_pth, 'FAS_data',args.dataset), exist_ok=True)

	args.load_ddb_data = os.path.join(args.save_pth, 'ddb_data', args.dataset, f'{args.model_name}_{args.split_use}_{args.atk}.csv')
	args.save_pth = os.path.join(args.save_pth, 'FAS_data', args.dataset, f'{args.model_name}_{args.split_use}_{args.atk}_{args.use_hf}.csv')

	print(args)
	print(f'==> Model: {args.model_name} \t Dataset: {args.dataset}')

	main(args)
		