from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from PyTorch_Src_train.cifar10_models.vgg import vgg11_bn
from PyTorch_Src_train.cifar10_models.resnet import resnet18
from PyTorch_Src_train.cifar10_models.googlenet import GoogLeNet, googlenet
import wandb
from _code.utils import setup_dataset_models_standard as tinyimagenet_load

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

class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)

def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        wandb.log({f'{args.dataset}/Training Loss': loss.item()})

def test(args,model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    wandb.log({f'{args.dataset}/Test Accuracy': 100. * correct / len(test_loader.dataset)})

def test_train_split(args,model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    wandb.log({f'{args.dataset}/Train Accuracy': 100. * correct / len(test_loader.dataset)})

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Training Example')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--model_name', default='resnet18', type=str)
    parser.add_argument('--dataset',help='cifar10 or tinyimagenet',default='svhn')
    parser.add_argument('--wandb',help='wandb',default=0, type=int)

    args = parser.parse_args()
    print(args)

    mode = 'online' if args.wandb else 'disabled'
    wandb.init(project='Training_Datasets', entity='vclab', name=f'train_{args.dataset}_{args.model_name}', mode=mode, config=args, tags=[args.dataset,args.model_name])

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    if args.dataset == 'svhn':
        norm_layer = Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        train_kwargs = {'batch_size': args.batch_size, 'shuffle': False}
        test_kwargs = {'batch_size': args.test_batch_size, 'shuffle': False}
        if use_cuda:
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
        dataset1 = datasets.SVHN('data', download=True, transform=transform)
        dataset2 = datasets.SVHN('data', transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    if args.dataset == 'tinyimagenet':
        norm_layer = NormalizeByChannelMeanStd(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
        args.num_classes = 200
        args.data = 'data/tiny-imagenet-200'
        train_loader, val_loader, test_loader = tinyimagenet_load(args, shuffle=True)
        # test_loader = train_loader

    if args.model_name == 'vgg11':
        my_model = vgg11_bn(num_classes=args.num_classes, pretrained=False) # automatically picks up weights from PyTorch_Src_train/cifar10_models/state_dicts
        model = nn.Sequential(
            norm_layer,
            my_model).to(device) 
    
    if args.model_name == 'resnet18':
        my_model = resnet18(num_classes=args.num_classes, pretrained=False) # automatically picks up weights from PyTorch_Src_train/cifar10_models/state_dicts
        model = nn.Sequential(
            norm_layer,
            my_model).to(device) 
            
    if args.model_name == 'googlenet':
        my_model = GoogLeNet(num_classes=args.num_classes) # automatically picks up weights from PyTorch_Src_train/cifar10_models/state_dicts
        model = nn.Sequential(
            norm_layer,
            my_model).to(device) 

    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):

        train(args, model, device, train_loader, optimizer, epoch)
        test_train_split(args, model, device, train_loader)
        test(args, model, device, test_loader)
        scheduler.step()

        if args.save_model:
            save_path = f"_code/checkpoint/{args.dataset}_{args.model_name}_natural.pt"
            print('Model Saved at:',save_path)
            torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    main()