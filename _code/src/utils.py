import os 
import numpy as np

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import SubsetRandomSampler
torch.backends.cudnn.deterministic = True
from collections import OrderedDict
import dill


from src.cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from src.cifar10_models.resnet import resnet18, resnet34, resnet50
from src.cifar10_models.densenet import densenet121
from src.cifar10_models.inception import inception_v3


CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD  = [0.2471, 0.2435, 0.2616]

MNIST_MEAN = [0.5,]
MNIST_STD = [0.5,]

cifar_idx2cls = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

idx2cls_mnist = {x:x for x in range(10)}
idx2cls_fmnist = {0: "T-shirt/Top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot"}


class WeightedDataset(torch.utils.data.Dataset):
    """Dataset wrapper to induce class-imbalance"""

    def __init__(self, dataset, split, mode, preprocess=None):
        

        self.dataset = dataset ## Recieve (Transformed) dataset
        self.mode = mode ## Corresponding Mode i.e. Balanced, Linear_Imbalance, Long_tailed
        self.split = split
        self.preprocess = preprocess
        self.selects = np.load('./PyTorch_CIFAR10/assets/'+split+'_'+mode+'_selects_cifar10.npy') ## Load selects array
        
        
    def __getitem__(self, i):
        x, y = self.dataset[i] ## Original Sample
        select = self.selects[i] ## Load pre-computed select or non-select

        if self.preprocess:
            x = self.preprocess(x)

        
        return (x,y,select)
        
    def __len__(self):
        return len(self.dataset)


def load_data(root='./input/cifar10/', batch_size=32, valid_size=0.2, mode='balanced', return_data=False):
    
    dataset = root.split('/')[-2]

    ## Normalization will go in model construction
    preprocess = transforms.Compose([transforms.Resize((32,32)),
                                        transforms.ToTensor()])
    
    ## Init the data
    if dataset == 'cifar10':
        train_data = torchvision.datasets.CIFAR10(root, train=True, transform=preprocess)
        test_data = torchvision.datasets.CIFAR10(root=root, train=False, transform=preprocess)
    else:
        print(f'{root} / {dataset} doesnt exist')

    w_train_data = WeightedDataset(train_data, mode=mode, split='train')
    w_test_data = WeightedDataset(test_data, mode=mode, split='test')


    ## Split Train and Valid Data
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size*num_train))
    train_idx,valid_idx = indices[split:],indices[:split]
    
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
        
    train_loader = torch.utils.data.DataLoader(w_train_data, batch_size=batch_size, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(w_train_data, batch_size=batch_size, sampler=valid_sampler)
    test_loader = torch.utils.data.DataLoader(w_test_data, batch_size=batch_size)

    if return_data:
        return train_loader ,valid_loader, test_loader, w_train_data, w_test_data

    return train_loader,valid_loader,test_loader


def batch_accuracy(output,target,topk=(1,)):
    '''
    Caluclate running accuracy of one batch
    '''
    maxk  = max(topk)
    batch_size = output.size(0)
    
    _, pred = output.topk(maxk,1,True,True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Normalize(nn.Module) :
    def __init__(self, mean, std) :
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))
        
    def forward(self, input):
        # Broadcasting
        if len(self.mean)>1:
            mean = self.mean.reshape(1, 3, 1, 1)
            std = self.std.reshape(1, 3, 1, 1)
        else:
            mean, std = self.mean, self.std
        return (input - mean) / std


def load_model(model_name, device, args):

    if args.mode == 'long_tailed':
        save_path = './src/'+args.dataset+'_lt_state_dict/'+args.model_name+'.pt'    
    else:
        # save_path = './src/'+args.dataset+'_models/'+'state_dicts/'+args.model_name+'.pt'    
        save_path = './src/'+args.dataset+'_models/'+'state_dicts/cifar_l2_1_0.pt'
        # args.model_name+f'_1.0_8.0_cifar10.pt'    
        x = torch.load(save_path, pickle_module=dill)
        print(x['model'].keys())
        exit(0)

    # MEAN, STD = CIFAR_MEAN, CIFAR_STD 
    MEAN, STD = [0]*3, [1]*3
    
    if model_name == 'resnet18':
        base_model = resnet18(pretrained=True, save_path=save_path)
    elif model_name == 'resnet34':
        base_model = resnet34(pretrained=True, save_path=save_path)
    elif model_name == 'vgg16_bn':
        base_model = vgg16_bn(pretrained=True, save_path=save_path)
    elif model_name == 'densenet121':
        base_model = densenet121(pretrained=True)
    elif model_name == 'inception_v3':
        base_model = inception_v3(pretrained=True)
    else:
        print('MODEL NOT FOUND')

    
    norm_layer = Normalize(mean=MEAN, std=STD)

    model = nn.Sequential(
        norm_layer,
        base_model
    ).to(device)
        
    return model


class AverageMeter(object):
    """Computes and stores the average and current value""" 
    def __init__(self):
        self.reset()
                   
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_correct(outputs, labels, get_pred=False):
    
    _, pred = torch.max(outputs, 1)
    correct = (pred == labels).float().sum(0).item()
    if get_pred:
        return pred, correct
    else:
        return correct