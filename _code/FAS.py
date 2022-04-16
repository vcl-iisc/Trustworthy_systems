import os
import torch
import tqdm
import argparse
import numpy as np
import matplotlib.pyplot as plt
from models import Holdout, Target
import seaborn as sns
from utils import ResNet18
torch.manual_seed(0)
np.random.seed(0)
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import torch.utils.data as data_utils
import torch.optim as optim
torch.set_printoptions(profile="full")
import warnings
warnings.filterwarnings("ignore")

from src.utils import load_data, load_model, AverageMeter, get_correct
from src.frequencyHelper import generateDataWithDifferentFrequencies_3Channel as freq_3t
from src.frequencyHelper import generateDataWithDifferentFrequencies_GrayScale as freq_t
 

def freq_attr(loader, data_store, model, args, device):

    model.to(device)
    model.eval()
    
    pbar = tqdm.tqdm(loader, unit="batch", leave=True, total=len(loader))
    radii = list(range(2, args.r_range+1))
    # scores = {cls_: {str(r): [] for r in radii} for cls_ in range(args.num_classes)}
    start=True
    with torch.no_grad():
        for data, ddb in pbar:

            data = data.to(device)
            output_og = model(data)
            # print(output_og)
            conf_og, pred_og = torch.max(output_og, 1)
            start_in = True
            ## For each radius
            for r in radii:

                ## Check How the correct logit conf. has changed on some low-pass data
                img_l = freq_3t(data, r=r, device=device)
                # data_l = img_l.to(device, dtype=torch.float)
                output = model(img_l)
                # print(output)
                conf = torch.index_select(output, dim=1, index = pred_og)
                # conf = output[:, pred_og]
                # print(torch.diag(conf))
                
                occ_score = (conf_og - torch.diag(conf))
                
                if start_in==True:
                    scores = occ_score.unsqueeze(0).T
                    start_in = False
                else:
                    scores = torch.cat((scores, occ_score.unsqueeze(0).T), dim=1)

            FAS_score = torch.argmax(scores, axis=1)
            FAS_score_k = torch.topk(scores, args.k, dim=1)[1]
            FAS_score_k = FAS_score_k.float().mean(dim=1)
            if start==True:
                FAS_scores=FAS_score
                FAS_scores_k = FAS_score_k
                start=False
            else:
                FAS_scores = torch.cat((FAS_scores, FAS_score))
                FAS_scores_k = torch.cat((FAS_scores_k, FAS_score_k))
            # print(len(FAS_scores))
    data_store['FAS_score_cummul'] = FAS_scores.cpu().numpy()
    data_store['FAS_score_cummul_'+str(args.k)] = FAS_scores_k.cpu().numpy()
    # torch.save(data_store, args.save_pth)
    print(f'Completed Frequency: Saved to {args.save_pth}')

    return data_store

def main(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    if args.dataset == 'tinyimagenet':
        args.num_classes = 200
        model = ResNet18(num_classes=args.num_classes).to(device)
        checkpoint = torch.load(args.load_model_pth)
        best_sa = checkpoint['best_sa']
        model.load_state_dict(checkpoint['state_dict'])

    if args.dataset == 'cifar10':   
        args.num_classes = 10
        model = Target().to(device)
        model.load_state_dict(torch.load(args.load_model_pth))
    
    model = nn.DataParallel(model)

    test_loader, data = load_ddb_data_fn(args)
    print(f'Data loaded from {args.load_ddb_data} and Model Loaded from {args.load_model_pth}')

    ####################################################
    if args.dataset == 'tinyimagenet':
        data['ddbs'] = torch.stack(data['ddbs'])#[:1000]
        data['images'] = torch.stack(data['images'])#[:1000,:,:,:]
        data['x'] = list(range(len(data['ddbs'])))

    data = freq_attr(test_loader, data, model, args, device)
    
    args.batch_size = 256
    test_loader, _ = load_ddb_data_fn(args)
    data = estimate_local_lip_v2( data, model, test_loader, perturb_steps=10, step_size=0.003, epsilon=0.01, device=device)

    print('-'*50)

def load_ddb_data_fn(args):
    data = torch.load(args.load_ddb_data)
    # print(data['images'].shape)
    
    if args.dataset == 'cifar10':
        tensor_dset = torch.utils.data.TensorDataset(torch.stack(data['images']), torch.tensor(data['ddbs']))

    if args.dataset == 'tinyimagenet':
        tensor_dset = torch.utils.data.TensorDataset(torch.stack(data['images']), torch.tensor(data['ddbs']))

    tensor_loader = torch.utils.data.DataLoader(tensor_dset, shuffle=False, batch_size=args.batch_size, num_workers=16, drop_last=False)
    return tensor_loader, data

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
    del data['images']
    torch.save(data, args.save_pth)
    print(f'Completed Lipschitz: Saved to {args.save_pth}')

    return data 
    
if __name__ == '__main__':

    ## Add Arguments
    parser = argparse.ArgumentParser(description='Perform Robustness Experiment')

    parser.add_argument('--dataset',help='cifar10, tinyimagenet',default='cifar10')
    parser.add_argument('--batch_size',help='Batch Size',default=4,type=int)
    parser.add_argument('--r_range', help='max radius range', default=31, type = int)
    parser.add_argument('--gpu', help='gpu-id', default='0', type=str)
    parser.add_argument('--load_ddb_data', default='/data/rohit_lal/cvprw-robustness/savedir/ddb_fas_final_CIFAR.pth', type=str)
    parser.add_argument('--load_model_pth', default="_code/checkpoint/target.pth", type=str)
    parser.add_argument('--save_pth', default='savedir/CIFAR_ddb_fas.pth', type=str)
    parser.add_argument('--k',help='top k',default=3,type=int)

    args = parser.parse_args()
    print(args)
    main(args)
        