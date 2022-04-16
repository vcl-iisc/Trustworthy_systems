import seaborn as sns
import torch,os
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
from collections import OrderedDict
sns.set_style('darkgrid')
from IPython.display import display
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

dataset = 'cifar10' # [cifar10, svhn, tinyimagenet]
model = 'robust_resnet18' # [resnet18, robust_resnet18, mobilenet_ddb, mobilenet_trust, robust_wideresnet, googlenet, vgg11]
split = 'test' # ['test', 'train']
atk = 'PGD' # [DeepFool, PGD]
hf = 0 # [0,1] 0 is old version and 1 is new version

print(f'Dataset: {dataset} \t Model: {model} \t Attack: {atk} \t Split: {split} \t HF: {hf}')
data = pd.read_csv(f'csv_data_adv/FAS_data/{dataset}/{model}_{split}_{atk}_{hf}.csv', index_col=False)
df = pd.DataFrame(data)
# len(df['Flipping_Freq'][df['Flipping_Freq'] == -1])


def get_trust_score(ddbs: list,Flipping_Freq: list) -> list:
    
    def normalize_list(x:list) -> list:
        x = np.array(x)
        return (x-np.min(x))/(np.max(x)-np.min(x))
    
    norm_Flipping_Freq = normalize_list(Flipping_Freq)
    norm_Flipping_Freq = 1-norm_Flipping_Freq
    norm_ddbs = normalize_list(ddbs)
    
    # TScore = lambda x,y: (y+x)/2
    TScore = lambda x,y: (2*x*y)/(x+y+1e-10)
    # TScore = lambda x,y: (x+y)/2
    # TScore = lambda x,y: 1/(x+1e-8) + y
 

    T_score = TScore(norm_ddbs ,norm_Flipping_Freq)

    return T_score, norm_ddbs, norm_Flipping_Freq

T_score, norm_ddbs, norm_Flipping_Freq = get_trust_score(df['ddbs'].tolist() , df['Flipping_Freq'].tolist())
df['T_score'] = T_score
df['norm_ddbs'] = norm_ddbs
df['norm_Flipping_Freq'] = norm_Flipping_Freq

# df['T_score'] = df['T_score'].fillna(1)


def K_means_flagging(df, values,r):
    kmeans = KMeans(n_clusters=2, random_state=r)
    df[f'flag_{values}'] = kmeans.fit_predict(np.array(df[values]).reshape(-1,1))
    c1,c2 = kmeans.cluster_centers_

    if c1< c2: df[f'flag_{values}'] = 1-df[f'flag_{values}']  ## we should flag those which are closer to smaller centroids

    filter_flag_ddbs = df[(df[f'flag_{values}']) == 1]
    total_flags = len(filter_flag_ddbs)
    correct_flags = len(filter_flag_ddbs[(filter_flag_ddbs['Model_preds'] != filter_flag_ddbs['gt'])])

    print(f'{values} \t\t correct_flags: {correct_flags}, total_flags: {total_flags}, %correct: {correct_flags*100/ total_flags:.2f}')
    return df

def GMM(df, values,r):
    gmm = GaussianMixture(n_components=2, random_state = r)
    df[f'flag_{values}'] = gmm.fit_predict(np.array(df[values]).reshape(-1,1))
    c1,c2 = gmm.means_

    if c1< c2: df[f'flag_{values}'] = 1-df[f'flag_{values}']  ## we should flag those which are closer to smaller centroids

    filter_flag_ddbs = df[(df[f'flag_{values}']) == 1]
    total_flags = len(filter_flag_ddbs)
    correct_flags = len(filter_flag_ddbs[(filter_flag_ddbs['Model_preds'] != filter_flag_ddbs['gt'])])

    print(f'{values} \t\t correct_flags: {correct_flags}, total_flags: {total_flags}, %correct: {correct_flags*100/ total_flags:.2f}')
    return df

# print(f'Dataset: {dataset} \t Attack: {atk} \t Model: {model} \t split: {split}')
# for r in [1,6,70,24,56,12,45,100]:
    # print('############',r,'############')
r=10
df = GMM(df,'norm_ddbs',r)
df = GMM(df,'norm_Flipping_Freq',r)
df = GMM(df,'T_score',r)
