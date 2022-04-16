from email.mime import image
import torch
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
import numpy as np
from scipy import signal
import torch_dct

import seaborn as sns
import matplotlib as mpl

COLOR = 'black'
mpl.rcParams['text.color'] = COLOR
mpl.rcParams['text.color'] = COLOR
mpl.rcParams['axes.labelcolor'] = COLOR
mpl.rcParams['xtick.color'] = COLOR
mpl.rcParams['ytick.color'] = COLOR

# plt.style.use('ggplot')

def fft(img):
    return np.fft.fft2(img)


def fftshift(img):
    return fft(img)


def ifft(img):
    return np.fft.ifft2(img)


def ifftshift(img):
    return ifft(img)



def DCT(image):
    return dct(dct(image, norm="ortho", axis=0), norm="ortho", axis=1)


def iDCT(image):
    return idct(idct(image, norm="ortho", axis=0), norm="ortho", axis=1)



def distance(i, j, imageSize, r):
    dis = np.sqrt((i - imageSize/2) ** 2 + (j - imageSize/2) ** 2)
    if dis <= r:
        return 1.0
    else:
        return 0

def mask_radial(img, r):
    rows, cols = img.shape
    mask = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            mask[i, j] = distance(i, j, imageSize=rows, r=r)
    return mask

def mask_occ(img, r, device):
    _, _, rows, cols = img.size()
    mask = torch.ones(img.size()).to(device)
    for i in range(rows):
        for j in range(cols):
            if i == r and j<=r:
                mask[:,:,i,j] = 0
            if j == r and i<=r:
                mask[:,:,i,j] = 0
    return mask 

def mask_occ_cummul(img, r, device):
    _, _, rows, cols = img.size()
    # mask = torch.ones(img.size()).to(device)
    img[:,:,:r,:r] = 1
    return img

def generateSmoothKernel(data, r):
    result = np.zeros_like(data)
    [k1, k2, m, n] = data.shape
    mask = np.zeros([3,3])
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                mask[i,j] = 1
            else:
                mask[i,j] = r
    mask = mask
    for i in range(m):
        for j in range(n):
            result[:,:, i,j] = signal.convolve2d(data[:,:, i,j], mask, boundary='symm', mode='same')
    return result


def generateDataWithDifferentFrequencies_GrayScale(Images, r):
    Images_freq_low = []
    mask = mask_occ(np.zeros([32, 32]), r) ## Since I rescaled MNSIT images to 32x32 to use CIFAR modified models directly
    for i in range(Images.shape[0]):
        fd = fftshift(Images[i, :].reshape([32, 32]))
        fd = fd * mask
        img_low = ifftshift(fd)
        Images_freq_low.append(np.real(img_low).reshape([32 * 32]))


    Images_freq_high = [] 
    for i in range(Images.shape[0]):
        fd = fftshift(Images[i, :].reshape([32, 32]))
        fd = fd * (1-mask)
        img_high = ifftshift(fd)
        Images_freq_high.append(np.real(img_high).reshape([32 * 32]))

    return np.array(Images_freq_low), np.array(Images_freq_high)

def generateDataWithDifferentFrequencies_3Channel(images, r, device):
    images_freq_low = []
    mask = mask_occ(torch.zeros(images.shape).to(device), r, device)
    # mask = mask_occ_cummul(torch.zeros(images.shape).to(device), r, device)
    fd = torch_dct.dct_2d(images)
    fd = fd * mask
    # fd = fd * (1-mask)
    img_r = torch_dct.idct_2d(fd)
    return img_r

def generateDataWithCummulativeFrequencies_3Channel(images, r, device):
    mask = mask_occ_cummul(torch.zeros(images.shape).to(device), r, device)
    fd = torch_dct.dct_2d(images)
    fd = fd * mask
    img_r = torch_dct.idct_2d(fd)
    return img_r

def generateDataWithCummulativeFrequencies_3Channel_high(images, r, device):
    mask = mask_occ_cummul(torch.zeros(images.shape).to(device), r, device)
    fd = torch_dct.dct_2d(images)
    fd = fd * (1-mask) ## 1- -> In order to get the high frequency mask
    img_r = torch_dct.idct_2d(fd)
    return img_r


def generatePSD(Image):
    ## 1, 32, 32, 3
    tmp = np.zeros([Image.shape[1], Image.shape[2], 3])

    for j in range(3):
        fourier_image = np.fft.fftn(Image[0, :, :, j])
        fourier_amplitudes = np.abs(fourier_image)**2