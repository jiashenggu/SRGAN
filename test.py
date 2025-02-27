import argparse
from doctest import OPTIONFLAGS_BY_NAME
import os
from math import log10
import random
import time

import pandas as pd
import numpy as np
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform, PairedTrainDatasetFromFolder, PairedValDatasetFromFolder
from loss import GeneratorLoss
from model import Generator, Discriminator

from sklearn.model_selection import KFold
torch.set_num_threads(4)
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=88, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=2, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')
parser.add_argument('--netG_name', default=None, type=str, help='generator model name')
parser.add_argument('--netD_name', default=None, type=str, help='discriminator model name')
parser.add_argument('--exp_name', default=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) , type=str, help='experiment name')


def reset_weights(m):
    '''
    Try resetting model weights to avoid
    weight leakage.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}') 
            layer.reset_parameters()


if __name__ == '__main__':
    opt = parser.parse_args()
    
    CROP_SIZE = opt.crop_size
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs

    exp_name = opt.exp_name

    val_set = PairedValDatasetFromFolder('data/DIV2K_valid_HR', 'data/DIV2K_valid_LR_bicubic/X2', upscale_factor=UPSCALE_FACTOR)

    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

    netG = Generator(UPSCALE_FACTOR)
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    
    netD = Discriminator()
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

    generator_criterion = GeneratorLoss()
    
    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()
    
    if opt.netG_name and opt.netD_name:
        netG.load_state_dict(torch.load(opt.netG_name))
        netD.load_state_dict(torch.load(opt.netD_name))
    else:
        exit()
    
    results = {'psnr': [], 'ssim': []}

    netG.eval()
    out_path = 'testing_results/SRF_' + str(UPSCALE_FACTOR) +'_' + exp_name + '/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    with torch.no_grad():
        val_bar = tqdm(val_loader)
        valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
        val_images = []
        for val_lr, val_hr_restore, val_hr in val_bar:
            batch_size = val_lr.size(0)
            valing_results['batch_sizes'] += batch_size
            lr = val_lr
            hr = val_hr
            if torch.cuda.is_available():
                lr = lr.cuda()
                hr = hr.cuda()
            sr = netG(lr)
    
            batch_mse = ((sr - hr) ** 2).data.mean()
            valing_results['mse'] += batch_mse * batch_size
            batch_ssim = pytorch_ssim.ssim(sr, hr).item()
            valing_results['ssims'] += batch_ssim * batch_size
            valing_results['psnr'] = 10 * log10((hr.max()**2) / (valing_results['mse'] / valing_results['batch_sizes']))
            valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
            val_bar.set_description(
                desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                    valing_results['psnr'], valing_results['ssim']))
    
            val_images.extend(
                [display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
                display_transform()(sr.data.cpu().squeeze(0))])

        val_images = torch.stack(val_images)
        val_images = torch.chunk(val_images, val_images.size(0) // 3)
        val_save_bar = tqdm(val_images, desc='[saving training results]')
        index = 1
        for image in val_save_bar:
            image = utils.make_grid(image, nrow=3, padding=5)
            utils.save_image(image, out_path + 'index_%d.png' % (index), padding=5)
            index += 1



    # save psnr\ssim
    results['psnr'].append(valing_results['psnr'])
    results['ssim'].append(valing_results['ssim'])


    statistic_path = 'statistics/'
    data_frame = pd.DataFrame(
        data={'PSNR': results['psnr'], 'SSIM': results['ssim']})
    data_frame.to_csv(statistic_path + 'srf_' + str(UPSCALE_FACTOR) + '_' + exp_name + '_' + 'test_results.csv')
