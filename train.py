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

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize, InterpolationMode

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

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

if __name__ == '__main__':
    opt = parser.parse_args()
    
    CROP_SIZE = opt.crop_size
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs
    kfold = KFold(n_splits=4, shuffle=True)

    train_set = PairedTrainDatasetFromFolder('data/DIV2K_train_HR', 'data/DIV2K_train_LR_bicubic/X2', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    val_set = PairedValDatasetFromFolder('data/DIV2K_train_HR', 'data/DIV2K_train_LR_bicubic/X2', upscale_factor=UPSCALE_FACTOR)
    for fold, (train_ids, test_ids) in enumerate(kfold.split(train_set)):
        print(f'FOLD {fold}')
        print('--------------------------------')
        exp_name = opt.exp_name + '_FOLD_' + str(fold)
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=16, shuffle=True)
        # val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)
        train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=16, sampler=train_subsampler)
        val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, sampler=test_subsampler)
    

        netG = Generator(UPSCALE_FACTOR)
        print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
        netG.apply(reset_weights)
        
        netD = Discriminator()
        print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))
        netD.apply(reset_weights)

        generator_criterion = GeneratorLoss()
        
        if torch.cuda.is_available():
            netG.cuda()
            netD.cuda()
            generator_criterion.cuda()
        
        optimizerG = optim.Adam(netG.parameters())
        optimizerD = optim.Adam(netD.parameters())
        if opt.netG_name and opt.netD_name:
            netG.load_state_dict(torch.load(opt.netG_name))
            netD.load_state_dict(torch.load(opt.netD_name))
        
        results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}
        if opt.netG_name and opt.netD_name:
            start_epoch = int(opt.netG_name.split('.')[0][-1]) + 1
        else:
            start_epoch = 0
        for epoch in range(start_epoch+1, NUM_EPOCHS + 1):
            train_bar = tqdm(train_loader)
            running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}
        
            netG.train()
            netD.train()
            for data, target in train_bar:
                g_update_first = True
                batch_size = data.size(0)
                running_results['batch_sizes'] += batch_size
        
                ############################
                # (1) Update D network: maximize D(x)-1-D(G(z))
                ###########################
                real_img = Variable(target)
                if torch.cuda.is_available():
                    real_img = real_img.cuda()
                z = Variable(data)
                if torch.cuda.is_available():
                    z = z.cuda()
                fake_img = netG(z)
        
                netD.zero_grad()
                real_out = netD(real_img).mean()
                fake_out = netD(fake_img).mean()
                d_loss = 1 - real_out + fake_out
                d_loss.backward(retain_graph=True)
                optimizerD.step()
        
                ############################
                # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
                ###########################
                netG.zero_grad()
                ## The two lines below are added to prevent runetime error in Google Colab ##
                fake_img = netG(z)
                fake_out = netD(fake_img).mean()
                ##
                g_loss = generator_criterion(fake_out, fake_img, real_img)
                g_loss.backward()
                
                fake_img = netG(z)
                fake_out = netD(fake_img).mean()
                
                
                optimizerG.step()

                # loss for current batch before optimization 
                running_results['g_loss'] += g_loss.item() * batch_size
                running_results['d_loss'] += d_loss.item() * batch_size
                running_results['d_score'] += real_out.item() * batch_size
                running_results['g_score'] += fake_out.item() * batch_size
        
                train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                    epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
                    running_results['g_loss'] / running_results['batch_sizes'],
                    running_results['d_score'] / running_results['batch_sizes'],
                    running_results['g_score'] / running_results['batch_sizes']))
        
            netG.eval()
            out_path = 'training_results/SRF_' + str(UPSCALE_FACTOR) +'_' + exp_name + '/'
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
                if epoch % 10 == 0 and epoch != 0:
                    val_images = torch.stack(val_images)
                    val_images = torch.chunk(val_images, val_images.size(0) // 15)
                    val_save_bar = tqdm(val_images, desc='[saving training results]')
                    index = 1
                    for image in val_save_bar:
                        image = utils.make_grid(image, nrow=3, padding=5)
                        utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
                        index += 1
        
            # save model parameters
            if epoch % 10 == 0 and epoch != 0:
                model_path = 'epochs' +'_' + exp_name
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                torch.save(netG.state_dict(), model_path + '/netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
                torch.save(netD.state_dict(), model_path + '/netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
            # save loss\scores\psnr\ssim
            results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
            results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
            results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
            results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
            results['psnr'].append(valing_results['psnr'])
            results['ssim'].append(valing_results['ssim'])
        
            if epoch % 10 == 0 and epoch != 0:
                statistic_path = 'statistics/'
                data_frame = pd.DataFrame(
                    data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                        'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
                    index=range(start_epoch+1, epoch + 1))
                data_frame.to_csv(statistic_path + 'srf_' + str(UPSCALE_FACTOR) + '_' + exp_name + '_' + 'train_results.csv', index_label='Epoch')
