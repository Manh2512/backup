import glob
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib import imag

from train_GAN import device, apply_threshold, ImageDataset, Generator
from unet import UNet
import torch
import torch.nn as nn
from torchvision import transforms


latent_dim = 512
batch_size = 2
transform = transforms.Compose([
    transforms.ToTensor(),
])

path_model = 'Desktop/URECA/GAN-main/models/UNet'
path_output = 'Desktop/URECA/GAN-main/outputs/UNet'

input_dir = 'Desktop/URECA/dataset_AI_students_1/try/input'
target_dir = 'Desktop/URECA/dataset_AI_students_1/try/target'
dataset = ImageDataset(input_dir=input_dir, target_dir=target_dir, transform=transform)
# Create DataLoader
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

generator = UNet(1, 1).to(device)

#load generator
checkpoint_G = torch.load(os.path.join(path_model,"checkpoint_UNet_final.pth"), map_location='cpu')
start_ep = checkpoint_G['epoch']
print("Resuming from the checkpoint: ep", start_ep+1)
generator.load_state_dict(checkpoint_G['model'], strict=False)

#generate training result
for batch in data_loader:
    xx, yy = batch
    xx = xx.to(device)
    yy = yy.to(device)

    im = generator(xx).detach()
    im_th = apply_threshold(im, 0.2)
    for i in range(xx.size(0)):
        inp = xx[i]
        target = yy[i]
        out = im[i]
        out_th = im_th[i]
        inp, target, out, out_th = inp.expand(3, -1, -1), target.expand(3, -1, -1), out.expand(3, -1, -1), out_th.expand(3, -1, -1)
        inp, target, out, out_th = inp.permute(1, 2, 0), target.permute(1, 2, 0), out.permute(1, 2, 0), out_th.permute(1, 2, 0)
        inp, target, out, out_th = inp.cpu().numpy(), target.cpu().numpy(), out.cpu().numpy(), out_th.cpu().numpy()
        #save images
        plt.imsave(os.path.join(path_output, f'input_{i}.jpg'), inp, cmap='gray')
        plt.imsave(os.path.join(path_output, f'target_{i}.jpg'), target, cmap='gray')
        plt.imsave(os.path.join(path_output, f'output_{i}.jpg'), out, cmap='gray')
        plt.imsave(os.path.join(path_output, f'output_th_{i}.jpg'), out_th, cmap='gray')
        