import glob
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib import imag

from yNet_with_FFT import device, apply_threshold, ImageDataset, FYNet
import torch
import torch.nn as nn
from torchvision import transforms


latent_dim = 512
batch_size = 1
transform = transforms.Compose([
    transforms.ToTensor(),
])

path_model = 'Desktop/URECA/GAN-main/models/neoFYNet'
path_output = 'Desktop/URECA/GAN-main/outputs/neoFYNet'

input_dir_1 = 'Desktop/URECA/dataset_AI_students_1/try/input1'
input_dir_2 = 'Desktop/URECA/dataset_AI_students_1/try/input2'
target_dir = 'Desktop/URECA/dataset_AI_students_1/try/target'
dataset = ImageDataset(input_dir_1=input_dir_1, input_dir_2=input_dir_2, target_dir=target_dir, transform=transform)
# Create DataLoader
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

generator = FYNet(latent_dim, 1, 1).to(device)

#load generator
checkpoint_G = torch.load(os.path.join(path_model,"checkpoint_neoFYNet_final.pth"), map_location='cpu')
start_ep = checkpoint_G['epoch']
print("Resuming from the checkpoint: ep", start_ep+1)
generator.load_state_dict(checkpoint_G['model'], strict=False)

#generate training result
for batch in data_loader:
    x1, x2, yy = batch
    x1 = x1.to(device)
    x2 = x2.to(device)
    yy = yy.to(device)

    im = generator(x1, x2).detach()
    im_th = apply_threshold(im, 0.4)
    for i in range(x1.size(0)):
        inp1 = x1[i]
        inp2 = x2[i]
        target = yy[i]
        out = im[i]
        out_th = im_th[i]
        inp1, inp2, target, out, out_th = inp1.expand(3, -1, -1), inp2.expand(3, -1, -1), target.expand(3, -1, -1), out.expand(3, -1, -1), out_th.expand(3, -1, -1)
        inp1, inp2, target, out, out_th = inp1.permute(1, 2, 0), inp2.permute(1, 2, 0), target.permute(1, 2, 0), out.permute(1, 2, 0), out_th.permute(1, 2, 0)
        inp1, inp2, target, out, out_th = inp1.cpu().numpy(), inp2.cpu().numpy(), target.cpu().numpy(), out.cpu().numpy(), out_th.cpu().numpy()
        #save images
        plt.imsave(os.path.join(path_output, f'input1_{i}.jpg'), inp1, cmap='gray')
        plt.imsave(os.path.join(path_output, f'input2_{i}.jpg'), inp2, cmap='gray')
        plt.imsave(os.path.join(path_output, f'target_{i}.jpg'), target, cmap='gray')
        plt.imsave(os.path.join(path_output, f'output_{i}.jpg'), out, cmap='gray')
        plt.imsave(os.path.join(path_output, f'output_th_{i}.jpg'), out_th, cmap='gray')
        