import glob
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib import imag

from yNet_with_FFT import device, normalize, apply_threshold, FYNet
import torch
import torch.nn as nn
from torchvision import transforms

from PIL import Image


latent_dim = 256
batch_size = 1
transform = transforms.Compose([
    transforms.ToTensor(),
])

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir_1, input_dir_2, transform=None):
        self.input_dir_1 = input_dir_1
        self.input_dir_2 = input_dir_2
        self.input_images_1 = sorted(os.listdir(input_dir_1))  # Sort to align input and target
        self.input_images_2 = sorted(os.listdir(input_dir_2))
        self.transform = transform

    def __len__(self):
        return len(self.input_images_1)

    def __getitem__(self, idx):
        # Load input and target images
        input_image_path_1 = os.path.join(self.input_dir_1, self.input_images_1[idx])
        input_image_path_2 = os.path.join(self.input_dir_2, self.input_images_2[idx])
        
        input_image_1 = Image.open(input_image_path_1)
        input_image_2 = Image.open(input_image_path_2)
        
        idx_str = self.input_images_1[idx].split('_')[1].split('.')[0]
        image_index = int(idx_str)
        
        # Apply transformations
        if self.transform:
            input_image_1 = self.transform(input_image_1)
            input_image_2 = self.transform(input_image_2)
            
        return input_image_1, input_image_2, image_index

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Lambda(lambda x: 1-x)
])

path_model = 'Desktop/URECA/GAN-main/models/FYNet'
path_output = 'Desktop/URECA/GAN-main/outputs/FYNet/test'

input_dir_1 = 'Desktop/URECA/dataset_AI_students/test/input1'
input_dir_2 = 'Desktop/URECA/dataset_AI_students/test/input2'

dataset = ImageDataset(input_dir_1=input_dir_1, input_dir_2=input_dir_2, transform=transform)
# Create DataLoader
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = FYNet(latent_dim, 1, 1).to(device)

#load generator
checkpoint = torch.load(os.path.join(path_model,"checkpoint_FYNet_final.pth"), map_location='cpu')
start_ep = checkpoint['epoch']
print("Resuming from the checkpoint: ep", start_ep+1)
model.load_state_dict(checkpoint['model'], strict=False)

#generate training result
for batch in data_loader:
    x1, x2, idx = batch
    x1 = x1.to(device)
    x2 = x2.to(device)
    
    im = normalize(model(x1, x2).detach())
    for i in range(x1.size(0)):
        out = im[i]
        out= out.expand(3, -1, -1)
        out= out.permute(1, 2, 0)
        out = out.cpu().numpy()
        #save images
        plt.imsave(os.path.join(path_output, f'output_{int(idx):04d}.jpg'), out, cmap='gray')
        
