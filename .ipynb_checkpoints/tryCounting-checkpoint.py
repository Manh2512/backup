import glob
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib import imag
from PIL import Image

from counting import Counter
import torch
import torch.nn as nn
from torchvision import transforms

#device = torch.device('cpu')
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
latent_dim = 256
batch_size = 1
dataset_size = 5000


# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.unsqueeze(x[0:1,:,:], 0))
])


path_model = 'Desktop/URECA/GAN-main/models/counter'
path_output = 'Desktop/URECA/GAN-main/outputs/YNet'

input_dir = 'Desktop/URECA/GAN-main/outputs/YNet'
train_dir = os.path.join(input_dir, 'train')
val_dir = os.path.join(input_dir, 'val')
test_dir = os.path.join(input_dir, 'test')


def load_image_to_tensor(directory, filename, transform=None):
    # Load image and convert it to a tensor
    img_path = os.path.join(directory, filename)
    img = Image.open(img_path)
    img_tensor = transform(img)  # Assuming you want to convert it to a tensor
    idx_str = filename.split('_')[1].split('.')[0]  # Extract index from filename
    image_index = int(idx_str) - 1
    
    return img_tensor, image_index

def image_loader(directory, transform):
    # Loop through the images in the directory
    for filename in os.listdir(directory):
        yield load_image_to_tensor(directory, filename, transform)

# Generating tensor of output
results = torch.zeros(dataset_size)

counter = Counter(latent_dim).to(device)

#load generator
checkpoint_G = torch.load(os.path.join(path_model,"checkpoint_count_final.pth"), map_location='cpu')
start_ep = checkpoint_G['epoch']
print("Resuming from the checkpoint: ep", start_ep+1)
counter.load_state_dict(checkpoint_G['model'], strict=False)

#yy=[653, 794]
#generate training result
for xx, idx in image_loader(train_dir, transform):
    xx = xx.to(device)
    count = counter(xx).detach()
    results[idx] = count

#generate validation result
for xx, idx in image_loader(val_dir, transform):
    xx = xx.to(device)
    count = counter(xx).detach()
    results[idx] = count

#generate testing result
for xx, idx in image_loader(test_dir, transform):
    xx = xx.to(device)
    count = counter(xx).detach()
    results[idx] = count

results = torch.ceil(results)
#save the result
array = results.numpy()
np.savetxt(os.path.join(path_output, f'output_counts_ep={start_ep+1}.csv'), array, delimiter=',', fmt='%d')
print("Saving outputs completed!")