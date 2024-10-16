import glob
import os

import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
import numpy as np
import scipy
import scipy.io
import matplotlib.pyplot as plt

from PIL import Image
import skimage
from torchvision import transforms
import torchvision

torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

latent_dim = 256
batch_size = 1

lr = 1e-4
beta1 = 0.6
beta2 = 0.99
num_epochs = 20
loss_hist = []
min_loss = 0
"""
def normalize(x):
    min_value = torch.min(x)
    max_value = torch.max(x)
    return (x-min_value)/(max_value-min_value)
"""
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir, transform=None):
        self.input_dir = input_dir
        self.input_images = sorted(os.listdir(input_dir))  # Sort to align input and target
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        # Load input and target images
        input_image_path = os.path.join(self.input_dir, self.input_images[idx])
        input_image = Image.open(input_image_path)
        
        idx_str = self.input_images[idx].split('_')[1].split('.')[0]
        image_index = int(idx_str)-1
        
        # Apply transformations
        if self.transform:
            input_image = self.transform(input_image)
            
        return input_image, image_index

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x[0:1,:,:])
])

class Counter(nn.Module):
    def __init__(self, latent_dim):
        super(Counter, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(32 * latent_dim//2 * latent_dim//2, 64)
        self.fc2 = nn.Linear(64, 1)
    def forward(self, x):
        # Forward pass through conv layers
        conv1 = self.pool(torch.relu(self.conv1(x)))
        conv2 = self.pool(torch.relu(self.conv2(x)))
        merge = torch.cat((conv1, conv2), dim=1)
        conv3 = self.conv3(merge)
        # Flatten the output from conv layers
        count = conv3.view(-1, 32 * latent_dim//2 * latent_dim//2)
        
        # Forward pass through fully connected layers
        count = torch.relu(self.fc1(count))
        count = self.fc2(count)
        return count

class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()
    def forward(self, x, y):
        loss = (x-y)**2
        return loss

def main():
    #===================== LOADING DATA ===========================
    path_model = 'Desktop/URECA/GAN-main/models/counter'
    path_output = 'Desktop/URECA/GAN-main/outputs/count'
    
    # Create dataset
    input_dir = 'Desktop/URECA/GAN-main/outputs/train'
    dataset = ImageDataset(input_dir=input_dir, transform=transform)

    # Create DataLoader
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    ground_truth = scipy.io.loadmat('Desktop/Manh/dataset_AI_students/particle_counts.mat')
    
    particle_counts = torch.tensor(ground_truth['particle_counts'], dtype=torch.int64)
    output = torch.zeros(len(particle_counts)).to(device)
    # Initialize counter
    counter = Counter(latent_dim).to(device)
    # Loss functions
    mse_loss = L2Loss().to(device)
    # Optimizers
    optimizer = optim.Adam(counter.parameters(), lr=lr)

    start_ep = 0
    min_loss = 1e10
    """
    #loading Model
    if os.path.isfile(os.path.join(path_model,"checkpoint_Unet.pth")):
        #checkpoint = torch.load(os.path.join(path_model,"checkpoint_Unet.pth"), map_location='cpu')
        checkpoint = torch.load(os.path.join(path_model,"checkpoint_Unet.pth"),map_location=torch.device('cpu'))
        start_ep = checkpoint['epoch']
        print("Resuming from the checkpoint: ep", start_ep+1)
        np.random.set_state(checkpoint['np_rand_state'])
        torch.set_rng_state(checkpoint['torch_rand_state'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])
    """
    # Training loop
    for epoch in range(num_epochs):
        for i, (xx, index_list) in enumerate(data_loader):
            #data and its label
            xx = xx.to(device)
            index_list = index_list.to(torch.int64)
            yy = particle_counts[index_list].to(device)

            # -----------------
            #  Training
            # -----------------
            optimizer.zero_grad()
            # Generate a batch of images
            count = torch.squeeze(counter(xx), 1)
            output[index_list] = count

            # Total loss
            loss =  mse_loss(count, yy)
    
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            # ---------------------
            #  Progress Monitoring
            # ---------------------
            if (epoch + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}]\
                            Batch {i+1}/{len(data_loader)} "
                    f"Training loss: {loss.item():4f}"
                )
        if (epoch + 1) % 1 == 0:
            #save losses
            loss_hist.append(loss.item())
            if loss.item() < min_loss:
                #save model
                min_loss = loss.item()
                torch.save({'epoch': epoch+start_ep,
                            'model': counter.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'np_rand_state': np.random.get_state(),
                            'torch_rand_state': torch.get_rng_state(),
                            }, os.path.join(path_model,"checkpoint_count_final.pth"))
            #save model
            torch.save({'epoch': epoch+start_ep,
                        'model': counter.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'np_rand_state': np.random.get_state(),
                        'torch_rand_state': torch.get_rng_state(),
                        }, os.path.join(path_model,"checkpoint_count.pth"))
            
    #visualize losses
    fig, axs = plt.subplots(1, 1, figsize=(10, 12))

    # Plot Training Loss
    axs.plot(range(1, num_epochs + 1), loss_hist, marker='o', linestyle='-', color='r')
    axs.set_title('Training Loss vs. Epochs')
    axs.set_xlabel('Epochs')
    axs.set_ylabel('Training Loss')
    axs.grid(True)


    plt.tight_layout()
    plt.savefig(path_output+f'/epochs={num_epochs}plot.png')
    plt.show()

if __name__ == '__main__':
    main()
