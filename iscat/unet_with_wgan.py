"""
WGAN with Generator being UNet.
Adversarial Loss + MSE Loss + BCE Loss.
"""
import os
import glob

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
import torchvision
import scipy
import matplotlib.pyplot as plt

from PIL import Image
import skimage

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

torch.manual_seed(0)
np.random.seed(0)

latent_dim = 512
batch_size = 7

lr = 1e-5
beta1 = 0.6
beta2 = 0.9
clip_value = 0.01
num_epochs = 50

train_loss_hist = []
val_loss_hist = []
d_loss_hist = []

def normalize(x):
    max_value = x.max()
    min_value = x.min()
    return (x - min_value) / (max_value - min_value)

def apply_threshold(tensor, threshold=0.5):
    # Apply threshold to generate binary tensor (0 or 1)
    tensor = normalize(tensor)
    binary_tensor = (tensor > threshold).float()
    
    return binary_tensor

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
        )
    
    def forward(self, x):
        return self.layer(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DoubleConv, self).__init__()
        self.layer = nn.Sequential(
            ConvLayer(in_channels, out_channels, kernel_size, stride, padding),
            ConvLayer(out_channels,  out_channels, kernel_size, stride, padding)
        )
    def forward(self, x):
        return self.layer(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Down, self).__init__()
        self.down_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvLayer(in_channels, out_channels),
            ConvLayer(out_channels, out_channels)
        )

    def forward(self, x):
        return self.down_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                ConvLayer(in_channels, in_channels//2, 1, 1, 0)
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        
        self.doubleconv = nn.Sequential(
            ConvLayer(in_channels, in_channels//2),
            ConvLayer(in_channels//2, in_channels//2)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.doubleconv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.out_conv = ConvLayer(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return normalize(self.out_conv(x))


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir, target_dir, transform=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.input_images = sorted(os.listdir(input_dir))  # Sort to align input and target
        self.target_images = sorted(os.listdir(target_dir))
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        # Load input and target images
        input_image_path = os.path.join(self.input_dir, self.input_images[idx])
        target_image_path = os.path.join(self.target_dir, self.target_images[idx])
        
        input_image = Image.open(input_image_path)
        target_image = Image.open(target_image_path)
        
        # Apply transformations
        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)
            
        return input_image, target_image

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Lambda(lambda x: 1-x)
])

class Generator(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(Generator, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.conv0_1 = nn.Conv2d(self.n_channels, 32, 1, 1, 0)
        self.conv0_2 = ConvLayer(32, 32, 3, 1, 1)

        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)

        self.conv0 = DoubleConv(32, 32)
        self.conv1 = DoubleConv(64, 64)
        self.conv2 = DoubleConv(128, 128)

        self.up1 = Up(256, 128)
        self.up2 = Up(128, 64)
        self.up3 = Up(64, 32)

        self.conv_end1 = ConvLayer(32, 8)
        self.conv_end2 = ConvLayer(8, 8)
        self.outc = OutConv(8, n_classes)

    def forward(self, x):
        x1 = self.conv0_1(x)
        x1 = self.conv0_2(x1)
        
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, self.conv2(x3))
        x = self.up2(x, self.conv1(x2))
        x = self.up3(x, self.conv0(x1))

        x_end1 = self.conv_end1(x)
        x_end2 = self.conv_end2(x_end1)
        x = self.outc(x_end1 + x_end2)
        
        return x.to(device)


# Define the discriminator receiving input of [N, 2, W, H], N is batch size
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.BatchNorm2d(64, momentum=0.82),
            nn.LeakyReLU(0.25),
            nn.Dropout(0.25),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128, momentum=0.82),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(256, momentum=0.8),
            nn.LeakyReLU(0.25),
            nn.Flatten(),
            nn.Linear(256 * 32 * 32, 1)
        )

    def forward(self, img):
        validity = self.model(img)
        return validity


def main():
    #===================== LOADING DATA ===========================
    path_model = 'models/iscat/UWGAN'
    plot_output = 'outputs/plots'
    if not os.path.exists(path_model):
        os.makedirs(path_model)
    if not os.path.exists(plot_output):
        os.makedirs(plot_output)
    
    # Create dataset
    train_input_dir = 'iSCAT_processed/train/Input'
    train_target_dir = 'iSCAT_processed/train/GT'
    
    val_input_dir = 'iSCAT_processed/val/Input'
    val_target_dir = 'iSCAT_processed/val/GT'
    
    train_dataset = ImageDataset(input_dir=train_input_dir, target_dir=train_target_dir, transform=transform)
    val_dataset = ImageDataset(input_dir=val_input_dir, target_dir=val_target_dir, transform=transform)
    
    # Create DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    # Initialize generator and discriminator
    generator = Generator(1, 1).to(device)
    discriminator = Discriminator().to(device)
    # Loss functions
    adversarial_loss = nn.MSELoss().to(device)
    mse_loss = nn.MSELoss().to(device)
    bce_loss = nn.BCELoss().to(device)
    #perceptual_loss = PerceptualLoss().to(device)
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=1e-5) #regularization
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=1e-5)

    start_ep = 1
    min_val_loss = 1e10
    patience_counter = 0
    ran_epochs = 0
    """
    #loading Generator
    if os.path.isfile(os.path.join(path_model,"checkpoint_UWG_final.pth")):
        checkpoint = torch.load(os.path.join(path_model,"checkpoint_UWG_final.pth"),map_location=torch.device('cpu'))
        start_ep = checkpoint['epoch']
        print("Resuming from the checkpoint: ep", start_ep+1)
        np.random.set_state(checkpoint['np_rand_state'])
        torch.set_rng_state(checkpoint['torch_rand_state'])
        optimizer_G.load_state_dict(checkpoint['optimizer'])
        generator.load_state_dict(checkpoint['model'])

    #loading Discriminator
    if os.path.isfile(os.path.join(path_model,"checkpoint_D_final.pth")):
        checkpoint = torch.load(os.path.join(path_model,"checkpoint_D_final.pth"),map_location=torch.device('cpu'))
        start_ep = checkpoint['epoch']
        print("Resuming from the checkpoint: ep", start_ep+1)
        np.random.set_state(checkpoint['np_rand_state'])
        torch.set_rng_state(checkpoint['torch_rand_state'])
        optimizer_D.load_state_dict(checkpoint['optimizer'])
        discriminator.load_state_dict(checkpoint['model'])
    """
    # Training loop
    for epoch in range(num_epochs):
        ran_epochs += 1
        for i, (xx,yy) in enumerate(train_loader):
            #data and its label
            xx = xx.to(device)
            yy = yy.to(device)
    
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            # Generate a batch of images
            fake_images = generator(xx)
    
            # Measure discriminator's ability to classify real and fake images
            d_loss = 1.0 - adversarial_loss(discriminator(fake_images.detach()), discriminator(yy))
            # Backward pass and optimize
            d_loss.backward()
            optimizer_D.step()
            """
            for p in discriminator.parameters():
                p.data.clamp_(-clip_value, clip_value)
            """
            # -----------------
            #  Train Generator
            # -----------------
    
            optimizer_G.zero_grad()
            # Generate a batch of images
            gen_images = generator(xx)
            # Total loss
            g_loss = 0.5*adversarial_loss(discriminator(gen_images), discriminator(yy)) + bce_loss(gen_images, yy)
            
            # Backward pass and optimize
            g_loss.backward()
            optimizer_G.step()
            # ---------------------
            #  Progress Monitoring
            # ---------------------
            if (i + 1) % 125 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}]\
                            Batch {i+1}/{len(train_loader)} "
                    f"Discriminator Loss: {d_loss.item():.4f} "
                    f"Generator Loss: {g_loss.item():.4f}"
                )
                
        #======================VALIDATION=======================
        valid_mse = 0
        with torch.no_grad():
            for i, (xx, yy) in enumerate(val_loader):
                xx, yy = xx.to(device), yy.to(device)
                im = generator(xx)
                valid_mse += (mse_loss(im, yy)).item()
        valid_mse /= len(val_loader)
                
        print(
            f"Epoch [{epoch+1}/{num_epochs}]\
                    Valid MSE Loss: {valid_mse:.4f} "
        )
        
        #save losses
        d_loss_hist.append(d_loss.item())
        train_loss_hist.append(g_loss.item())
        val_loss_hist.append(valid_mse)
        #save generator
        torch.save({'epoch': epoch+start_ep,
                    'model': generator.state_dict(),
                    'optimizer': optimizer_G.state_dict(),
                    'np_rand_state': np.random.get_state(),
                    'torch_rand_state': torch.get_rng_state(),
                    }, os.path.join(path_model,f"checkpoint_YG_ep={epoch+start_ep}.pth"))
        #save discriminator
        torch.save({'epoch': epoch+start_ep,
                    'model': discriminator.state_dict(),
                    'optimizer': optimizer_D.state_dict(),
                    'np_rand_state': np.random.get_state(),
                    'torch_rand_state': torch.get_rng_state(),
                    }, os.path.join(path_model,f"checkpoint_YD_ep={epoch+start_ep}.pth"))
        
        if valid_mse < min_val_loss:
            min_val_loss = valid_mse
            #save generator
            torch.save({'epoch': epoch+start_ep,
                        'model': generator.state_dict(),
                        'optimizer': optimizer_G.state_dict(),
                        'np_rand_state': np.random.get_state(),
                        'torch_rand_state': torch.get_rng_state(),
                        }, os.path.join(path_model,"checkpoint_UWG_final.pth"))
            #save discriminator
            torch.save({'epoch': epoch+start_ep,
                        'model': discriminator.state_dict(),
                        'optimizer': optimizer_D.state_dict(),
                        'np_rand_state': np.random.get_state(),
                        'torch_rand_state': torch.get_rng_state(),
                        }, os.path.join(path_model,"checkpoint_D_final.pth"))
        else:
            patience_counter += 1
        if patience_counter > 3:
            break
    
    #=======================end of training loop================================
    #visualize losses
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # Plot Discrimination Loss
    axs[0].plot(range(1, ran_epochs + 1), d_loss_hist, marker='o', linestyle='-', color='yellow')
    axs[0].set_title('Discrimination Loss vs. Epochs')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Discrimination Loss')
    axs[0].grid(True)

    # Plot Generation Loss
    axs[1].plot(range(1, ran_epochs + 1), train_loss_hist, marker='o', linestyle='-', color='b')
    axs[1].set_title('Generation Loss vs. Epochs')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Generation Loss')
    axs[1].grid(True)

    # Plot Validation Loss
    axs[2].plot(range(1, ran_epochs + 1), val_loss_hist, marker='o', linestyle='-', color='g')
    axs[2].set_title('Validation Loss vs. Epochs')
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('Loss')
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig(plot_output+f'/UWGAN_ep={num_epochs}.png')
    plt.show()

if __name__ == '__main__':
    main()
