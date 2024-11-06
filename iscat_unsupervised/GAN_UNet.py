"""
WGAN with Generator being UNet.
Adversarial Loss + MSE Loss + BCE Loss.
"""
import os
import glob
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

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
from skimage.metrics import structural_similarity as ssim
from physics import pixel_size, objective_mag, z, wavelengths, physics_module

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
torch.autograd.set_detect_anomaly(True)

torch.manual_seed(0)
np.random.seed(0)

latent_dim = 512 #dimension of image
batch_size = 5

lr = 1e-5
num_epochs = 20

d_loss_hist = []
train_loss_hist = []
val_loss_hist = []

#_________________________________________Generator built from here_________________________________________
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels, momentum=0.9),
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
    def __init__(self, input_dir, wavelengths, transform=None):
        self.input_dir = input_dir
        self.input_images = os.listdir(input_dir)
        self.input_images = [f for f in self.input_images if f.endswith('.png')]
        self.wavelengths = wavelengths
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        # Load input and target images
        input_image_path = os.path.join(self.input_dir, self.input_images[idx])
        input_image = Image.open(input_image_path)
        idx_str = self.input_images[idx].split('_')[1].split('.')[0]
        image_index = int(idx_str)-1
        wavelength = self.wavelengths[image_index]
        
        # Apply transformations
        if self.transform:
            input_image = self.transform(input_image)
            
        return input_image, wavelength #image_index used for getting the corresponding wavelength

# Define transformations
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
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

class Discriminator(nn.Module):
    def __init__(self, latent_dim):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.LeakyReLU(0.25),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(256, momentum=0.9),
            nn.LeakyReLU(0.25),
            nn.Flatten(),
            nn.Linear(256 * 32 * 32, 1)
        )

    def forward(self, img):
        validity = self.model(img)
        return validity

class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1  #grayscale
    
    def create_window(self, channel: int):
        _1D_window = torch.hann_window(self.window_size, periodic=False).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, self.window_size, self.window_size).contiguous()
        return window

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        window = self.create_window(channel).to(device)
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return ssim_map.mean()

        return ssim_map.mean(1).mean(1).mean(1)


#_________________________________________END here_________________________________________

def main():
    #===================== LOADING DATA ===========================
    path_model = 'unsupervised/UNet'
    plot_output = 'outputs/plots'

    # Create dataset
    train_input_dir = 'iSCAT_processed/train/Input'
    val_input_dir = 'iSCAT_processed/val/Input'
    
    train_dataset = ImageDataset(input_dir=train_input_dir, wavelengths=wavelengths, transform=transform)
    val_dataset = ImageDataset(input_dir=val_input_dir, wavelengths=wavelengths, transform=transform)
    
    # Create DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)
    # Initialize generator and discriminator
    generator = Generator(1, 1).to(device)
    discriminator = Discriminator(latent_dim).to(device)
    # Loss functions
    ssim_loss = SSIM().to(device)
    adversarial_loss = nn.BCEWithLogitsLoss()
    # Optimizers
    optimizer_G = optim.SGD(generator.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5) #L2 regularization
    optimizer_D = optim.SGD(discriminator.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5) #L2 regularization
    scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=10, gamma=0.1)
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=10, gamma=0.1)

    start_ep = 1
    min_val_loss = 1e10
    """
    if os.path.isfile(os.path.join(path_model,"checkpoint_YG_final.pth")):
        checkpoint = torch.load(os.path.join(path_model,"checkpoint_YG_final.pth"),map_location=torch.device('cpu'))
        start_ep = checkpoint['epoch']
        print("Resuming from the checkpoint: ep", start_ep)
        np.random.set_state(checkpoint['np_rand_state'])
        torch.set_rng_state(checkpoint['torch_rand_state'])
        optimizer_G.load_state_dict(checkpoint['optimizer'])
        generator.load_state_dict(checkpoint['model'])
    """
    
    # Training loop
    for epoch in range(num_epochs):
        for i, (xx, wavelengths) in enumerate(train_loader):
            xx = xx.to(device)
            # Adversarial ground truths
            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            # Generate a batch of images
            reconstructed = []
            for j in range(wavelengths.size(0)):  # should be 8 
                inp = xx[i]  # Shape: [1, 512, 512]
                w = wavelengths[i]
                out = generator(inp.unsqueeze(0))
                re_inp = physics_module(out, pixel_size, w, objective_mag, z)
                reconstructed.append(re_inp.squeeze(0))
    
            reconstructed = torch.stack(reconstructed, dim=0)
    
            # Measure discriminator's ability to classify real and fake images
            real_loss = adversarial_loss(discriminator(xx), valid)
            fake_loss = adversarial_loss(discriminator(reconstructed.detach()), fake)
            d_loss = (real_loss+fake_loss)/2
            # Backward pass and optimize
            d_loss.backward()
            optimizer_D.step()
            # ---------------------
            #  Training progress
            # ---------------------
            optimizer_G.zero_grad()
            # Unsupervised loss
            #loss = (1 - ssim_loss(reconstructed, xx)) + bce_loss(reconstructed, xx)
            loss = adversarial_loss(discriminator(reconstructed), valid)
            # Backward pass and optimize
            loss.backward()
            optimizer_G.step()

            #  Progress Monitoring
            # ---------------------
            if (i + 1) % 200 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}]\
                            Batch {i+1}/{len(train_loader)} "
                    f"Discriminator Loss: {d_loss.item():.4f}   "
                    f"Training Loss: {loss.item():.4f} "
                )
                
        #======================VALIDATION=======================
        valid_loss = 0
        with torch.no_grad():
            for i, (xx, w) in enumerate(val_loader):
                xx = xx.to(device)
                im = generator(xx)
                reconstructed = physics_module(im, pixel_size, w, objective_mag, z)
                valid_loss += 1 - ssim_loss(reconstructed, xx).item()
        valid_loss /= len(val_loader)
                
        print(
            f"Epoch [{epoch+1}/{num_epochs}]\
                    Valid SSIM Loss: {valid_loss:.4f} "
        )
        
        if valid_loss < min_val_loss:
            min_val_loss = valid_loss
            #save generator
            torch.save({'epoch': epoch+start_ep,
                        'model': generator.state_dict(),
                        'optimizer': optimizer_G.state_dict(),
                        'scheduler': scheduler_G.state_dict(),
                        'np_rand_state': np.random.get_state(),
                        'torch_rand_state': torch.get_rng_state(),
                        }, os.path.join(path_model,"checkpoint_YG_final.pth"))
            torch.save({'epoch': epoch+start_ep,
                        'model': discriminator.state_dict(),
                        'optimizer': optimizer_D.state_dict(),
                        'scheduler': scheduler_D.state_dict(),
                        'np_rand_state': np.random.get_state(),
                        'torch_rand_state': torch.get_rng_state(),
                        }, os.path.join(path_model,"checkpoint_YD_final.pth"))
        #save losses
        d_loss_hist.append(d_loss.item())
        train_loss_hist.append(loss.item())
        val_loss_hist.append(valid_loss)
        #save generator
        torch.save({'epoch': epoch+start_ep,
                    'model': generator.state_dict(),
                    'optimizer': optimizer_G.state_dict(),
                    'scheduler': scheduler_G.state_dict(),
                    'np_rand_state': np.random.get_state(),
                    'torch_rand_state': torch.get_rng_state(),
                    }, os.path.join(path_model,f"checkpoint_YG_ep={epoch+start_ep}.pth"))
        torch.save({'epoch': epoch+start_ep,
                    'model': discriminator.state_dict(),
                    'optimizer': optimizer_D.state_dict(),
                    'scheduler': scheduler_D.state_dict(),
                    'np_rand_state': np.random.get_state(),
                    'torch_rand_state': torch.get_rng_state(),
                    }, os.path.join(path_model,f"checkpoint_YD_ep={epoch+start_ep}.pth"))

    
    #visualize losses
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    #plot Discrimination loss
    axs[0].plot(range(1, num_epochs + 1), d_loss_hist, marker='o', linestyle='-', color='g')
    axs[0].set_title('Dis Loss vs. Epochs')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Discriminator Loss')
    axs[0].grid(True)
    
    # Plot Training Loss
    axs[1].plot(range(1, num_epochs + 1), train_loss_hist, marker='o', linestyle='-', color='r')
    axs[1].set_title('Train Loss vs. Epochs')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Train Loss')
    axs[1].grid(True)

    # Plot Validation Loss
    axs[2].plot(range(1, num_epochs + 1), val_loss_hist, marker='o', linestyle='-', color='b')
    axs[2].set_title('Validation Loss vs. Epochs')
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('Validation Loss')
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig(plot_output+f'/unUNet_ep={num_epochs+start_ep-1}.png')
    plt.show()

if __name__ == '__main__':
    main()
