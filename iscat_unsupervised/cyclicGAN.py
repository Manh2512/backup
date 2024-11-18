"""
WGAN with Generator being UNet.
Adversarial Loss + MSE Loss + BCE Loss.
Network currently cannot learn.
"""
import os
import glob
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

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
from physics import pixel_size, objective_mag, z, physics_module

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
torch.autograd.set_detect_anomaly(True)

torch.manual_seed(0)
np.random.seed(0)

latent_dim = 512 #dimension of image
batch_size = 5

lr = 1e-5
#clip_value = 0.01
num_epochs = 20
wavelength = 530e-9

d_loss_hist_X = []
d_loss_hist_Y = []
train_loss_hist = []
val_loss_hist = []

def normalize(x):
    max_value = x.max()
    min_value = x.min()
    return (x - min_value) / (max_value - min_value)

def save_checkpoint(epoch, model, optimizer, scheduler, path, name):
    torch.save({'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'np_rand_state': np.random.get_state(),
                'torch_rand_state': torch.get_rng_state(),
                }, os.path.join(path, name))

def load_checkpoint(path_model, name):
    checkpoint = torch.load(os.path.join(path_model, name),map_location=torch.device('cpu'))
    print("Resuming from the checkpoint: ep", start_ep)
    np.random.set_state(checkpoint['np_rand_state'])
    torch.set_rng_state(checkpoint['torch_rand_state'])
    return checkpoint

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
        return normalize(self.layer(x))

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
        return self.out_conv(x)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir, transform=None):
        self.input_dir = input_dir
        self.input_images = os.listdir(input_dir)
        self.input_images = [f for f in self.input_images if f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        # Load input and target images
        input_image_path = os.path.join(self.input_dir, self.input_images[idx])
        input_image = Image.open(input_image_path)
        idx_str = self.input_images[idx].split('_')[1].split('.')[0]
        
        # Apply transformations
        if self.transform:
            input_image = self.transform(input_image)
            
        return input_image

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
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.25),
            nn.Flatten(),
            nn.Linear(256 * 32 * 32, 1)
        )

    def forward(self, img):
        validity = self.model(img)
        return validity

#_________________________________________END here_________________________________________

def main():
    #===================== LOADING DATA ===========================
    path_model = 'models/unsupervised/cyclicUNet'
    plot_output = 'outputs/plots'
    if not os.path.exists(path_model):
        os.makedirs(path_model)

    # Create dataset
    wavelengths = torch.tensor(np.loadtxt('iSCAT_processed/iscat_wavelengths.csv', delimiter=','), dtype=torch.float32).to(device)
    train_input_dir = 'iSCAT_processed/train/Input'
    val_input_dir = 'iSCAT_processed/val/Input'
    
    train_dataset = ImageDataset(input_dir=train_input_dir, transform=transform)
    val_dataset = ImageDataset(input_dir=val_input_dir, transform=transform)
    
    # Create DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)
    # Initialize generator and discriminator
    generator = Generator(1, 1).to(device)
    discriminator_X = Discriminator(latent_dim).to(device)
    discriminator_Y = Discriminator(latent_dim).to(device)
    
    # Loss functions
    adversarial_loss = nn.BCEWithLogitsLoss()
    cyclic_consistency = nn.L1Loss()
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, weight_decay=1e-5) #L2 regularization
    optimizer_DX = optim.Adam(discriminator_X.parameters(), lr=lr, weight_decay=1e-5) #L2 regularization
    optimizer_DY = optim.Adam(discriminator_Y.parameters(), lr=lr, weight_decay=1e-5) #L2 regularization
    
    scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=5, gamma=0.1)
    scheduler_DX = optim.lr_scheduler.StepLR(optimizer_DX, step_size=5, gamma=0.1)
    scheduler_DY = optim.lr_scheduler.StepLR(optimizer_DX, step_size=5, gamma=0.1)

    start_ep = 1
    min_val_loss = 1e10
    """
    if os.path.isfile(os.path.join(path_model,"checkpoint_UNetG_ep=20.pth")):
        checkpoint_G = load_checkpoint(path_model,"checkpoint_UNetG_ep=20.pth")
        start_ep = checkpoint_G['epoch']
        print("Resuming from the checkpoint: ep", start_ep)
        optimizer_G.load_state_dict(checkpoint_G['optimizer'])
        generator.load_state_dict(checkpoint_G['model'])
    
        checkpoint_DX = torch.load(os.path.join(path_model, "checkpoint_DX_ep=20.pth"),map_location=torch.device('cpu'))
        optimizer_DX.load_state_dict(checkpoint_DX['optimizer'])
        discriminator_X.load_state_dict(checkpoint_DX['model'])

        checkpoint_DY = torch.load(os.path.join(path_model, "checkpoint_DY_ep=20.pth"),map_location=torch.device('cpu'))
        optimizer_DY.load_state_dict(checkpoint_DY['optimizer'])
        discriminator_Y.load_state_dict(checkpoint_DY['model'])
    """
    
    # Training loop
    for epoch in range(num_epochs):
        for i, xx in enumerate(train_loader):
            xx = xx.to(device)
            # Adversarial ground truths
            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_DX.zero_grad()
            # Generate a batch of images
            y1 = generator(xx)
            reconstructed = []
            #find xhat = physics(y1)
            for j in range(y1.shape[0]):  # should be 8
                out = y1[j].unsqueeze(0)  # Shape: [1, 512, 512]
                re_inp = physics_module(out, pixel_size, wavelength, objective_mag, z)
                reconstructed.append(re_inp.squeeze(0))
    
            reconstructed = torch.stack(reconstructed, dim=0)
            fake_images = reconstructed.detach()

            #generate y2 from xhat
            y2 = generator(fake_images)
            
            #train discriminator_X
            # Measure discriminator's ability to classify real and fake images
            real_loss_X = adversarial_loss(discriminator_X(xx), valid)
            fake_loss_X = adversarial_loss(discriminator_X(fake_images), fake)
            d_loss_X = (real_loss_X+fake_loss_X)/2
            # Backward pass and optimize
            d_loss_X.backward()
            optimizer_DX.step()

            #train discriminator_Y
            optimizer_DY.zero_grad()
            # Measure discriminator's ability to classify real and fake images, consider y1 real, y2 fake-result of Gen Net
            real_loss_Y = adversarial_loss(discriminator_Y(y1.detach()), valid)
            fake_loss_Y = adversarial_loss(discriminator_Y(y2.detach()), fake)
            d_loss_Y = (real_loss_Y+fake_loss_Y)/2
            # Backward pass and optimize
            d_loss_Y.backward()
            optimizer_DY.step()
            # ---------------------
            #  Training progress
            # ---------------------
            optimizer_G.zero_grad()
            loss = 0.5*adversarial_loss(discriminator_X(reconstructed), valid) + adversarial_loss(discriminator_Y(y2), valid) \
            + 10*(cyclic_consistency(reconstructed, xx) + cyclic_consistency(y1, y2))
            # Backward pass and optimize
            loss.backward()
            optimizer_G.step()

            #  Progress Monitoring
            # ---------------------
            if (i + 1) % 200 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}]\
                            Batch {i+1}/{len(train_loader)} "
                    f"Discriminator X Loss: {d_loss_X.item():.4f}   "
                    f"Discriminator Y Loss: {d_loss_Y.item():.4f}   "
                    f"Training Loss: {loss.item():.4f} "
                )
                
        #======================VALIDATION=======================
        valid_loss = 0
        with torch.no_grad():
            for i, xx in enumerate(val_loader):
                xx = xx.to(device)
                im = generator(xx)
                reconstructed = physics_module(im, pixel_size, wavelength, objective_mag, z)
                valid_loss += cyclic_consistency(reconstructed, xx).item()
        valid_loss /= len(val_loader)
                
        print(
            f"Epoch [{epoch+1}/{num_epochs}]\
                    Valid L1 Loss: {valid_loss:.4f} "
        )
        
        if valid_loss < min_val_loss:
            min_val_loss = valid_loss
            save_checkpoint(epoch+start_ep, generator, optimizer_G, scheduler_G, path_model, "checkpoint_UNetG_final.pth")
            save_checkpoint(epoch+start_ep, discriminator_X, optimizer_DX, scheduler_DX, path_model, "checkpoint_DX_final.pth")
            save_checkpoint(epoch+start_ep, discriminator_Y, optimizer_DY, scheduler_DY, path_model, "checkpoint_DY_final.pth")
            
        #save losses
        d_loss_hist_X.append(d_loss_X.item())
        d_loss_hist_Y.append(d_loss_Y.item())
        train_loss_hist.append(loss.item())
        val_loss_hist.append(valid_loss)
        
        #save generator
        save_checkpoint(epoch+start_ep, generator, optimizer_G, scheduler_G, path_model, f"checkpoint_YG_ep={epoch+start_ep}.pth")
        save_checkpoint(epoch+start_ep, discriminator_X, optimizer_DX, scheduler_DX, path_model, f"checkpoint_DX_ep={epoch+start_ep}.pth")
        save_checkpoint(epoch+start_ep, discriminator_Y, optimizer_DY, scheduler_DY, path_model, f"checkpoint_DY_ep={epoch+start_ep}.pth")
    
    #visualize losses
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    #plot Discrimination loss
    axs[0].plot(range(1, num_epochs + 1), d_loss_hist_X, marker='o', linestyle='-', color='g')
    axs[0].plot(range(1, num_epochs + 1), d_loss_hist_Y, marker='o', linestyle='-', color='orange')
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
    plt.savefig(plot_output+f'/cycleUNet_ep={num_epochs}.png')
    plt.show()

if __name__ == '__main__':
    main()
