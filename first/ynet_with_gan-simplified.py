"""
YWGANet receiving Input1 and Input2
CNNs layers
Wasserstein Adversarial Loss + MSE Loss + BCE Loss
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(0)
np.random.seed(0)

latent_dim = 512
batch_size = 8

lr = 1e-5
num_epochs = 18

train_loss_hist = []
val_loss_hist = []
d_loss_hist = []

def normalize(x):
    max_value = x.max()
    min_value = x.min()
    return (x - min_value) / (max_value - min_value)

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


class Down(nn.Module):
    """Max pooling then double convolutions"""
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
    """Concatenate, then upsampling, then double convolution -->in_channels//2"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                ConvLayer(3*in_channels, 3*in_channels//2, 1, 1, 0)
            )
        else:
            self.up = nn.ConvTranspose2d(3*in_channels, 3*in_channels//2, kernel_size=2, stride=2)
        
        self.doubleconv = nn.Sequential(
            ConvLayer(3*in_channels//2, in_channels//2),
            ConvLayer(in_channels//2, in_channels//2)
        )

    def forward(self, x0, x1, x2):
        x = torch.cat([x0, x1, x2], dim=1)
        x = self.up(x)
        return self.doubleconv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.out_conv = ConvLayer(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return normalize(self.out_conv(x))


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir_1, input_dir_2, target_dir, transform=None):
        self.input_dir_1 = input_dir_1
        self.input_dir_2 = input_dir_2
        self.target_dir = target_dir
        self.input_images_1 = sorted(os.listdir(input_dir_1))  # Sort to align input and target
        self.input_images_2 = sorted(os.listdir(input_dir_2))
        self.target_images = sorted(os.listdir(target_dir))
        self.transform = transform

    def __len__(self):
        return len(self.target_images)

    def __getitem__(self, idx):
        # Load input and target images
        input_image_path_1 = os.path.join(self.input_dir_1, self.input_images_1[idx])
        input_image_path_2 = os.path.join(self.input_dir_2, self.input_images_2[idx])
        target_image_path = os.path.join(self.target_dir, self.target_images[idx])
        
        input_image_1 = Image.open(input_image_path_1)
        input_image_2 = Image.open(input_image_path_2)
        target_image = Image.open(target_image_path)
        
        # Apply transformations
        if self.transform:
            input_image_1 = self.transform(input_image_1)
            input_image_2 = self.transform(input_image_2)
            target_image = self.transform(target_image)
            
        return input_image_1, input_image_2, target_image

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

        #for Input 1
        in_ch = self.n_channels
        self.conv1_1 = ConvLayer(in_ch, 32)
        self.conv2_1 = ConvLayer(32, 32)
        self.down1_1 = Down(32, 64)
        self.down2_1 = Down(64, 128)
        self.down3_1 = Down(128, 256)

        #for Input 2
        self.conv1_2 = ConvLayer(in_ch, 32)
        self.conv2_2 = ConvLayer(32, 32)
        self.down1_2 = Down(32, 64)
        self.down2_2 = Down(64, 128)
        self.down3_2 = Down(128, 256)

        #for upsampling
        self.conv3 = ConvLayer(512, 256)
        self.up1 = Up(256, 128)
        self.up2 = Up(128, 64)
        self.up3 = Up(64, 32)

        #ending
        self.conv_end0 = ConvLayer(32*3, 8)
        self.conv_end1 = ConvLayer(8, 8)
        self.conv_end2 = ConvLayer(8, 8)
        
        self.outc = OutConv(8, n_classes)

    def forward(self, x1, x2):
        #for Input 1
        x1 = self.conv1_1(x1)
        down0_1 = self.conv2_1(x1)
        down1_1 = self.down1_1(down0_1)
        down2_1 = self.down2_1(down1_1)
        down3_1 = self.down3_1(down2_1)

        #for Input 2
        x2 = self.conv1_2(x2)
        down0_2 = self.conv2_2(x2)
        down1_2 = self.down1_2(down0_2)
        down2_2 = self.down2_2(down1_2)
        down3_2 = self.down3_2(down2_2)

        #concatenate and upsampling
        x0 = torch.cat((down3_1, down3_2), dim=1)
        x0 = self.conv3(x0)
        x0 = self.up1(x0, down3_1, down3_2)
        x0 = self.up2(x0, down2_1, down2_2)
        x0 = self.up3(x0, down1_1, down1_2)

        #ending
        merge = torch.cat((x0, down0_1, down0_2), dim=1)
        x0 = self.conv_end0(merge)
        conv_end1 = self.conv_end1(x0)
        conv_end2 = self.conv_end2(x0 + conv_end1)

        x = self.outc(x0 + conv_end1 + conv_end2)
        
        return x.to(device)


# Define the discriminator receiving input of [N, 2, W, H], N is batch size
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        validity = self.model(img)
        return validity


def main():
    #===================== LOADING DATA ===========================
    path_model = 'models/first/YGAN-simplified'
    plot_output = 'outputs/plots'
    if not os.path.exists(path_model):
        os.makedirs(path_model)
    # Create dataset
    train_input_dir_1 = 'dataset_AI_students/train/input1'
    train_input_dir_2 = 'dataset_AI_students/train/input2'
    train_target_dir = 'dataset_AI_students/train/target'
    
    val_input_dir_1 = 'dataset_AI_students/val/input1'
    val_input_dir_2 = 'dataset_AI_students/val/input2'
    val_target_dir = 'dataset_AI_students/val/target'

    test_input_dir_1 = 'dataset_AI_students/test/input1'
    test_input_dir_2 = 'dataset_AI_students/test/input2'
    test_target_dir = 'dataset_AI_students/test/target'
    train_dataset = ImageDataset(input_dir_1=train_input_dir_1, input_dir_2=train_input_dir_2, target_dir=train_target_dir, transform=transform)
    val_dataset = ImageDataset(input_dir_1=val_input_dir_1, input_dir_2=val_input_dir_2, target_dir=val_target_dir, transform=transform)
    test_dataset = ImageDataset(input_dir_1=test_input_dir_1, input_dir_2=test_input_dir_2, target_dir=test_target_dir, transform=transform)

    # Create DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    # Initialize generator and discriminator
    generator = Generator(1, 1).to(device)
    discriminator = Discriminator().to(device)
    # Loss functions
    adversarial_loss = nn.MSELoss().to(device)
    mse_loss = nn.MSELoss().to(device)
    bce_loss = nn.BCELoss().to(device)
    #perceptual_loss = PerceptualLoss().to(device)
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, weight_decay=1e-5) #regularization
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, weight_decay=1e-5)

    start_ep = 0
    min_val_loss = 1e10
    """
    #loading Generator
    if os.path.isfile(os.path.join(path_model,"checkpoint_G.pth")):
        #checkpoint = torch.load(os.path.join(path_model,"checkpoint.pth"), map_location='cpu')
        checkpoint = torch.load(os.path.join(path_model,"checkpoint_G.pth"),map_location=torch.device('cpu'))
        start_ep = checkpoint['epoch']
        print("Resuming from the checkpoint: ep", start_ep+1)
        np.random.set_state(checkpoint['np_rand_state'])
        torch.set_rng_state(checkpoint['torch_rand_state'])
        optimizer_G.load_state_dict(checkpoint['optimizer'])
        generator.load_state_dict(checkpoint['model'])

    #loading Discriminator
    if os.path.isfile(os.path.join(path_model,"checkpoint_D.pth")):
        #checkpoint = torch.load(os.path.join(path_model,"checkpoint.pth"), map_location='cpu')
        checkpoint = torch.load(os.path.join(path_model,"checkpoint_D.pth"),map_location=torch.device('cpu'))
        start_ep = checkpoint['epoch']
        print("Resuming from the checkpoint: ep", start_ep+1)
        np.random.set_state(checkpoint['np_rand_state'])
        torch.set_rng_state(checkpoint['torch_rand_state'])
        optimizer_D.load_state_dict(checkpoint['optimizer'])
        discriminator.load_state_dict(checkpoint['model'])
    """
    # Training loop
    for epoch in range(num_epochs):
        for i, (x1,x2,yy) in enumerate(train_loader):
            #data and its label
            x1 = x1.to(device)
            x2 = x2.to(device)
            yy = yy.to(device)
            
            # Adversarial ground truths
            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)
    
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            # Generate a batch of images
            fake_images = generator(x1, x2)
    
            # Measure discriminator's ability to classify real and fake images
            real_loss = adversarial_loss(discriminator(yy), valid)
            fake_loss = adversarial_loss(discriminator(fake_images.detach()), fake)
            d_loss = (real_loss+fake_loss)/2
            # Backward pass and optimize
            d_loss.backward()
            optimizer_D.step()
    
            # -----------------
            #  Train Generator
            # -----------------
    
            optimizer_G.zero_grad()
            # Generate a batch of images
            gen_images = generator(x1, x2)
            # Total loss
            g_loss = 0.3*adversarial_loss(discriminator(gen_images), valid) + mse_loss(gen_images, yy) + bce_loss(gen_images, yy)
            
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
            for i, (x1, x2, yy) in enumerate(val_loader):
                x1, x2, yy = x1.to(device), x2.to(device), yy.to(device)
                im = generator(x1, x2)
                valid_mse += (mse_loss(im, yy)).item()
        valid_mse /= len(val_loader)
                
        print(
            f"Epoch [{epoch+1}/{num_epochs}]\
                    Valid MSE Loss: {valid_mse:.4f} "
        )
        
        if valid_mse < min_val_loss:
            min_val_loss = valid_mse
            #save generator
            torch.save({'epoch': epoch+start_ep,
                        'model': generator.state_dict(),
                        'optimizer': optimizer_G.state_dict(),
                        'np_rand_state': np.random.get_state(),
                        'torch_rand_state': torch.get_rng_state(),
                        }, os.path.join(path_model,"checkpoint_YWGAN_final.pth"))
            #save discriminator
            torch.save({'epoch': epoch+start_ep,
                        'model': discriminator.state_dict(),
                        'optimizer': optimizer_D.state_dict(),
                        'np_rand_state': np.random.get_state(),
                        'torch_rand_state': torch.get_rng_state(),
                        }, os.path.join(path_model,"checkpoint_D_final.pth"))
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
                    }, os.path.join(path_model,f"checkpoint_YWGAN_ep={epoch}.pth"))
        #save discriminator
        torch.save({'epoch': epoch+start_ep,
                    'model': discriminator.state_dict(),
                    'optimizer': optimizer_D.state_dict(),
                    'np_rand_state': np.random.get_state(),
                    'torch_rand_state': torch.get_rng_state(),
                    }, os.path.join(path_model,f"checkpoint_D_ep={epoch}.pth"))

    
    #visualize losses
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # Plot Discrimination Loss
    axs[0].plot(range(1, num_epochs + 1), d_loss_hist, marker='o', linestyle='-', color='r')
    axs[0].set_title('Discrimination Loss vs. Epochs')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Discrimination Loss')
    axs[0].grid(True)

    # Plot Generation Loss
    axs[1].plot(range(1, num_epochs + 1), train_loss_hist, marker='o', linestyle='-', color='b')
    axs[1].set_title('Generation Loss vs. Epochs')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Loss')
    axs[1].grid(True)

    # Plot Validation and Test Loss
    axs[2].plot(range(1, num_epochs + 1), val_loss_hist, marker='o', linestyle='-', color='g')
    axs[2].set_title('Validation Loss vs. Epochs')
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('Loss')
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig(plot_output+f'/YGAN-simplified_ep={num_epochs}.png')
    plt.show()

if __name__ == '__main__':
    main()
