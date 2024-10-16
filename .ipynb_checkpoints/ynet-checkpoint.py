"YNet model with MSE Loss + BCE Loss + Perceptual Loss"
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
import matplotlib.pyplot as plt

from PIL import Image
import skimage
from torchvision import transforms
import torchvision


torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


latent_dim = 512
batch_size = 1

lr = 1e-6
beta1 = 0.6
beta2 = 0.99
num_epochs = 10
loss_hist = []
min_loss = 0

def normalize(x):
    max_value = x.max()
    min_value = x.min()
    return (x - min_value) / (max_value - min_value)

def apply_threshold(tensor, threshold=0.5):
    # Apply threshold to generate binary tensor (0 or 1)
    tensor = normalize(tensor)
    binary_tensor = (tensor < threshold).float()
    
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
        return len(self.input_images_1)

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


class YNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(YNet, self).__init__()
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


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        
        # Load a pre-trained VGG network
        self.vgg = models.vgg16(pretrained=True).features.to(device)
        
        # Freeze the VGG model weights
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        self.criterion = nn.L1Loss()
    
    def forward(self, input_img, target_img):
        assert input_img.shape == target_img.shape, "Input and target images must have the same shape"

        # Preprocess the images to match VGG's input requirements (Normalize, etc.)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_img = input_img.to(torch.device('cpu'))
        target_img = target_img.to(torch.device('cpu'))
        input_img = transform(input_img).to(device)
        target_img = transform(target_img).to(device)
        
        # Extract feature representations at specified layers
        input_features = self.vgg(input_img)
        target_features = self.vgg(target_img)
        
        # Compute the perceptual loss between input and target features
        loss = self.criterion(input_features, target_features)
        
        return loss


def main():
    #===================== LOADING DATA ===========================
    path_model = 'Desktop/URECA/GAN-main/models/YNet'
    path_output = 'Desktop/URECA/GAN-main/outputs/YNet'
    
    # Create dataset
    input_dir_1 = 'Desktop/URECA/dataset_AI_students_1/try/input1'
    input_dir_2 = 'Desktop/URECA/dataset_AI_students_1/try/input2'
    target_dir = 'Desktop/URECA/dataset_AI_students_1/try/target'
    dataset = ImageDataset(input_dir_1=input_dir_1, input_dir_2=input_dir_2, target_dir=target_dir, transform=transform)

    # Create DataLoader
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
   
    # Initialize generator and discriminator
    model = YNet(1, 1).to(device)
    # Loss functions
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    perceptual_loss = PerceptualLoss()
    # Optimizers
    optimizer = optim.Adam(model.parameters(), lr=lr)

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
        for i, (x1, x2,yy) in enumerate(data_loader):
            #data and its label
            x1 = x1.to(device)
            x2 = x2.to(device)
            yy = yy.to(device)

            # -----------------
            #  Training
            # -----------------
            optimizer.zero_grad()
            # Generate a batch of images
            gen_images = model(x1, x2)

            # Total loss
            loss =  mse_loss(gen_images, yy) + bce_loss(gen_images, yy) + 0.5*perceptual_loss(gen_images, yy)
    
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            # ---------------------
            #  Progress Monitoring
            # ---------------------
            if (i + 1) % 1 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}]\
                            Batch {i+1}/{len(data_loader)} "
                    f"Training Loss: {loss.item():.4f}"
                )
        if (epoch + 1) % 1 == 0:
            #save losses
            loss_hist.append(loss.item())
            if loss.item() < min_loss:
                #save model
                min_loss = loss.item()
                #max_g_loss = g_loss.item()
                torch.save({'epoch': epoch+start_ep,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'np_rand_state': np.random.get_state(),
                            'torch_rand_state': torch.get_rng_state(),
                            }, os.path.join(path_model,"checkpoint_YNet_final.pth"))
            #save model
            torch.save({'epoch': epoch+start_ep,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'np_rand_state': np.random.get_state(),
                        'torch_rand_state': torch.get_rng_state(),
                        }, os.path.join(path_model,"checkpoint_YNet.pth"))
            
    #visualize losses
    fig, axs = plt.subplots(1, 1, figsize=(10, 12))

    # Plot Training Loss
    axs.plot(range(1, num_epochs + 1), loss_hist, marker='o', linestyle='-', color='r')
    axs.set_title('Training Loss vs. Epochs')
    axs.set_xlabel('Epochs')
    axs.set_ylabel('Training Loss')
    axs.grid(True)


    plt.tight_layout()
    plt.savefig(path_output+'/plot.png')
    plt.show()

if __name__ == '__main__':
    main()