"GAN network base"
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
import torchvision

from PIL import Image
import skimage

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

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
])

class ConvLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super(ConvLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.layer(x)
        return x

class FourierLayer(nn.Module):
    def __init__(self, in_channel, out_channel, modes1, modes2, device='cpu'):
        super(FourierLayer, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.modes1 = modes1
        self.modes2 = modes2
        self.device = device
        
        self.scale = (1 / in_channel * out_channel)
        self.weights1 = nn.Parameter(self.scale * torch.rand(1, self.out_channel, self.modes1, self.modes2, dtype=torch.cfloat, device=device))
        self.weights2 = nn.Parameter(self.scale * torch.rand(1, self.out_channel, self.modes1, self.modes2, dtype=torch.cfloat, device=device))
    
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        
        out_ft = torch.zeros(batchsize, self.out_channel,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)

        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.in_channel = 1
        self.latent_dim = latent_dim

        in_ch = self.in_channel

        modes = 64
        self.conv1 = ConvLayer(in_ch, in_ch*4)
        self.conv2 = ConvLayer(in_ch*4, in_ch*8)
        
        in_ch *= 8
        self.fl1 = FourierLayer(in_ch, in_ch*2, modes, modes, device)
        self.conv3 = ConvLayer(in_ch, in_ch*2) #layer for skip connection

        in_ch *= 4
        self.fl2 = FourierLayer(in_ch, in_ch//2, modes, modes, device)
        self.prelu1 = nn.PReLU()

        in_ch //= 2
        self.fl3 = FourierLayer(in_ch, in_ch//2, modes, modes, device)

        in_ch //= 2
        self.conv4 = ConvLayer(in_ch, in_ch//2)
        self.fl4 = FourierLayer(in_ch, in_ch//2, modes, modes, device)
        
        self.prelu2 = nn.PReLU()

        self.fl5 = FourierLayer(in_ch, in_ch*2, modes, modes, device)
        self.conv5 = ConvLayer(in_ch, in_ch*2)

        in_ch *= 4
        self.fl6 = FourierLayer(in_ch, in_ch//4, modes, modes, device)
        
        self.prelu3 = nn.PReLU()
        in_ch //= 4
        self.conv6 = ConvLayer(in_ch, in_ch//2, 3, 1, 1)
        
        in_ch //= 2
        self.conv_end = nn.Conv2d(in_ch, self.in_channel, 1, 1, 0)
        
        self.bn_end = nn.BatchNorm2d(self.in_channel)
        #self.activation_end = nn.PReLU()
    
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        fl1 = self.fl1(conv2)
        conv3 = self.conv3(conv2)
        merge1 = torch.cat((conv3, fl1), dim=1)
        fl2 = self.fl2(merge1)

        prelu1 = self.prelu1(fl2)

        fl3 = self.fl3(prelu1)
        fl4 = self.fl4(fl3)
        conv4 = self.conv4(fl3)
        merge2 = torch.cat((conv4, fl4), dim=1)
        prelu2 = self.prelu2(merge2)
        
        fl5 = self.fl5(prelu2)
        conv5 = self.conv5(prelu2)
        merge3 = torch.cat((conv5, fl5), dim=1)
        
        fl6 = self.fl6(merge3)
        
        prelu3 = self.prelu3(fl6)

        conv6 = self.conv6(prelu3)
        
        conv_end = self.conv_end(conv6)
        x = self.bn_end(conv_end)
        #x = self.activation_end(x)

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
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128, momentum=0.82),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(256, momentum=0.8),
            nn.LeakyReLU(0.25),
            nn.Flatten(),
            nn.Linear(256 * 32 * 32, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        validity = self.model(img)
        return validity


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        
        # Load a pre-trained VGG network
        self.vgg = models.vgg16(pretrained=True).features.to(device)
        
        # Freeze the VGG model weights
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        # Loss function (e.g., L2 loss)
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
