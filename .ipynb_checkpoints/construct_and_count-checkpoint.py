"""
YNet model with MSE Loss + BCE Loss + Perceptual Loss.
Implement FFT at Bridge.
"""
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

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

#===========READ DATA FROM MATLAB============
data = scipy.io.loadmat('Desktop/URECA/dataset_AI_students/particle_counts.mat')

particles = data['particle_counts']

n = len(particles)
particles = torch.tensor(particles, dtype = torch.int64).to(device)
#============================================

latent_dim = 512
batch_size = 1

lr = 1e-5
beta1 = 0.6
beta2 = 0.99
num_epochs = 15
loss_hist = []
min_loss = 0

def normalize(x):
    max_value = x.max()
    min_value = x.min()
    return (x - min_value) / (max_value - min_value)

def apply_threshold(tensor, threshold=0.5):
    # Apply threshold to generate binary tensor (0 or 1)
    tensor = normalize(tensor)
    binary_tensor = (tensor > threshold).float()
    
    return binary_tensor

class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(1, out_channels, self.modes, self.modes, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(1, out_channels, self.modes, self.modes, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes, :self.modes] = \
            self.compl_mul2d(x_ft[:, :, :self.modes, :self.modes], self.weights1)
        out_ft[:, :, -self.modes:, :self.modes] = \
            self.compl_mul2d(x_ft[:, :, -self.modes:, :self.modes], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


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

class FourierBlock(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(FourierBlock, self).__init__()

        """
        SpectralConv2d_fast -> PReLU -> SpectralConv2d (of same modes and channels) and 2 conv layers
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        self.sconv1 = SpectralConv2d_fast(self.in_channels, self.in_channels, modes)
        self.prelu = nn.PReLU()
        self.sconv2 = SpectralConv2d_fast(self.out_channels, self.in_channels, modes)

        self.conv1 = ConvLayer(self.in_channels, self.out_channels)
        self.conv2 = ConvLayer(self.in_channels, self.out_channels)

    def forward(self, x):
        ft1 = self.sconv1(x)
        prelu = self.prelu(ft1)
        ft2 = self.sconv2(x + prelu)

        conv1 = self.conv1(x)
        conv2 = self.conv2(ft2)

        return (conv1 + conv2)


class Up(nn.Module):
    """Concatenate, then upsampling, then double convolution -->in_channels//2"""
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

    def forward(self, x):
        x = self.up(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.out_conv = ConvLayer(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return normalize(self.out_conv(x))

class Counter(nn.Module):
    def __init__(self, latent_dim, in_channels):
        super(Counter, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.conv0 = nn.Conv2d(self.in_channels, 32, 3, 1, 1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv4 = nn.Conv2d(32, 32, 1, 1, 0)
        #end
        self.fc1 = nn.Linear(32 * latent_dim//2 * latent_dim//2, 64)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = self.conv0(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        merge = torch.cat((conv1, conv2), dim=1)
        conv3 = self.conv3(merge)
        conv4 = self.conv4(conv3)
        
        count = conv4.view(-1, 32 * latent_dim//2 * latent_dim//2)
        count = torch.relu(self.fc1(count))
        count = self.fc2(count)
        return count

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
        #image index to retrieve particle_counts's values
        idx_str = self.input_images_1[idx].split('_')[1].split('.')[0]
        image_index = int(idx_str)-1
        
        # Apply transformations
        if self.transform:
            input_image_1 = self.transform(input_image_1)
            input_image_2 = self.transform(input_image_2)
            target_image = self.transform(target_image)
            
        return input_image_1, input_image_2, target_image, image_index

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Lambda(lambda x: 1-x)
])


class FYNet(nn.Module):
    def __init__(self, latent_dim, n_channels, n_classes, bilinear=True):
        super(FYNet, self).__init__()
        self.latent_dim = latent_dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        #for Input 1
        self.conv1_1 = ConvLayer(self.n_channels, 32)
        self.conv2_1 = ConvLayer(32, 32)

        #for Input 2
        self.conv1_2 = ConvLayer(self.n_channels, 32)
        self.conv2_2 = ConvLayer(32, 32)
        #downsampling
        self.down = nn.MaxPool2d(2)
        self.fb1 = FourierBlock(64, 64, self.latent_dim//4)
        self.fb2 = FourierBlock(64, 64, self.latent_dim//4)

        #upsampling
        self.up = Up(64, 32, bilinear)
        self.conv3 = ConvLayer(32, 32)
        self.conv4 = ConvLayer(32, 32)

        self.fb3 = FourierBlock(64, 64, self.latent_dim//4)
        self.fb4 = FourierBlock(64, 32, self.latent_dim//4)

        #ending of reconstruction
        self.conv_end0 = ConvLayer(32*3, 8)
        self.conv_end1 = ConvLayer(8, 8)
        self.conv_end2 = ConvLayer(8, 8)

        self.outc = OutConv(8, self.n_classes)
        #ending of counting
        self.counter = Counter(self.latent_dim, 32*3)

    def forward(self, x1, x2):
        #for Input 1
        x1 = self.conv1_1(x1)
        x1 = self.conv2_1(x1)
        down0_1 = self.down(x1)

        #for Input 2
        x2 = self.conv1_2(x2)
        x2 = self.conv2_2(x2)
        down0_2 = self.down(x2)

        #concatenate
        merge1 = torch.cat((down0_1, down0_2), dim=1)
        fb1 = self.fb1(merge1)
        fb2 = self.fb2(fb1)

        #upsampling
        up = self.up(fb2)
        conv3 = self.conv3(up)
        conv4 = self.conv4(conv3)
        merge2 = torch.cat((up, conv4), dim=1)
        fb3 = self.fb3(merge2)
        x0 = self.fb4(fb3)

        #ending
        merge = torch.cat((x0, x1, x2), dim=1)
        x0 = self.conv_end0(merge)
        conv_end1 = self.conv_end1(x0)
        conv_end2 = self.conv_end2(x0 + conv_end1)

        x = self.outc(x0 + conv_end1 + conv_end2)
        x = apply_threshold(x, 0.4)
        count = self.counter(merge)
        count = torch.clamp(count, min=6, max=800)
        
        return x.to(device), count


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
        
#L2Loss for counting particles
class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()
    def forward(self, x, y):
        loss = ((x/y) -1)**2
        return loss

def main():
    #===================== LOADING DATA ===========================
    path_model = 'Desktop/URECA/GAN-main/models/FYNetC'
    path_output = 'Desktop/URECA/GAN-main/outputs/FYNetC'
    
    # Create dataset
    input_dir_1 = 'Desktop/URECA/dataset_AI_students_1/try/input1'
    input_dir_2 = 'Desktop/URECA/dataset_AI_students_1/try/input2'
    target_dir = 'Desktop/URECA/dataset_AI_students_1/try/target'
    dataset = ImageDataset(input_dir_1=input_dir_1, input_dir_2=input_dir_2, target_dir=target_dir, transform=transform)

    # Create DataLoader
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
   
    # Initialize generator and discriminator
    model = FYNet(latent_dim, 1, 1).to(device)
    # Loss functions
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    perceptual_loss = PerceptualLoss()
    counting_loss = L2Loss()
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
        for i, (x1, x2, yy, idx) in enumerate(data_loader):
            #data and its label
            x1 = x1.to(device)
            x2 = x2.to(device)
            yy = yy.to(device)
            gt_count = particles[idx]

            # -----------------
            #  Training
            # -----------------
            optimizer.zero_grad()
            # Generate a batch of images
            gen_images, counts = model(x1, x2)

            # Total loss
            loss = mse_loss(gen_images, yy) + bce_loss(gen_images, yy) + perceptual_loss(gen_images, yy) + 0*counting_loss(counts, gt_count)
            if(epoch > 5):
                loss += 0.99*counting_loss(counts, gt_count)
                
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
                    f"Training Loss: {loss.item():.4f}", f"Counting: {int(counts[0])}"
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
                            }, os.path.join(path_model,"checkpoint_FYNet_final.pth"))
            #save model
            torch.save({'epoch': epoch+start_ep,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'np_rand_state': np.random.get_state(),
                        'torch_rand_state': torch.get_rng_state(),
                        }, os.path.join(path_model,"checkpoint_FYNet.pth"))
            
    #visualize losses
    fig, axs = plt.subplots(1, 1, figsize=(10, 12))

    # Plot Training Loss
    axs.plot(range(1, num_epochs + 1), loss_hist, marker='o', linestyle='-', color='r')
    axs.set_title('Training Loss vs. Epochs')
    axs.set_xlabel('Epochs')
    axs.set_ylabel('Training Loss')
    axs.grid(True)


    plt.tight_layout()
    plt.savefig(path_output+f'/eps={num_epochs}_plot_FYNetC.png')
    plt.show()

if __name__ == '__main__':
    main()