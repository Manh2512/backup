import glob

import torch.optim as optim
import numpy as np
import scipy
import matplotlib.pyplot as plt
from network import *


torch.manual_seed(0)
np.random.seed(0)

latent_dim = 512
batch_size = 10

lr = 1e-6
beta1 = 0.5
beta2 = 0.9
num_epochs = 15

g_loss_hist = []
d_loss_hist = []

def apply_threshold(tensor, threshold=0.5):
    # Apply threshold to generate binary tensor (0 or 1)
    binary_tensor = (tensor > threshold).float()
    
    return binary_tensor


def main():
    #===================== LOADING DATA ===========================
    path_model = 'Desktop/URECA/GAN-main/models'
    path_output = 'Desktop/URECA/GAN-main/outputs/Sep17'
    
    # Create dataset
    input_dir = 'Desktop/URECA/dataset_AI_students/test/input'
    target_dir = 'Desktop/URECA/dataset_AI_students/test/target'
    dataset = ImageDataset(input_dir=input_dir, target_dir=target_dir, transform=transform)

    # Create DataLoader
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
   
    # Initialize generator and discriminator
    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)
    # Loss functions
    adversarial_loss = nn.BCELoss().to(device)
    mse_loss = nn.MSELoss().to(device)
    perceptual_loss = PerceptualLoss().to(device)
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

    start_ep = 0
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
        for i, (xx,yy) in enumerate(data_loader):
            #data and its label
            xx = xx.to(device)
            yy = yy.to(device)
            # Adversarial ground truths
            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)
    
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            # Generate a batch of images
            fake_images = generator(xx)
    
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
            gen_images = generator(xx)
            # Total loss
            g_loss = adversarial_loss(discriminator(gen_images), valid) + mse_loss(gen_images, yy) \
            + perceptual_loss(gen_images, yy)
            
            # Backward pass and optimize
            g_loss.backward()
            optimizer_G.step()
            # ---------------------
            #  Progress Monitoring
            # ---------------------
            if (i + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}]\
                            Batch {i+1}/{len(data_loader)} "
                    f"Discriminator Loss: {d_loss.item():.4f} "
                    f"Generator Loss: {g_loss.item():.4f}"
                )

        #save losses
        d_loss_hist.append(d_loss.item())
        g_loss_hist.append(g_loss.item())
        #save generator
        torch.save({'epoch': epoch+start_ep,
                    'model': generator.state_dict(),
                    'optimizer': optimizer_G.state_dict(),
                    'np_rand_state': np.random.get_state(),
                    'torch_rand_state': torch.get_rng_state(),
                    }, os.path.join(path_model,"checkpoint_G.pth"))
        #save discriminator
        torch.save({'epoch': epoch+start_ep,
                    'model': discriminator.state_dict(),
                    'optimizer': optimizer_D.state_dict(),
                    'np_rand_state': np.random.get_state(),
                    'torch_rand_state': torch.get_rng_state(),
                    }, os.path.join(path_model,"checkpoint_D.pth"))

    
    #visualize losses
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    # Plot Discrimination Loss
    axs[0].plot(range(1, num_epochs + 1), d_loss_hist, marker='o', linestyle='-', color='r')
    axs[0].set_title('Discrimination Loss vs. Epochs')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Discrimination Loss')
    axs[0].grid(True)

    # Plot Generation Loss
    axs[1].plot(range(1, num_epochs + 1), g_loss_hist, marker='o', linestyle='-', color='b')
    axs[1].set_title('Generation Loss vs. Epochs')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Generation Loss')
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(path_output+'/plot.png')
    plt.show()

if __name__ == '__main__':
    main()