"""
Split iSCAT_data to train and validation and add Gaussian noise to images.
"""
import os
import numpy as np
import shutil
import random
from PIL import Image
from torchvision import transforms

def add_gaussian_noise(image, mean=0, std=10):
    image_np = np.array(image)
    noise = np.random.normal(mean, std, image_np.shape).astype(np.uint8)
    noisy_image_np = np.clip(image_np + 0.05*noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image_np)

def split_and_resize(input_dir, target_dir, output_dir, train_ratio=0.75):
    # Create directories for train and test splits
    train_input_dir = os.path.join(output_dir, 'train', 'Input')
    train_target_dir = os.path.join(output_dir, 'train', 'GT')
    
    val_input_dir = os.path.join(output_dir, 'val', 'Input')
    val_target_dir = os.path.join(output_dir, 'val', 'GT')

    os.makedirs(train_input_dir, exist_ok=True)
    os.makedirs(train_target_dir, exist_ok=True)
    os.makedirs(val_input_dir, exist_ok=True)
    os.makedirs(val_target_dir, exist_ok=True)

    # List of input and target files
    input_images = sorted(os.listdir(input_dir))
	input_images = [f for f in input_images if f.endswith('png')]
    target_images = sorted(os.listdir(target_dir))
	target_images = [f for f in target_images if f.endswith('png')]

    # Ensure input and target have the same number of files
    assert len(input_images) == len(target_images), "Input and target folders must have the same number of images."

    # Shuffle and split the dataset
    combined = list(zip(input_images, target_images))
    random.shuffle(combined)
    split_train_idx = int(len(combined) * train_ratio)
    train_set = combined[:split_train_idx]
    val_set = combined[split_train_idx:]
    resize_transform = transforms.Resize((512,512))

    def copy(image_path, output_path):
        img = Image.open(image_path)
        img = add_gaussian_noise(img, 0, 10)
        img = resize_transform(img)
        img.save(output_path)

    # Copy and resize images for the training set
    for input_img, target_img in train_set:
        copy(os.path.join(input_dir, input_img), os.path.join(train_input_dir, input_img))
        copy(os.path.join(target_dir, target_img), os.path.join(train_target_dir, target_img))
	
    # Copy and resize images for the val set
    for input_img, target_img in val_set:
        copy(os.path.join(input_dir, input_img), os.path.join(val_input_dir, input_img))
        copy(os.path.join(target_dir, target_img), os.path.join(val_target_dir, target_img))

    print(f"Split completed: {len(train_set)} train and {len(val_set)} validating images.")

#usage
input_folder = 'iSCAT_data/Input'
target_folder = 'iSCAT_data/GT'
output_folder = 'iSCAT_processed'
split_and_resize(input_folder, target_folder, output_folder, train_ratio=0.75)
