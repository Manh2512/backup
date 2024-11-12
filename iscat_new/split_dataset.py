"""
Split the dataset_AI_students to train, val and test dataset.
"""
import os
import shutil
import random
from PIL import Image
from torchvision import transforms

mags = [20, 40, 60, 80, 100]

def split_dataset(input_dir, target_dir, output_dir, mags, train_ratio=0.75, val_ratio=0.15):
    #list of shuffled files
    numbers = list(range(1, 5001))
    random.shuffle(numbers)

    train_end = int(train_ratio * 5000)
    val_end = train_end + int(val_ratio * 5000)
    resize_transform = transforms.Resize((256,256))

    def copy(image_path, output_path):
        img = Image.open(image_path)
        resized = resize_transform(img)
        img.save(output_path)
        
    # Create directories for train and test splits
    #input
    for i in range(len(mags)):
        train_input_dir = os.path.join(output_dir, 'train', f'Mag_{mags[i]}')
        val_input_dir = os.path.join(output_dir, 'val', f'Mag_{mags[i]}')
        test_input_dir = os.path.join(output_dir, 'test', f'Mag_{mags[i]}')
        
        os.makedirs(train_input_dir, exist_ok=True)
        os.makedirs(val_input_dir, exist_ok=True)
        os.makedirs(test_input_dir, exist_ok=True)

        for j in range(train_end):
            copy(os.path.join(input_dir, f'Mag_{mags[i]}', f'Input_{numbers[j]:04d}.png'), os.path.join(train_input_dir, f'Input_{numbers[j]:04d}.png'))
        for j in range(train_end, val_end):
            copy(os.path.join(input_dir, f'Mag_{mags[i]}', f'Input_{numbers[j]:04d}.png'), os.path.join(val_input_dir, f'Input_{numbers[j]:04d}.png'))
        for j in range(val_end, 5000):
            copy(os.path.join(input_dir, f'Mag_{mags[i]}', f'Input_{numbers[j]:04d}.png'), os.path.join(test_input_dir, f'Input_{numbers[j]:04d}.png'))
            
    #ground truth
    train_gt_dir = os.path.join(output_dir, 'train', 'GT')
    val_gt_dir = os.path.join(output_dir, 'val', 'GT')
    test_gt_dir = os.path.join(output_dir, 'test', 'GT')
    
    os.makedirs(train_gt_dir, exist_ok=True)
    os.makedirs(val_gt_dir, exist_ok=True)
    os.makedirs(test_gt_dir, exist_ok=True)

    for j in range(train_end):
        copy(os.path.join(target_dir, f'GT_{numbers[j]:04d}.png'), os.path.join(train_gt_dir, f'GT_{train_set[i]:04d}.png'))
    for i in range(train_end, val_end):
        copy(os.path.join(target_dir, f'GT_{numbers[j]:04d}.png'), os.path.join(val_gt_dir, f'GT_{val_set[i]:04d}.png'))
    for i in range(val_end, 5000):
        copy(os.path.join(target_dir, f'GT_{numbers[j]:04d}.png'), os.path.join(test_gt_dir, f'GT_{test_set[i]:04d}.png'))

    print(f"Split completed: {train_end} train, {val_end-train_end} val images and {5000-val_end} test images.")

#usage
input_folder = 'ISCAT_dataset_5mag/Input'
target_folder = 'ISCAT_dataset_5mag/GT'
output_folder = 'dataset_split'
split_dataset(input_folder, target_folder, output_folder, mags, train_ratio=0.7, val_ratio=0.15)
