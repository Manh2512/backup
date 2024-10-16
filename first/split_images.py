"""
Split the dataset_AI_students to train, val and test dataset.
"""
import os
import shutil
import random
from PIL import Image
from torchvision import transforms

def split_dataset(input_dir_1, input_dir_2, target_dir, output_dir, train_ratio=0.8, val_ratio=0.1):
    # Create directories for train and test splits
    train_input_dir_1 = os.path.join(output_dir, 'train', 'input1')
    train_input_dir_2 = os.path.join(output_dir, 'train', 'input2')
    train_target_dir = os.path.join(output_dir, 'train', 'target')
    
    val_input_dir_1 = os.path.join(output_dir, 'val', 'input1')
    val_input_dir_2 = os.path.join(output_dir, 'val', 'input2')
    val_target_dir = os.path.join(output_dir, 'val', 'target')
    
    test_input_dir_1 = os.path.join(output_dir, 'test', 'input1')
    test_input_dir_2 = os.path.join(output_dir, 'test', 'input2')
    test_target_dir = os.path.join(output_dir, 'test', 'target')

    os.makedirs(train_input_dir_1, exist_ok=True)
    os.makedirs(train_input_dir_2, exist_ok=True)
    os.makedirs(train_target_dir, exist_ok=True)
    os.makedirs(val_input_dir_1, exist_ok=True)
    os.makedirs(val_input_dir_2, exist_ok=True)
    os.makedirs(val_target_dir, exist_ok=True)
    os.makedirs(test_input_dir_1, exist_ok=True)
    os.makedirs(test_input_dir_2, exist_ok=True)
    os.makedirs(test_target_dir, exist_ok=True)

    # List of input and target files
    input_images_1 = sorted(os.listdir(input_dir_1))
    input_images_1 = [f for f in input_images_1 if f.endswith('png')]
    input_images_2 = sorted(os.listdir(input_dir_2))
    input_images_2 = [f for f in input_images_2 if f.endswith('png')]
    target_images = sorted(os.listdir(target_dir))
    target_images = [f for f in target_images if f.endswith('png')]

    # Ensure input and target have the same number of files
    assert len(input_images_1) == len(target_images), "Input and target folders must have the same number of images."

    # Shuffle and split the dataset
    combined = list(zip(input_images_1, input_images_2, target_images))
    random.shuffle(combined)
    split_train_idx = int(len(combined) * train_ratio)
    split_val_idx = int(len(combined) * val_ratio)
    train_set = combined[:split_train_idx]
    val_set = combined[split_train_idx:split_train_idx+split_val_idx]
    test_set = combined[split_train_idx+split_val_idx:]
    
    #resize_transform = transforms.Resize((256,256))

    def copy(image_path, output_path):
        img = Image.open(image_path)
        #resized = resize_transform(img)
        img.save(output_path)

    # Copy and resize images for the training set
    for input_img_1, input_img_2, target_img in train_set:
        copy(os.path.join(input_dir_1, input_img_1), os.path.join(train_input_dir_1, input_img_1))
        copy(os.path.join(input_dir_2, input_img_2), os.path.join(train_input_dir_2, input_img_2))
        copy(os.path.join(target_dir, target_img), os.path.join(train_target_dir, target_img))
	
	# Copy and resize images for the val set
    for input_img_1, input_img_2, target_img in val_set:
        copy(os.path.join(input_dir_1, input_img_1), os.path.join(val_input_dir_1, input_img_1))
        copy(os.path.join(input_dir_2, input_img_2), os.path.join(val_input_dir_2, input_img_2))
        copy(os.path.join(target_dir, target_img), os.path.join(val_target_dir, target_img))
    
    # Copy and resize images for the test set
    for input_img_1, input_img_2, target_img in test_set:
        copy(os.path.join(input_dir_1, input_img_1), os.path.join(test_input_dir_1, input_img_1))
        copy(os.path.join(input_dir_2, input_img_2), os.path.join(test_input_dir_2, input_img_2))
        copy(os.path.join(target_dir, target_img), os.path.join(test_target_dir, target_img))


    print(f"Split completed: {len(train_set)} train and {len(test_set)} test images.")

#usage
input_folder_1 = 'dataset_AI_students/Input1'
input_folder_2 = 'dataset_AI_students/Input2'
target_folder = 'dataset_AI_students/GT'
output_folder = 'dataset_AI_students'
split_dataset(input_folder_1, input_folder_2, target_folder, output_folder, train_ratio=0.7, val_ratio=0.15)
