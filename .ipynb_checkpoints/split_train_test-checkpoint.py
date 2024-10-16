import os
import shutil
from sklearn.model_selection import train_test_split

# Function to create directories if they don't exist
def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Directories for input and target images
input_dir = 'Desktop/URECA/dataset_AI_students/Input2'
target_dir = 'Desktop/URECA/dataset_AI_students/GT'

# Load and sort image file names
input_images = sorted(os.listdir(input_dir))
target_images = sorted(os.listdir(target_dir))

# Check that input and target files match
assert len(input_images) == len(target_images), "Input and Target files count mismatch!"

input_train, input_test, target_train, target_test = train_test_split(
    input_images, target_images, test_size=0.2, random_state=42)

# Create directories for train/test splits
train_input_dir = 'Desktop/URECA/dataset_AI_students/train/input'
train_target_dir = 'Desktop/URECA/dataset_AI_students/train/target'
test_input_dir = 'Desktop/URECA/dataset_AI_students/test/input'
test_target_dir = 'Desktop/URECA/dataset_AI_students/test/target'

create_dir(train_input_dir)
create_dir(train_target_dir)
create_dir(test_input_dir)
create_dir(test_target_dir)

# Function to copy images to the corresponding train/test directories
def copy_images(image_list, source_dir, dest_dir):
    for image in image_list:
        shutil.copy(os.path.join(source_dir, image), os.path.join(dest_dir, image))

# Copy images to train/test directories
copy_images(input_train, input_dir, train_input_dir)
copy_images(target_train, target_dir, train_target_dir)
copy_images(input_test, input_dir, test_input_dir)
copy_images(target_test, target_dir, test_target_dir)

print("Dataset split completed with images copied!")
