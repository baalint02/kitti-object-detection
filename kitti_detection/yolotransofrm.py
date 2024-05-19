import config
import os
import random
from sklearn.model_selection import train_test_split
import shutil

image_width = 1242
image_height = 375

object_type_to_index = {
    'Car': 0,
    'Van': 1,
    'Truck': 2,
    'Pedestrian': 3,
    'Person_sitting': 4,
    'Cyclist': 5,
    'Tram': 6,
    'Misc': 7,
    'DontCare': 8
}

#converting to be acceptable for yolov8
def convert_bbox(left, top, right, bottom):
    x = (left+right)/2
    y = (top+bottom)/2
    width = right - left
    height = bottom - top
    return x, y, width, height

#normalizing, this is yolov8 input requirement
def normalize_bbox(x, y, width, height, img_width, img_height):
    x_norm = x / img_width
    y_norm = y / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    return x_norm, y_norm, width_norm, height_norm

#moving train test and valid images
def move_images(label_base_dir, image_base_dir, output_image_base_dir, subdirectories):
    for subdir in subdirectories:
        label_dir = os.path.join(label_base_dir, subdir)
        output_image_dir = os.path.join(output_image_base_dir, subdir)
        
        for filename in os.listdir(label_dir):
            if filename.endswith('.txt'):
                image_filename = filename.replace('.txt', '.png')  
                src_image_path = os.path.join(image_base_dir, image_filename)
                dst_image_path = os.path.join(output_image_dir, image_filename)
                
                if os.path.exists(src_image_path):
                    shutil.move(src_image_path, dst_image_path)
                else:
                    print(f"Image {image_filename} not found in {image_base_dir}")


#modify these to your directory names in config.py
label_dir = config.DATA_LABEL_DIR_PATH
output_dir = config.DATA_LABEL_NEW_PATH
image_base_dir = config.DATA_IMAGE_DIR_PATH
output_image_base_dir = config.DATA_IMAGE_NEW_PATH

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

subdirectories = ['train', 'test', 'valid']
for subdir in subdirectories:
    os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

# Read all filenames
filenames = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

# Split filenames into train, test, and valid sets
train_files, temp_files = train_test_split(filenames, test_size=0.3, random_state=42)
test_files, valid_files = train_test_split(temp_files, test_size=0.5, random_state=42)

splits = {
    'train': train_files,
    'test': test_files,
    'valid': valid_files
}

# Process each split
for split, files in splits.items():
    for filename in files:
        input_file_path = os.path.join(label_dir, filename)
        output_file_path = os.path.join(output_dir, split, filename)
        
        with open(input_file_path, 'r') as file, open(output_file_path, 'w') as output_file:
            for line in file:
                parts = line.strip().split()
                object_type = parts[0]
                left = float(parts[4])
                top = float(parts[5])
                right = float(parts[6])
                bottom = float(parts[7])
                
                #Converting object type to numbers
                if object_type in object_type_to_index:
                    object_type_idx = object_type_to_index[object_type]
                else:
                    continue
                
                # Convert bbox format and normalize
                x, y, width, height = convert_bbox(left, top, right, bottom)
                x_norm, y_norm, width_norm, height_norm = normalize_bbox(x, y, width, height, image_width, image_height)
                
                #Writing the new file
                new_line = f"{object_type_idx} {x_norm:.6f} {y_norm:.6f} {width_norm:.6f} {height_norm:.6f} " + '\n'
                output_file.write(new_line)
#after train test split is given for labels, the corrensponding images are moved
move_images(output_dir, image_base_dir, output_image_base_dir, subdirectories)