import numpy as np
import cv2
import os
import csv
from image_processing import func

# Creating necessary directories if they don't exist
if not os.path.exists("data"):
    os.makedirs("data")
if not os.path.exists("data/train"):
    os.makedirs("data/train")
if not os.path.exists("data/test"):
    os.makedirs("data/test")

# Define paths and labels
input_path = "data/train"  # original input directory
output_path = "data/test"  # updated to "data"
a = ['label'] + [f"pixel{i}" for i in range(64 * 64)]  # creating pixel column names

label = 0
total_images = 0
train_count = 0
test_count = 0

# Processing images and categorizing them into train/test sets
for dirpath, dirnames, filenames in os.walk(input_path):
    for dirname in dirnames:
        print(dirname)
        train_dir = os.path.join(output_path, "train", dirname)
        test_dir = os.path.join(output_path, "test", dirname)

        # Creating subdirectories within train and test folders
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        num_train = 100000000000000000  # a high value to treat all images as training by default
        i = 0

        # Walking through each file in the directory
        for file in os.listdir(os.path.join(input_path, dirname)):
            total_images += 1
            actual_path = os.path.join(input_path, dirname, file)
            train_path = os.path.join(train_dir, file)
            test_path = os.path.join(test_dir, file)

            # Read image and process
            img = cv2.imread(actual_path, 0)  # Read in grayscale
            bw_image = func(actual_path)  # Apply function from image_processing module

            # Save the processed image in train or test folder
            if i < num_train:
                train_count += 1
                cv2.imwrite(train_path, bw_image)
            else:
                test_count += 1
                cv2.imwrite(test_path, bw_image)

            i += 1
        label += 1

# Print counts
print("Total images processed:", total_images)
print("Training images:", train_count)
print("Testing images:", test_count)
