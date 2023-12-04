import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
import cv2

# Constants
DATASET_PATH = "Mini_BAGLS_dataset"
dataset_folder = Path(DATASET_PATH)
meta_files = [x for x in os.listdir(DATASET_PATH) if '.meta' in x]

# Randomly choose 4 samples from the dataset
chosen_samples = []
results = []

for _ in range(4):
    chosen_samples.append(random.choice(meta_files).split('.')[0])

# Process chosen samples
for sample_id in chosen_samples:
    # File paths
    img_path = Path(DATASET_PATH + '/' + sample_id + '.png')
    seg_path = Path(DATASET_PATH + '/' + sample_id + '_seg.png')
    meta_path = Path(DATASET_PATH + '/' + sample_id + '.meta')

    # Read image, segmentation, and metadata
    image = cv2.imread(str(img_path.absolute()))
    segmentation = cv2.imread(str(seg_path.absolute()))
    metadata = json.loads(meta_path.read_text())

    # Blend image and segmentation
    alpha = 0.40
    beta = 1 - alpha
    blended_image = cv2.addWeighted(image, alpha, segmentation, beta, 0.0)

    # Extract subject disorder status from metadata
    disorder_status = metadata['Subject disorder status']

    # Store the results
    results.append((sample_id, blended_image, disorder_status))

# Display the results in a 2x2 grid
fig = plt.figure()
for i in range(0, len(results)):
    sample_id = results[i][0]
    blended_img = results[i][1]
    disorder_status = results[i][2]

    plt.subplot(2, 2, i + 1)
    plt.title('Sample ' + sample_id + ' - ' + disorder_status)
    plt.imshow(blended_img)
plt.show()

# Read an image from file
original_img = cv2.imread('leaves.jpg')

# Get image dimensions
height = original_img.shape[0]
width = original_img.shape[1]

# Create three copies of the original image
img_copy_1 = original_img.copy()
img_copy_2 = original_img.copy()
img_copy_3 = original_img.copy()

# Convert each copy to a different grayscale representation
for y in range(0, height):
    for x in range(0, width):
        pixel = original_img[y, x]
        blue = pixel[0]
        green = pixel[1]
        red = pixel[2]

        # Grayscale conversions
        res_1 = (min(red, green, blue) + max(red, green, blue)) / 2
        res_2 = (red + green + blue) / 3
        res_3 = 0.2989 * red + 0.5870 * green + 0.1140 * blue

        # Update pixel values in each copy
        img_copy_1[y, x] = res_1
        img_copy_2[y, x] = res_2
        img_copy_3[y, x] = res_3

# Display the three grayscale representations in a vertical stack
fig2 = plt.figure()
plt.subplot(3, 1, 1)
plt.imshow(img_copy_1)
plt.subplot(3, 1, 2)
plt.imshow(img_copy_2)
plt.subplot(3, 1, 3)
plt.imshow(img_copy_3)
plt.show()
