from PIL import Image
import numpy as np
import os

def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            with Image.open(img_path) as img:
                images.append(np.array(img.convert('RGB')))
        except IOError:
            # Handle files that cannot be opened as images here
            print(f"Skipping file (not an image?): {filename}")
    return images

folder_path = '/home/mwilkerson1/IMPACT/Simulation/IMPACT-SAR-UNET/dataset/train/'
all_images = load_images(folder_path)

# Convert to a NumPy array and scale the pixel values to [0, 1]
all_images = np.stack(all_images)

# Calculate the mean and std dev per channel
mean = all_images.mean(axis=(0, 1, 2))
std = all_images.std(axis=(0, 1, 2))

print("Mean: ", mean)
print("Std: ", std)