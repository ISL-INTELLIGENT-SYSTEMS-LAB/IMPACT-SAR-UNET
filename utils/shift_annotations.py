import numpy as np
from PIL import Image
import os

annotation_dir = '/home/mwilkerson1/IMPACT/Simulation/IMPACT-SAR-UNET/dataset/train_masks/'

for filename in os.listdir(annotation_dir):
    if filename.endswith('.png'):  
        file_path = os.path.join(annotation_dir, filename)
        image = Image.open(file_path)
        image_array = np.array(image)

        if len(image_array.shape) == 2:
            shifted_image_array = image_array - 1
        else:
            shifted_image_array = image_array.copy()
            shifted_image_array[:, :, 0] = image_array[:, :, 0] - 1

        shifted_image_array = np.clip(shifted_image_array, 0, 5)
        shifted_image = Image.fromarray(shifted_image_array.astype(np.uint8))
        shifted_image.save(file_path)

print("Annotation values have been shifted and original images have been replaced.")
