import os
import shutil
from sklearn.model_selection import train_test_split

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def split_data(image_dir, mask_dir, train_img_dir, train_mask_dir, val_img_dir, val_mask_dir, test_img_dir, test_mask_dir, train_size=0.8889, test_size=0.1):
    """Splits the images and masks into training, validation, and test sets and renames masks."""

    create_dir(train_img_dir)
    create_dir(train_mask_dir)
    create_dir(val_img_dir)
    create_dir(val_mask_dir)
    create_dir(test_img_dir)
    create_dir(test_mask_dir)

    images = os.listdir(image_dir)
    masks = os.listdir(mask_dir)
    
    images.sort()
    masks.sort()

    # First split to separate out the test set
    initial_train_images, test_images, initial_train_masks, test_masks = train_test_split(images, masks, test_size=test_size, random_state=42)
    
    # Second split to separate out the validation set from the remaining images
    train_images, val_images, train_masks, val_masks = train_test_split(initial_train_images, initial_train_masks, train_size=train_size, random_state=42)
    
    def rename_mask(mask):
        return mask.replace('encoded_seg', 'converted_RGB')

    # Move files into their respective directories
    for img, msk in zip(train_images, train_masks):
        shutil.move(os.path.join(image_dir, img), os.path.join(train_img_dir, img))
        new_msk_name = rename_mask(msk)
        shutil.move(os.path.join(mask_dir, msk), os.path.join(train_mask_dir, new_msk_name))

    for img, msk in zip(val_images, val_masks):
        shutil.move(os.path.join(image_dir, img), os.path.join(val_img_dir, img))
        new_msk_name = rename_mask(msk)
        shutil.move(os.path.join(mask_dir, msk), os.path.join(val_mask_dir, new_msk_name))

    for img, msk in zip(test_images, test_masks):
        shutil.move(os.path.join(image_dir, img), os.path.join(test_img_dir, img))
        new_msk_name = rename_mask(msk)
        shutil.move(os.path.join(mask_dir, msk), os.path.join(test_mask_dir, new_msk_name))

image_dir = '/home/mwilkerson1/IMPACT/Simulation/Data_Collection/Sentinal_SARS_RGB3'
mask_dir = '/home/mwilkerson1/IMPACT/Simulation/Data_Collection/Sentinal_SARS_encoded'
train_img_dir = '/home/mwilkerson1/IMPACT/Simulation/IMPACT-SAR-UNET/dataset/train'
train_mask_dir = '/home/mwilkerson1/IMPACT/Simulation/IMPACT-SAR-UNET/dataset/train_masks'
val_img_dir = '/home/mwilkerson1/IMPACT/Simulation/IMPACT-SAR-UNET/dataset/val'
val_mask_dir = '/home/mwilkerson1/IMPACT/Simulation/IMPACT-SAR-UNET/dataset/val_masks'
test_img_dir = '/home/mwilkerson1/IMPACT/Simulation/IMPACT-SAR-UNET/dataset/test'
test_mask_dir = '/home/mwilkerson1/IMPACT/Simulation/IMPACT-SAR-UNET/dataset/test_masks'

split_data(image_dir, mask_dir, train_img_dir, train_mask_dir, val_img_dir, val_mask_dir, test_img_dir, test_mask_dir)
