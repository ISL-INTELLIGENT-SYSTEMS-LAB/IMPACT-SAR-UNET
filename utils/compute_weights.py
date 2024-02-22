import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data import DataLoader

from utils import get_loaders

TRAIN_IMG_DIR = '/home/mwilkerson1/IMPACT/Simulation/IMPACT-SAR-UNET/dataset/train'
TRAIN_MASK_DIR = '/home/mwilkerson1/IMPACT/Simulation/IMPACT-SAR-UNET/dataset/train_masks'
VAL_IMG_DIR = '/home/mwilkerson1/IMPACT/Simulation/IMPACT-SAR-UNET/dataset/val'
VAL_MASK_DIR = '/home/mwilkerson1/IMPACT/Simulation/IMPACT-SAR-UNET/dataset/val_masks'

train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE=1,
    )

def compute_weights(train_loader):
    segmentation_map = []
    for sample in train_loader:
        seg_map = sample['annotation']  # Assuming 'annotation' is the key for segmentation maps
        seg_map = seg_map.cpu().numpy()  # If your annotations are on GPU, move them to CPU and convert to numpy
        segmentation_map.extend(seg_map.flatten())

    y = np.array(segmentation_map)  # Convert the list of labels to a numpy array

    unique, counts = np.unique(y, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    print(f"Class distribution in the dataset: {class_distribution}")

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    print(f"Computed class weights: {class_weights}")

    return class_weights

if __name__ == "__main__":
    class_weights = compute_weights(train_loader)
    print(class_weights)