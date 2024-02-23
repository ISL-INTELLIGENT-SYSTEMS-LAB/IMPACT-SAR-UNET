import os
import random
import torch
import torchvision
from dataset.dataset import SARDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

COLOR_MAP = {
        0: (128, 0, 0),
        1: (222, 184, 135),
        2: (127, 255, 0),
        3: (173, 216, 230),
        4: (0, 191, 255),
}

def set_seed(seed_value):
    torch.manual_seed(seed_value)  # Set the seed for CPU
    torch.cuda.manual_seed(seed_value)  # Set the seed for all GPU devices
    torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups
    np.random.seed(seed_value)  # Seed for NumPy's random number generator
    random.seed(seed_value)  # Seed for Python's random module
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # Set Python hash seed

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(state, filename="best_model.pth"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

def get_loaders(
        train_dir,
        train_maskdir,
        val_dir,
        val_maskdir,
        batch_size,
        train_transform,
        val_transform,
        num_workers=4,
        pin_memory=True,
):
    train_ds = SARDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = SARDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, num_classes, device="cuda"):
    num_correct = 0
    num_pixels = 0
    conf_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x)  # Get the raw scores from the model
            preds = torch.softmax(preds, dim=1)  # Apply softmax to get probabilities
            preds = torch.argmax(preds, dim=1)  # Get the predicted class labels
            y = torch.squeeze(y)
            correct = (preds == y).sum().item()
            num_correct += correct  # Count correct predictions
            pixels = y.nelement()
            num_pixels += pixels  # Accumulate the total number of pixels

            for true_class in range(num_classes):
                for pred_class in range(num_classes):
                    conf_matrix[true_class, pred_class] += ((y == true_class) & (preds == pred_class)).sum().item()

        # Print total correct predictions and total pixels after all batches
        print(f'Total correct: {num_correct}')
        print(f'Total pixels: {num_pixels}')

    # Calculate per-class IoU
    ious = []
    for c in range(num_classes):
        true_positive = conf_matrix[c, c].item()
        false_positive = conf_matrix[:, c].sum().item() - true_positive
        false_negative = conf_matrix[c, :].sum().item() - true_positive
        denominator = true_positive + false_positive + false_negative
        if denominator > 0:
            iou = true_positive / denominator
            ious.append(iou)

    mean_iou = sum(ious) / len(ious) if ious else 0.0
    accuracy = num_correct / num_pixels * 100  # Multiply by 100 to get percentage

    # Calculate per-class IoU
    ious = []
    for c in range(num_classes):
        true_positive = conf_matrix[c, c].item()
        false_positive = conf_matrix[:, c].sum().item() - true_positive
        false_negative = conf_matrix[c, :].sum().item() - true_positive
        denominator = true_positive + false_positive + false_negative
        if denominator > 0:
            iou = true_positive / denominator
            ious.append(iou)

    mean_iou = sum(ious) / len(ious) if ious else 0.0
    accuracy = num_correct / num_pixels * 100  # Multiply by 100 to get percentage
    print(f'Val accuracy: {accuracy:.2f}%')
    print(f'Val Mean IoU: {mean_iou:.4f}')

    model.train()
    
    return mean_iou

def mask_to_rgb(mask, color_map):
    single_image = False
    if len(mask.shape) == 2:
        single_image = True
        mask = mask.unsqueeze(0)
    
    # Check if there is a channel dimension and squeeze it if necessary
    if mask.shape[1] == 1:
        mask = mask.squeeze(1)

    batch_size, height, width = mask.shape
    rgb_image = torch.zeros((batch_size, 3, height, width), dtype=torch.uint8)
    
    for class_label, color in color_map.items():
        mask_label = (mask == class_label)
        for channel in range(3):
            rgb_image[:, channel, :, :] += (mask_label * color[channel]).byte()
    
    if single_image:
        rgb_image = rgb_image.squeeze(0)
    
    return rgb_image

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    
    # Ensure the folder exists.
    os.makedirs(folder, exist_ok=True)
    
    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        
        with torch.no_grad():
            preds = model(x)
            preds = torch.argmax(preds, dim=1)  # Get the predicted class for each pixel
        
        preds_rgb = mask_to_rgb(preds.cpu(), COLOR_MAP)
        y_rgb = mask_to_rgb(y.cpu(), COLOR_MAP)
        
        for i in range(preds_rgb.size(0)):
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            axs[0].imshow(preds_rgb[i].numpy().transpose(1, 2, 0))
            axs[0].set_title('Prediction')
            axs[0].axis('off')

            axs[1].imshow(y_rgb[i].numpy().transpose(1, 2, 0))
            axs[1].set_title('Ground Truth')
            axs[1].axis('off')

            # Save the full figure
            plt.savefig(os.path.join(folder, f"comparison_{idx}_{i}.png"))
            plt.close(fig)