import torch
import torchvision
from dataset.dataset import SARDataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

COLOR_MAP = {
        0: (128, 0, 0),
        1: (222, 184, 135),
        2: (127, 255, 0),
        3: (173, 216, 230),
        4: (0, 191, 255),
}

def save_checkpoint(state, filename="my_checkpoint.pth"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

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

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()  
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds) 
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )  

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}%")
    print(f"Dice score: {dice_score:.6f}")
    model.train()

def mask_to_rgb(mask, color_map):

    single_image = False
    if len(mask.shape) == 2:
        single_image = True
        mask = mask.unsqueeze(0)

    rgb_image = torch.zeros(mask.size(0), 3, mask.size(1), mask.size(2), dtype=torch.uint8)
    
    for class_label, color in color_map.items():
        mask_label = mask == class_label

        for i, channel_color in enumerate(color):
            rgb_image[:, i, :, :] += (mask_label * channel_color).byte()
    
    if single_image:
        rgb_image = rgb_image.squeeze(0)
    
    return rgb_image

def save_predictions_as_imgs(loader, model, folder="save_images/", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = model(x)
            preds = torch.argmax(preds, dim=1)  
        
        preds_rgb = mask_to_rgb(preds.cpu(), COLOR_MAP)
        
        for i in range(preds_rgb.size(0)):
            img = Image.fromarray(preds_rgb[i].numpy().transpose(1, 2, 0))
            img.save(f"{folder}/pred_{idx}_{i}.png")