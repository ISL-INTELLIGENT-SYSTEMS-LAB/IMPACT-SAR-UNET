import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model.model import UNET
from utils.utils import(
    set_seed,
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs
)

# hyperparameters, etc...
SEED = 42
LEARNING_RATE = 7e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 4
NUM_WORKERS = 8
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = '/home/mwilkerson1/IMPACT/Simulation/IMPACT-SAR-UNET/dataset/train'
TRAIN_MASK_DIR = '/home/mwilkerson1/IMPACT/Simulation/IMPACT-SAR-UNET/dataset/train_masks'
VAL_IMG_DIR = '/home/mwilkerson1/IMPACT/Simulation/IMPACT-SAR-UNET/dataset/val'
VAL_MASK_DIR = '/home/mwilkerson1/IMPACT/Simulation/IMPACT-SAR-UNET/dataset/val_masks'

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.long().to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            targets = torch.squeeze(targets, dim=1)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaled_loss = scaler.scale(loss)
        scaled_loss.backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

def main():
    set_seed(42)
    
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            #A.Rotate(limit=35, p=1.0),
            #A.HorizontalFlip(p=0.5),
            #A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.19884332, 0.13797273, 0.19884332],
                std=[0.16673183, 0.12461259, 0.16673183],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.19884332, 0.13797273, 0.19884332],
                std=[0.16673183, 0.12461259, 0.16673183],
                max_pixel_value=1.0,
            ),
            ToTensorV2(),
        ],
    )
    
    weights_dict = {
        0.0: 7.175651984937747,
        1.0: 2.2439208696066477,
        2.0: 0.3100294873821099,
        3.0: 6.44734926713217,
        4.0: 0.9667541130804129
        }
        
    weights_tensor = torch.tensor([weights_dict[i] for i in sorted(weights_dict)], dtype=torch.float).to(DEVICE)

    model = UNET(in_channels=3, out_channels=5).to(DEVICE) 
    loss_fn = nn.CrossEntropyLoss(weight=weights_tensor) 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-2)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY
    )
        
    # debugging
    '''for images, masks in train_loader:
        print("Images min/max:", images.min().item(), images.max().item())
        print("Masks min/max:", masks.min().item(), masks.max().item())
        # Check for NaN or Inf values
        assert not torch.isnan(images).any(), "Images have NaN values"
        assert not torch.isnan(masks).any(), "Masks have NaN values"
        assert not torch.isinf(images).any(), "Images have Inf values"
        assert not torch.isinf(masks).any(), "Masks have Inf values"'''
    
    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth"), model)
        check_accuracy(val_loader, model, device=DEVICE)
        
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save mode
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)
        # check accuracy
        check_accuracy(val_loader, model, num_classes=5, device=DEVICE)
        # print examples
        if epoch == (NUM_EPOCHS - 1):
            save_predictions_as_imgs(
                val_loader, model, folder="saved_images/", device=DEVICE
            )

if __name__=="__main__":
    main()
