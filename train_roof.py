# models/train_roof.py

import os, random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF, InterpolationMode
import segmentation_models_pytorch as smp

# NEW (after moving train_roof.py to project root)
from models.model_utils import RoofDataset, pad_to_square_pil, compute_iou
from overlay_visualizer import run_stage3_overlay




# Reproducibility & Device
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Constants
TARGET_SIZE = (256, 256)
BATCH_SIZE = 4
NUM_EPOCHS = 200
PATIENCE = 150

class FocalDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal = smp.losses.FocalLoss('binary', gamma=2)
        self.dice  = smp.losses.DiceLoss(mode='binary')
    def forward(self, preds, targets):
        return self.focal(preds, targets) + self.dice(preds, targets)

if __name__ == '__main__':
    print("✅ Imports working. Starting training...")

    # Paths
    root_dir = "./dataset/Images_data"
    all_folders = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
    random.shuffle(all_folders)
    n = len(all_folders)
    train_f, val_f, test_f = np.split(all_folders, [int(.7*n), int(.85*n)])

    train_ds = RoofDataset(train_f, augment=True)
    val_ds = RoofDataset(val_f, augment=False)
    test_ds = RoofDataset(test_f, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=1, activation="sigmoid").to(device)
    loss_fn = FocalDiceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    best_iou = 0.0
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_iou': []}
    start_epoch = 1

    checkpoint_path = 'models/checkpoint_256.pth'
    model_path = 'models/best_model.pth'

    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            best_iou = checkpoint['best_iou']
            history = checkpoint['history']
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming training from epoch {start_epoch}")
        except Exception as e:
            print(f"Failed to load checkpoint. Starting fresh. Reason: {e}")

    for epoch in range(start_epoch, NUM_EPOCHS+1):
        model.train()
        tloss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = loss_fn(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tloss += loss.item()
        tloss /= len(train_loader)

        model.eval()
        vloss, ious = 0, []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                vloss += loss_fn(preds, masks).item()
                ious.append(compute_iou(preds, masks))
        vloss /= len(val_loader)
        viou = np.mean(ious)
        scheduler.step(vloss)

        history['train_loss'].append(tloss)
        history['val_loss'].append(vloss)
        history['val_iou'].append(viou)

        print(f"Epoch {epoch:03d} | Train Loss: {tloss:.4f} | Val Loss: {vloss:.4f} | Val IoU: {viou:.4f}")

        if viou > best_iou + 1e-4:
            best_iou = viou
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_iou': best_iou,
                'history': history
            }, checkpoint_path)
            torch.save(model.state_dict(), model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch}. Best IoU: {best_iou:.4f}")
                break


    run_stage3_overlay(test_loader)
