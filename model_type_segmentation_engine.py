# --------------------------------------
# MODEL TRAINING PIPELINE: ROOF SEGMENTATION (U-NET Style DeepLabV3+)
# --------------------------------------

# This script trains a segmentation model (DeepLabV3+ with ResNet101 backbone) on a dataset of roof images and masks.
# Importing all required libraries

import os
import zipfile
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# --------------------------------------
# CONFIGURATION SECTION
# --------------------------------------

# Location of the zipped dataset (contains folders with image and mask)
ZIP_PATH = "roof_masks.zip" 
EXTRACT_PATH = "roof_dataset_augmented"
AUGMENTED_PATH = "augmented_roof_dataset"
# MODEL_SAVE_PATH = "roof_deeplabv3plus_resnet101_augmented.pth"
MODEL_SAVE_PATH = "best_roof_path.pth"
BATCH_SIZE = 4
EPOCHS = 20
TRAIN_RATIO, VAL_RATIO = 0.75, 0.15


# --------------------------------------
#  DATA EXTRACTION STEP
# --------------------------------------

# Extract ZIP if not already done
if not os.path.exists(EXTRACT_PATH):
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_PATH)
    print(" ZIP extracted.")
else:
    print(" Extract path already exists, skipping extraction.")

# --------------------------------------
#  DATA AUGMENTATION STEP
# --------------------------------------

os.makedirs(AUGMENTED_PATH, exist_ok=True)

# Defines a list of augmentations using Albumentations package
# (This helps improve generalization and mimic real-world variations)

augmentations = [
    A.HorizontalFlip(p=1),
    A.VerticalFlip(p=1),
    A.Rotate(limit=30, p=1),
    A.RandomScale(scale_limit=0.2, p=1),
    A.GaussNoise(p=1),
    A.RandomBrightnessContrast(p=1),
    A.Affine(scale=(0.9, 1.1), translate_percent=0.1, shear=15, rotate=(-30, 30), p=1),
]

# Loop through each folder (each sample)

for folder in os.listdir(EXTRACT_PATH):
    fpath = os.path.join(EXTRACT_PATH, folder)
    img_path = os.path.join(fpath, "img.png")
    mask_path = os.path.join(fpath, "label.png")

    # If any image or mask is missing, skip this folder
    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        continue
    # Load image and corresponding mask
    image = np.array(Image.open(img_path).convert("RGB"))
    mask = np.array(Image.open(mask_path))
    mask = (mask > 0).astype(np.uint8)

    # Save original image and mask into augmented folder
    out_base = os.path.join(AUGMENTED_PATH, f"{folder}_orig")
    os.makedirs(out_base, exist_ok=True)
    Image.fromarray(image).save(os.path.join(out_base, "img.png"))
    Image.fromarray(mask * 255).save(os.path.join(out_base, "label.png"))

    # Apply all augmentations and save each variation
    for i, aug in enumerate(augmentations):
        augmented = aug(image=image, mask=mask)
        aug_img, aug_mask = augmented["image"], augmented["mask"]

        aug_dir = os.path.join(AUGMENTED_PATH, f"{folder}_aug{i+1}")
        os.makedirs(aug_dir, exist_ok=True)
        Image.fromarray(aug_img).save(os.path.join(aug_dir, "img.png"))
        Image.fromarray(aug_mask * 255).save(os.path.join(aug_dir, "label.png"))

print(" Augmentation completed.")

# --------------------------------------
#  DATASET CLASS DEFINITION
# --------------------------------------

# This class handles loading images and masks, applying transformations, and preparing data for training.

def get_transform():
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

# Custom PyTorch Dataset class for Roof Segmentation dataset
class RoofDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        # Walk through all folders and gather image+mask pairs
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            img_path = os.path.join(folder_path, "img.png")
            mask_path = os.path.join(folder_path, "label.png")
            if os.path.exists(img_path) and os.path.exists(mask_path):
                self.samples.append((img_path, mask_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        mask = (mask > 0).astype("float32")
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].unsqueeze(0)
        return image, mask

# --------------------------------------
#  SPLIT DATASET INTO TRAIN/VAL/TEST
# --------------------------------------

# Load entire dataset
dataset = RoofDataset(AUGMENTED_PATH, transform=get_transform())

# Calculate split sizes
train_len = int(TRAIN_RATIO * len(dataset))
val_len = int(VAL_RATIO * len(dataset))
test_len = len(dataset) - train_len - val_len
train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])

# Wrap into DataLoaders (efficient batch processing)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

print(f" Dataset split — Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

# --------------------------------------
#  MODEL TRAINING SETUP
# --------------------------------------

# Use GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load DeepLabV3+ model architecture with ResNet-101 encoder (pre-trained on ImageNet)

model = smp.DeepLabV3Plus(
    encoder_name="resnet101", # Backbone architecture
    encoder_weights="imagenet",  # Use pretrained encoder weights
    in_channels=3,  # RGB input
    classes=1 # Output: single-channel binary mask
).to(device)

# Loss functions:
# Dice Loss — good for imbalanced datasets (segmentation tasks)
# Focal Loss — focuses on hard-to-classify pixels
dice_loss = smp.losses.DiceLoss(mode="binary")
focal_loss = smp.losses.FocalLoss(mode="binary")

# Optimizer (Adam — commonly used for segmentation tasks)1
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# --------------------------------------
#  TRAINING LOOP
# --------------------------------------

for epoch in range(EPOCHS):
    model.train() # Put model into training mode
    total_loss = 0.0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs) # Forward pass — generate predicted masks

        # Compute combined loss: Dice + Focal
        loss = dice_loss(preds, masks) + focal_loss(preds, masks)
        optimizer.zero_grad() # Reset gradients before backprop
        loss.backward() # Backpropagation (compute gradients)
        optimizer.step() # Update model weights

        # Accumulate loss for this batch
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f" Epoch {epoch+1}/{EPOCHS} — Loss: {avg_loss:.4f}")

# --------------------------------------
#  SAVE THE FINAL TRAINED MODEL
# --------------------------------------
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f" Model saved: {MODEL_SAVE_PATH}")
