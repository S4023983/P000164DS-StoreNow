# DeepLabV3+ with ResNet-101 and data augmentation

# ===== Here its best to run inside VM, Dependency Installation (Run in terminal before running this script) =====
# python -m venv venv
# .\venv\Scripts\activate   (on Windows)
# python -m pip install --upgrade pip
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install segmentation-models-pytorch matplotlib pillow

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import segmentation_models_pytorch as smp

# Dataset class
class RoofDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.folders = [os.path.join(root_dir, f) for f in os.listdir(root_dir)
                        if os.path.isdir(os.path.join(root_dir, f))]

        self.label_mapping = {
            "_background_": "background", "roof": "roof",
            "roofline": "roof_line", "roof line": "roof_line",
            "vent": "vent", "vents": "vent", "vents'": "vent",
            "window": "window", "chimney": "chimney",
            "bend": "bend", "wall": "wall"
        }
        self.roof_labels = {'roof', 'roof_line', 'chimney', 'vent', 'bend'} # Labels that indicate roof obstruction. Mapping is done here.
        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
        ])

    def normalize_labels(self, labels):
        return [self.label_mapping.get(label.strip().lower(), "other") for label in labels]

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        folder = self.folders[idx]
        img = Image.open(os.path.join(folder, 'img.png')).convert('RGB')
        mask = Image.open(os.path.join(folder, 'label.png')).convert('L')
        labels_file = os.path.join(folder, 'label_names.txt')

        if os.path.exists(labels_file):
            with open(labels_file, 'r') as f:
                raw_labels = f.read().splitlines()
            labels = self.normalize_labels(raw_labels)
        else:
            labels = ['background']

        label_is_roof = any(lbl in self.roof_labels for lbl in labels)

        img = img.resize((128, 128))
        mask = mask.resize((128, 128))

        img = self.augment(img)
        mask = self.augment(mask)

        image = TF.to_tensor(img)
        mask = mask.resize((128, 128))
        mask = np.array(mask)
        mask = (mask > 25).astype(np.uint8) if label_is_roof else np.zeros_like(mask)
        mask = torch.from_numpy(mask).long()  # shape: [H, W], type: long (int64)


        return image, mask

# Loss functions
class DiceLoss(nn.Module):
    def __init__(self, smooth=1): super().__init__(); self.smooth = smooth
    def forward(self, p, t):
        p, t = p.contiguous().view(-1), t.contiguous().view(-1)
        intersection = (p * t).sum()
        return 1 - (2. * intersection + self.smooth) / (p.sum() + t.sum() + self.smooth)

class BCEDiceLoss(nn.Module):
    def __init__(self): super().__init__(); self.bce = nn.BCELoss(); self.dice = DiceLoss()
    def forward(self, p, t): return self.bce(p, t) + self.dice(p, t)

# IoU metric
def compute_iou(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum(dim=(1,2,3))
    union = ((preds + targets) >= 1).float().sum(dim=(1,2,3))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()

# Setup
root_dir = "dataset/New folder (4)"
dataset = RoofDataset(root_dir)
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=2)
test_loader = DataLoader(test_ds, batch_size=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# DeepLabV3+ Model
model = smp.DeepLabV3Plus(
    encoder_name="resnet101",
    encoder_weights="imagenet",
    in_channels=3,
    # classes=1,
    classes = 2,
    # activation="sigmoid"
    activation = None
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Training
num_epochs = 100
train_losses, val_losses, val_ious = [], [], []
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    for batch_idx, (images, masks) in enumerate(train_loader):
        images, masks = images.to(device), masks.to(device)
        preds = model(images)
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

    train_losses.append(total_train_loss / len(train_loader))

    model.eval()
    total_val_loss = 0
    iou_scores = []
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            total_val_loss += criterion(preds, masks).item()
            iou_scores.append(compute_iou(preds, masks))
    val_losses.append(total_val_loss / len(val_loader))
    val_ious.append(np.mean(iou_scores))

    print(f"Epoch {epoch+1:2d} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f} | Val IoU: {val_ious[-1]:.4f}")

    if epoch == 0 or val_ious[-1] > max(val_ious[:-1]):
        os.makedirs("saved_models", exist_ok=True)
        torch.save(model.state_dict(), "saved_models/best_deeplabv3plus.pth")
        print(f"Model saved at epoch {epoch+1} with Val IoU: {val_ious[-1]:.4f}")

    torch.cuda.empty_cache()

# Plot loss & IoU
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("Loss Curve")
plt.legend()

plt.subplot(1,2,2)
plt.plot(val_ious, label="Val IoU", color='green')
plt.title("Validation IoU")
plt.legend()
plt.tight_layout()
plt.show()

# Sample predictions
model.eval()
num_samples = min(5, len(test_loader))
samples = random.sample(list(test_loader), num_samples)

plt.figure(figsize=(15, 15))
for i, (images, masks) in enumerate(samples):
    images, masks = images.to(device), masks.to(device)
    with torch.no_grad():
        preds = model(images)
    preds_binary = (preds > 0.5).float()

    plt.subplot(5, 3, i * 3 + 1)
    plt.imshow(images[0].permute(1, 2, 0).cpu())
    plt.title("Input Image")
    plt.axis('off')

    plt.subplot(5, 3, i * 3 + 2)
    print("masks.shape:", masks.shape)
    print("masks[0].shape:", masks[0].shape)
    plt.imshow(masks[0].cpu(), cmap='gray')
    plt.title("Ground Truth")
    plt.axis('off')

    plt.subplot(5, 3, i * 3 + 3)
    plt.imshow(preds_binary[0][0].cpu(), cmap='gray')
    plt.title("Prediction")
    plt.axis('off')

plt.tight_layout()
plt.show()
