# train_obstruction.py

import os, random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.transforms import functional as TF, InterpolationMode
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

# Reproducibility & Device
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Toggle between roof segmentation and obstruction detection
FINE_MODE = True

# Constants
TARGET_SIZE = (256, 256)
BATCH_SIZE = 4
NUM_EPOCHS = 100
PATIENCE = 150

# Helper: pad image to square
def pad_to_square_pil(img, fill=0):
    w, h = img.size
    m = max(w, h)
    pad_left = (m - w) // 2
    pad_top = (m - h) // 2
    pad_right = m - w - pad_left
    pad_bottom = m - h - pad_top
    return TF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=fill)

class RoofDataset(Dataset):
    def __init__(self, folders, augment=False, fine_features=False):
        self.folders = folders
        self.augment = augment
        self.fine_features = fine_features
        self.label_mapping = {
            '_background_': 'background', 'roof': 'roof',
            'roofline': 'roof_line', 'roof line': 'roof_line',
            'vent': 'vent', 'vents': 'vent', "vents'": 'vent',
            'window': 'window', 'chimney': 'chimney',
            'bend': 'bend', 'wall': 'wall'
        }
        self.roof_labels = {'roof', 'roof_line', 'chimney', 'vent', 'bend'}
        self.fine_labels = {'chimney', 'vent', "vents", "vents'", 'bend'}
        self.jitter = transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25)

    def normalize_labels(self, labels):
        return [self.label_mapping.get(lbl.strip().lower(), 'other') for lbl in labels]

    def _augment_pair(self, img, msk):
        if random.random() < 0.5:
            img, msk = TF.hflip(img), TF.hflip(msk)
        if random.random() < 0.5:
            img, msk = TF.vflip(img), TF.vflip(msk)
        angle = random.uniform(-15, 15)
        img = TF.rotate(img, angle, interpolation=InterpolationMode.BILINEAR)
        msk = TF.rotate(msk, angle, interpolation=InterpolationMode.NEAREST)
        scale = random.uniform(0.9, 1.1)
        shear = random.uniform(-5, 5)
        tx = int(0.10 * img.width * random.uniform(-1, 1))
        ty = int(0.10 * img.height * random.uniform(-1, 1))
        img = TF.affine(img, 0, (tx, ty), scale, shear, InterpolationMode.BILINEAR, fill=0)
        msk = TF.affine(msk, 0, (tx, ty), scale, shear, InterpolationMode.NEAREST, fill=0)
        if random.random() < 0.3:
            img = TF.gaussian_blur(img, kernel_size=3)
        if random.random() < 0.3:
            arr = np.array(img)
            noise = np.random.normal(0, 5, arr.shape).astype(np.int16)
            arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
        img = self.jitter(img)
        return img, msk

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        folder = self.folders[idx]
        img = Image.open(os.path.join(folder, 'img.png')).convert('RGB')
        msk = Image.open(os.path.join(folder, 'label.png')).convert('L')
        labels_file = os.path.join(folder, 'label_names.txt')
        labels = self.normalize_labels(open(labels_file).read().splitlines()) if os.path.exists(labels_file) else ['background']
        is_roof = any(lbl in self.roof_labels for lbl in labels)
        if self.augment:
            img, msk = self._augment_pair(img, msk)
        img = pad_to_square_pil(img, fill=0)
        msk = pad_to_square_pil(msk, fill=0)
        img = TF.resize(img, TARGET_SIZE, interpolation=InterpolationMode.BILINEAR)
        msk = TF.resize(msk, TARGET_SIZE, interpolation=InterpolationMode.NEAREST)
        img = TF.to_tensor(img)
        msk = TF.to_tensor(msk)
        if self.fine_features:
            is_fine = any(lbl in self.fine_labels for lbl in labels)
            msk = ((msk > 0.05) & (msk < 0.2)).float() if is_fine else torch.zeros_like(msk)
        else:
            msk = (msk > 0.1).float() if is_roof else torch.zeros_like(msk)
        return img, msk

@torch.no_grad()
def compute_iou(pred, targ, thresh=0.5):
    pred = (pred > thresh).float()
    inter = (pred * targ).sum((1,2,3))
    union = ((pred + targ) >= 1).float().sum((1,2,3))
    return ((inter + 1e-6) / (union + 1e-6)).mean().item()

if __name__ == '__main__':
    print("✅ Starting obstruction model training...")

    root_dir = "./dataset/Images_data"
    all_folders = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
    random.shuffle(all_folders)
    n = len(all_folders)
    train_f, val_f, test_f = np.split(all_folders, [int(.7*n), int(.85*n)])

    train_ds = RoofDataset(train_f, augment=True, fine_features=FINE_MODE)
    val_ds = RoofDataset(val_f, augment=False, fine_features=FINE_MODE)
    test_ds = RoofDataset(test_f, augment=False, fine_features=FINE_MODE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=1, activation="sigmoid").to(device)

    class FocalTverskyLoss(nn.Module):
        def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, smooth=1e-6):
            super().__init__()
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma
            self.smooth = smooth

        def forward(self, preds, targets):
            preds = preds.view(-1)
            targets = targets.view(-1)
            tp = (preds * targets).sum()
            fp = ((1 - targets) * preds).sum()
            fn = (targets * (1 - preds)).sum()
            tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
            return (1 - tversky) ** self.gamma

    loss_fn = FocalTverskyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    best_iou = 0.0
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_iou': []}
    start_epoch = 1

    checkpoint_path = 'checkpoint_fine_new.pth'
    model_path = 'best_model_fine.pth'

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

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()
    else:
        print(f"Model file '{model_path}' not found. Evaluation skipped.")

    def evaluate(loader, name):
        ious = []
        with torch.no_grad():
            for imgs, masks in loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                ious.append(compute_iou(preds, masks))
        print(f"{name} IoU: {np.mean(ious):.4f}")

    evaluate(train_loader, 'Train')
    evaluate(test_loader, 'Test')

    def show_sample(loader, title):
        imgs, masks = next(iter(loader))
        imgs, masks = imgs.to(device), masks.to(device)
        with torch.no_grad(): preds = model(imgs)
        img  = imgs[0].cpu().permute(1,2,0)
        msk  = masks[0,0].cpu()
        prd  = (preds[0,0] > 0.5).float().cpu()
        plt.figure(figsize=(9,3))
        for i, (src, lbl) in enumerate(zip([img, msk, prd], ['Image', 'Mask', 'Pred'])):
            plt.subplot(1,3,i+1)
            plt.imshow(src, cmap='gray' if i>0 else None)
            plt.title(f"{title} {lbl}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    show_sample(train_loader, 'Train-')
    show_sample(test_loader,  'Test-')

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'],   label='Val Loss')
    plt.legend(); plt.title('Loss Curve')
    plt.subplot(1,2,2)
    plt.plot(history['val_iou'], label='Val IoU')
    plt.legend(); plt.title('Validation IoU')
    plt.tight_layout(); plt.show()
