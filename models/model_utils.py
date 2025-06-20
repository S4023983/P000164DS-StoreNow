# models/model_utils.py

import os
import torch
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF, InterpolationMode

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
        img = TF.resize(img, (256, 256), interpolation=InterpolationMode.BILINEAR)
        msk = TF.resize(msk, (256, 256), interpolation=InterpolationMode.NEAREST)
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
