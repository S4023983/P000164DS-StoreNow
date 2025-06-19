# roof_type_classifier.py


# -----------------------------
# Importing necessary libraries
# -----------------------------

import os
import pandas as pd
import numpy as np
import torch
import torchvision
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import json
from efficientnet_pytorch import EfficientNet

# -----------------------------
# Configuration & Paths
# -----------------------------

# Update this to your local paths
root_dir = "./"
# CSV file containing image filenames and corresponding roof type labels    
csv_file = os.path.join(root_dir, "roof_type_labels.csv")

# Image directory containing roof images for classification
image_dir = os.path.join(root_dir, "dataset", "classification_dataset")

# Device configuration: Use GPU if available, else fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Hyperparameters
# -----------------------------

batch_sz = 32       # Batch size for training and validation
n_classes = 5       # Number of unique roof classes
k_folds = 5         # Number of folds for cross-validation
epochs = 4          # Number of training epochs per fold

class RoofImageDataset(Dataset):
    """
    Custom dataset class for loading roof images and corresponding labels.
    Reads image paths from CSV, loads images from disk, applies augmentation.
    """
    def __init__(self, csv_path, img_dir, transform=None):
        self.data = pd.read_csv(csv_path).dropna(subset=[pd.read_csv(csv_path).columns[0]]) # Drop empty filenames
        self.img_dir = img_dir
        self.transform = transform
        self.label_set = sorted(self.data['label'].unique()) # Unique roof labels (classes)
        self.label_to_index = {name: i for i, name in enumerate(self.label_set)} # Mapping label names to numeric indices

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        fname = str(self.data.iloc[index, 0]) # Get filename from CSV
        filepath = os.path.join(self.img_dir, fname)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        label = self.data.iloc[index, 1] # Get label from CSV
        label_index = self.label_to_index[label]
        image = Image.open(filepath).convert("RGB") # Loading image
        if self.transform:
            image = self.transform(image)
        return image, label_index

# ===================================================
# Image Transformations (Data Augmentation)
# ===================================================

image_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8,1.0)), # Random crop with resize
    transforms.RandomHorizontalFlip(p=0.5), # Flip horizontally
    transforms.RandomVerticalFlip(p=0.2), # Flip vertically (less probable)
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Color augmentation
    transforms.RandomRotation(15), # Random rotation
    transforms.ToTensor(), # Convert to tensor
    transforms.Normalize([0.485, 0.456, 0.406], # Normalize using pretrained ImageNet stats
                         [0.229, 0.224, 0.225])
])

# ===================================================
# Dataset Preparation
# ===================================================
dataset = RoofImageDataset(csv_file, image_dir, transform=image_transforms)

# Read the CSV again for generating label map (name to index)
df = pd.read_csv(csv_file).dropna(subset=[pd.read_csv(csv_file).columns[0]])
label_names = sorted(df['label'].unique())
name_to_idx = {name: i for i, name in enumerate(label_names)}

# Save label mapping to JSON for future inference use
with open(os.path.join(root_dir, "label_map.json"), "w") as f:
    json.dump(name_to_idx, f, indent=2)

# Convert labels into numeric form for stratified K-Fold splitting
y_numeric = df['label'].map(name_to_idx).values

# ===================================================
# Load Pre-trained EfficientNet-B3 Model
# ===================================================
model = EfficientNet.from_pretrained('efficientnet-b3') # Load pretrained EfficientNet-B3 weights
in_features = model._fc.in_features # Get size of final FC layer
model._fc = torch.nn.Linear(in_features, 5) # Replace FC layer with correct output size for 5-class classification
model = model.to(device)

# ===================================================
# 5-Fold Cross-Validation Training Loop
# ===================================================
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
fold_accuracies = [] # Store validation accuracy for each fold

for fold, (train_idx, val_idx) in enumerate(skf.split(X=np.arange(len(dataset)), y=y_numeric)):
    print(f"\n**** Fold {fold+1}/{k_folds} ****")

    # Create training and validation subsets for this fold
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    
    # Create dataloaders
    train_loader = DataLoader(train_subset, batch_size=batch_sz, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_sz, shuffle=False)

    # Loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

    # ---------------------
    # Training loop for current fold
    # ---------------------

    for epoch in range(epochs):
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()

            # Track metrics for logging
            total_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == lbls).sum().item()
            total_samples += lbls.size(0)
        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples
        print(f"Fold {fold+1}, Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc*100:.4f}")

    # ---------------------
    # Validation loop for current fold
    # ---------------------

    model.eval()
    val_correct, val_samples = 0, 0
    true_labels, pred_labels = [], []
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == lbls).sum().item()
            val_samples += lbls.size(0)
            true_labels.extend(lbls.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

    # Calculate accuracy for this fold

    fold_accuracy = val_correct / val_samples
    fold_accuracies.append(fold_accuracy)
    print(f"Fold {fold+1} Val Accuracy: {fold_accuracy*100:.4f}")

    # Generate classification report for this fold
    unique_labels = sorted(set(true_labels) | set(pred_labels))
    reduced_target_names = [label_names[i] for i in unique_labels]

    print(classification_report(true_labels, pred_labels, target_names=reduced_target_names, zero_division=0))

    # Plot Confusion Matrix
    cm = confusion_matrix(true_labels, pred_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names)
    plt.title(f"Confusion Matrix - Fold {fold+1}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# ===================================================
# Cross-validation summary metrics
# ===================================================

mean_acc = np.mean(fold_accuracies)
std_acc = np.std(fold_accuracies)
print(f"\nCross Validation Results: Mean Accuracy = {mean_acc:.4f}, Std = {std_acc:.4f}")

# ===================================================
# Save the trained model after cross-validation
# ===================================================
model_path = os.path.join(root_dir, "roof_model.pth")
torch.save(model.state_dict(), model_path)
print(f"Model successfully saved: {model_path}")
