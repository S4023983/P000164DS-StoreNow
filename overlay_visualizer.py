import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from torchvision.transforms import functional as TF
from models.model_utils import pad_to_square_pil

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_usable_area(roof_mask, obs_mask):
    roof_bin = (roof_mask > 0.5).float()
    obs_bin = (obs_mask > 0.5).float()
    usable_mask = roof_bin * (1 - obs_bin)
    roof_area = roof_bin.sum().item()
    usable_area = usable_mask.sum().item()
    usable_ratio = (usable_area / roof_area) * 100 if roof_area > 0 else 0.0
    return usable_mask, usable_ratio

def show_usable_overlay(img_tensor, roof_mask, obs_mask):
    usable_mask, usable_ratio = calculate_usable_area(roof_mask, obs_mask)
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    roof_bin = (roof_mask > 0.5).float().cpu().squeeze().numpy()
    obs_bin = (obs_mask > 0.5).float().cpu().squeeze().numpy()
    usable_bin = usable_mask.cpu().squeeze().numpy()

    overlay = img.copy()
    overlay[roof_bin == 1] = [0.7, 0.7, 0.7]
    overlay[obs_bin == 1] = [1.0, 0.0, 0.0]
    overlay[usable_bin == 1] = [0.0, 1.0, 0.0]

    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.title(f"Usable Roof Area: {usable_ratio:.2f}%")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def run_stage3_overlay(test_loader):
    roof_model_path = 'best_model.pth'
    obs_model_path = 'best_model_fine.pth'

    if not os.path.exists(roof_model_path) or not os.path.exists(obs_model_path):
        print("❌ One or both models not found. Please train roof and obstruction models first.")
        return

    roof_model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1, activation="sigmoid").to(device)
    roof_model.load_state_dict(torch.load(roof_model_path))
    roof_model.eval()

    obs_model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1, activation="sigmoid").to(device)
    obs_model.load_state_dict(torch.load(obs_model_path))
    obs_model.eval()

    imgs, _ = next(iter(test_loader))
    img = imgs[0].unsqueeze(0).to(device)

    with torch.no_grad():
        roof_pred = roof_model(img)
        obs_pred = obs_model(img)

    show_usable_overlay(img[0], roof_pred, obs_pred)
