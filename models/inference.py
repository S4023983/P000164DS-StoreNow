# models/inference.py

import torch
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import numpy as np

def load_models(device='cpu'):
    roof_model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1, activation="sigmoid").to(device)
    roof_model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
    roof_model.eval()

    obstruction_model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1, activation="sigmoid").to(device)
    obstruction_model.load_state_dict(torch.load("models/best_model_fine.pth", map_location=device))
    obstruction_model.eval()

    return roof_model, obstruction_model

def predict_masks(img_tensor, roof_model, obstruction_model, device='cpu'):
    img_tensor = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        roof_pred = roof_model(img_tensor)
        obs_pred = obstruction_model(img_tensor)
    return roof_pred[0, 0].cpu(), obs_pred[0, 0].cpu()

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

    roof_bin = (roof_mask > 0.5).float().cpu().numpy()
    obs_bin = (obs_mask > 0.5).float().cpu().numpy()
    usable_bin = usable_mask.cpu().numpy()

    overlay = img.copy()
    overlay[roof_bin == 1] = [0.7, 0.7, 0.7]      # Gray for roof
    overlay[obs_bin == 1] = [1.0, 0.0, 0.0]       # Red for obstruction
    overlay[usable_bin == 1] = [0.0, 1.0, 0.0]    # Green for usable

    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.title(f"Usable Roof Area: {usable_ratio:.2f}%")
    plt.axis('off')
    plt.show()
    return usable_ratio

def predict_and_overlay(img_tensor, roof_model, obstruction_model, device='cpu'):
    roof_mask, obs_mask = predict_masks(img_tensor, roof_model, obstruction_model, device=device)
    usable_ratio = show_usable_overlay(img_tensor, roof_mask, obs_mask)
    return usable_ratio
