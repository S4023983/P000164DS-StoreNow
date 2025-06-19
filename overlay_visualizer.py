import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from torchvision.transforms import functional as TF
from models.model_utils import pad_to_square_pil

# Set device to GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to calculate usable roof area by subtracting obstruction area from total roof area
def calculate_usable_area(roof_mask, obs_mask):
    # Convert predicted probabilities to binary masks (threshold > 0.5)
    roof_bin = (roof_mask > 0.5).float()
    obs_bin = (obs_mask > 0.5).float()

    # Compute usable area mask: roof minus obstructions
    usable_mask = roof_bin * (1 - obs_bin)

    # Total number of pixels predicted as roof
    roof_area = roof_bin.sum().item()
    # Total number of pixels after removing obstructions (usable area)
    usable_area = usable_mask.sum().item()

    # Calculate usable area percentage (avoid division by zero)
    usable_ratio = (usable_area / roof_area) * 100 if roof_area > 0 else 0.0
    return usable_mask, usable_ratio

# Function to visualize original image with roof, obstruction, and usable areas overlaid
def show_usable_overlay(img_tensor, roof_mask, obs_mask):
    # Get usable mask and ratio
    usable_mask, usable_ratio = calculate_usable_area(roof_mask, obs_mask)

    # Convert image tensor to NumPy array for visualization (channel last format)
    img = img_tensor.permute(1, 2, 0).cpu().numpy()

    # Prepare binary masks for overlay visualization
    roof_bin = (roof_mask > 0.5).float().cpu().squeeze().numpy()
    obs_bin = (obs_mask > 0.5).float().cpu().squeeze().numpy()
    usable_bin = usable_mask.cpu().squeeze().numpy()

    # Overlay colors: grey for roof, red for obstructions, green for usable area
    overlay = img.copy()
    overlay[roof_bin == 1] = [0.7, 0.7, 0.7] # Roof (gray)
    overlay[obs_bin == 1] = [1.0, 0.0, 0.0] # Obstructions (red)
    overlay[usable_bin == 1] = [0.0, 1.0, 0.0] # Usable area (green)

    # Plot the overlay result
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.title(f"Usable Roof Area: {usable_ratio:.2f}%")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Pipeline function to run inference and visualize overlay
def run_stage3_overlay(test_loader):
    # Paths to trained model weights
    roof_model_path = 'best_model.pth'
    obs_model_path = 'best_model_fine.pth'

    # Check if both models exist
    if not os.path.exists(roof_model_path) or not os.path.exists(obs_model_path):
        print("One or both models not found. Please train roof and obstruction models first.")
        return

    # Load trained roof segmentation model (DeepLab or UNet)
    roof_model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1, activation="sigmoid").to(device)
    roof_model.load_state_dict(torch.load(roof_model_path))
    roof_model.eval()

    # Load trained obstruction segmentation model (UNet)
    obs_model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1, activation="sigmoid").to(device)
    obs_model.load_state_dict(torch.load(obs_model_path))
    obs_model.eval()

    # Fetch a batch of test images (here only first batch and first image are used)
    imgs, _ = next(iter(test_loader))
    img = imgs[0].unsqueeze(0).to(device)

    # Run inference for both models (roof + obstructions)
    with torch.no_grad():
        roof_pred = roof_model(img)
        obs_pred = obs_model(img)

    # Show visual overlay of usable roof area
    show_usable_overlay(img[0], roof_pred, obs_pred)
