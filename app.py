# app.py

import streamlit as st
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from PIL import Image
from torchvision.transforms import functional as TF
from models.model_utils import pad_to_square_pil

# Constants
TARGET_SIZE = (256, 256)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load models once
@st.cache_resource
def load_models():
    roof_model_path = 'best_model.pth'
    obs_model_path = 'best_model_fine.pth'
    
    if not os.path.exists(roof_model_path) or not os.path.exists(obs_model_path):
        st.error("One or both model files are missing. Train the models first.")
        return None, None

    roof_model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1, activation="sigmoid").to(device)
    roof_model.load_state_dict(torch.load(roof_model_path, map_location=device))
    roof_model.eval()

    obs_model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1, activation="sigmoid").to(device)
    obs_model.load_state_dict(torch.load(obs_model_path, map_location=device))
    obs_model.eval()

    return roof_model, obs_model

# Usable area calculation
def calculate_usable_area(roof_mask, obs_mask):
    roof_bin = (roof_mask > 0.5).float()
    obs_bin = (obs_mask > 0.5).float()
    usable_mask = roof_bin * (1 - obs_bin)
    roof_area = roof_bin.sum().item()
    usable_area = usable_mask.sum().item()
    usable_ratio = (usable_area / roof_area) * 100 if roof_area > 0 else 0.0
    return usable_mask, usable_ratio

# Streamlit interface
st.set_page_config(page_title="Rooftop Usable Area Estimator", layout="centered")
st.title("🏠 Rooftop Usable Area Estimator")
st.write("Upload a rooftop image. The app will show roof area, obstructions, and usable space.")

uploaded_file = st.file_uploader("Upload a rooftop image (PNG or JPG)", type=['png', 'jpg', 'jpeg'])

roof_model, obs_model = load_models()

if uploaded_file and roof_model and obs_model:
    image = Image.open(uploaded_file).convert('RGB')
    original = image.copy()

    # Preprocessing
    image = pad_to_square_pil(image)
    image = TF.resize(image, TARGET_SIZE)
    img_tensor = TF.to_tensor(image).unsqueeze(0).to(device)

    with torch.no_grad():
        roof_pred = roof_model(img_tensor)
        obs_pred = obs_model(img_tensor)

    usable_mask, usable_ratio = calculate_usable_area(roof_pred, obs_pred)

    # Convert masks to numpy
    roof_bin = (roof_pred > 0.5).float().cpu().squeeze().numpy()
    obs_bin = (obs_pred > 0.5).float().cpu().squeeze().numpy()
    usable_bin = usable_mask.cpu().squeeze().numpy()

    # Display overlays
    overlay = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy().copy()
    overlay[roof_bin == 1] = [0.7, 0.7, 0.7]      # Gray
    overlay[obs_bin == 1] = [1.0, 0.0, 0.0]       # Red
    overlay[usable_bin == 1] = [0.0, 1.0, 0.0]    # Green

    st.image(original, caption="Uploaded Image", use_column_width=True)
    st.image(overlay, caption=f"Overlay (Usable Area: {usable_ratio:.2f}%)", use_column_width=True)

elif uploaded_file and (not roof_model or not obs_model):
    st.warning("Model files missing. Please make sure 'best_model.pth' and 'best_model_fine.pth' are in the root directory.")
