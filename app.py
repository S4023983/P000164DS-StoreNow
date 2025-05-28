# app.py

import streamlit as st
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
from models.model_utils import pad_to_square_pil
import json
import io
import requests
from efficientnet_pytorch import EfficientNet

# Constants
TARGET_SIZE = (256, 256)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Classifier-related setup ---
GOOGLE_API_KEY = "AIzaSyCo2cd_BPa3V3gOd8IOvQhxGGRAShAf9c4"  # Replace with your actual key
model_path = "roof_model.pth"
label_map_path = "label_map.json"

@st.cache_resource
def load_classifier():
    with open(label_map_path, "r") as f:
        label_to_index = json.load(f)
    idx_to_name = {v: k for k, v in label_to_index.items()}
    n_classes = len(label_to_index)
    model = EfficientNet.from_name("efficientnet-b3", num_classes=n_classes)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, idx_to_name

infer_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def classify_roof(image_bytes, model, idx_to_name):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = infer_transforms(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
    idx = logits.argmax(dim=1).item()
    return idx_to_name[idx]

def geocode_address(address: str, api_key: str):
    resp = requests.get("https://maps.googleapis.com/maps/api/geocode/json",
                        params={"address": address, "key": api_key})
    data = resp.json()
    status = data.get("status")
    if status != "OK":
        err_msg = data.get("error_message", "No error_message returned")
        raise RuntimeError(f"Geocoding error: {status} — {err_msg}")
    loc = data["results"][0]["geometry"]["location"]
    return loc["lat"], loc["lng"]

def fetch_satellite_view(lat, lng, api_key,
                         size="640x640", zoom=21):
    resp = requests.get("https://maps.googleapis.com/maps/api/staticmap",
                        params={"center": f"{lat},{lng}",
                                "zoom": zoom,
                                "size": size,
                                "maptype": "satellite",
                                "key": api_key})
    if resp.status_code != 200:
        raise RuntimeError(f"Satellite View error: HTTP {resp.status_code}")
    return resp.content

def fetch_building_insights(lat, lng):
    url = "https://solar.googleapis.com/v1/buildingInsights:findClosest"
    params = {
        "location.latitude": lat,
        "location.longitude": lng,
        "requiredQuality": "HIGH",
        "key": GOOGLE_API_KEY
    }
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        raise RuntimeError(f"Solar API error {resp.status_code}: {resp.text}")
    return resp.json()

def get_best_panel_orientation(address):
    lat, lng = geocode_address(address, GOOGLE_API_KEY)
    insights = fetch_building_insights(lat, lng)
    segments = insights.get("solarPotential", {}).get("roofSegmentStats", [])
    if not segments:
        st.warning("No roof segments found — using latitude-based fallback.")
        lat = insights["center"]["latitude"]
        tilt = abs(lat - 0) * 0.9
        azimuth = 180
        area = 0.0
    else:
        best = max(segments, key=lambda s: s["stats"]["areaMeters2"])
        tilt = best["pitchDegrees"]
        azimuth = best["azimuthDegrees"]
        area = best["stats"]["areaMeters2"]
    return tilt, azimuth, area

def azimuth_to_direction(azimuth):
    directions = ["North", "Northeast", "East", "Southeast",
                  "South", "Southwest", "West", "Northwest"]
    idx = round(azimuth / 45) % 8
    return directions[idx]

# Dummy function: you should define it or import it from your project
def load_models():
    # Replace this with actual model loading logic
    return None, None

def calculate_usable_area(roof_pred, obs_pred):
    roof_mask = (roof_pred > 0.5).float()
    obs_mask = (obs_pred > 0.5).float()
    usable_mask = roof_mask * (1 - obs_mask)
    total = torch.sum(roof_mask)
    usable = torch.sum(usable_mask)
    ratio = (usable / total * 100).item() if total > 0 else 0
    return usable_mask, ratio

# --- Streamlit UI ---
st.set_page_config(page_title="Rooftop Usable Area Estimator", layout="centered")
st.title("🏠 Rooftop Usable Area Estimator")
st.write("Upload a rooftop image or enter an address. We'll predict roof type, show roof/obstruction areas, and solar insights.")

roof_model, obs_model = load_models()
clf_model, clf_idx_to_name = load_classifier()

uploaded_file = st.file_uploader("Upload a rooftop image (PNG or JPG)", type=['png', 'jpg', 'jpeg'])
address = st.text_input("Or enter an address (e.g., 48 Sargood Street, Altona)")

if address:
    try:
        lat, lng = geocode_address(address, GOOGLE_API_KEY)
        st.success(f"Coordinates: {lat:.6f}, {lng:.6f}")

        sat_img = fetch_satellite_view(lat, lng, GOOGLE_API_KEY)
        roof_type = classify_roof(sat_img, clf_model, clf_idx_to_name)
        tilt, azimuth, area = get_best_panel_orientation(address)

        st.image(sat_img, caption="Satellite View", use_column_width=True)
        st.write(f"**Predicted roof type:** {roof_type}")
        st.write(f"**Best tilt:** {tilt:.1f}°")
        st.write(f"**Best azimuth:** {azimuth:.1f}° ({azimuth_to_direction(azimuth)})")
        st.write(f"**Usable roof area:** {area:.1f} m²")

    except Exception as e:
        st.error(f"Error: {e}")

elif uploaded_file and roof_model and obs_model:
    image = Image.open(uploaded_file).convert('RGB')
    original = image.copy()

    image = pad_to_square_pil(image)
    image = transforms.Resize(TARGET_SIZE)(image)
    img_tensor = TF.to_tensor(image).unsqueeze(0).to(device)

    with torch.no_grad():
        roof_pred = roof_model(img_tensor)
        obs_pred = obs_model(img_tensor)

    usable_mask, usable_ratio = calculate_usable_area(roof_pred, obs_pred)

    roof_bin = (roof_pred > 0.5).float().cpu().squeeze().numpy()
    obs_bin = (obs_pred > 0.5).float().cpu().squeeze().numpy()
    usable_bin = usable_mask.cpu().squeeze().numpy()

    overlay = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy().copy()
    overlay[roof_bin == 1] = [0.7, 0.7, 0.7]
    overlay[obs_bin == 1] = [1.0, 0.0, 0.0]
    overlay[usable_bin == 1] = [0.0, 1.0, 0.0]

    st.image(original, caption="Uploaded Image", use_column_width=True)
    st.image(overlay, caption=f"Overlay (Usable Area: {usable_ratio:.2f}%)", use_column_width=True)

elif uploaded_file:
    st.warning("Model files missing. Please make sure 'best_model.pth' and 'best_model_fine.pth' are in the root directory.")
