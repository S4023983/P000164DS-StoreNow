
# image_classification.py
# importing necessary libraries
# This script performs roof type classification using a pre-trained EfficientNet model.
import os
import io
import json
import torch
from torchvision import transforms
from PIL import Image
import requests
from efficientnet_pytorch import EfficientNet

# Google API key (you need billing enabled for full access)
GOOGLE_API_KEY = "AIzaSyDs6lG7Jxpz54iUAPLIWLBkdUSt-UtdV5o"


# File paths for model and label mapping
root_dir = os.getcwd()
model_path = os.path.join(root_dir, "roof_model.pth")
label_map_path = os.path.join(root_dir, "label_map.json")

# Loading label mapping (e.g. "Flat Roof" -> 0, "Gable Roof" -> 1, etc.)
with open(label_map_path, "r") as f:
    label_to_index = json.load(f)
idx_to_name = {v: k for k, v in label_to_index.items()}

# Loading trained EfficientNet model for roof classification
n_classes = len(label_to_index)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EfficientNet.from_name("efficientnet-b3", num_classes=n_classes)
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

# Imaging preprocessing pipeline (same normalization as during training)
infer_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Classifying the roof type from satellite image bytes
def classify_roof(image_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = infer_transforms(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
    idx = logits.argmax(dim=1).item()
    return idx_to_name[idx]

# Converting address to lat/lon using Google Geocoding API
def geocode_address(address: str, api_key: str) -> tuple[float, float]:
    resp = requests.get(
        "https://maps.googleapis.com/maps/api/geocode/json",
        params={"address": address, "key": api_key}
    )
    data = resp.json()
    status = data.get("status")
    if status != "OK":
        err_msg = data.get("error_message", "No error_message returned")
        raise RuntimeError(f"Geocoding error: {status} — {err_msg}")
    loc = data["results"][0]["geometry"]["location"]
    return loc["lat"], loc["lng"]

# Fetching satellite image from Google Static Maps API
def fetch_satellite_view(lat: float, lng: float, api_key: str,
                         size: str = "640x640", zoom: int = 21) -> bytes:
    resp = requests.get(
        "https://maps.googleapis.com/maps/api/staticmap",
        params={
            "center": f"{lat},{lng}",
            "zoom": zoom,
            "size": size,
            "maptype": "satellite",
            "key": api_key
        }
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Satellite View error: HTTP {resp.status_code}")
    return resp.content

# Fetching solar rooftop insights from Google Solar API
def fetch_building_insights(lat: float, lng: float) -> dict:
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

# Get best tilt, azimuth, area from Google Solar API response   
def get_best_panel_orientation(address):
    lat, lng = geocode_address(address, GOOGLE_API_KEY)
    insights = fetch_building_insights(lat, lng)
    segments = insights.get("solarPotential", {}).get("roofSegmentStats", [])

    if not segments:
        # If no segments found, fallback using latitude
        print("No roof segments found — using latitude-based fallback.")
        lat = insights["center"]["latitude"]
        tilt = abs(lat - 0) * 0.9
        azimuth = 180
        area = 0.0
        roof_type = "Flat Roof (fallback)"
    else:
        # Pick segment with largest usable area
        best = max(segments, key=lambda s: s["stats"]["areaMeters2"])
        tilt = best["pitchDegrees"]
        azimuth = best["azimuthDegrees"]
        area = best["stats"]["areaMeters2"]

    return tilt, azimuth, area

# Convert azimuth angle to cardinal direction (N, NE, E, etc.)
def azimuth_to_direction(azimuth):
    directions = ["North", "Northeast", "East", "Southeast",
                  "South", "Southwest", "West", "Northwest"]
    idx = round(azimuth / 45) % 8
    return directions[idx]

# Debugging function to print URL of satellite image directly
def show_rooftop_satellite_image(lat, lng, api_key):
    url = (
        "https://maps.googleapis.com/maps/api/staticmap?"
        f"center={lat},{lng}&zoom=20&size=600x600&maptype=satellite"
        f"&markers=color:red%7C{lat},{lng}&key={api_key}"
    )
    print(f"Image URL: {url}")

# CLI mode for testing
if __name__ == "__main__":
    address = input("Enter an address: ")
    lat, lng = geocode_address(address, GOOGLE_API_KEY)
    print(f"Coordinates: {lat:.6f}, {lng:.6f}")

    # Fetch and show satellite image
    img_bytes = fetch_satellite_view(lat, lng, GOOGLE_API_KEY)
    img = Image.open(io.BytesIO(img_bytes))
    img.show()  # ✅ Shows image in default viewer

    # Classify roof
    roof_type = classify_roof(img_bytes)
    print(f"Predicted roof type: {roof_type}")

    # Get solar orientation
    try:
        tilt, azimuth, area = get_best_panel_orientation(address)
        print(f"Best tilt: {tilt:.1f}°")
        print(f"Best azimuth: {azimuth:.1f}°")
        print(f"Usable roof area: {area:.1f} m²")
        print(f"Facing: {azimuth_to_direction(azimuth)}")
    except RuntimeError as e:
        print(f"Solar orientation unavailable: {e}")

