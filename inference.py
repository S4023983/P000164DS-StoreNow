# inference.py
import os
import sys
import io
import torch
import requests
import json
from PIL import Image
from torchvision import transforms
from utils.downloader import download_image
from utils.visualizer import predict_and_visualize
from main import UNet  # here we are using the same architecture as training
# ===== Load Classification Model (EfficientNet) =====
from efficientnet_pytorch import EfficientNet

# ===== Device Setup =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #making my program adaptable to any device such a CPU or GPU
# ===== Load Segmentation Model =====
segmentation_model = UNet().to(device) #initializing the unet Model
seg_model_path = os.path.join("saved_models", "best_unet_model.pth") #Here defining the path where the best trained model is saved.

if not os.path.exists(seg_model_path):
    print("Segmentation model not found. Run 'main.py' first to train it.")
    sys.exit()

segmentation_model.load_state_dict(torch.load(seg_model_path, map_location=device))
segmentation_model.eval()
print("Segmentation model loaded.")



root_dir = "saved_models"
cls_model_path = os.path.join(root_dir, "roof_model.pth")
label_map_path = os.path.join(root_dir, "label_map.json")

if not os.path.exists(cls_model_path) or not os.path.exists(label_map_path):
    print("Classification model or label map not found.")
    sys.exit()

with open(label_map_path, "r") as f:
    label_to_index = json.load(f)
idx_to_name = {v: k for k, v in label_to_index.items()}

n_classes = len(label_to_index)
classification_model = EfficientNet.from_name("efficientnet-b3", num_classes=n_classes)
classification_model.load_state_dict(torch.load(cls_model_path, map_location=device))
classification_model = classification_model.to(device)
classification_model.eval()
print("Classification model loaded.")

# ===== Transforms =====
infer_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

seg_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ===== Functions =====
def classify_roof(image_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = infer_transforms(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = classification_model(tensor)
    idx = logits.argmax(dim=1).item()
    return idx_to_name[idx]

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

def fetch_satellite_view(lat: float, lng: float, api_key: str,
                         size: str = "640x640", zoom: int = 20) -> bytes:
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

# ===== Main Execution =====
GOOGLE_API_KEY = "AIzaSyBa_cYa7t2qnCrhYVfnkYJoXVhGyGFtwrM"

address = input("Enter an address: ")
lat, lon = geocode_address(address, GOOGLE_API_KEY)
print(f"Coordinates: {lat:.6f}, {lon:.6f}")

img_bytes = fetch_satellite_view(lat, lon, GOOGLE_API_KEY)
roof_type = classify_roof(img_bytes)
print(f"Predicted roof type: {roof_type}")

# Save satellite image temporarily
temp_image_path = "temp_satellite_image.png"
with open(temp_image_path, "wb") as f:
    f.write(img_bytes)

# Run segmentation and visualize
predict_and_visualize(temp_image_path, segmentation_model, seg_transform, device)

# Cleanup temp image
if os.path.exists(temp_image_path):
    os.remove(temp_image_path)
