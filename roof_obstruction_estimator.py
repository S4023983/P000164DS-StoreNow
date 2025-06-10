import os
import cv2
import math
import torch
import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Config
GOOGLE_API_KEY = "AIzaSyBa_cYa7t2qnCrhYVfnkYJoXVhGyGFtwrM"
MODEL_PATH = "best_roof_model.pth"
SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"
LAT, LON = 52.57833206921684, 13.509644598173818


ZOOM = 20
IMG_SIZE = 640
PADDING = 20
THRESH = 0.5
MIN_ROOF_REMAIN_RATIO = 0.6
EPSILON_RATIO = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

roof_model = smp.DeepLabV3Plus(encoder_name="resnet101", encoder_weights=None,
                                in_channels=3, classes=1).to(device)
roof_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
roof_model.eval()

transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.5,) * 3, std=(0.5,) * 3),
    ToTensorV2()
])

sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT).to(device)
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,# we are placing a grid of “prompt points” across the image 
                       #(here, points_per_side=32 means a 32×32 grid of points).
    pred_iou_thresh=0.90,
    stability_score_thresh=0.92, #We filter out unstable mask proposals that have low predicted IoU or low stability score.
    min_mask_region_area=200
)

def fetch_satellite_image(lat, lon, zoom=20, size=640, api_key=GOOGLE_API_KEY):
    url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom={zoom}&size={size}x{size}&maptype=satellite&key={api_key}"
    response = requests.get(url)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to fetch map image: {response.status_code}")
    return np.array(Image.open(BytesIO(response.content)).convert("RGB"))

TILE = 256
center_world_x, center_world_y = None, None
pixels_per_tile = 2 ** ZOOM

def _lnglat_to_world(lng, lat):
    siny = math.sin(lat * math.pi / 180)
    siny = min(max(siny, -0.9999), 0.9999)
    x = TILE * (0.5 + lng / 360)
    y = TILE * (0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi))
    return x, y

def pixel_to_latlon(px, py):
    world_x = center_world_x + (px - IMG_SIZE / 2) / pixels_per_tile
    world_y = center_world_y + (py - IMG_SIZE / 2) / pixels_per_tile
    lng = (world_x / TILE - 0.5) * 360
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * world_y / TILE)))
    lat = lat_rad * 180 / math.pi
    return lat, lng

def predict_mask(image_np):
    H, W = image_np.shape[:2]
    input_tensor = transform(image=image_np)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        pred = torch.sigmoid(roof_model(input_tensor)).squeeze().cpu().numpy()
        pred = cv2.resize(pred, (W, H), interpolation=cv2.INTER_LINEAR)
        return (pred > THRESH).astype(np.uint8) * 255

def morph_clean(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def run_sam_everything(cropped, roof_mask):
    masks = mask_generator.generate(cropped)
    roof_pixels = cropped[roof_mask > 0]
    std_color = np.std(roof_pixels, axis=0).mean()

    if std_color < 10:
        max_obstruction_area_ratio = 0.05
    elif std_color < 20:
        max_obstruction_area_ratio = 0.1
    elif std_color < 30:
        max_obstruction_area_ratio = 0.15
    else:
        max_obstruction_area_ratio = 0.2

    print(f"Color std deviation: {std_color:.2f} → Obstruction Area Tolerance: {max_obstruction_area_ratio*100:.0f}%")

    roof_area = np.sum(roof_mask)
    valid_masks = []
    temp_roof_mask = roof_mask.copy()

    for m in masks:
        binary = m["segmentation"].astype(np.uint8)
        intersection = cv2.bitwise_and(binary, roof_mask)
        ratio_inside = np.sum(intersection) / np.sum(binary)
        mask_area = np.sum(binary)

        if ratio_inside >= 0.9 and (mask_area / roof_area) < max_obstruction_area_ratio:
            candidate_mask = cv2.bitwise_and(temp_roof_mask, cv2.bitwise_not(binary))
            if np.sum(candidate_mask) / roof_area >= MIN_ROOF_REMAIN_RATIO:
                valid_masks.append(binary)
                temp_roof_mask = candidate_mask

    clean_roof_mask = temp_roof_mask

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(cropped)
    plt.title("Original Cropped Roof")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(roof_mask, cmap='gray')
    plt.title("Original Roof Mask")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(clean_roof_mask, cmap='gray')
    plt.title("Clean Roof (Obstructions Removed)")
    plt.axis('off')
    plt.show()

    return clean_roof_mask, valid_masks

def run_pipeline(lat, lon):
    global center_world_x, center_world_y
    center_world_x, center_world_y = _lnglat_to_world(lon, lat)

    orig = fetch_satellite_image(lat, lon)
    H, W = orig.shape[:2]
    img_center = (W // 2, H // 2)

    mask1 = morph_clean(predict_mask(orig))
    contours, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No roof detected.")
        return

    def centroid(c):
        M = cv2.moments(c)
        if M["m00"] == 0:
            return float('inf')
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return np.sqrt((cx - img_center[0])**2 + (cy - img_center[1])**2)

    best_contour = min(contours, key=centroid)
    roof_only_mask = np.zeros_like(mask1)
    cv2.drawContours(roof_only_mask, [best_contour], -1, 255, thickness=-1)

    ys, xs = np.where(roof_only_mask == 255)
    x1, y1 = max(xs.min() - PADDING, 0), max(ys.min() - PADDING, 0)
    x2, y2 = min(xs.max() + PADDING, W), min(ys.max() + PADDING, H)

    cropped = orig[y1:y2, x1:x2]
    roof_crop_mask = roof_only_mask[y1:y2, x1:x2]
    roof_mask_bin = (roof_crop_mask > 0).astype(np.uint8)

    clean_roof_mask, obstructions = run_sam_everything(cropped, roof_mask_bin)

    contours2, _ = cv2.findContours(clean_roof_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours2:
        print("No valid roof after cleaning.")
        return

    final = max(contours2, key=cv2.contourArea)
    eps = EPSILON_RATIO * cv2.arcLength(final, True)
    approx = cv2.approxPolyDP(final, eps, True).squeeze()
    approx[:, 0] += x1
    approx[:, 1] += y1

    closed = np.vstack([approx, approx[0]])
    plt.figure(figsize=(10, 6))
    plt.imshow(orig)
    plt.plot(closed[:, 0], closed[:, 1], color='red', linewidth=2)
    plt.title("Refined Roof Edge")
    plt.axis('off')
    plt.show()

    print("GPS Coordinates (lat, lon):")
    for x, y in approx:
        lat_, lon_ = pixel_to_latlon(x, y)
        print(f"{lat_:.6f}, {lon_:.6f}")

    return clean_roof_mask, obstructions

if __name__ == "__main__":
    clean_roof, obstruction_masks = run_pipeline(LAT, LON)



import math
import matplotlib.pyplot as plt

# GPS coordinates of the roof corners (lat, lon)
coords = [
    (37.853348, 145.000787),
    (37.853363, 145.000771),
    (37.853471, 145.000759),
    (37.853478, 145.000816),
    (37.853489, 145.000840),
    (37.853487, 145.000873),
    (37.853474, 145.000885),
    (37.853427, 145.000888),
    (37.853370, 145.000907),
    (37.853357, 145.000884)
]

# Earth radius in meters
R = 6378137

# Convert lat/lon to local flat meters using the Haversine formula
def latlon_to_meters(lat1, lon1, lat2, lon2):
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

# Project lat/lon to (x, y) in meters relative to the first point
base_lat, base_lon = coords[0]
xy_meters = []

for lat, lon in coords:
    x = latlon_to_meters(base_lat, base_lon, base_lat, lon)
    y = latlon_to_meters(base_lat, base_lon, lat, base_lon)
    if lon < base_lon:
        x *= -1
    if lat < base_lat:
        y *= -1
    xy_meters.append((x, y))

# Shoelace formula for area of polygon
def polygon_area(pts):
    n = len(pts)
    area = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i+1)%n]
        area += x1 * y2 - x2 * y1
    return abs(area / 2)

flat_area = polygon_area(xy_meters)
slope_angle = 30  # degrees
usable_area = flat_area / math.cos(math.radians(slope_angle))

# Solar panel size (standard panel 1.9m x 1.6m)
panel_area = 1.9 * 1.6
num_panels = int(usable_area // panel_area)

# Output results
print(f"Flat roof area: {flat_area:.2f} m²")
print(f"Usable roof area (30° slope): {usable_area:.2f} m²")
print(f"Solar panels (1.9m x 1.6m) that can be installed: {num_panels}")

# Plot the projected roof area
x_vals, y_vals = zip(*xy_meters + [xy_meters[0]])  # close the loop
plt.figure(figsize=(6, 6))
plt.plot(x_vals, y_vals, 'bo-', linewidth=2)
plt.fill(x_vals, y_vals, alpha=0.3)
plt.title("Roof Outline (meters)")
plt.xlabel("X (meters)")
plt.ylabel("Y (meters)")
plt.axis("equal")
plt.grid(True)
plt.show()
