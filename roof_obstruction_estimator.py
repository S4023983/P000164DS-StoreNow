# ======================================================================
# ROOF OBSTRUCTION ESTIMATION PIPELINE USING SEGMENTATION + SAM
# ----------------------------------------------------------------------
# This code estimates usable rooftop area by:
# 1️⃣ Fetching satellite imagery from Google Maps
# 2️⃣ Segmenting the rooftop using DeepLabV3+
# 3️⃣ Identifying obstructions using Facebook's Segment Anything Model (SAM)
# 4️⃣ Calculating roof area, obstruction area, and usable area (corrected for tilt)
# ======================================================================

# -------------------- IMPORT LIBRARIES ---------------------------------

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
# Segment Anything Model (SAM) - Facebook Meta AI's model
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# --------------------------------------
# CONFIGURATION PARAMETERS
# --------------------------------------


# Local directories for model weights and checkpoints
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT_DIR, "best_roof_model.pth") # Trained DeepLabV3+ weights
SAM_CHECKPOINT = os.path.join(ROOT_DIR, "sam_vit_b_01ec64.pth") # SAM model weights

# Google Maps API configuration (static map API used and google solar api used)
GOOGLE_API_KEY = "AIzaSyDs6lG7Jxpz54iUAPLIWLBkdUSt-UtdV5o"  # Replace with your actual API key

# Parameters for image fetching and processing
ZOOM = 20  
IMG_SIZE = 640  # Size of image in pixels (640x640)
PADDING = 20 # Padding added around detected rooftop bounding box
THRESH = 0.5 # Threshold for converting model output probability to binary mask
MIN_ROOF_REMAIN_RATIO = 0.6 # Minimum valid roof area ratio after removing obstructions
EPSILON_RATIO = 0.01 # Simplification parameter for contour approximation

# Google projection constants for lat-long to world coordinates conversion
TILE = 256
pixels_per_tile = 2 ** ZOOM
center_world_x, center_world_y = None, None

# --------------------------------------
# GEOSPATIAL UTILITY FUNCTIONS
# --------------------------------------

def _lnglat_to_world(lng, lat):
    """Convert longitude & latitude to Google Maps world coordinates."""
    siny = math.sin(lat * math.pi / 180)
    siny = min(max(siny, -0.9999), 0.9999)
    x = TILE * (0.5 + lng / 360)
    y = TILE * (0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi))
    return x, y

def pixel_to_latlon(px, py):
    """Convert pixel coordinates back to latitude & longitude."""
    world_x = center_world_x + (px - IMG_SIZE / 2) / pixels_per_tile
    world_y = center_world_y + (py - IMG_SIZE / 2) / pixels_per_tile
    lng = (world_x / TILE - 0.5) * 360
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * world_y / TILE)))
    lat = lat_rad * 180 / math.pi
    return lat, lng

def haversine_area(coords):
    """Calculate area of polygon on Earth's surface using Haversine formula."""
    R = 6378137
    coords_rad = np.radians(coords)
    lats, lons = coords_rad[:, 0], coords_rad[:, 1]
    area = 0
    for i in range(-1, len(coords) - 1):
        area += (lons[i+1] - lons[i]) * (2 + np.sin(lats[i]) + np.sin(lats[i+1]))
    return abs(area * (R ** 2) / 2)



# --------------------------------------
# GEOSPATIAL UTILITY FUNCTIONS
# --------------------------------------

# Load DeepLabV3+ model for initial roof segmentation   
device = torch.device("cpu")
roof_model = smp.DeepLabV3Plus("resnet101", encoder_weights=None, in_channels=3, classes=1).to(device)
roof_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
roof_model.eval()

# Albumentations transform for preprocessing input image
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.5,) * 3, std=(0.5,) * 3),
    ToTensorV2()
])

# Load Facebook SAM model for secondary obstruction segmentation
sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT).to(device)
# mask_generator = SamAutomaticMaskGenerator(
#     model=sam,
#     points_per_side=32,
#     pred_iou_thresh=0.90,
#     stability_score_thresh=0.92,
#     min_mask_region_area=200
# )

# --------------------------------------
# SATELLITE IMAGE FETCHING
# --------------------------------------

def fetch_satellite_image(lat, lon):
    """Download satellite image from Google Static Maps API."""
    url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom={ZOOM}&size={IMG_SIZE}x{IMG_SIZE}&maptype=satellite&key={GOOGLE_API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        raise RuntimeError("Failed to fetch satellite image.")
    return np.array(Image.open(BytesIO(response.content)).convert("RGB"))

# --------------------------------------
# ROOF SEGMENTATION STAGE
# --------------------------------------

def predict_mask(image_np):
    """Apply DeepLabV3+ model to predict initial roof mask."""
    H, W = image_np.shape[:2]
    input_tensor = transform(image=image_np)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        pred = torch.sigmoid(roof_model(input_tensor)).squeeze().cpu().numpy()
        pred = cv2.resize(pred, (W, H), interpolation=cv2.INTER_LINEAR)
        return (pred > THRESH).astype(np.uint8) * 255

def morph_clean(mask):
    """Apply morphological operations (closing+opening) to remove noise from mask."""
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel), cv2.MORPH_OPEN, kernel)



# --------------------------------------
# OBSTRUCTION DETECTION USING SAM
# --------------------------------------

def run_sam_everything(cropped, roof_mask):
    """Run Segment Anything Model (SAM) inside cropped roof region to find obstructions."""
    if cropped.size == 0:
        print("Warning: The cropped image area is zero. Skipping obstruction analysis.")
        return roof_mask, [] # Return original mask and no obstructions
 
    #  Create a new mask generator for every run to avoid state issues
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.90,
        stability_score_thresh=0.92,
        min_mask_region_area=200,
    )
 
    masks = mask_generator.generate(cropped)
 
    # Compute color variation within the roof region to adjust obstruction threshold dynamically
    roof_pixels = cropped[roof_mask > 0]
    std_color = np.std(roof_pixels, axis=0).mean()
    max_ratio = 0.05 if std_color < 10 else 0.1 if std_color < 20 else 0.15 if std_color < 30 else 0.2
 
    # The console output of this line confirms the function is running
    print(f"Color std deviation: {std_color:.2f} → Obstruction tolerance: {max_ratio*100:.0f}%")
 
    # Filter valid obstruction masks
    roof_area = np.sum(roof_mask)
    valid_masks, temp_roof_mask = [], roof_mask.copy()
 
    for m in masks:
        binary = m["segmentation"].astype(np.uint8)
        intersection = cv2.bitwise_and(binary, roof_mask)
        if np.sum(intersection) / np.sum(binary) >= 0.9 and (np.sum(binary) / roof_area) < max_ratio:
            candidate = cv2.bitwise_and(temp_roof_mask, cv2.bitwise_not(binary))
            if np.sum(candidate) / roof_area >= MIN_ROOF_REMAIN_RATIO:
                valid_masks.append(binary)
                temp_roof_mask = candidate
 
    return temp_roof_mask, valid_masks

# --------------------------------------
# OUTPUT VISUALIZATION FUNCTION
# --------------------------------------

def show_obstruction_edges_on_original(orig_img, x_offset, y_offset, obstruction_masks):
    gps_polygons = []
    for obs_mask in obstruction_masks:
        contours, _ = cv2.findContours(obs_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if contour.shape[0] < 3: continue
            closed = contour + np.array([[[x_offset, y_offset]]])
            closed = closed.astype(np.int32)
            gps_coords = [pixel_to_latlon(x, y) for x, y in closed[:, 0, :]]
            gps_polygons.append(gps_coords)
    return gps_polygons


def save_final_overlay(orig_img, roof_contour, obstruction_masks, x_offset, y_offset, output_path=None):
    """Draw and save annotated image showing roof boundaries and obstructions."""
    overlay = orig_img.copy()
    # Draw roofline in green
    closed = np.vstack([roof_contour, roof_contour[0]])
    cv2.polylines(overlay, [closed.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

    # Draw obstructions in blue
    for i, mask in enumerate(obstruction_masks):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if contour.shape[0] < 3:
                continue
            closed = contour + np.array([[[x_offset, y_offset]]])
            closed = closed.astype(np.int32)
            cv2.polylines(overlay, [closed], isClosed=True, color=(0, 0, 255), thickness=2)
            for pt in closed[:, 0, :]:
                cv2.circle(overlay, tuple(pt), 2, (0, 0, 255), -1)
            cx, cy = np.mean(closed[:, 0, :], axis=0).astype(int)
            # cv2.putText(overlay, f"O{i+1}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Preparing and ensuring we have the output directory
    output_dir = os.path.join(ROOT_DIR, "roof_obstruction_estimator_images")
    os.makedirs(output_dir, exist_ok=True)

    # Set default output path if not provided
    if output_path is None:
        output_path = os.path.join(output_dir, "final_overlay.png")

    cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"\nSaved annotated result to: {output_path}")


# --------------------------------------
# FULL PIPELINE FUNCTION
# --------------------------------------

def run_pipeline(lat, lon, roof_angle_deg):
    """Main analysis pipeline: fetch image → segment roof → detect obstructions → calculate usable area."""
    global center_world_x, center_world_y
    center_world_x, center_world_y = _lnglat_to_world(lon, lat)

    orig = fetch_satellite_image(lat, lon)
    H, W = orig.shape[:2]

    # Step 1: Predict rough roof mask
    mask = morph_clean(predict_mask(orig))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the roof contour closest to image center
    if not contours:
        print("No roof detected.")
        return None # Return None if no roof is found

    def centroid(c):
        M = cv2.moments(c)
        if M["m00"] == 0: return float('inf')
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        return np.sqrt((cx - W // 2)**2 + (cy - H // 2)**2)

    best = min(contours, key=centroid)
    roof_mask = np.zeros_like(mask)
    cv2.drawContours(roof_mask, [best], -1, 255, -1)

    # Crop around detected roof for further fine segmentation
    ys, xs = np.where(roof_mask == 255)
    x1, y1, x2, y2 = max(xs.min()-PADDING,0), max(ys.min()-PADDING,0), min(xs.max()+PADDING,W), min(ys.max()+PADDING,H)
    cropped, crop_mask = orig[y1:y2, x1:x2], roof_mask[y1:y2, x1:x2]

    # Step 2: Run obstruction detection using SAM
    clean_mask, obstructions = run_sam_everything(cropped, (crop_mask > 0).astype(np.uint8))

    contours2, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours2:
        print("No valid roof after cleaning.")
        return None # Return None if cleaning fails

    # Simplify final roof contour for cleaner visualization
    final = max(contours2, key=cv2.contourArea)
    approx = cv2.approxPolyDP(final, EPSILON_RATIO * cv2.arcLength(final, True), True).squeeze()
    approx[:, 0] += x1; approx[:, 1] += y1
    roof_coords = [pixel_to_latlon(x, y) for x, y in approx]

    # Step 3: Calculate areas (haversine projection)
    gps_polygons = show_obstruction_edges_on_original(orig, x1, y1, obstructions)
    roof_area_flat = haversine_area(np.array(roof_coords))
    tilt_factor = 1 / math.cos(math.radians(roof_angle_deg))
    roof_area_actual = roof_area_flat * tilt_factor
    obstruction_area = sum(haversine_area(np.array(poly)) for poly in gps_polygons)
    usable_area = roof_area_actual - obstruction_area
    
    # Save annotated result image
    output_dir = os.path.join(ROOT_DIR, "roof_obstruction_estimator_images")
    os.makedirs(output_dir, exist_ok=True)
    # Use a unique filename to prevent browser caching issues
    output_path = os.path.join(output_dir, f"final_overlay_{lat}{lon}{roof_angle_deg}.png")

    save_final_overlay(orig, approx, obstructions, x1, y1, output_path=output_path)
    
    # RETURN the results instead of printing them
    return {
        "adjusted_roof_area": roof_area_actual,
        "obstruction_area": obstruction_area,
        "usable_area": usable_area,
        "output_image_path": output_path
    }

# --------------------------------------
# CLI ENTRY POINT
# --------------------------------------
if __name__ == "__main__":
    coord_input = input("Enter coordinates (lat, lon): ").strip()
    try:
        lat_str, lon_str = coord_input.split(",")
        lat = float(lat_str.strip())
        lon = float(lon_str.strip())
    except ValueError:
        print("Invalid format! Use: 52.43464, 9.73011")
        exit()

    angle = float(input("Enter Roof Angle (°): "))
    run_pipeline(lat, lon, angle)

