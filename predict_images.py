import os
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import segmentation_models_pytorch as smp
import cv2
from skimage import measure
from skimage.morphology import remove_small_objects

# ----------- CONFIGURATION -----------
MODEL_PATH = "saved_models/best_deeplabv3plus.pth"
IMAGE_PATH = "test_images/image_2.png"  # update this
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 128
THRESHOLD = 0.4
APPLY_MORPH = True
FILTER_COMPONENTS = True
MIN_COMPONENT_AREA = 100
APPLY_OVERLAY = True
# -------------------------------------

# Morphology post-processing
def apply_morphology(mask_tensor):
    mask_np = mask_tensor.squeeze().cpu().numpy().astype(bool)
    cleaned = remove_small_objects(mask_np, min_size=200)
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cleaned.astype(np.uint8)
    closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return torch.tensor(opened).unsqueeze(0)

# Filter small connected components
def filter_small_components(mask_tensor, min_area=100):
    mask_np = mask_tensor.squeeze().cpu().numpy().astype(np.uint8)
    labeled = measure.label(mask_np)
    cleaned = np.zeros_like(mask_np)
    for region in measure.regionprops(labeled):
        if region.area >= min_area:
            cleaned[labeled == region.label] = 1
    return torch.tensor(cleaned).unsqueeze(0)

# Load image with preprocessing
def load_image(path):
    img = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.CenterCrop(min(img.size)),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)  # add batch dimension

# Load model
model = smp.DeepLabV3Plus(
    encoder_name="resnet101",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation="sigmoid"
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Inference
image_tensor = load_image(IMAGE_PATH).to(DEVICE)

with torch.no_grad():
    pred = model(image_tensor)
    print("Prediction stats - min:", pred.min().item(), "max:", pred.max().item())
    pred_mask = (pred > THRESHOLD).float()

    if APPLY_MORPH:
        pred_mask = apply_morphology(pred_mask)

    if FILTER_COMPONENTS:
        pred_mask = filter_small_components(pred_mask, MIN_COMPONENT_AREA)

# Visualization
plt.figure(figsize=(15, 5))

# Input
plt.subplot(1, 3, 1)
plt.imshow(image_tensor[0].permute(1, 2, 0).cpu())
plt.title("Input Image")
plt.axis("off")

# Prediction Mask
plt.subplot(1, 3, 2)
plt.imshow(pred_mask.squeeze().cpu(), cmap='gray')
plt.title("Predicted Mask")
plt.axis("off")

# Overlay
if APPLY_OVERLAY:
    overlay = image_tensor[0].permute(1, 2, 0).cpu().numpy().copy()
    mask_np = pred_mask.squeeze().cpu().numpy()
    color_mask = np.zeros_like(overlay)
    color_mask[..., 1] = mask_np * 255  # Green channel
    alpha = 0.4
    overlayed = (overlay * 255 * (1 - alpha) + color_mask * alpha).astype(np.uint8)

    plt.subplot(1, 3, 3)
    plt.imshow(overlayed)
    plt.title("Overlay: Prediction")
    plt.axis("off")

plt.tight_layout()
plt.show()
