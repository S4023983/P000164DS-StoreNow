# utils/visualizer.py
import torch
from PIL import Image
import matplotlib.pyplot as plt

def predict_and_visualize(img_path, model, transform, device):
    model.eval()
    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(input_tensor)
        mask = (pred > 0.5).float().squeeze().cpu()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Satellite Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Predicted Roof Mask")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return mask
