import os
import json
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet

# === Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = "dataset_classification"
batch_size = 16
epochs = 10

# === Transforms ===
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === Load Dataset ===
dataset = datasets.ImageFolder(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# === Label Mapping ===
label_to_index = dataset.class_to_idx
os.makedirs("saved_models", exist_ok=True)
with open("saved_models/label_map.json", "w") as f:
    json.dump(label_to_index, f)

# === Model Setup ===
n_classes = len(label_to_index)
model = EfficientNet.from_name("efficientnet-b3", num_classes=n_classes)
model = model.to(device)

# === Loss and Optimizer ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# === Training Loop ===
model.train()
for epoch in range(epochs):
    total_loss = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

# === Save Model ===
torch.save(model.state_dict(), "saved_models/roof_model.pth")
print("✅ Model and label map saved to 'saved_models/'")
