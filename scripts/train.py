import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.unet import UNet
from utils.dataset import PlantDiseaseDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Hyperparameters
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 25
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

# Define transformations
transform = A.Compose(
    [
        A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)

# Create datasets
train_dataset = PlantDiseaseDataset(
    image_dir="data/train_images",
    mask_dir="data/train_masks",
    transform=transform,
)
val_dataset = PlantDiseaseDataset(
    image_dir="data/val_images",
    mask_dir="data/val_masks",
    transform=transform,
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
criterion = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        outputs = model(images)  # Shape: [batch_size, 1, height, width]
        outputs = outputs.squeeze(1)  # Remove channel dimension: [batch_size, height, width]
        loss = criterion(outputs, masks)  # Masks shape: [batch_size, height, width]
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {running_loss/len(train_loader):.4f}")

# Save the model
torch.save(model.state_dict(), "outputs/checkpoints/unet_plant_disease.pth")