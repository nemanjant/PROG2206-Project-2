import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.unet import UNet
from utils.dataset import PlantDiseaseDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
model.load_state_dict(torch.load("outputs/checkpoints/unet_plant_disease.pth"))
model.eval()

# Define transformations
transform = A.Compose(
    [
        A.Resize(256, 256),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)

# Load and preprocess the image
image_path = "data/test_images/test_image.jpg"
image = Image.open(image_path).convert("RGB")
image_transformed = transform(image=np.array(image))["image"].unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    output = model(image_transformed)
    output = output.squeeze(0).squeeze(0).cpu().numpy()  # Remove batch and channel dimensions
    output = (output > 0.2).astype(np.uint8)  # Apply threshold (adjust as needed)

# Convert transformed image back to numpy for display
image_transformed_np = image_transformed.squeeze(0).permute(1, 2, 0).cpu().numpy()

# Display the test image and predicted mask side by side
plt.figure(figsize=(10, 5))

# Subplot 1: Test Image
plt.subplot(1, 2, 1)
plt.imshow(image_transformed_np)
plt.title("Test Image")
plt.axis('off')  # Hide axes

# Subplot 2: Predicted Mask
plt.subplot(1, 2, 2)
plt.imshow(output, cmap='gray')
plt.title("Predicted Mask")
plt.axis('off')  # Hide axes

plt.savefig("outputs/predictions/test_image_with_mask.png")

plt.tight_layout()  # Adjust spacing between subplots
plt.show()