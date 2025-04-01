import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class PlantDiseaseDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace(".jpg", ".png"))  # Adjust extension if needed
        
        # Load image and mask
        image = np.array(Image.open(img_path).convert("RGB"))  # Ensure image is in RGB format
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)  # Convert mask to grayscale
        
        # Normalize mask to [0, 1]
        mask = mask / 255.0
        
        # Apply transformations (if any)
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        
        return image, mask