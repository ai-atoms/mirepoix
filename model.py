import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader


class EfficientNetRegression(nn.Module):
    def __init__(self, weights=None, output_dim=768, base_model='b0'):
        super(EfficientNetRegression, self).__init__()
        
        # Select base model
        if base_model == 'b0':
            self.model = models.efficientnet_b0(weights=weights)
            in_features = 1280  # EfficientNet-B0 features
        elif base_model == 'b3':
            self.model = models.efficientnet_b3(weights=weights)
            in_features = 1536  # EfficientNet-B3 features
        elif base_model == 'b4':
            self.model = models.efficientnet_b4(weights=weights)
            in_features = 1792  # EfficientNet-B4 features
        else:
            raise ValueError(f"Unsupported base model: {base_model}")
        
        # Replace classifier with regression head
        self.model.classifier = nn.Identity()  # Remove original classifier
        
        # Simplified regression head for smaller output dimension
        self.regression_head = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            
            nn.Linear(1024, output_dim)
        )
        
        # Modify first convolutional layer for grayscale if needed
        self.model.features[0][0] = nn.Conv2d(
            1, self.model.features[0][0].out_channels,
            kernel_size=3, stride=2, padding=1, bias=False
        )
        
    def forward(self, x):
        features = self.model(x)
        return self.regression_head(features)


class EfficientNetFeatureExtractor(nn.Module):
    def __init__(self, weights=None, base_model='b0'):
        super(EfficientNetFeatureExtractor, self).__init__()
        
        # Select base model
        if base_model == 'b0':
            self.model = models.efficientnet_b0(weights=weights)
            in_features = 1280  # EfficientNet-B0 features
        elif base_model == 'b3':
            self.model = models.efficientnet_b3(weights=weights)
            in_features = 1536  # EfficientNet-B3 features
        elif base_model == 'b4':
            self.model = models.efficientnet_b4(weights=weights)
            in_features = 1792  # EfficientNet-B4 features
        else:
            raise ValueError(f"Unsupported base model: {base_model}")
        
        # Remove original classifier
        self.model.classifier = nn.Identity()
        
        # Modify first convolutional layer for grayscale
        self.model.features[0][0] = nn.Conv2d(
            1, self.model.features[0][0].out_channels,
            kernel_size=3, stride=2, padding=1, bias=False
        )
        
    def forward(self, x):
        return self.model(x)  # Returns 1280-dim features


class DefectDataset(Dataset):
    def __init__(self, img_dir, npy_dir, transform=None):
        """
        Args:
            img_dir: Directory with image files (.png)
            npy_dir: Directory with corresponding .npy files
            transform: Optional transform to be applied to images
        """
        self.img_dir = img_dir
        self.npy_dir = npy_dir
        self.transform = transform

        # Get all image files and verify corresponding .npy files
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
        self.target_files = [os.path.join(npy_dir, os.path.splitext(f)[0] + '.npy') 
                           for f in self.img_files]

        # Define 6 sections (d1-d6), each with 128 values
        self.sections = {
            'd1': slice(0, 128),
            'd2': slice(128, 256), 
            'd3': slice(256, 384),
            'd4': slice(384, 512),
            'd5': slice(512, 640),
            'd6': slice(640, 768)
        }

        # Load all targets and calculate per-section scaling
        self.scalers = {name: {'min': np.inf, 'max': -np.inf} 
                       for name in self.sections}
        raw_targets = []
        
        # First pass: find min/max for each section
        for npy_file in self.target_files:
            target = np.load(npy_file)
            raw_targets.append(target)
            for name, slc in self.sections.items():
                section = target[slc]
                valid = section[~np.isnan(section)]
                if len(valid) > 0:
                    self.scalers[name]['min'] = min(self.scalers[name]['min'], valid.min())
                    self.scalers[name]['max'] = max(self.scalers[name]['max'], valid.max())

        # Second pass: apply scaling
        self.scaled_targets = []
        for target in raw_targets:
            scaled = np.zeros_like(target)
            for name, slc in self.sections.items():
                section = target[slc]
                mask = ~np.isnan(section)
                if self.scalers[name]['max'] > self.scalers[name]['min']:  # Avoid division by zero
                    scaled[slc][mask] = (section[mask] - self.scalers[name]['min']) / \
                                      (self.scalers[name]['max'] - self.scalers[name]['min'])
                else:
                    scaled[slc][mask] = 0.0  # Handle constant sections
            self.scaled_targets.append(scaled)

        # Convert to tensor
        self.targets = torch.tensor(np.array(self.scaled_targets), dtype=torch.float32)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = Image.open(img_path).convert("L")  # Grayscale

        if self.transform:
            image = self.transform(image)

        # Get target and mask
        target = self.targets[idx]
        mask = ~torch.isnan(target)  # Mask for valid values
        target = torch.nan_to_num(target, nan=0.0)  # Replace NaN with 0

        return image, target, mask

    def inverse_scale(self, tensor):
        """Convert scaled predictions back to original scale per-section"""
        numpy_arr = tensor.numpy() if isinstance(tensor, torch.Tensor) else tensor
        restored = np.zeros_like(numpy_arr)
        
        for name, slc in self.sections.items():
            scale = self.scalers[name]['max'] - self.scalers[name]['min']
            center = self.scalers[name]['min']
            restored[:, slc] = numpy_arr[:, slc] * scale + center
        
        return torch.tensor(restored, dtype=torch.float32) if isinstance(tensor, torch.Tensor) else restored


class MaskedMSELoss(nn.Module):
    def forward(self, preds, targets, mask):
        squared_error = (preds - targets)**2
        masked_error = squared_error * mask
        valid_count = max(mask.sum(), 1)  # Prevent division by zero
        return (masked_error.sum() / valid_count, valid_count)


# Example usage
if __name__ == "__main__":
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])
    
    dataset = DefectDataset("data/images/b1_64/300K", "data/targets/b1_64", transform=transform)
    dataloader = DataLoader(dataset, batch_size=50, shuffle=True)
    
    model = EfficientNetRegression(weights=None) 
    
    images, targets, mask = next(iter(dataloader))
    output = model(images)
    print(output.shape)  # Should output torch.Size([batch_size, 768])
