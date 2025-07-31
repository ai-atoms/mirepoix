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
    def __init__(self, weights=None, output_dim=1536, base_model='b0'):
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
        elif base_model == 'b5':
            self.model = models.efficientnet_b5(weights=weights)
            in_features = 2048  # EfficientNet-B5 features
        else:
            raise ValueError(f"Unsupported base model: {base_model}")
        
        # Replace classifier with regression head
        self.model.classifier = nn.Identity()  # Remove original classifier
        
        # Simplified regression head for smaller output dimension
        self.regression_head = nn.Sequential(
            nn.Linear(in_features, output_dim)
        )
        
        # Modify first convolutional layer for grayscale if needed
        self.model.features[0][0] = nn.Conv2d(
            1, self.model.features[0][0].out_channels,
            kernel_size=3, stride=2, padding=1, bias=False
        )
        
    def forward(self, x):
        features = self.model(x)
        return self.regression_head(features)


class EfficientNetRegression2HL(nn.Module):
    def __init__(self, weights=None, output_dim=1536, base_model='b0'):
        super(EfficientNetRegression2HL, self).__init__()
        
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
        elif base_model == 'b5':
            self.model = models.efficientnet_b5(weights=weights)
            in_features = 2048  # EfficientNet-B5 features
        else:
            raise ValueError(f"Unsupported base model: {base_model}")
        
        # Replace classifier with regression head
        self.model.classifier = nn.Identity()  # Remove original classifier
        
        # Simplified regression head for smaller output dimension
        self.regression_head = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),   # 2nd layer
            nn.ReLU(),
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
        elif base_model == 'b5':
            self.model = models.efficientnet_b5(weights=weights)
            in_features = 2048  # EfficientNet-B5 features
        else:
            raise ValueError(f"Unsupported base model: {base_model}")
        
        # Replace classifier with regression head
        self.model.classifier = nn.Identity()  # Remove original classifier
        
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

        # Get all image files and corresponding .npy files
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        self.target_files = [os.path.join(npy_dir, os.path.splitext(f)[0] + '.npy') 
                           for f in self.img_files]

        # Load all targets
        self.targets = []
        
        for npy_file in self.target_files:
            target = np.load(npy_file)            
            self.targets.append(target)

        # Convert to tensors
        self.targets = torch.tensor(np.array(self.targets), dtype=torch.float32)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = Image.open(img_path).convert("L")  # Grayscale

        if self.transform:
            image = self.transform(image)

        target = self.targets[idx]          # Original targets
        return image, target

class DefectDatasetNamed(Dataset):
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

        # Get all image files and corresponding .npy files
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        self.target_files = [os.path.join(npy_dir, os.path.splitext(f)[0] + '.npy') 
                           for f in self.img_files]

        # Load all targets
        self.targets = []
        
        for npy_file in self.target_files:
            target = np.load(npy_file)            
            self.targets.append(target)

        # Convert to tensors
        self.targets = torch.tensor(np.array(self.targets), dtype=torch.float32)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = Image.open(img_path).convert("L")  # Grayscale

        if self.transform:
            image = self.transform(image)

        target = self.targets[idx]          # Original targets
        
        # -- recover the name
        filename = os.path.basename(img_path).split('.png')[0]        
        return image, target, filename


# Example usage
if __name__ == "__main__":
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])
    
    root_dir = 'data/datasets/dsc_d9/'
    dataset = DefectDataset(f'{root_dir}/images', f'{root_dir}/rtargets', transform=transform)
    dataloader = DataLoader(dataset, batch_size=50, shuffle=True)
    
    model = EfficientNetRegression(weights=None) 
    
    images, targets = next(iter(dataloader))
    
    # testing regression model
    output = model(images)
    print(output.shape)  # Should output torch.Size([batch_size, 1536])
