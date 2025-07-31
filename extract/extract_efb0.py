'''
python3
github.com @ camilofs
'''

import os
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as T
import pretrained_microscopy_models as pmm

# Initialize EN-B0 with DEFAULT weights 
from torchvision.models import efficientnet_b0
efb0_full = efficientnet_b0(weights='DEFAULT')
efb0 = nn.Sequential(*list(efb0_full.features.children()))  # Only encoder

# Remove the original classifier (and add pooling to obtain [1, 2048, 7, 7] -> [1, 2048])
# efb0.classifier = nn.Identity() # ! 
efb0.classifier = nn.Sequential(
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten()
)

# Sanity check
print(efb0(torch.randn(1, 3, 256, 256)).shape)  # Should be [1, 2048]

# Define image transform
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5])
])

# Input and output directories
input_dir = 'data/datasets/d567b/images'
output_dir = os.path.join(os.path.dirname(input_dir), 'enc_images/efb0_in')

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process all PNG images in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.png'):
        # Load and transform image
        image_path = os.path.join(input_dir, filename)
        image = Image.open(image_path).convert("RGB")
        img_t = transform(image)
        
        # Get EFN embedding
        with torch.no_grad():
            enc_img_t = efb0(img_t.unsqueeze(0)).detach().numpy()
        
        # Save embedding
        output_path = os.path.join(output_dir, filename.replace('.png', '.npy'))
        np.save(output_path, enc_img_t)
        
        print(f"Processed {filename}")

print("All images processed successfully!")