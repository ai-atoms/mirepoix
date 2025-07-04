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

# Initialize EN-B0 with MicroNet weights 
# /home/camilofs/.cache/torch/hub/checkpoints/efficientnet-b0_pretrained_microscopynet_v1.0.pth.tar
from torchvision.models import efficientnet_b5
efb5_full = efficientnet_b5(weights=None)
efb5 = nn.Sequential(*list(efb5_full.features.children()))  # Only encoder

# Remove the original classifier (and add pooling to obtain [1, 2048, 7, 7] -> [1, 2048])
# efb0.classifier = nn.Identity() # ! 
efb5.classifier = nn.Sequential(
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten()
)

# Load and filter state dict
url = pmm.util.get_pretrained_microscopynet_url('efficientnet-b5', 'micronet')
state_dict = torch.hub.load_state_dict_from_url(url, map_location=torch.device('cpu'))
filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('classifier.')}
efb5.load_state_dict(filtered_state_dict, strict=False)

# Sanity check
print(efb5(torch.randn(1, 3, 456, 456)).shape)  # Should be [1, 2048]

# Define image transform
transform = T.Compose([
    T.Resize((456, 456)),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5])
])

# Input and output directories
input_dir = 'outputs_f60'
output_dir = os.path.join(os.path.dirname(os.path.dirname(input_dir)), 'img_descriptors/efb5_456')

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
            enc_img_t = efb5(img_t.unsqueeze(0)).detach().numpy()
        
        # Save embedding
        output_path = os.path.join(output_dir, filename.replace('.png', '.npy'))
        np.save(output_path, enc_img_t)
        
        print(f"Processed {filename}")

print("All images processed successfully!")