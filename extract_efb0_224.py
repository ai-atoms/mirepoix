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
efb0 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=False)

# Remove the original classifier (and add pooling to obtain [1, 1280, 7, 7] -> [1, 1280])
# efb0.classifier = nn.Identity() # ! 
efb0.classifier = nn.Sequential(
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten()
)

# Load and filter state dict
url = pmm.util.get_pretrained_microscopynet_url('efficientnet-b0', 'micronet')
state_dict = torch.hub.load_state_dict_from_url(url, map_location=torch.device('cpu'))
filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('classifier.')}
efb0.load_state_dict(filtered_state_dict, strict=False)

# Sanity check
print(efb0(torch.randn(1,3,224,224)).shape)  # Should be [1, 1280]

# Define image transform
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5])
])

# Input and output directories
input_dir = 'data/sample/abtem/'
output_dir = os.path.join(os.path.dirname(os.path.dirname(input_dir)), 'img_descriptors/efb0_224')

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