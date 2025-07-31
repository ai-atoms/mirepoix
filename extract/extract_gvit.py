'''
python3
github.com @ camilofs
'''

import os
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

# Initialize Google's Vision Transformer
# /home/camilofs/.cache/torch/hub/checkpoints/vit_l_16-852ce7e3.pth
gvit_l16 = models.vit_l_16(pretrained=True) # large model ~ 300 M
gvit_l16.heads.head = nn.Identity() # remove head
gvit_l16.eval()

# Sanity check
print(gvit_l16(torch.randn(1,3,224,224)).shape)  # Should be [1, 1024]

# Define image transform
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5])
])

# Input and output directories
input_dir = 'data/datasets/d567b/images'
output_dir = os.path.join(os.path.dirname(input_dir), 'enc_images/vitl16_in')

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process all PNG images in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.png'):
        # Load and transform image
        image_path = os.path.join(input_dir, filename)
        image = Image.open(image_path).convert("RGB")
        img_t = transform(image)
        
        # Get Google's ViT embedding
        with torch.no_grad():
            enc_img_t = gvit_l16(img_t.unsqueeze(0)).detach().numpy()
        
        # Save embedding
        output_path = os.path.join(output_dir, filename.replace('.png', '.npy'))
        np.save(output_path, enc_img_t)
        
        print(f"Processed {filename}")

print("All images processed successfully!")
