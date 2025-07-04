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

# Initialize ConvNeXt-Large model
# /home/camilofs/.cache/torch/hub/checkpoints/convnext_large-ea097f82.pth
convnext = models.convnext_large(weights=models.ConvNeXt_Large_Weights.DEFAULT)
# convnext.classifier = nn.Identity() # remove head
convnext.classifier = nn.Sequential(
    nn.Flatten()  # Removes [1,1536,1,1] → [1,1536]
)
convnext.eval()  # Set to eval mode

# Sanity check
print(convnext(torch.randn(1,3,224,224)).shape)  # Should be [1, 1536]

# Define image transform
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5])
])

# Input and output directories
input_dir = 'data/sample/abtem/'
output_dir = os.path.join(os.path.dirname(os.path.dirname(input_dir)), 'img_descriptors/convnext')

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process all PNG images in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.png'):
        # Load and transform image
        image_path = os.path.join(input_dir, filename)
        image = Image.open(image_path).convert("RGB")
        img_t = transform(image)
        
        # Get DINOv2 embedding
        with torch.no_grad():
            enc_img_t = convnext(img_t.unsqueeze(0)).detach().numpy()
        
        # Save embedding
        output_path = os.path.join(output_dir, filename.replace('.png', '.npy'))
        np.save(output_path, enc_img_t)
        
        print(f"Processed {filename}")

print("All images processed successfully!")
