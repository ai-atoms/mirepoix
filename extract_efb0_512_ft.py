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
from dmodel import EfficientNetFeatureExtractor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize EN-B0 (fine-tuned FeatExtract)
feature_extractor = EfficientNetFeatureExtractor().to(device)
full_state_dict = torch.load("data/models/dsc_d9_20250624_1113/best_model.pth", map_location=device)
filtered_state_dict = {k: v for k, v in full_state_dict.items() if not k.startswith('regression_head')}
feature_extractor.model.load_state_dict(filtered_state_dict, strict=False)
feature_extractor.eval()

# Test it
with torch.no_grad():
    features = feature_extractor(torch.randn(1, 1, 512, 512).to(device))
    print(features.shape)  # Should be [1, 1280]

# Define image transform
transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5])
])

# Input and output directories
input_dir = 'data/sample/abtem/'
output_dir = os.path.join(os.path.dirname(os.path.dirname(input_dir)), 'img_descriptors/efb0_512_ft')

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process all PNG images in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.png'):
        # Load and transform image
        image_path = os.path.join(input_dir, filename)
        image = Image.open(image_path).convert("L") # Grayscale
        img_t = transform(image)
        
        # Get EFN (FT-FE) embedding
        with torch.no_grad():
            enc_img_t = feature_extractor(img_t.unsqueeze(0).to(device)) # using GPU
            enc_img_t = enc_img_t.cpu().detach().numpy() # safe conversion
        
        # Save embedding
        output_path = os.path.join(output_dir, filename.replace('.png', '.npy'))
        np.save(output_path, enc_img_t)
        
        print(f"Processed {filename}")

print("All images processed successfully!")