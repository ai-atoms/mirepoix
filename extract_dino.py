'''
python3
github.com @ camilofs
'''

import os
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as T
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize DINOv2 model
# /home/camilofs/.cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth
dinov2_vitx14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14') # large model ~ 300 M
dinov2_vitx14.eval().to(device) # using GPU

# Define image transform
transform = T.Compose([
    T.Resize((588, 588)), # multiple of patch height = 14
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5])
])

# Input and output directories
input_dir = 'data/sample/abtem/'
output_dir = os.path.join(os.path.dirname(os.path.dirname(input_dir)), 'img_descriptors/dino')

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
            enc_img_t = dinov2_vitx14(img_t.unsqueeze(0).to(device)) # using GPU
            enc_img_t = enc_img_t.cpu().detach().numpy() # safe conversion
        
        # Save embedding
        output_path = os.path.join(output_dir, filename.replace('.png', '.npy'))
        np.save(output_path, enc_img_t)
        
        print(f"Processed {filename}")

print("All images processed successfully!")
