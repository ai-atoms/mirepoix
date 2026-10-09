'''
python3
github.com @ camilofs
'''

import argparse
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
    T.Resize((384, 384)), # or (224, 224)
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5])
])

ap = argparse.ArgumentParser(description='Encode PNG images with ConvNeXt-Large (1536-d vector per image)')
ap.add_argument('--input', default='data/fpa70a/images')
ap.add_argument('--output', default='data/fpa70a/enc_images/convnext_in')
ap.add_argument('--limit', type=int, help='only encode the first N images')
args = ap.parse_args()
os.makedirs(args.output, exist_ok=True)

filenames = sorted(f for f in os.listdir(args.input) if f.endswith('.png'))[:args.limit]
for filename in filenames:
    image = Image.open(os.path.join(args.input, filename)).convert("RGB")
    img_t = transform(image)
    with torch.no_grad():
        enc_img_t = convnext(img_t.unsqueeze(0)).detach().numpy()
    np.save(os.path.join(args.output, filename.replace('.png', '.npy')), enc_img_t)
    print(f"Processed {filename}")

print("All images processed successfully!")
