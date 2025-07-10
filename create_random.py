import torch
import numpy as np
import os
from torchvision.utils import save_image

# Set fixed random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create directories
output_img_dir = "temp/random_images"
output_target_dir = "temp/random_targets"
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_target_dir, exist_ok=True)

# Parameters
num_images = 64
start_idx = 101
image_size = (1, 1236, 1236)  # Grayscale (1 channel), 64x64 pixels
target_size = (640,)          # Target vector shape

for i in range(num_images):
    # --- Generate random image ---
    random_image = torch.rand(image_size)
    
    # --- Generate random target vector ---
    random_target = np.random.rand(*target_size)  # Values in [0, 1)
    
    # Format the filename (e.g., 00_000_000803)
    idx_str = f"{start_idx + i:06d}"
    filename = f"99_000_{idx_str}"
    
    # Save image (.png)
    img_path = os.path.join(output_img_dir, f"{filename}.png")
    save_image(random_image, img_path)
    
    # Save target (.npy)
    target_path = os.path.join(output_target_dir, f"{filename}.npy")
    np.save(target_path, random_target)
    
    print(f"Saved: {filename}.png & {filename}.npy")

print(f"\nGenerated {num_images} images + structured targets!")