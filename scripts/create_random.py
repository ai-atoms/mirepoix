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
image_size = (1, 128, 128)    # Grayscale (1 channel), 1236x1236 pixels
target_size = (640,)          # Target vector shape

for i in range(num_images):
    # --- Generate random image ---
    random_image = torch.rand(image_size)
    
    # --- Generate random target vector ---
    random_target = np.random.normal(loc=0.5, scale=0.02, size=target_size)
    random_target = np.clip(random_target, 0.4, 0.6)
    
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