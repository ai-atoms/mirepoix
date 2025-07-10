import torch
import numpy as np
import os
from torchvision.utils import save_image

# Set fixed seed for reproducibility (though brightness is deterministic)
torch.manual_seed(42)
np.random.seed(42)

# Create directories
output_img_dir = "temp/dummy_grayscale_images"
output_target_dir = "temp/dummy_targets"
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_target_dir, exist_ok=True)

# Parameters
num_images = 64
start_idx = 1
image_size = (1, 1236, 1236)  # Grayscale (1 channel), 64x64 pixels
target_size = (640,)          # Target vector shape

# Brightness range (64 to 192 out of 255)
min_brightness = 64 / 255.0
max_brightness = 192 / 255.0
brightness_step = (max_brightness - min_brightness) / (num_images - 1)

for i in range(num_images):
    # --- Generate dummy image with controlled brightness ---
    brightness = min_brightness + i * brightness_step
    dummy_image = torch.full(image_size, brightness)  # Uniform brightness
    
    # --- Generate all-zero target with fixed pattern ---
    dummy_target = np.zeros(target_size)
    
    # Format the filename (e.g., 00_000_000803)
    idx_str = f"{start_idx + i:06d}"
    filename = f"99_000_{idx_str}"
    
    # Save image (.png)
    img_path = os.path.join(output_img_dir, f"{filename}.png")
    save_image(dummy_image, img_path)
    
    # Save target (.npy)
    target_path = os.path.join(output_target_dir, f"{filename}.npy")
    np.save(target_path, dummy_target)
    
    print(f"Saved: {filename}.png (brightness={brightness:.3f}) & {filename}.npy")

print(f"\nGenerated {num_images} dummy images + structured targets!")