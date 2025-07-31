import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from dmodel import EfficientNetRegression2HL, DefectDatasetNamed
from tqdm import tqdm
import numpy as np

# Deterministic runs
seed = 169006142
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Define transformations including random crop and rescale
transform = T.Compose([
    # T.RandomCrop(618),
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5])  # normalization [-1, 1]
])

# Load dataset
dataset = DefectDatasetNamed("data/datasets/d567b/images", "data/datasets/d567b/rtargets", transform=transform)

# Split dataset into training and testing sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# -- Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EfficientNetRegression2HL(base_model='b0', weights=None, output_dim=640)
model.load_state_dict(torch.load("saved_models/d567b/mirepoix_efn_b0_2hl/best_model.pth", weights_only=True, map_location=device))

# -- Predictions
# Create the output directory if it doesn't exist
os.makedirs('results/d567b/predictions', exist_ok=True)

# Move model to device
model = model.to(device)
model.eval()  # Set to evaluation mode

# Create dataloader
dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

total_mse = 0.0
num_samples = 0
sample_index = 0  # For naming output files

with torch.no_grad():
    for batch in dataloader:
        images, targets, filenames = batch
        images = images.to(device)
        targets = targets.to(device)

        predictions = model(images)

        batch_mse = F.mse_loss(predictions, targets, reduction='sum')
        total_mse += batch_mse.item()
        num_samples += targets.size(0)

        predictions_np = predictions.cpu().numpy()

        for pred, fname in zip(predictions_np, filenames):
            save_path = os.path.join("results/d567b/predictions", fname)
            np.save(save_path, pred)


# Calculate overall MSE
final_mse = total_mse / num_samples
print(f"Mean Squared Error: {final_mse:.4f}")