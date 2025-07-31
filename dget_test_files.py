import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from dmodel import EfficientNetRegression, DefectDatasetNamed
from tqdm import tqdm

# Deterministic runs
seed = 169006142
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Define transformations including random crop and rescale
transform = T.Compose([
    # T.RandomCrop(618),
    T.Resize((512, 512)),
    T.RandomApply([T.RandomRotation((0, 90))], p=0.25),
    T.RandomHorizontalFlip(p=0.25), 
    T.RandomVerticalFlip(p=0.25),
    T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2)], p=0.25),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5])  # normalization [-1, 1]
])

# Load dataset
dataset = DefectDatasetNamed("data/datasets/d567/images", "data/datasets/d567/rtargets", transform=transform)

# Split dataset into training and testing sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Export test filenames without extension
with open("test_filenames.txt", "w") as f:
    for idx in test_dataset.indices:
        stem = os.path.splitext(dataset.img_files[idx])[0]
        f.write(f"{stem}\n")



