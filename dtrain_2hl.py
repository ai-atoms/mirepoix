import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from dmodel import EfficientNetRegression2HL, DefectDataset
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
dataset = DefectDataset("data/datasets/d567b/images", "data/datasets/d567b/rtargets", transform=transform)

# Split dataset into training and testing sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# -- Initialize model, loss function, and optimizer
model = EfficientNetRegression2HL(base_model='b0', weights='DEFAULT', output_dim=640) # weights='DEFAULT' or None
# model.load_state_dict(torch.load('efficientnet_b3_rwightman-b3899882.pth'), strict=False)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training config
epochs = 256
eval_interval = 4
log_file = "training_log.csv"
best_model_path = 'best_model.pth'

# Initialize logging
with open(log_file, 'w') as f:
    csv.writer(f).writerow(["epoch", "train_loss", "test_loss"])

best_loss = float('inf')

for epoch in range(epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # Evaluation phase
    if (epoch + 1) % eval_interval == 0:
        model.eval()
        test_loss = 0.0
        
        with torch.no_grad():
            for images, targets in test_loader:
                outputs = model(images.to(device))
                test_loss += criterion(outputs, targets.to(device)).item()
        
        test_loss /= len(test_loader)
        
        # Save best model
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), best_model_path)
    
    # Logging
    test_loss_log = test_loss if (epoch + 1) % eval_interval == 0 else "NA"
    with open(log_file, 'a') as f:
        csv.writer(f).writerow([epoch+1, train_loss, test_loss_log])
    
    print(f"Epoch {epoch+1}/{epochs}: "
          f"Train Loss: {train_loss:.4f} | "
          f"Test Loss: {test_loss_log if isinstance(test_loss_log, float) else 'NA'}")

# Final evaluation
model.eval()
final_loss = 0.0
with torch.no_grad():
    for images, targets in test_loader:
        outputs = model(images.to(device))
        final_loss += criterion(outputs, targets.to(device)).item()

final_loss /= len(test_loader)

# Save final results
with open(log_file, 'a') as f:
    csv.writer(f).writerow(["final", "NA", final_loss])

# Version-controlled model save
import datetime
os.makedirs("saved_models", exist_ok=True)  # Create directory if needed
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
save_path = f"saved_models/enb0_2hl_d567b_reg640_{timestamp}.pth"
torch.save(model.state_dict(), save_path)

# Empty cache
print(f"\nTraining complete. Final test loss: {final_loss:.4f}")
torch.cuda.empty_cache()

