import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import pandas as pd

seed = 169006142
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# ------------------ Dataset ------------------
class NPYRegressionDataset(Dataset):
    def __init__(self, input_folder, target_folder, normalize_y=True):
        self.input_folder = input_folder
        self.target_folder = target_folder
        self.normalize_y = normalize_y

        # Paired filenames (only those that exist in both folders)
        self.filenames = sorted([
            f for f in os.listdir(input_folder)
            if f.endswith('.npy') and os.path.exists(os.path.join(target_folder, f))
        ])

        # Preload all y vectors to compute mean and std if needed
        if self.normalize_y:
            all_y = [np.load(os.path.join(target_folder, f)).astype(np.float32).squeeze() for f in self.filenames]
            all_y = np.stack(all_y)  # shape: (N, 128)
            self.y_mean = all_y.mean(axis=0)  # shape: (128,)
            self.y_std = all_y.std(axis=0) + 1e-8  # avoid division by zero

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        x = np.load(os.path.join(self.input_folder, name)).astype(np.float32).squeeze()
        y = np.load(os.path.join(self.target_folder, name)).astype(np.float32).squeeze()

        if self.normalize_y:
            y = (y - self.y_mean) / self.y_std

        return torch.from_numpy(x), torch.from_numpy(y)


# ------------------ Model ------------------
class RegressionModel(nn.Module):
    def __init__(self, in_features, output_dim):
        super().__init__()
        self.regression_head = nn.Sequential(
             nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),   # 2nd layer
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.regression_head(x)


# ------------------ Evaluation ------------------
def compute_rmse(model, dataloader, device="cpu"):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            total_loss += loss.item() * x.size(0)
    rmse = np.sqrt(total_loss / len(dataloader.dataset))
    return rmse


# ------------------ Training ------------------
def train(model, train_loader, test_loader, model_name, epochs=20, lr=1e-3, device="cpu"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_rmse = float('inf')
    log = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        train_rmse = np.sqrt(total_loss / len(train_loader.dataset))
        test_rmse = compute_rmse(model, test_loader, device)

        log.append({'epoch': epoch+1, 'train_rmse': train_rmse, 'test_rmse': test_rmse})
        print(f"Epoch {epoch+1}: Train RMSE = {train_rmse:.4f}, Test RMSE = {test_rmse:.4f}")

        # Save best model
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            torch.save(model.state_dict(), "temp/best_head.pth")
            print(f"  ↳ New best model saved (RMSE={best_rmse:.4f}) → best_head.pth")

    # Save final model (for reference)
    torch.save(model.state_dict(), f"temp/{model_name}.pth")

    # Save log to CSV
    pd.DataFrame(log).to_csv("temp/head_training_log.csv", index=False)
    print(f"\nBest model saved as: {model_name}.pth")
    print(f"Final Train RMSE: {log[-1]['train_rmse']:.4f}")
    print(f"Final Test RMSE : {log[-1]['test_rmse']:.4f}")


# ------------------ Main ------------------
def main():
    # ------------- Parameters to Modify -------------
    input_folder = "./data/datasets/d567b/rtargets"
    target_folder = "./data/datasets/d567b/ptargets"
    in_features = 640 # 1024, 1280, 1536, 2048
    output_dim = 8
    batch_size = 64
    test_split = 0.2
    epochs = 1024
    lr = 1e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # -----------------------------------------------

    model_name = os.path.basename(os.path.normpath(input_folder))

    # Prepare data
    dataset = NPYRegressionDataset(input_folder, target_folder, normalize_y=False)
    
    dataloader = DataLoader(dataset, batch_size=50, shuffle=True)
    enc_images, rtargets = next(iter(dataloader))
    print (enc_images.shape, rtargets.shape)
    
    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Save test filenames to file
    with open("temp/test_split_filenames.txt", "w") as f:
        for idx in test_dataset.indices:
            f.write(f"{dataset.filenames[idx]}\n")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = RegressionModel(in_features, output_dim)
    train(model, train_loader, test_loader, model_name=model_name,
          epochs=epochs, lr=lr, device=device)

    # -------- Load Best Model and Evaluate on Filtered Subset --------
    best_model = RegressionModel(in_features, output_dim)
    best_model.load_state_dict(torch.load("temp/best_head.pth"))
    best_model.to(device)
    best_model.eval()

    # Evaluate only on test samples with filename ID <= 73
    criterion = nn.MSELoss()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for idx in test_dataset.indices:
            filename = dataset.filenames[idx]
            if int(filename.split('_')[0]) <= 73:
                x, y = dataset[idx]
                x, y = x.to(device), y.to(device)
                y_pred = best_model(x.unsqueeze(0))
                loss = criterion(y_pred, y.unsqueeze(0))
                total_loss += loss.item()
                count += 1

    if count > 0:
        filtered_rmse = (total_loss / count) ** 0.5
        print(f"\nFiltered Test RMSE (filename ID ≤ 73): {filtered_rmse:.4f}")
    else:
        print("\nNo test samples matched the filter (ID ≤ 73).")


# ------------------ Entry Point ------------------
if __name__ == "__main__":
    main()
