import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

seed = 169006142
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# ------------------ Dataset ------------------
class NPYRegressionDataset(Dataset):
    def __init__(self, input_folder, target_folder):
        self.input_folder = input_folder
        self.target_folder = target_folder

        self.filenames = sorted([
            f for f in os.listdir(input_folder)
            if f.endswith('.npy') and os.path.exists(os.path.join(target_folder, f))
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        x = np.load(os.path.join(self.input_folder, name)).astype(np.float32).squeeze()
        y = np.load(os.path.join(self.target_folder, name)).astype(np.float32).squeeze()
        return torch.from_numpy(x), torch.from_numpy(y)


# ------------------ Model ------------------
class RegressionModel(nn.Module):
    def __init__(self, in_features, output_dim):
        super().__init__()
        self.regression_head = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),   # 2nd layer
            nn.ReLU(),
            nn.Linear(1024, 1024),   # 3rd layer
            nn.ReLU(),
            nn.Linear(1024, output_dim)
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


# ------------------ Main Evaluation ------------------
def main():
    # --------- User-specified paths and parameters ---------
    input_folder = "./data/datasets/d567b/enc_images/vitl16_in"
    target_folder = "./data/datasets/d567b/rtargets"
    model_path = "saved_models/head_3hl_vitl16_in_20250721_1107/best_head.pth" # check
    in_features = 1024
    output_dim = 640
    batch_size = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # -------------------------------------------------------

    # Prepare dataset & dataloader
    dataset = NPYRegressionDataset(input_folder, target_folder)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Load model
    model = RegressionModel(in_features, output_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Compute RMSE
    rmse = compute_rmse(model, dataloader, device)
    print(f"\nTest RMSE (unfiltered): {rmse:.4f}")


if __name__ == "__main__":
    main()
