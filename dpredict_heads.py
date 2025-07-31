import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# ------------------ Config ------------------
input_folder = "./data/datasets/d567b/enc_images/convnext_in"
model_path = "./saved_models/d567b/heads/convnext_in_3hl_20250721_1239/best_head.pth"
in_features = 1536
output_dim = 640
batch_size = 32
device = "cuda" if torch.cuda.is_available() else "cpu"
# --------------------------------------------

# Set reproducibility
seed = 169006142
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# ------------------ Dataset ------------------
class NPYRegressionDataset(Dataset):
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.filenames = sorted([
            f for f in os.listdir(input_folder)
            if f.endswith('.npy')
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        x = np.load(os.path.join(self.input_folder, name)).astype(np.float32).squeeze()
        return torch.from_numpy(x), name

# ------------------ Model ------------------
class RegressionModel(nn.Module):
    def __init__(self, in_features, output_dim):
        super().__init__()
        self.regression_head = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim)
        )

    def forward(self, x):
        return self.regression_head(x)

# ------------------ Prediction ------------------
def predict_and_save(model, dataloader, output_folder, device="cpu"):
    model.eval()
    os.makedirs(output_folder, exist_ok=True)
    with torch.no_grad():
        for x, names in dataloader:
            x = x.to(device)
            preds = model(x).cpu().numpy()
            for i, name in enumerate(names):
                np.save(os.path.join(output_folder, name), preds[i])

# ------------------ Run ------------------
def main():
    input_name = os.path.basename(os.path.normpath(input_folder))
    output_folder = os.path.join("results/d567b/heads", input_name)

    dataset = NPYRegressionDataset(input_folder)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    model = RegressionModel(in_features, output_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    print(f"Running prediction on: {input_name}")
    predict_and_save(model, dataloader, output_folder, device)
    print(f" → Predictions saved to: {output_folder}")

if __name__ == "__main__":
    main()
