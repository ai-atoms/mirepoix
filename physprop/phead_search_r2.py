import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# --- Settings ---
input_folder = "./data/datasets/d567b/r2targets"
target_folder = "./data/datasets/d567b/ptargets"
n_epochs = 256
batch_size = 64
lr = 1e-3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Load data ---
filenames = sorted([
    f for f in os.listdir(input_folder)
    if f.endswith('.npy') and os.path.exists(os.path.join(target_folder, f))
])

X = np.stack([np.load(os.path.join(input_folder, f)) for f in filenames])
Y = np.stack([np.load(os.path.join(target_folder, f)) for f in filenames])  # shape (N, 8)

n_targets = Y.shape[1] # 8 physical variables
input_dim = X.shape[1] # 640 variables in the descriptor

# --- Normalize each target separately (min-max) ---
scalers = [MinMaxScaler() for _ in range(n_targets)]
Y_scaled = np.zeros_like(Y)
for i in range(n_targets):
    Y_scaled[:, i:i+1] = scalers[i].fit_transform(Y[:, i:i+1])

# --- Define simple MLP model ---
class Regressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

# --- Train and evaluate one model per target ---
print("RMSE per target (normalized):")
for i in range(n_targets):
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y_scaled[:, i], test_size=0.2, random_state=42
    )

    x_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    x_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = Regressor(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Training loop
    model.train()
    for epoch in range(n_epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test_tensor.to(device)).cpu().numpy().flatten()
        y_true = y_test  # already on CPU, still normalized
        rmse = mean_squared_error(y_true, y_pred)
        print(f"Target {i}: RMSE (test set) = {rmse:.4f}")
