import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class NPYTimeSeriesDataset(Dataset):
    def __init__(self, input_folder, target_folder, p=1):
        """
        Args:
            input_folder: Path to folder containing 1280-dim input vectors (.npy files)
            target_folder: Path to folder containing 64-dim target vectors (.npy files)
            p: Order of VAR model (number of lags to use)
        """
        self.p = p
        
        # Load and sort input files (1280-dim)
        self.input_files = sorted(
            [f for f in os.listdir(input_folder) if f.endswith('.npy')],
            key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        # Load and sort target files (64-dim)
        self.target_files = sorted(
            [f for f in os.listdir(target_folder) if f.endswith('.npy')],
            key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        # Verify we have matching pairs
        assert len(self.input_files) == len(self.target_files), "Input and target file counts must match"
        assert all(in_f == tar_f for in_f, tar_f in zip(self.input_files, self.target_files)), "File names must match"
        
        # Load input data with scaling
        self.scale_factor = 1e+14  # Only applied to input data
        self.input_data = np.stack([np.load(os.path.join(input_folder, f)).squeeze() * self.scale_factor 
                                  for f in self.input_files])
        
        # Load target data without scaling (already 0-1)
        self.target_data = np.stack([np.load(os.path.join(target_folder, f)).squeeze() 
                                   for f in self.target_files])
        
        # Compute min/max for input data only (after scaling)
        self.input_min = np.min(self.input_data, axis=0)
        self.input_max = np.max(self.input_data, axis=0)
        self.epsilon = 1e-8  # Small constant for numerical stability
        
        # Pre-compute valid indices (skip first p timesteps)
        self.valid_indices = list(range(self.p, len(self.input_files)))
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """Returns a tuple of (input_window, target) where:
           - input_window: p consecutive normalized input vectors (1280-dim each)
           - target: single target vector (64-dim)
        """
        actual_idx = self.valid_indices[idx]
        
        # Get window of p previous inputs (normalized)
        input_window = self.input_data[actual_idx-self.p:actual_idx]
        input_window = (input_window - self.input_min) / (self.input_max - self.input_min + self.epsilon)
        
        # Get current target (no normalization needed)
        target = self.target_data[actual_idx]
        
        return (
            torch.FloatTensor(input_window),  # Shape: (p, 1280)
            torch.FloatTensor(target)         # Shape: (64,)
        )
    
    def inverse_transform_input(self, normalized_input):
        """Convert normalized input back to original UNSCALED values"""
        if isinstance(normalized_input, torch.Tensor):
            normalized_input = normalized_input.numpy()
        
        # Step 1: Undo 0-1 normalization (returns scaled values)
        scaled_input = normalized_input * (self.input_max - self.input_min + self.epsilon) + self.input_min
        
        # Step 2: Remove the scaling factor (returns original values)
        original_input = scaled_input / self.scale_factor
        
        return original_input


class RegularizedVAR(nn.Module):
    def __init__(self, input_dim, target_dim, lag_order, l1_lambda=0.01, l2_lambda=0.01):
        """
        Args:
            input_dim: Dimension of input vectors (1280)
            target_dim: Dimension of target vectors (64)
            lag_order: Number of lags (p) in VAR model
            l1_lambda: Weight for L1 regularization
            l2_lambda: Weight for L2 regularization
        """
        super(RegularizedVAR, self).__init__()
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.lag_order = lag_order
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        
        # Each coefficient matrix is now (target_dim, input_dim)
        # This maps from input space (1280) to target space (64)
        self.coefficients = nn.ParameterList([
            nn.Parameter(torch.randn(target_dim, input_dim) * 0.01)
            for _ in range(lag_order)
        ])
        
        self.intercept = nn.Parameter(torch.zeros(target_dim))
        
    def forward(self, x):
        """
        x: input tensor of shape (batch_size, lag_order, input_dim)
           This comes directly from our Dataset's __getitem__
        Returns: predictions of shape (batch_size, target_dim)
        """
        # Ensure correct input dimensions
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension if missing
        elif x.dim() > 3:
            x = x.squeeze()  # Remove extra dimensions
        
        batch_size, seq_len, _ = x.shape

        if seq_len != self.lag_order:
            raise ValueError(
                f"Input sequence length ({seq_len}) must equal lag order ({self.lag_order})"
            )

        # Initialize prediction with intercept
        pred = self.intercept.unsqueeze(0).repeat(batch_size, 1)  # (batch_size, target_dim)
        
        for lag in range(self.lag_order):
            # x[:, lag] has shape (batch_size, input_dim)
            # coeff has shape (target_dim, input_dim)
            pred += torch.matmul(x[:, lag], self.coefficients[lag].t())
        
        return pred
    
    def regularization_loss(self):
        """Compute L1 and L2 regularization terms"""
        l1_reg = torch.tensor(0., device=self.coefficients[0].device)
        l2_reg = torch.tensor(0., device=self.coefficients[0].device)
        
        for param in self.coefficients:
            l1_reg += torch.norm(param, p=1)
            l2_reg += torch.norm(param, p=2)
            
        return self.l1_lambda * l1_reg + self.l2_lambda * l2_reg
    

def train_var_model(dataloader, input_dim, target_dim, lag_order, epochs=100, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model with both input and target dimensions
    model = RegularizedVAR(
        input_dim=input_dim,
        target_dim=target_dim,
        lag_order=lag_order
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for input_windows, targets in dataloader:
            # Move data to device
            input_windows = input_windows.to(device)  # shape: (batch_size, lag_order, input_dim)
            targets = targets.to(device)             # shape: (batch_size, target_dim)
            
            optimizer.zero_grad()
            
            # Forward pass - outputs will be (batch_size, target_dim)
            outputs = model(input_windows)
            
            # Compute loss - compare directly with targets
            loss = criterion(outputs, targets)
            
            # Add regularization
            loss += model.regularization_loss()
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation (optional)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for input_windows, targets in dataloader:
                input_windows = input_windows.to(device)
                targets = targets.to(device)
                outputs = model(input_windows)
                val_loss += criterion(outputs, targets).item()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {total_loss/len(dataloader):.6f}, Val Loss: {val_loss/len(dataloader):.6f}')
    
    return model


def evaluate_var_model(trained_model, dataset, num_samples=16, lag_order=3, window_size=32, yfix=True):
    trained_model.eval()
    
    # Get all indices of the last window_size frames
    all_indices = range(len(dataset))
    last_indices = all_indices[-window_size:]
    
    # Prepare larger figure for 16 subplots
    plt.figure(figsize=(20, 20))
    
    with torch.no_grad():
        # Collect all predictions and targets for the window
        all_targets = []
        all_preds = []
        
        for idx in last_indices:
            _, target = dataset[idx]
            pred = trained_model(dataset[idx][0].unsqueeze(0)).squeeze()
            all_targets.append(target.cpu().numpy())
            all_preds.append(pred.cpu().numpy())
        
        # Convert to arrays
        targets = np.stack(all_targets)  # (window_size, target_dim)
        preds = np.stack(all_preds)     # (window_size, target_dim)
        
        # Select random features to plot
        feature_indices = np.random.choice(trained_model.target_dim, num_samples, replace=False)
        
        for i, feat_idx in enumerate(feature_indices):
            plt.subplot(4, 4, i+1)  # 4x4 grid
            
            # Plot last window_size frames with thinner lines
            plt.plot(range(window_size), targets[:, feat_idx], 'b-', 
                    label='GT' if i == 0 else "", linewidth=1, alpha=0.7)
            plt.plot(range(window_size), preds[:, feat_idx], 'r--', 
                    label='Pred' if i == 0 else "", linewidth=1, alpha=0.7)
            
            mse = torch.mean((torch.tensor(targets[:, feat_idx]) - 
                           torch.tensor(preds[:, feat_idx]))**2).item()
            
            if yfix:
                plt.ylim(0, 1)  # Set fixed y-axis range
                plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])  # Simplify y-axis ticks
            
            plt.subplots_adjust(top=0.9)
            plt.title(f'F{feat_idx} (MSE: {mse:.1e})', fontsize=9)
            plt.xticks(fontsize=7)
            plt.yticks(fontsize=7)
            
            # Only show legend on first subplot to save space
            if i == 0:
                plt.legend(fontsize=8, framealpha=0.5)
    
    # plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.4, wspace=0.3)
    plt.suptitle(f'VAR({lag_order}) Predictions vs Ground Truth (Last {window_size} Frames)', y=0.98)
    plt.show()
    
    # Return quantitative metrics
    return calculate_metrics(trained_model, dataset)


def calculate_metrics(model, dataset):
    """Calculate comprehensive evaluation metrics"""
    model.eval()
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for input_window, target in dataset:
            pred = model(input_window.unsqueeze(0))
            all_targets.append(target.cpu().numpy())
            all_preds.append(pred.squeeze().cpu().numpy())
    
    targets = np.stack(all_targets)
    preds = np.stack(all_preds)
    
    metrics = {
        'mse': torch.mean((torch.tensor(targets) - 
                           torch.tensor(preds))**2).item(),
        'mae': np.mean(np.abs(targets - preds)),
        'featurewise_mse': np.mean((targets - preds)**2, axis=0),
        'featurewise_mae': np.mean(np.abs(targets - preds), axis=0)
    }
    
    print("\nGlobal Metrics:")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    
    return metrics


def evaluate_var_32range(trained_model, dataset, num_samples=16, lag_order=3, yfix=False):
    trained_model.eval()
    
    # Prepare figure
    plt.figure(figsize=(20, 20))
    
    with torch.no_grad():
        # Collect data for first 32 frames
        all_targets = []
        all_preds = []
        valid_range = range(32)  # 0-31 frames
        
        for idx in valid_range:
            input_window, target = dataset[idx]
            # Only predict for frames >32
            if idx > 16:
                pred = trained_model(input_window.unsqueeze(0)).squeeze().cpu().numpy()
            else:
                pred = np.full(trained_model.target_dim, np.nan)  # Empty prediction
            all_targets.append(target.cpu().numpy())
            all_preds.append(pred)
        
        # Convert to arrays
        targets = np.stack(all_targets)  # (64, target_dim)
        preds = np.stack(all_preds)     # (64, target_dim)
        
        # Select random features to plot
        # feature_indices = np.random.choice(trained_model.target_dim, num_samples, replace=False)
        # # feature_indices = list(range(num_samples))
        # # offset = 48
        # # feature_indices = [i + offset for i in feature_indices]
        feature_indices = [7, 10, 11, 15, 17, 22, 27, 31, 34, 37, 40, 46, 49, 56, 57, 62]
        
        for i, feat_idx in enumerate(feature_indices):
            plt.subplot(4, 4, i+1)
            
            # Full ground truth (0-64)
            plt.plot(valid_range, targets[:, feat_idx], 'b-', label='GT', linewidth=1, alpha=0.7)
            
            # Predictions (>32 only)
            pred_range = range(16, 32)
            plt.plot(pred_range, preds[16:, feat_idx], 'r--', label='Pred (>16)', linewidth=1.5, alpha=0.9)
            
            # Add vertical line at prediction start
            plt.axvline(x=16, color='gray', linestyle=':', alpha=0.5)

            if yfix:
                plt.ylim(0, 1)  # Set fixed y-axis range
                plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])  # Simplify y-axis ticks
            
            plt.subplots_adjust(top=0.9)
            
            plt.title(f'Feature {feat_idx}', fontsize=9)
            plt.xticks([0, 8, 16, 24, 32], fontsize=7)
            
            if i == 0:
                plt.legend(fontsize=8, framealpha=0.5)
    
    # plt.tight_layout()
    plt.suptitle(f'VAR({lag_order}) + ResLayer: Predictions (f>16)')
    plt.show()


def evaluate_var_32range_selected(trained_model, dataset, lag_order=3, yfix=False, confidence=0.95):
    trained_model.eval()
    
    # Calculate residuals using only the validation portion (frames >16)
    val_abs_errors = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            if idx <= 16:
                continue
            input_window, target = dataset[idx]
            pred = trained_model(input_window.unsqueeze(0)).squeeze().cpu().numpy()
            abs_error = np.abs(target.numpy() - pred)  # Symmetric error magnitude
            val_abs_errors.append(abs_error)

    val_abs_errors = np.stack(val_abs_errors)  # shape (n_samples, n_features)

    # Compute percentile of absolute errors
    q = 100 * confidence  # e.g., 95 for 95% CI
    error_margin = np.percentile(val_abs_errors, q, axis=0)  # shape (n_features,)
    
    # Prepare figure
    plt.figure(figsize=(16, 12))
    
    with torch.no_grad():
        # Collect data for first 32 frames
        all_targets = []
        all_preds = []
        valid_range = range(32)  # 0-31 frames
        
        for idx in valid_range:
            input_window, target = dataset[idx]
            if idx > 16:
                pred = trained_model(input_window.unsqueeze(0)).squeeze().cpu().numpy()
            else:
                pred = np.full(trained_model.target_dim, np.nan)
            all_targets.append(target.cpu().numpy())
            all_preds.append(pred)
        
        targets = np.stack(all_targets) # (32, 64)
        preds = np.stack(all_preds)     # (32, 64)
        
        # Selected features to plot
        feature_indices = [7, 10, 11, 15, 17, 22, 27, 31, 34, 37, 40, 46, 49, 56, 57, 62]
        
        for i, feat_idx in enumerate(feature_indices):
            plt.subplot(4, 4, i+1)
            
            # Ground truth
            plt.plot(valid_range, targets[:, feat_idx], 'b-', label='GT', linewidth=1, alpha=0.7)
            
            # Predictions and CI
            pred_range = range(16, 32)
            valid_preds = preds[16:, feat_idx]
            plt.plot(pred_range, valid_preds, 'r--', label='Pred (>16)', linewidth=1.5, alpha=0.9)
            
            # Corrected fill_between - now using same-length arrays
            plt.fill_between(
                pred_range,
                valid_preds - error_margin[feat_idx],
                valid_preds + error_margin[feat_idx],
                color='red', alpha=0.15
            )
            
            plt.axvline(x=16, color='gray', linestyle=':', alpha=0.5)
            
            if yfix:
                plt.ylim(0, 1)
                plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            
            plt.title(f'Feature {feat_idx}, CI: ±{error_margin[feat_idx]:.4f}', fontsize=9)
            plt.xticks([0, 8, 16, 24, 32], fontsize=7)
            
            if i == 0:
                plt.legend(fontsize=8, framealpha=0.5)
    
    # plt.tight_layout()
    plt.suptitle(f'VAR({lag_order}) Predictions with {int(confidence*100)}% Confidence Intervals', y=0.96)
    plt.show()


# -- testing the framework
# Parameters
input_dim = 1280
target_dim = 64
lag_order = 4
batch_size = 32

# Deterministic runs
seed = 169006142
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

# Dataset
root_dir = 'data/datasets/dsf60_d9'
dataset = NPYTimeSeriesDataset(
    input_folder=f'{root_dir}/img_descriptors/efb0_512_ft',
    target_folder=f'{root_dir}/r2targets/',
    p=lag_order  # Use t lags
)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Check first batch
for inputs, targets in dataloader:
    print("Input stats - Mean:", inputs.mean().item(), "Std:", inputs.std().item())
    print("Target stats - Mean:", targets.mean().item(), "Std:", targets.std().item())
    break

'''
# Train the model
model = train_var_model(
    dataloader=dataloader,
    input_dim=input_dim,
    target_dim=target_dim,
    lag_order=lag_order,
    epochs=2000,
    learning_rate=0.001
)

# Save the trained model
torch.save(model.state_dict(), 'var_model_lag3.pth')
print("Training complete! Model saved to var_model_lag3.pth")
'''

# Get the PT model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu' # not working on GPU
model = RegularizedVAR(1280, 64, 4).to(device)
model.load_state_dict(torch.load('var_model_lag4.pth', map_location=device))
model.to(device)

'''
# evaluate
metrics = evaluate_var_model(
    trained_model=model,
    dataset=dataset,  # Your validation dataset
    num_samples=16,
    lag_order=8, 
    yfix = False
)

# Access detailed metrics
print("\nFeature-wise MSE:")
print(metrics['featurewise_mse'])
'''

metrics = evaluate_var_32range_selected(
    trained_model=model,
    dataset=dataset,  # Your validation dataset
    lag_order=lag_order,
    yfix=False
)
