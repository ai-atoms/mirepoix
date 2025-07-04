import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class RegularizedVAR(nn.Module):
    def __init__(self, input_dim, lag_order, l1_lambda=0.01, l2_lambda=0.01):
        super(RegularizedVAR, self).__init__()
        self.input_dim = input_dim
        self.lag_order = lag_order
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        
        # Each coefficient matrix is now (input_dim, input_dim)
        self.coefficients = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim, input_dim) * 0.01)
            for _ in range(lag_order)
        ])
        
        self.intercept = nn.Parameter(torch.zeros(input_dim))
        
    def forward(self, x):
        """
        x: input tensor of shape (batch_size, sequence_length, input_dim)
        Returns: predictions of shape (batch_size, sequence_length - lag_order, input_dim)
        """
        # Ensure correct input dimensions
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension if missing
        elif x.dim() > 3:
            x = x.squeeze()  # Remove extra dimensions
        
        batch_size, seq_len, _ = x.shape

        if seq_len <= self.lag_order:
            raise ValueError(
            f"Input sequence length ({seq_len}) must be greater than lag order ({self.lag_order})"
            )

        predictions = []
        
        for t in range(self.lag_order, seq_len):
            pred = self.intercept.unsqueeze(0).repeat(batch_size, 1) # Correct dimensions
            for lag in range(1, self.lag_order + 1):
                # Reshape for proper matrix multiplication
                x_reshaped = x[:, t - lag].view(batch_size, 1, -1)  # (batch, 1, input_dim)
                coeff_reshaped = self.coefficients[lag - 1].view(1, self.input_dim, self.input_dim)  # (1, d, d)
                
                # Batch matrix multiplication
                pred += torch.bmm(x_reshaped, coeff_reshaped).view(batch_size, -1)
            
            predictions.append(pred)
        
        return torch.stack(predictions, dim=1)
    
    def regularization_loss(self):
        """Compute L1 and L2 regularization terms"""
        l1_reg = torch.tensor(0., device=self.coefficients[0].device)
        l2_reg = torch.tensor(0., device=self.coefficients[0].device)
        
        for param in self.coefficients:
            l1_reg += torch.norm(param, p=1)
            l2_reg += torch.norm(param, p=2)
            
        return self.l1_lambda * l1_reg + self.l2_lambda * l2_reg


def train_var_model(dataloader, input_dim, lag_order, epochs=100, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = RegularizedVAR(input_dim, lag_order).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            
            # Input is the entire sequence
            inputs = batch[:, :-1]  # all but last step
            targets = batch[:, 1:]  # all but first step
            
            optimizer.zero_grad()
            
            # Forward pass - get predictions for all time steps after lag_order
            outputs = model(inputs)
            
            # Compare with targets (need to align time steps)
            loss = criterion(outputs, targets[:, model.lag_order:])
            
            # Add regularization
            loss += model.regularization_loss()
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.6f}')
    
    return model


def forecast(model, last_observations, steps=22):
    """
    last_observations: tensor of shape (1, lag_order, input_dim)
    steps: number of steps to forecast
    """
    predictions = []
    current_window = last_observations.clone()
    
    for _ in range(steps):
        with torch.no_grad():
            next_pred = model(current_window)[:, -1:]  # Get most recent prediction
            
        predictions.append(next_pred)
        
        # Update window: remove oldest, add new prediction
        current_window = torch.cat([current_window[:, 1:], next_pred], dim=1)
    
    return torch.cat(predictions, dim=1)


class NPYTimeSeriesDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.files = sorted(
            [f for f in os.listdir(folder_path) if f.endswith('.npy')],
            key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        # Load and scale data (applying 1e+14 factor)
        self.scale_factor = 1e+14
        self.data = np.stack([np.load(os.path.join(folder_path, f)).squeeze() * self.scale_factor for f in self.files])
        
        # Compute min/max after scaling
        self.data_min = np.min(self.data, axis=0)
        self.data_max = np.max(self.data, axis=0)
        self.epsilon = 1e-8  # Small constant for numerical stability
        
        # -- sanity check
        # print(f"Original range (after scaling): Min {self.data_min.min():.4f}, Max {self.data_max.max():.4f}")
    
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        """Returns data normalized to 0-1 range (after scaling)"""
        normalized = (self.data - self.data_min) / (self.data_max - self.data_min + self.epsilon)
        return torch.FloatTensor(normalized)
    
    def inverse_transform(self, normalized_data):
        """Convert back to original UNSCALED values"""
        if isinstance(normalized_data, torch.Tensor):
            normalized_data = normalized_data.numpy()
        
        # Step 1: Undo 0-1 normalization (returns scaled values)
        scaled_data = normalized_data * (self.data_max - self.data_min + self.epsilon) + self.data_min
        
        # Step 2: Remove the scaling factor (returns original values)
        original_data = scaled_data / self.scale_factor
        
        # -- sanity check
        # print(f"Restored range check: Min {original_data.min():.4e}, Max {original_data.max():.4e}")
        return original_data


def evaluate_var_model(trained_model, dataset, lag_order=16):
    # Set model to evaluation mode
    trained_model.eval()
    
    # Get normalized data from dataset
    normalized_data = dataset[0].numpy()  # (127, 1280)
    
    # Select 4 random features to visualize
    feature_indices = np.random.choice(normalized_data.shape[1], 4, replace=False)
    
    # Prepare figure
    plt.figure(figsize=(15, 10))
    
    with torch.no_grad():
        # Convert data to tensor and add batch dimension
        full_sequence = torch.FloatTensor(normalized_data).unsqueeze(0)
        
        # Get model predictions (will predict from lag_order onward)
        predictions = trained_model(full_sequence).cpu().numpy()[0]  # (111, 1280)
        
        # For each selected feature
        for i, feat_idx in enumerate(feature_indices):
            plt.subplot(2, 2, i+1)
            
            # Ground truth (only showing predictable portion)
            true_values = normalized_data[lag_order:, feat_idx]
            plt.plot(range(lag_order, len(normalized_data)), true_values, 
                    'b-', label='Ground Truth', alpha=0.7)
            
            # Predictions
            pred_values = predictions[:, feat_idx]
            plt.plot(range(lag_order, lag_order + len(pred_values)), pred_values,
                    'r--', label='Predictions', alpha=0.7)
            
            # Calculate MSE for this feature
            mse = torch.mean((torch.tensor(true_values) - torch.tensor(pred_values)) ** 2).item()
            
            plt.title(f'Feature {feat_idx}\nMSE: {mse:.2e}')
            plt.xlabel('Frame Number')
            plt.ylabel('Normalized Value')
            plt.legend()
            
            # Add grid for better readability
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle(f'VAR({lag_order}) Model Predictions vs Ground Truth\n(Randomly Selected Features)', y=1.02)
    plt.show()



# -- start
root_dir = 'data/datasets/dsf60_d9/img_descriptors/efb0_512_ft'
dataset = NPYTimeSeriesDataset(root_dir)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

'''
# Train the model with lag_order=16
trained_model = train_var_model(
    dataloader=dataloader,
    input_dim=1280,  # EfficientNet embedding dimension
    lag_order=64,    # Using 16 time lags
    epochs=1000,     # Number of training epochs
    learning_rate=0.002
)

# Save the trained model
torch.save(trained_model.state_dict(), 'var_model_lag64.pth')
print("Training complete! Model saved to var_model_lag64.pth")

'''
# For GPU (if available):
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
model = RegularizedVAR(1280, 64).to(device)
model.load_state_dict(torch.load('var_model_lag64.pth', map_location=device))
model.to(device)


evaluate_var_model(model, dataset, lag_order=64)


