import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.tsa.vector_ar.var_model import VAR


class RobustVARDataset:
    def __init__(self, target_folder, feature_indices, p=3):
        self.p = p
        self.feature_indices = feature_indices
        
        # Load and prepare data
        self.target_files = sorted([f for f in os.listdir(target_folder) if f.endswith('.npy')],
                                 key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        full_data = np.stack([np.load(os.path.join(target_folder, f)).squeeze() 
                            for f in self.target_files])
        self.data = full_data[:, feature_indices]  # (127, 16)
        
        # Normalize
        self.data_mean = self.data.mean(axis=0)
        self.data_std = self.data.std(axis=0)
        self.normalized_data = (self.data - self.data_mean) / (self.data_std + 1e-8)
    
    def fit_var_model(self):
        """Robust VAR fitting that handles edge cases"""
        n_obs, n_vars = self.normalized_data.shape
        
        # Calculate absolute maximum possible lags
        max_possible = (n_obs - 1) // n_vars
        
        if max_possible < 1:
            raise ValueError(f"Only {n_obs} observations for {n_vars} variables - need more data")
        
        # Set safe maxlags (minimum of desired p and possible)
        safe_maxlags = min(self.p, max_possible)
        
        print(f"Fitting with {safe_maxlags} lags (requested {self.p})")
        
        model = VAR(self.normalized_data)
        try:
            return model.fit(maxlags=safe_maxlags, ic='aic')
        except Exception as e:
            # Fallback to OLS if still having issues
            print(f"Using OLS fallback: {str(e)}")
            return model.fit(maxlags=safe_maxlags, method='ols')


def evaluate_statsmodels_var_first32(results, dataset, confidence=0.95):
    """
    Evaluate Statsmodels VAR model predictions for first 32 observations
    with predictions shown only from 16-32 in true (original) scale
    
    Args:
        results: Fitted VARResults object
        dataset: RobustVARDataset instance
        confidence: Confidence level for intervals
    """
    # Get original unnormalized data
    true_data = dataset.data  # (n_obs, 16) in original scale
    n_obs, n_features = true_data.shape
    lag_order = results.k_ar
    feature_indices = dataset.feature_indices
    
    # Calculate residuals in normalized space
    norm_residuals = results.resid  # (n_obs - lag_order, n_features)
    
    # Convert residuals to true scale
    true_residuals = norm_residuals * dataset.data_std
    abs_errors = np.abs(true_residuals)
    error_margin = np.percentile(abs_errors, 100*confidence, axis=0)
    
    # Prepare figure
    plt.figure(figsize=(16, 12))
    
    # Plot first 32 observations
    plot_range = range(0, 32)
    plot_data = true_data[plot_range]
    
    # Initialize predictions array (fill with NaNs for first 16)
    preds = np.full((32, n_features), np.nan)
    
    # Make predictions only for 16-32
    for t in range(16, 32):
        if t >= lag_order:  # Ensure we have enough lags
            # Forecast in normalized space
            norm_pred = results.forecast(dataset.normalized_data[t-lag_order:t], steps=1)
            # Convert prediction to true scale
            true_pred = norm_pred * dataset.data_std + dataset.data_mean
            preds[t] = true_pred.squeeze()
    
    # Plot each feature
    for i in range(n_features):
        plt.subplot(4, 4, i+1)
        
        # Ground truth (true scale)
        plt.plot(plot_range, plot_data[:, i], 'b-', label='Actual', linewidth=1, alpha=0.7)
        
        # Predictions (only show 16-32)
        pred_times = range(16, 32)
        valid_preds = preds[16:, i]
        plt.plot(pred_times, valid_preds, 'r--', label='Predicted', linewidth=1.5, alpha=0.9)
        
        # Confidence intervals
        plt.fill_between(
            pred_times,
            valid_preds - 2*error_margin[i],
            valid_preds + 2*error_margin[i],
            color='red', alpha=0.15
        )
        
        # Vertical line showing prediction start
        plt.axvline(x=16, color='gray', linestyle=':', alpha=0.5)
        
        # Formatting
        plt.title(f'Feature {feature_indices[i]}', fontsize=9)
        plt.xticks([0, 8, 16, 24, 32], fontsize=7)
        
        if i == 0:
            plt.legend(fontsize=8, framealpha=0.5)
    
    plt.suptitle(f'VAR({lag_order}) Predictions with {int(confidence*100)}% Confidence Intervals', y=0.96)
    # plt.tight_layout()
    plt.show()
    
    # Calculate metrics only for prediction period (16-32)
    pred_period_errors = []
    for t in range(16, min(32, n_obs)):
        if not np.isnan(preds[t]).any():
            error = true_data[t] - preds[t]
            pred_period_errors.append(error)
    
    if pred_period_errors:
        pred_period_errors = np.array(pred_period_errors)
        mae = np.abs(pred_period_errors).mean(axis=0)
        mse = (pred_period_errors**2).mean(axis=0)
        
        print("\nError Metrics (True Scale, Prediction Period Only):")
        print(f"{'Feature':<8}{'MAE':<12}{'MSE':<12}")
        for i, idx in enumerate(feature_indices):
            print(f"{idx:<8}{mae[i]:<12.4f}{mse[i]:<12.4f}")
        
        return {
            'mae': mae,
            'mse': mse,
            'error_margin': error_margin
        }
    else:
        print("No valid predictions in the 16-32 range")
        return None


def check_requirements(self):
    """Verify dataset meets VAR requirements"""
    n_obs, n_vars = self.normalized_data.shape
    required = n_vars * self.p + 1
    print(f"\nDiagnostics:")
    print(f"- Observations: {n_obs}")
    print(f"- Variables: {n_vars}")
    print(f"- Requested lags: {self.p}")
    print(f"- Required obs: {required}")
    print(f"- Meets requirement: {n_obs > required} ({(n_obs/required)*100:.1f}%)")


def force_fit(self, method='ols'):
    """Force fit with specified method"""
    model = VAR(self.normalized_data)
    return model.fit(maxlags=self.p, method=method)


# -- testing the framework
# Deterministic runs
seed = 169006142
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Configuration
feature_indices = [7, 10, 11, 15, 17, 22, 27, 31, 34, 37, 40, 46, 49, 56, 57, 62]
root_dir = 'data/datasets/dsf60_d9'
target_folder = f'{root_dir}/r2targets/'
p = 5  # Initial lag order guess

# Initialize dataset
dataset = RobustVARDataset(target_folder=target_folder,
                          feature_indices=feature_indices,
                          p=p)

# Try fitting
results = dataset.fit_var_model()

'''
# If still failing, force OLS
if not results:
    print("Falling back to OLS estimation")
    results = dataset.force_fit(method='ols')

# Verify results
if results:
    print(results.summary())
    print(f"\nActual lags used: {results.k_ar}")
else:
    print("Failed to fit model - need more data")
'''

# Evaluate and plot
metrics = evaluate_statsmodels_var_first32(results, dataset, confidence=0.95)

# To de-normalize any metric:
# error_margin_original_scale = metrics['error_margin'] * dataset.data_std
