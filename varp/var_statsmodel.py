import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.tsa.vector_ar.var_model import VAR

from matplotlib.cm import viridis
from matplotlib.colors import to_hex
import scienceplots
import pandas as pd

# ---------------------------
# 1. Setup SciencePlots + Viridis palette
# ---------------------------
plt.style.use(['science'])

mycolors = [to_hex(viridis(i / 5)) for i in range(5)]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=mycolors)
plt.rcParams['figure.figsize'] = (6, 6)
plt.rcParams['font.size'] = 24
plt.rcParams['lines.linewidth'] = 2
edge_colors = plt.cycler(color=mycolors)


class RobustVARDataset:
    def __init__(self, target_folder, desired_fname='f_70', p=4):
        self.p = p
        self.desired_fname = desired_fname
        self.target_files = sorted(
            [f for f in os.listdir(target_folder) 
             if (self.desired_fname in f) and f.endswith('.npy')],
            key=lambda x: int(x.split('_')[-1].split('.')[0])
        )
        
        # Load and stack data from filtered files
        self.data = np.stack([
            np.load(os.path.join(target_folder, f)).squeeze() 
            for f in self.target_files
        ])  # Shape: (n_samples, n_features)

        # Optional: Print info
        print(f"Loaded {len(self.target_files)} files matching '{desired_fname}'.")
        print(f"Data shape: {self.data.shape}")
        
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


def evaluate_statsmodels_var_old(results, dataset, confidence=0.95):
    """
    Initial implementation (-> old)
    """
    # Feature groups (ascending/descending order)
    g1 = [9, 7, 14, 11, 5]   # Ascending 
    g2 = [0, 4, 13, 10, 8]   # Ascending 
    g3 = [6, 15, 12]         # Descending
    g4 = [2, 1, 3]           # Descending
    groups = [g1, g3, g2, g4]
    group_names = ['(a)', '(b)',
                   '(c)', '(d)']

    # Get data in original scale
    true_data = dataset.data  # (n_obs, n_features)
    n_obs, n_features = true_data.shape
    lag_order = results.k_ar

    # Calculate residuals and error margins
    norm_residuals = results.resid  # (n_obs - lag_order, n_features)
    true_residuals = norm_residuals * dataset.data_std
    abs_errors = np.abs(true_residuals)
    error_margin = np.percentile(abs_errors, 100 * confidence, axis=0)

    # Initialize predictions array (NaNs for first 16 steps)
    preds = np.full((60, n_features), np.nan)
    for t in range(8, 60):
        if t >= lag_order:
            norm_pred = results.forecast(dataset.normalized_data[t - lag_order:t], steps=1)
            true_pred = norm_pred * dataset.data_std + dataset.data_mean
            preds[t] = true_pred.squeeze()

    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    plot_range = range(0, 60)
    pred_times = range(8, 60)

    for ax, group, name in zip(axes, groups, group_names):
        for i in group:
            # Plot ground truth
            ax.plot(plot_range, true_data[plot_range, i], 
                    linewidth=1.5, alpha=1.0, label=f'f{i}')

            # ax.scatter(plot_range, true_data[plot_range, i], 
            #     s=16, label=f'f{i}')

            # -- camilofs, Plot predictions (8-60)
            valid_preds = preds[8:, i]
            # ax.plot(pred_times, valid_preds, color='gray', ls='--', linewidth=1.5, alpha=0.4)
 
            # Confidence intervals (shared per group)
            ax.fill_between(
                pred_times,
                valid_preds - 2 * error_margin[i],
                valid_preds + 2 * error_margin[i],
                color='gray', alpha=0.24
            )

            ax.fill_between(
                pred_times,
                valid_preds - 3 * error_margin[i],
                valid_preds + 3 * error_margin[i],
                color='silver', alpha=0.32
            )

            if group in [g1, g2]:
                ax.set_ylim(0.0, 0.8)
                ax.legend(loc='lower right', ncol=2, fontsize=14)
            else:
                ax.set_ylim(0.6, 1.0)
                ax.legend(loc='upper right', fontsize=14)

        ax.set_xlim(0, 60)
        ax.axvline(x=8, color='gray', linestyle=':', alpha=0.5)
        ax.set_title(name, fontsize=24)
        # ax.grid(True)

    # plt.suptitle(f'VAR({lag_order}) Predictions ({int(confidence*100)} pct. CI)', y=0.98)
    plt.tight_layout()
    plt.show()
    # plt.savefig('fig6_f70_varp.png', dpi=300)

    # Calculate metrics (16-32 prediction period)
    pred_period_errors = []
    for t in range(8, min(60, n_obs)):
        if not np.isnan(preds[t]).any():
            pred_period_errors.append(true_data[t] - preds[t])

    if pred_period_errors:
        pred_period_errors = np.array(pred_period_errors)
        mae = np.abs(pred_period_errors).mean(axis=0)
        mse = (pred_period_errors ** 2).mean(axis=0)

        print("\nError Metrics (True Scale, Prediction Period Only):")
        print(f"{'Feature':<8}{'MAE':<12}{'MSE':<12}")
        for i in range(n_features):
            print(f"{i:<8}{mae[i]:<12.4f}{mse[i]:<12.4f}")

        return {'mae': mae, 'mse': mse, 'error_margin': error_margin}
    else:
        print("No valid predictions in the 16-32 range")
        return None


def evaluate_statsmodels_var_epistemic_recursive(results, dataset, confidence=0.95):
    """
    Evaluate a VAR(p) model using recursive prediction (epistemic error propagation).
    """

    # Groupings for plotting
    g1 = [9, 7, 14, 11, 5]
    g2 = [0, 4, 13, 10, 8]
    g3 = [6, 15, 12]
    g4 = [2, 1, 3]
    groups = [g1, g3, g2, g4]
    group_names = ['(a)', '(b)', '(c)', '(d)']

    # Data & params
    true_data = dataset.data
    true_data = true_data[:64]  # Ensure shape compatibility with predictions
    n_obs, n_features = true_data.shape
    lag_order = results.k_ar

    preds = np.full((64, n_features), np.nan)
    epistemic_errors = np.zeros((64, n_features))

    # Aleatoric margin
    true_residuals = results.resid * dataset.data_std
    abs_errors = np.abs(true_residuals)
    aleatoric_margin = np.percentile(abs_errors, 100 * confidence, axis=0)

    # Bootstrap: insert first 8 real observations as prediction seed
    preds[0:8] = true_data[0:8]

    # Recursive prediction loop
    for t in range(8, 64):
        # Build the input window of lag_order frames
        input_window = preds[t - lag_order:t]
        norm_input = (input_window - dataset.data_mean) / dataset.data_std

        # Forecast 1-step ahead
        norm_pred = results.forecast(norm_input, steps=1)
        true_pred = norm_pred * dataset.data_std + dataset.data_mean
        preds[t] = true_pred.squeeze()

        # Epistemic error: std of previous prediction errors
        if t > 8:
            past_errors = true_data[8:t] - preds[8:t]
            epistemic_errors[t] = np.std(past_errors, axis=0)

    # Compute dpa array
    frame_range = np.arange(0, 64)
    dpa_range = (frame_range * 2 * 200) / 686000
    
    pred_frames = np.arange(8, 64)
    dpa_pred = (pred_frames * 2 * 200) / 686000
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    for ax, group, name in zip(axes, groups, group_names):
        for i in group:
            valid_preds = preds[:, i]
    
            z_95 = 1.96
            z_99 = 2.576
    
            total_uncertainty_95 = np.sqrt(
                (z_95 * aleatoric_margin[i])**2 + 
                (z_95 * epistemic_errors[8:64, i])**2
            )
            total_uncertainty_99 = np.sqrt(
                (z_99 * aleatoric_margin[i])**2 + 
                (z_99 * epistemic_errors[8:64, i])**2
            )
    
            # 95% confidence interval
            ax.fill_between(
                dpa_pred,
                valid_preds[8:64] - total_uncertainty_95,
                valid_preds[8:64] + total_uncertainty_95,
                color='gray', alpha=0.3, edgecolor='none',
            )
    
            # 99% confidence interval
            ax.fill_between(
                dpa_pred,
                valid_preds[8:64] - total_uncertainty_99,
                valid_preds[8:64] + total_uncertainty_99,
                color='silver', alpha=0.4, edgecolor='none',
            )
    
            # Plot true data
            ax.plot(dpa_range, true_data[:, i], linewidth=1.5, alpha=1.0, label=f'f{i}')
    
            if group in [g1, g2]:
                ax.set_ylim(0.0, 0.8)
                ax.legend(loc='lower right', ncol=2, fontsize=10)
            else:
                ax.set_ylim(0.6, 1.0)
                ax.legend(loc='upper right', fontsize=10)
    
        ax.set_xlim(0, 0.035)
        ax.set_xticks([0.005, 0.01, 0.02, 0.03, 0.035])
        ax.set_xticklabels(['0.005', '0.01', '0.02', '0.03', '0.035'])
        ax.axvline(x=(8 * 2 * 200) / 686000, color='red', linestyle=':', alpha=0.5)
        ax.set_title(name, fontsize=24)
        ax.set_xlabel('Dose (dpa)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('fig6b_f70_varp.png', dpi=300)

    # Error metrics
    
    # -- camilofs (recalculate the total uncertainty -> shape: (52, n_features))
    z_95 = 1.96
    total_uncertainty_95 = np.sqrt(
        (z_95 * aleatoric_margin[np.newaxis, :])**2 +
        (z_95 * epistemic_errors[8:64])**2
    )

    pred_period_errors = true_data[8:64] - preds[8:64]
    if not np.all(np.isnan(pred_period_errors)):
        mae = np.nanmean(np.abs(pred_period_errors), axis=0)
        mse = np.nanmean(pred_period_errors ** 2, axis=0)

        print("\nError Metrics (8-64):")
        print(f"{'Feature':<8}{'MAE':<12}{'MSE':<12}")
        for i in range(n_features):
            print(f"{i:<8}{mae[i]:<12.4f}{mse[i]:<12.4f}")

        return {
            'mae': mae,
            'mse': mse,
            'aleatoric_margin': aleatoric_margin,
            'epistemic_errors': epistemic_errors[8:64],
            'unc_95': total_uncertainty_95
        }
    else:
        print("No valid predictions in range 8-64")
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


def export_statsmodels_var(results, dataset, confidence=0.95):
    """
    Export recursive VAR predictions and total 99% uncertainty to a pandas DataFrame.
    -- camilofs (this function should be generalized for any desired interval)
    
    Parameters:
        results: Trained statsmodels VARResults object
        dataset: An object with .data, .normalized_data, .data_mean, .data_std
        confidence: Confidence level for uncertainty bands (default: 0.95)
        
    Returns:
        pd.DataFrame with columns: ['frame', 'feature', 'prediction', 'total_uncertainty']
        Only includes frames t > 8 (i.e. 8 to 59)
    """
    # Prepare data
    true_data = dataset.data[:60]  # Limit to 60 frames
    n_obs, n_features = true_data.shape
    lag_order = results.k_ar

    preds = np.full((60, n_features), np.nan)
    epistemic_errors = np.zeros((60, n_features))

    # Aleatoric error estimation
    residuals = results.resid * dataset.data_std
    abs_errors = np.abs(residuals)
    aleatoric_margin = np.percentile(abs_errors, 100 * confidence, axis=0)  # (n_features,)

    # Seed predictions with true values up to t=8
    preds[:8] = true_data[:8]

    # Recursive prediction loop
    for t in range(8, 60):
        input_window = preds[t - lag_order:t]
        norm_input = (input_window - dataset.data_mean) / (dataset.data_std + 1e-8)

        norm_pred = results.forecast(norm_input, steps=1)
        true_pred = norm_pred * dataset.data_std + dataset.data_mean
        preds[t] = true_pred.squeeze()

        # Epistemic error from previous predictions
        if t > 8:
            past_errors = true_data[8:t] - preds[8:t]
            epistemic_errors[t] = np.std(past_errors, axis=0)

    # Z-score for desired confidence level (e.g., 2.576 for 99%)
    from scipy.stats import norm
    z_score = norm.ppf((1 + confidence) / 2)  # e.g., ≈2.576 for 99%

    # Total uncertainty
    total_uncertainty = np.sqrt(
        (z_score * aleatoric_margin[np.newaxis, :])**2 +
        (z_score * epistemic_errors[8:60])**2
    )  # Shape: (52, n_features)

    # Prepare DataFrame
    data = []
    for t in range(8, 60):  # t=8 to 59
        for f in range(n_features):
            data.append({
                'frame': t,
                'feature': f,
                'prediction': preds[t, f],
                'total_uncertainty': total_uncertainty[t - 8, f],
                'true': true_data[t, f]
            })

    df = pd.DataFrame(data)
    return df


def predict_extended_range(results, dataset, confidence=0.95):
    """
    Extended range
    """
    # Prepare data
    true_data = dataset.data[:151]  # Limit to 151 frames
    n_obs, n_features = true_data.shape
    lag_order = results.k_ar

    # Initialize arrays for predictions and errors
    preds = np.full((151, n_features), np.nan)
    epistemic_errors = np.zeros((151, n_features))

    # Aleatoric error estimation (same as before)
    residuals = results.resid * dataset.data_std
    abs_errors = np.abs(residuals)
    aleatoric_margin = np.percentile(abs_errors, 100 * confidence, axis=0)  # (n_features,)

    # Seed predictions with true values up to t=24 (new starting point)
    preds[:24] = true_data[:24]

    # Recursive prediction loop (now from 24 to 150)
    for t in range(24, 151):
        input_window = preds[t - lag_order:t]
        norm_input = (input_window - dataset.data_mean) / (dataset.data_std + 1e-8)

        norm_pred = results.forecast(norm_input, steps=1)
        true_pred = norm_pred * dataset.data_std + dataset.data_mean
        preds[t] = true_pred.squeeze()

        # Epistemic error from previous predictions
        if t > 24:
            past_errors = true_data[24:t] - preds[24:t]
            epistemic_errors[t] = np.std(past_errors, axis=0)

    # Z-score for desired confidence level
    from scipy.stats import norm
    z_score = norm.ppf((1 + confidence) / 2)

    # Total uncertainty (now covering 24 to 150)
    total_uncertainty = np.sqrt(
        (z_score * aleatoric_margin[np.newaxis, :])**2 +
        (z_score * epistemic_errors[24:151])**2
    )  # Shape: (127, n_features)

    # Prepare DataFrame with new range
    data = []
    for t in range(24, 151):  # t=24 to 150
        for f in range(n_features):
            data.append({
                'frame': t,
                'feature': f,
                'prediction': preds[t, f],
                'total_uncertainty': total_uncertainty[t - 24, f],
                'true': true_data[t, f]
            })

    df = pd.DataFrame(data)
    return df


# -- testing the framework
# Deterministic runs
seed = 169006142
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Configuration
root_dir = 'data/datasets/d567b'
target_folder = f'{root_dir}/r3targets/'
p = 4  # Initial lag order guess

# Initialize dataset
dataset = RobustVARDataset(target_folder=target_folder,
                          desired_fname='70_',
                          p=p)

# Try fitting
results = dataset.fit_var_model()

# Epistemic errors
metrics = evaluate_statsmodels_var_epistemic_recursive(results, dataset)
avg_unc_95 = np.mean(metrics['unc_95'], axis=0)  # shape (16,)
avg_unc_95_original_scale = avg_unc_95 * dataset.data_std
print(f"{'Feature':<8}{'Avg Unc. (95%)':<16}")
for i, unc in enumerate(avg_unc_95_original_scale):
    print(f"{i:<8}{unc*100:<16.5f} %")

# -- camilofs first range
df_pred = export_statsmodels_var(results, dataset, confidence=0.95)
print(df_pred.head())
df_pred.to_csv('varp_pred_70.csv')

# -- camilofs extended range
df_pred = predict_extended_range(results, dataset, confidence=0.95)
print(df_pred.head())
df_pred.to_csv('varp_pred_70b.csv')


