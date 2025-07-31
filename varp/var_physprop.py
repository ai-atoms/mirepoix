import os
import re
import torch
import pandas as pd
from tqdm import tqdm
import torch.nn as nn


class Regressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.net(x)


# --- Settings ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
regressor_path = 'saved_models/d567b/physprop/'
varp_data = 'varp_pred_70.csv'
input_dim = 16  # You have 16 features per frame

# Load dataset
df = pd.read_csv(varp_data)

# Build a dict to store predictions by frame
frame_feature_dict = {}  # {frame_id: {feature_id: (prediction, uncertainty)}}

for _, row in df.iterrows():
    frame = row['frame']
    feature = int(row['feature'])
    pred = float(row['prediction'])
    uncert = float(row['total_uncertainty'])
    if frame not in frame_feature_dict:
        frame_feature_dict[frame] = {}
    frame_feature_dict[frame][feature] = (pred, uncert)

# Load all regressors
regressor_files = [f for f in os.listdir(regressor_path) if re.match(r"regressor_\d+.*\.pt", f)]
feature_to_model = {}

for file in regressor_files:
    match = re.search(r"regressor_(\d+)", file)
    if match:
        f_id = int(match.group(1))
        model = Regressor(input_dim).to(device)
        model.load_state_dict(torch.load(os.path.join(regressor_path, file), map_location=device))
        model.eval()
        feature_to_model[f_id] = model

# Run predictions
results = []

for frame_id, features in tqdm(frame_feature_dict.items(), desc="Processing frames"):
    if len(features) != input_dim:
        print(f"Skipping frame {frame_id}: incomplete feature vector")
        continue

    # Construct base 16D input vector from predictions
    x_base = [features[i][0] for i in range(input_dim)]
    x_base_tensor = torch.tensor([x_base], dtype=torch.float32).to(device)

    for f in range(input_dim):
        if f not in feature_to_model:
            print(f"Warning: no model for feature {f}")
            continue

        model = feature_to_model[f]

        # Perturb the f-th input feature
        x_shifted = x_base.copy()
        x_shifted[f] += features[f][1]  # add uncertainty
        x_shifted_tensor = torch.tensor([x_shifted], dtype=torch.float32).to(device)

        with torch.no_grad():
            out_nominal = model(x_base_tensor).cpu().item()
            out_shifted = model(x_shifted_tensor).cpu().item()

        regression_unc = abs(out_shifted - out_nominal)

        results.append({
            "frame": frame_id,
            "feature": f,
            "regression": out_nominal,
            "regression_unc": regression_unc
        })

# Save results
out_df = pd.DataFrame(results)
out_df.to_csv("regression_results.csv", index=False)
print("Saved results to regression_results.csv")




