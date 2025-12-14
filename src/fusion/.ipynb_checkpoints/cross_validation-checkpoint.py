# 5-fold cross-validation for the gated fusion model
import numpy as np
import torch
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from train_gated_fusion import (
    GatedFusionStarNet, merged, cnn_dim, bert_dim, mlp_dim, gnn_dim, device, y_mean, y_std
)

# Feature matrix and target
X = merged.drop(
    columns=["business_id", "Star_Rating", "Star_Zscore", "postal_code"]
).fillna(0).values
y = merged["Star_Rating"].values

kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold = 1
rmse_scores = []

for train_idx, test_idx in kf.split(X):
    print(f"Training fold {fold}...")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Normalize targets using global statistics
    y_train_scaled = (y_train - y_mean) / y_std
    y_test_scaled  = (y_test  - y_mean) / y_std

    model = GatedFusionStarNet(
        cnn_dim, bert_dim, mlp_dim, gnn_dim
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = torch.nn.MSELoss()

    # Move data to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32).to(device)
    y_test_t  = torch.tensor(y_test_scaled,  dtype=torch.float32).unsqueeze(1).to(device)

    # Train for a small number of epochs per fold
    for _ in range(20):
        optimizer.zero_grad()
        pred, _ = model(X_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()

    # Evaluate on validation split
    model.eval()
    with torch.no_grad():
        pred, _ = model(X_test_t)
        pred = pred.cpu().numpy().flatten()
        pred = pred * y_std + y_mean  # revert normalization
        true = y_test

    rmse = np.sqrt(mean_squared_error(true, pred))
    rmse_scores.append(rmse)
    print(f"Fold {fold} RMSE: {rmse:.4f}")

    fold += 1

print("\nCross-validation summary")
print(f"RMSE per fold: {rmse_scores}")
print(f"Mean RMSE   : {np.mean(rmse_scores):.4f}")
print(f"Std RMSE    : {np.std(rmse_scores):.4f}")