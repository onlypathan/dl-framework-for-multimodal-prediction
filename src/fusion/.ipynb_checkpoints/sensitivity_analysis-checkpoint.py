# Sensitivity analysis for the gated fusion model
import numpy as np
import torch
from train_gated_fusion import (
    GatedFusionStarNet, X_train, X_test, y_train, y_test,
    cnn_dim, bert_dim, mlp_dim, gnn_dim, device
)
from sklearn.metrics import mean_squared_error


def evaluate_model(hidden_dim, dropout, lr):
    model = GatedFusionStarNet(
        cnn_dim=cnn_dim,
        bert_dim=bert_dim,
        mlp_dim=mlp_dim,
        gnn_dim=gnn_dim,
        hidden_dim=hidden_dim,
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_tensor  = torch.tensor(X_test.values,  dtype=torch.float32).to(device)
    y_test_tensor  = torch.tensor(y_test.values,  dtype=torch.float32).unsqueeze(1).to(device)

    # Train briefly to observe sensitivity trends
    for _ in range(10):
        optimizer.zero_grad()
        pred, _ = model(X_train_tensor)
        loss = criterion(pred, y_train_tensor)
        loss.backward()
        optimizer.step()

    # Evaluate on the test split
    model.eval()
    with torch.no_grad():
        pred, _ = model(X_test_tensor)
        pred = pred.cpu().numpy().flatten()
        true = y_test_tensor.cpu().numpy().flatten()

    rmse = np.sqrt(mean_squared_error(true, pred))
    return rmse


# Hyperparameter values to evaluate
hidden_dims = [512, 1024, 1536]
dropouts = [0.1, 0.25, 0.4]
learning_rates = [1e-5, 1e-4, 1e-3]

print("Sensitivity analysis results")
for h in hidden_dims:
    for d in dropouts:
        for lr in learning_rates:
            rmse = evaluate_model(h, d, lr)
            print(f"Hidden={h}, Dropout={d}, LR={lr} → RMSE={rmse:.4f}")