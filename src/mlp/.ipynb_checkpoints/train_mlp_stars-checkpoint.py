import sqlite3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Load normalized restaurant metadata
DB_PATH = "../data/yelp.db"
TABLE = "normalized_restaurant_data"

conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query(f"SELECT * FROM {TABLE}", conn)
conn.close()

print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")


# Select metadata features used for modeling
input_cols = [
    col for col in df.columns
    if col.endswith("_scaled") or col.startswith("Cat_") or col == "is_open_binary"
]

if not input_cols:
    raise ValueError("No input features found")

print(f"Total input features selected: {len(input_cols)}")

X = df[input_cols].fillna(0).astype(float)


# Define prediction target
TARGET = "stars_zscore"

if TARGET not in df.columns:
    raise ValueError(f"Target column {TARGET} not found")

y = df[TARGET].astype(float)

# Store mean and std for inverse scaling
star_mean = df["stars"].mean()
star_std = df["stars"].std()


# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")


# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(device)


# MLP model for metadata-based regression
class MetadataMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.net(x)


input_dim = X_train_tensor.shape[1]
model = MetadataMLP(input_dim).to(device)

print(f"Initialized MLP with {input_dim} input features")


# Training configuration
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=5e-4)
epochs = 200

best_val_loss = float("inf")
early_stopping_counter = 0
EARLY_STOPPING_PATIENCE = 10


# Training loop with validation and early stopping
for epoch in tqdm(range(epochs), desc="Training"):
    model.train()
    optimizer.zero_grad()

    y_pred = model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(X_test_tensor)
        val_loss = criterion(val_pred, y_test_tensor).item()

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch+1:03} | "
            f"Train Loss: {loss.item():.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f"mlp_best_{TARGET}.pt")
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break


# Final evaluation on test data
model.eval()
with torch.no_grad():
    y_pred_test_scaled = model(X_test_tensor).cpu().numpy().flatten()

# Convert predictions back to original star scale
y_pred_test = y_pred_test_scaled * star_std + star_mean
y_test_actual = y_test_tensor.cpu().numpy().flatten() * star_std + star_mean

rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_test))
mae = mean_absolute_error(y_test_actual, y_pred_test)
r2 = r2_score(y_test_actual, y_pred_test)

print("\nEvaluation Metrics (Star Rating Prediction)")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R²:   {r2:.4f}")


# Save final trained model
torch.save(model.state_dict(), f"mlp_final_{TARGET}.pt")
print(f"Saved MLP model to mlp_final_{TARGET}.pt")
