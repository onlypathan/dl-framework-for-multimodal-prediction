# Multimodal gated fusion trainer (CNN + BERT + MLP + GNN) for star rating prediction

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from tqdm import tqdm
import sqlite3
import os
import optuna


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")


# Output folders and database path
os.makedirs("fusion_results_gated/plots", exist_ok=True)
DB_PATH = "../data/yelp.db"


# Load target variables (star ratings)
conn = sqlite3.connect(DB_PATH)

targets_df = pd.read_sql_query("""
    SELECT business_id,
           stars AS Star_Rating,
           stars_zscore AS Star_Zscore
    FROM normalized_restaurant_data
    WHERE business_id IN (
        SELECT DISTINCT business_id FROM review_lookup
        INTERSECT
        SELECT DISTINCT business_id FROM image_lookup
    );
""", conn)

# Load postal codes for GNN merge
postal_df = pd.read_sql_query("""
    SELECT business_id, postal_code
    FROM normalized_restaurant_data
    WHERE business_id IN (
        SELECT DISTINCT business_id FROM review_lookup
        INTERSECT
        SELECT DISTINCT business_id FROM image_lookup
    );
""", conn)

conn.close()
print("Target rows:", targets_df.shape)


# Load modality embeddings
cnn_df  = pd.read_csv("../cnn/cnn_embeddings.csv")
bert_df = pd.read_csv("../bert/bert_embeddings.csv")
mlp_df  = pd.read_csv("../mlp/mlp_embeddings.csv")
gnn_df  = pd.read_csv("../gnn/gnn_embeddings.csv")


# Infer embedding dimensions
cnn_dim  = cnn_df.shape[1]  - 1
bert_dim = bert_df.shape[1] - 1
mlp_dim  = mlp_df.shape[1]  - 1
gnn_dim  = gnn_df.shape[1]  - 1

print(f"Embedding dims → CNN:{cnn_dim}, BERT:{bert_dim}, MLP:{mlp_dim}, GNN:{gnn_dim}")


# Merge all modalities
merged = targets_df.merge(postal_df, on="business_id", how="inner")

for emb in [cnn_df, bert_df, mlp_df]:
    merged = merged.merge(emb, on="business_id", how="inner")

merged = merged.merge(gnn_df, on="postal_code", how="left")
print("Fused dataset shape:", merged.shape)


# Prepare features and target
X = merged.drop(columns=["business_id", "Star_Rating", "Star_Zscore", "postal_code"]).fillna(0)
y_raw = merged["Star_Rating"].astype(float)

y_mean = y_raw.mean()
y_std  = y_raw.std() + 1e-8
y_scaled = (y_raw - y_mean) / y_std

X_train, X_test, y_train, y_test = train_test_split(
    X, y_scaled, test_size=0.2, random_state=42
)

def to_tensor(df):
    return torch.tensor(df.values, dtype=torch.float32)

X_train_tensor = to_tensor(X_train).to(device)
X_test_tensor  = to_tensor(X_test).to(device)
y_train_tensor = to_tensor(y_train).unsqueeze(1).to(device)
y_test_tensor  = to_tensor(y_test).unsqueeze(1).to(device)

batch_size = 256
train_loader = DataLoader(
    TensorDataset(X_train_tensor, y_train_tensor),
    batch_size=batch_size, shuffle=True
)
test_loader = DataLoader(
    TensorDataset(X_test_tensor, y_test_tensor),
    batch_size=batch_size, shuffle=False
)


# Gated fusion network definition
class GatedFusionStarNet(nn.Module):
    def __init__(self, cnn_dim, bert_dim, mlp_dim, gnn_dim, hidden_dim=1024, dropout=0.2):
        super().__init__()

        self.cnn_dim = cnn_dim
        self.bert_dim = bert_dim
        self.mlp_dim = mlp_dim
        self.gnn_dim = gnn_dim

        self.gate_cnn  = nn.Parameter(torch.tensor(1.0))
        self.gate_bert = nn.Parameter(torch.tensor(1.0))
        self.gate_mlp  = nn.Parameter(torch.tensor(1.0))
        self.gate_gnn  = nn.Parameter(torch.tensor(1.0))

        fusion_dim = cnn_dim + bert_dim + mlp_dim + gnn_dim

        self.shared = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
        )

        self.star_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        i1 = self.cnn_dim
        i2 = i1 + self.bert_dim
        i3 = i2 + self.mlp_dim
        i4 = i3 + self.gnn_dim

        cnn_part  = x[:, :i1]
        bert_part = x[:, i1:i2]
        mlp_part  = x[:, i2:i3]
        gnn_part  = x[:, i3:i4]

        cnn_w  = torch.sigmoid(self.gate_cnn)  * cnn_part
        bert_w = torch.sigmoid(self.gate_bert) * bert_part
        mlp_w  = torch.sigmoid(self.gate_mlp)  * mlp_part
        gnn_w  = torch.sigmoid(self.gate_gnn)  * gnn_part

        fused = torch.cat([cnn_w, bert_w, mlp_w, gnn_w], dim=1)
        shared_out = self.shared(fused)
        star_pred = self.star_head(shared_out)

        return star_pred, shared_out