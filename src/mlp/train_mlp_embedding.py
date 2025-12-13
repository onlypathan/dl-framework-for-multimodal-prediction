# Metadata MLP embedding extractor

import sqlite3
import pandas as pd
import torch
import torch.nn as nn

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# MLP architecture (must match training exactly)
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


# Load normalized metadata from SQLite
DB_PATH = "../data/yelp.db"
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("SELECT * FROM normalized_restaurant_data", conn)
conn.close()

print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")


# Select the same feature set used during training
input_cols = [
    col for col in df.columns
    if col.endswith("_scaled") or col.startswith("Cat_") or col == "is_open_binary"
]

if not input_cols:
    raise ValueError("No matching input features found")

X = df[input_cols].fillna(0).astype(float)
input_dim = X.shape[1]


# Load trained model weights
MODEL_PATH = "mlp_final_stars_zscore.pt"
model = MetadataMLP(input_dim=input_dim).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print(f"Loaded trained MLP from {MODEL_PATH}")


# Extract embeddings from the penultimate layer
def extract_embeddings(model, X_tensor):
    layers = list(model.net.children())[:-1]
    with torch.no_grad():
        for layer in layers:
            X_tensor = layer(X_tensor)
    return X_tensor.cpu().numpy()


X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
embeddings = extract_embeddings(model, X_tensor)

print(f"Extracted embeddings: {embeddings.shape}")


# Save embeddings to CSV
emb_df = pd.DataFrame(
    embeddings,
    columns=[f"MLP_emb_{i}" for i in range(embeddings.shape[1])]
)
emb_df["business_id"] = df["business_id"]

OUTPUT_PATH = "mlp_embeddings.csv"
emb_df.to_csv(OUTPUT_PATH, index=False)

print(f"Saved MLP embeddings to {OUTPUT_PATH}")
print(f"Final shape: {emb_df.shape}")