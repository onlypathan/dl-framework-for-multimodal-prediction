# ZIP-level GNN embedding generator using spatial proximity and reconstruction loss

import sqlite3
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import radians, sin, cos, sqrt, atan2
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import GCNConv

DB_PATH = "../data/yelp.db"


# Load restaurant data with real latitude and longitude
conn = sqlite3.connect(DB_PATH)

df = pd.read_sql_query("""
    SELECT
        n.business_id,
        n.postal_code,
        n.review_count_scaled,
        n.stars,
        n.stars_zscore,
        n.is_open_binary,
        b.latitude,
        b.longitude
    FROM normalized_restaurant_data n
    JOIN restaurant_info b
      ON n.business_id = b.business_id
    WHERE n.postal_code IS NOT NULL
""", conn)

conn.close()
print("Loaded rows:", df.shape)


# Build ZIP-level aggregated features
zip_group = df.groupby("postal_code")

zip_feat = pd.DataFrame({
    "avg_lat_raw": zip_group["latitude"].mean(),
    "avg_lon_raw": zip_group["longitude"].mean(),
    "avg_review_count_scaled": zip_group["review_count_scaled"].mean(),
    "avg_stars": zip_group["stars"].mean(),
    "avg_stars_zscore": zip_group["stars_zscore"].mean(),
    "avg_is_open": zip_group["is_open_binary"].mean(),
    "rest_count": zip_group.size()
})

print("ZIP feature shape (before scaling):", zip_feat.shape)


# Normalize features for GNN input
scaler = StandardScaler()
zip_feat_scaled = pd.DataFrame(
    scaler.fit_transform(zip_feat.values),
    index=zip_feat.index,
    columns=zip_feat.columns
)

print("ZIP feature shape (after scaling):", zip_feat_scaled.shape)


# Build graph edges based on geographic distance
def distance_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


zip_codes = list(zip_feat.index)
lat_raw = zip_feat["avg_lat_raw"]
lon_raw = zip_feat["avg_lon_raw"]

edges = []
THRESHOLD_KM = 7.0

for i, zi in enumerate(zip_codes):
    for j, zj in enumerate(zip_codes):
        if i == j:
            continue
        d = distance_km(lat_raw[zi], lon_raw[zi], lat_raw[zj], lon_raw[zj])
        if d <= THRESHOLD_KM:
            edges.append([i, j])

if not edges:
    raise RuntimeError("No edges created. Increase THRESHOLD_KM if needed.")

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
print("Edge index shape:", edge_index.shape)


# Prepare node feature tensor
x = torch.tensor(zip_feat_scaled.values, dtype=torch.float32)
print("Node feature tensor shape:", x.shape)

in_dim = x.shape[1]


# Two-layer GCN with reconstruction head
class ZipGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=32):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.decoder = nn.Linear(out_dim, in_dim)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        emb = self.conv2(h, edge_index)
        recon = self.decoder(emb)
        return emb, recon


model = ZipGCN(in_dim=in_dim, hidden_dim=64, out_dim=32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# Smoothness loss encourages nearby ZIPs to have similar embeddings
def smoothness_loss(emb, edge_index):
    src, dst = edge_index
    return ((emb[src] - emb[dst]) ** 2).mean()


# Reconstruction loss preserves original feature information
def reconstruction_loss(recon_x, x):
    return F.mse_loss(recon_x, x)


epochs = 200
alpha = 0.1

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    emb, recon_x = model(x, edge_index)

    loss_smooth = smoothness_loss(emb, edge_index)
    loss_recon = reconstruction_loss(recon_x, x)
    loss = loss_smooth + alpha * loss_recon

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Smooth={loss_smooth.item():.6f}, "
            f"Recon={loss_recon.item():.6f}, "
            f"Total={loss.item():.6f}"
        )

print("Training complete.")


# Extract final ZIP embeddings
model.eval()
with torch.no_grad():
    zip_emb, _ = model(x, edge_index)

zip_emb = zip_emb.numpy()

zip_emb_df = pd.DataFrame(
    zip_emb,
    index=zip_feat.index,
    columns=[f"zip_emb_{i}" for i in range(zip_emb.shape[1])]
)

zip_emb_df["postal_code"] = zip_emb_df.index
cols = ["postal_code"] + [c for c in zip_emb_df.columns if c.startswith("zip_emb_")]
zip_emb_df = zip_emb_df[cols]

zip_emb_df.to_csv("gnn_embeddings.csv", index=False)
print("Saved ZIP-GNN embeddings to gnn_embeddings.csv")
print("Final shape:", zip_emb_df.shape)