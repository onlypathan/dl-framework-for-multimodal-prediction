# Imports
import sqlite3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import os

# Paths and setup
DB_PATH = "../data/yelp.db"
IMAGE_DIR = "raw_images"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load star ratings for businesses with both reviews and images
conn = sqlite3.connect(DB_PATH)

restaurants_df = pd.read_sql_query("""
    SELECT business_id, stars, stars_zscore
    FROM normalized_restaurant_data
    WHERE business_id IN (
        SELECT DISTINCT rl.business_id
        FROM review_lookup rl
        JOIN image_lookup il ON rl.business_id = il.business_id
        WHERE rl.text IS NOT NULL
          AND TRIM(rl.text) != ''
    );
""", conn)

stars_map = {
    str(row["business_id"]): float(row["stars"])
    for _, row in restaurants_df.iterrows()
}

# Load image-to-business mapping
images_df = pd.read_sql_query("""
    SELECT DISTINCT photo_id, business_id
    FROM image_lookup
    WHERE business_id IN (
        SELECT DISTINCT business_id
        FROM review_lookup
    );
""", conn)

conn.close()

print(f"Loaded {len(images_df)} image rows and {len(restaurants_df)} restaurants")

# Match images with star ratings
final_photo_ids = []
final_stars = []

for _, row in images_df.iterrows():
    bid = str(row["business_id"])
    if bid in stars_map:
        final_photo_ids.append(row["photo_id"])
        final_stars.append(stars_map[bid])

print(f"Matched {len(final_photo_ids)} images with star ratings")

# Dataset definition (skips missing images)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class ImageDataset(Dataset):
    def __init__(self, photo_ids, stars, transform):
        self.photo_ids = photo_ids
        self.stars = stars
        self.transform = transform

    def __len__(self):
        return len(self.photo_ids)

    def __getitem__(self, idx):
        pid = self.photo_ids[idx]

        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = os.path.join(IMAGE_DIR, f"{pid}{ext}")
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            return None

        try:
            img = Image.open(img_path).convert("RGB")
        except:
            return None

        img = self.transform(img)
        star = torch.tensor(self.stars[idx], dtype=torch.float32)
        return img, star

def collate_skip(batch):
    return [b for b in batch if b is not None]

# Train–test split
X_train, X_test, y_train, y_test = train_test_split(
    final_photo_ids, final_stars,
    test_size=0.2, random_state=42
)

train_ds = ImageDataset(X_train, y_train, transform)
test_ds  = ImageDataset(X_test,  y_test,  transform)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_skip)
test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, collate_fn=collate_skip)

print(f"Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")

# ResNet-50 regression model
class ResNetRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.feature = nn.Sequential(*list(base.children())[:-1])
        self.regressor = nn.Linear(2048, 1)

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        return self.regressor(x)

model = ResNetRegressor().to(device)
print("Initialized ResNet-50 regressor")

# Training configuration
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
epochs = 3

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        if len(batch) == 0:
            continue

        imgs, stars = zip(*batch)
        imgs = torch.stack(imgs).to(device)
        stars = torch.tensor(stars, dtype=torch.float32).unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, stars)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f}")

# Evaluation
model.eval()
all_preds, all_trues = [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating"):
        if len(batch) == 0:
            continue

        imgs, stars = zip(*batch)
        imgs = torch.stack(imgs).to(device)
        preds = model(imgs).cpu().numpy().flatten()

        all_preds.extend(preds)
        all_trues.extend(stars)

rmse = np.sqrt(mean_squared_error(all_trues, all_preds))
mae  = mean_absolute_error(all_trues, all_preds)
r2   = r2_score(all_trues, all_preds)

print("\nEvaluation Metrics (CNN Image Model)")
print(f"Target: stars")
print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")
print(f"R²  : {r2:.4f}")

# Save model
save_name = "cnn_stars_baseline.pt"
torch.save(model.state_dict(), save_name)
print(f"Saved model to {save_name}")
