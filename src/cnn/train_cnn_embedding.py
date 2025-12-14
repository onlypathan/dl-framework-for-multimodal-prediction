# Imports
import os
import sqlite3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from tqdm import tqdm

# Paths and setup
DB_PATH = "../data/yelp.db"
IMAGE_FOLDER = "raw_images"
OUTPUT_PATH = "cnn_embeddings.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load image records for businesses that have reviews
conn = sqlite3.connect(DB_PATH)
query = """
SELECT DISTINCT i.photo_id, i.business_id
FROM image_lookup AS i
WHERE i.business_id IN (
    SELECT DISTINCT business_id
    FROM review_lookup
);
"""
image_df = pd.read_sql_query(query, conn)
conn.close()

print(f"Loaded {len(image_df)} image records")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load pretrained ResNet-50 without classification head
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.fc = nn.Identity()
model = model.to(device)
model.eval()

# Extract embedding for a single image
def extract_embedding(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        img_t = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model(img_t).cpu().numpy().flatten()
        return emb
    except Exception as e:
        print(f"Skipped {image_path}: {e}")
        return None

# Process images and collect embeddings per business
business_to_embs = {}

for _, row in tqdm(image_df.iterrows(), total=len(image_df), desc="Processing images"):
    photo_id = row["photo_id"]
    business_id = row["business_id"]

    for ext in [".jpg", ".jpeg", ".png"]:
        image_path = os.path.join(IMAGE_FOLDER, f"{photo_id}{ext}")
        if os.path.exists(image_path):
            emb = extract_embedding(image_path)
            if emb is not None:
                business_to_embs.setdefault(business_id, []).append(emb)
            break

# Aggregate embeddings per business using mean pooling
business_ids = []
embeddings = []

for biz_id, emb_list in business_to_embs.items():
    if emb_list:
        business_ids.append(biz_id)
        embeddings.append(np.mean(emb_list, axis=0))

print(f"Generated embeddings for {len(business_ids)} businesses")

# Save embeddings to CSV
emb_array = np.array(embeddings)
columns = [f"CNN_emb_{i}" for i in range(emb_array.shape[1])]
df = pd.DataFrame(emb_array, columns=columns)
df.insert(0, "business_id", business_ids)

df.to_csv(OUTPUT_PATH, index=False)
print(f"Saved embeddings to {OUTPUT_PATH}")
print(f"Final shape: {df.shape}")
