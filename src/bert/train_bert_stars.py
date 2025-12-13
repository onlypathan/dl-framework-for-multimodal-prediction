import sqlite3, pandas as pd, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

# Database
DB_PATH = "../data/yelp.db"
conn = sqlite3.connect(DB_PATH)

restaurants_df = pd.read_sql_query("""
    SELECT business_id, stars, stars_zscore
    FROM normalized_restaurant_data
    WHERE business_id IN (
        SELECT DISTINCT rl.business_id
        FROM review_lookup rl
        JOIN image_lookup il ON rl.business_id = il.business_id
        WHERE rl.text IS NOT NULL AND TRIM(rl.text) != ''
    );
""", conn)

reviews_df = pd.read_sql_query("""
    WITH ranked AS (
        SELECT
            business_id,
            text AS review_text,
            ROW_NUMBER() OVER (PARTITION BY business_id ORDER BY RANDOM()) AS rn
        FROM review_lookup
        WHERE text IS NOT NULL
          AND TRIM(text) != ''
          AND business_id IN (
              SELECT DISTINCT business_id
              FROM image_lookup
          )
    )
    SELECT business_id, review_text
    FROM ranked;
""", conn)

conn.close()

print(f"Reviews: {len(reviews_df)}, Businesses: {len(restaurants_df)}")

merged = reviews_df.merge(restaurants_df, on="business_id", how="inner")
print(f"Final merged rows: {len(merged)}")

TARGET = "stars_zscore"
texts = merged["review_text"].astype(str)
y = merged[TARGET].astype(float)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


class ReviewDataset(Dataset):
    def __init__(self, texts, targets, max_len=128):
        self.texts = texts
        self.targets = targets
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = tokenizer(
            self.texts[i],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "target": torch.tensor(self.targets[i], dtype=torch.float)
        }


X_tr, X_te, y_tr, y_te = train_test_split(texts, y, test_size=0.2, random_state=42)

y_mean = y_tr.mean()
y_std = y_tr.std() or 1

y_tr_s = (y_tr - y_mean) / y_std
y_te_s = (y_te - y_mean) / y_std

BATCH_SIZE = 192

train_loader = DataLoader(
    ReviewDataset(X_tr.values, y_tr_s.values),
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    ReviewDataset(X_te.values, y_te_s.values),
    batch_size=BATCH_SIZE,
    shuffle=False
)


class BertRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, ids, mask):
        x = self.bert(input_ids=ids, attention_mask=mask).last_hidden_state.mean(1)
        return self.fc(self.drop(x))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertRegressor().to(device)

optimizer = optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.MSELoss()
epochs = 3

for ep in range(epochs):
    model.train()
    train_loss = 0

    for b in tqdm(train_loader, desc=f"Epoch {ep+1}/{epochs}"):
        ids = b["input_ids"].to(device)
        mask = b["attention_mask"].to(device)
        tgt = b["target"].to(device).unsqueeze(1)

        optimizer.zero_grad()
        out = model(ids, mask)
        loss = loss_fn(out, tgt)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    val_loss = 0

    with torch.no_grad():
        for b in test_loader:
            ids = b["input_ids"].to(device)
            mask = b["attention_mask"].to(device)
            tgt = b["target"].to(device).unsqueeze(1)
            val_loss += loss_fn(model(ids, mask), tgt).item()

    print(
        f"Epoch {ep+1} | "
        f"Train {train_loss / len(train_loader):.4f} | "
        f"Val {val_loss / len(test_loader):.4f}"
    )


model.eval()
preds_s, trues_s = [], []

with torch.no_grad():
    for b in tqdm(test_loader, desc="Evaluating"):
        ids = b["input_ids"].to(device)
        mask = b["attention_mask"].to(device)
        preds_s.extend(model(ids, mask).cpu().numpy().flatten())
        trues_s.extend(b["target"].cpu().numpy().flatten())

pred = np.array(preds_s) * y_std + y_mean
true = np.array(trues_s) * y_std + y_mean

rmse = np.sqrt(((true - pred) ** 2).mean())
mae = np.abs(true - pred).mean()
r2 = 1 - ((true - pred) ** 2).sum() / ((true - true.mean()) ** 2).sum()

print(f"\nRMSE={rmse:.4f} | MAE={mae:.4f} | R²={r2:.4f}")

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "y_mean": y_mean,
        "y_std": y_std
    },
    f"bert_model_{TARGET}.pt"
)

print("Model saved.")
