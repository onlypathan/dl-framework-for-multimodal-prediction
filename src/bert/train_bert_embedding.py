import os, gc, math, sqlite3
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

# Configuration
DB_PATH        = "../data/yelp.db"
OUTPUT_DIR     = "emb_out"
FINAL_CSV      = "bert_embeddings.csv"
MODEL_NAME     = "bert-base-uncased"
MAX_LEN        = 128
INIT_BATCH     = 16
SHARD_ROWS     = 100_000
PER_BIZ_MAX    = None      # set to limit reviews per business
SEED           = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
SHARD_DIR = os.path.join(OUTPUT_DIR, "shards")
os.makedirs(SHARD_DIR, exist_ok=True)


# Mean pooling with attention mask
def masked_mean(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked_sum = (last_hidden_state * mask).sum(1)
    denom = mask.sum(1).clamp(min=1e-9)
    return masked_sum / denom


# Save shard to disk
def write_shard(idx, biz, emb):
    df = pd.DataFrame(emb, columns=[f"BERT_emb_{i}" for i in range(emb.shape[1])])
    df.insert(0, "business_id", biz)
    df.to_csv(os.path.join(SHARD_DIR, f"shard_{idx:05d}.csv"), index=False)


# Detect existing shards (resume support)
def existing_shards():
    return sorted([f for f in os.listdir(SHARD_DIR) if f.startswith("shard_")])


# Load review text
conn = sqlite3.connect(DB_PATH)
reviews = pd.read_sql_query(
    """
    SELECT rl.business_id, rl.text AS review_text
    FROM review_lookup rl
    WHERE rl.text IS NOT NULL
      AND TRIM(rl.text) != ''
      AND rl.business_id IN (SELECT DISTINCT business_id FROM image_lookup)
    """, conn
)
conn.close()

# Optional sampling per business
if PER_BIZ_MAX:
    reviews = (
        reviews.groupby("business_id", group_keys=False)
               .apply(lambda g: g.sample(min(PER_BIZ_MAX, len(g)), random_state=SEED))
               .reset_index(drop=True)
    )

total_reviews = len(reviews)
unique_business = reviews["business_id"].nunique()

print(f"Loaded reviews: {total_reviews}")
print(f"Unique businesses: {unique_business}")


# Load BERT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME).to(device)
model.eval()


# Shard setup
num_shards = math.ceil(total_reviews / SHARD_ROWS)
done_shards = {int(f.split("_")[1].split(".")[0]) for f in existing_shards()}


# Generate embeddings
progress = tqdm(total=total_reviews, desc="Embedding reviews", unit="review")

# Resume progress if shards exist
for s in done_shards:
    shard_end = min((s + 1) * SHARD_ROWS, total_reviews)
    progress.update(shard_end - s * SHARD_ROWS)

for shard_idx in range(num_shards):
    if shard_idx in done_shards:
        continue

    start = shard_idx * SHARD_ROWS
    end   = min(total_reviews, start + SHARD_ROWS)
    df = reviews.iloc[start:end].reset_index(drop=True)

    texts = df["review_text"].astype(str).tolist()
    bizs  = df["business_id"].tolist()

    i = 0
    batch = INIT_BATCH
    all_emb = []
    all_id = []

    while i < len(texts):
        j = min(i + batch, len(texts))
        batch_texts = texts[i:j]

        try:
            with torch.no_grad():
                enc = tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=MAX_LEN,
                    return_tensors="pt"
                ).to(device)

                out = model(**enc)
                pooled = masked_mean(out.last_hidden_state, enc["attention_mask"])
                all_emb.append(pooled.cpu().numpy())
                all_id.extend(bizs[i:j])

            progress.update(j - i)
            i = j

        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                batch = max(1, batch // 2)
            else:
                raise

    emb_np = np.vstack(all_emb)
    write_shard(shard_idx, all_id, emb_np)

    del df, texts, bizs, all_emb, all_id, emb_np
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

progress.close()


# Combine shards and aggregate
parts = [pd.read_csv(os.path.join(SHARD_DIR, f)) for f in existing_shards()]
all_df = pd.concat(parts, ignore_index=True)

# Mean embedding per business
agg = all_df.groupby("business_id", as_index=False).mean()

out = os.path.join(OUTPUT_DIR, FINAL_CSV)
agg.to_csv(out, index=False)

print(f"Final shape: {agg.shape}")
print(f"Saved to: {out}")
