import json
import os
from datasets import load_dataset

N = 6000  # a bit over 5,000 because cleaning removes some rows
os.makedirs("data/raw", exist_ok=True)

ds = load_dataset("gfissore/arxiv-abstracts-2021", split="train", streaming=True)

kept = 0
with open("data/raw/arxiv.jsonl", "w", encoding="utf-8") as f:
    for row in ds:
        text = row.get("abstract") or ""
        if len(text) < 200:
            continue
        f.write(json.dumps({"id": row["id"], "title": row["title"], "text": text},
                           ensure_ascii=False) + "\n")
        kept += 1
        if kept >= N:
            break

print(f"Saved {kept} documents")