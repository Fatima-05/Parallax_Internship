import chromadb
import argparse
import logging
import os
import time
import pandas as pd
import psutil
import torch

from src.chunking import split_text
from src.vectorstore import COLLECTION, MODEL_NAME, check_model, get_client, get_model

log=logging.getLogger("build_index")
STORE_BATCH=500


def memory():
    ram=psutil.Process().memory_info().rss/1e6
    gpu=torch.cuda.max_memory_allocated()/1e6 if torch.cuda.is_available() else 0
    return f"RAM {ram:.0f} MB, GPU peak {gpu:.0f} MB"


def main():
    print("main started")
    p=argparse.ArgumentParser()
    p.add_argument("--input", default="data/processed/clean_corpus.parquet")
    p.add_argument("--chunk-size", type=int, default=500)
    p.add_argument("--overlap", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--reset", action="store_true", help="delete the collection and rebuild it")
    a=p.parse_args()

    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True,
                        handlers=[logging.StreamHandler(), logging.FileHandler("logs/build_index.log")])

    df=pd.read_parquet(a.input)
    chunks=[]
    for row in df.itertuples():
        for i, text in enumerate(split_text(row.text, a.chunk_size, a.overlap)):
            chunks.append({"id": f"{row.id}_{i}", "text": text,
                           "meta": {"doc_id": row.id, "title": row.title, "chunk": i}})
    log.info("%d documents -> %d chunks (size %d, overlap %d)", len(df), len(chunks), a.chunk_size, a.overlap)

    client=get_client()
    if a.reset:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine", "model": MODEL_NAME, "hnsw:search_ef": 100})
    check_model(collection)

    done=set(collection.get(include=[])["ids"])
    todo=[c for c in chunks if c["id"] not in done]
    log.info("%d chunks already stored, %d to embed", len(done), len(todo))
    if not todo:
        return

    model=get_model()
    start=time.time()
    for i in range(0, len(todo), STORE_BATCH):
        batch=todo[i:i+STORE_BATCH]
        t=time.time()
        vectors=model.encode([c["text"] for c in batch], batch_size=a.batch_size,
                               normalize_embeddings=True)
        secs=time.time()-t
        collection.upsert(ids=[c["id"] for c in batch],
                          documents=[c["text"] for c in batch],
                          embeddings=vectors.tolist(),
                          metadatas=[c["meta"] for c in batch])
        log.info("stored %d/%d | %.1f ms per chunk | %s",
                 i+len(batch), len(todo), secs / len(batch) * 1000, memory())

    total=time.time()-start
    log.info("done: %d chunks in %.1f s (%.1f ms per chunk) | collection size %d",
             len(todo), total, total / len(todo) * 1000, collection.count())


if __name__=="__main__":
    main()