from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

DB_PATH="chroma_db"
COLLECTION="arxiv_chunks"
MODEL_NAME="all-MiniLM-L6-v2"


def get_client():
    return chromadb.PersistentClient(path=DB_PATH, settings=Settings(anonymized_telemetry=False))


def get_model():
    return SentenceTransformer(MODEL_NAME)


def check_model(collection):
    saved=collection.metadata.get("model")
    if saved!=MODEL_NAME:
        raise SystemExit(f"Collection was built with {saved}, not {MODEL_NAME}. Rebuild with --reset.")


def open_collection(client):
    try:
        collection=client.get_collection(COLLECTION)
    except Exception:
        raise SystemExit("Collection not found. Run: python -m scripts.build_index")
    check_model(collection)
    if collection.count()==0:
        raise SystemExit("Collection is empty. Run: python -m scripts.build_index --reset")
    return collection


def search(collection, model, query, k=5):
    vector=model.encode(query, normalize_embeddings=True)
    return collection.query(query_embeddings=[vector.tolist()], n_results=min(k, collection.count()))