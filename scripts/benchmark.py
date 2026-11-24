import time

import numpy as np

from src.vectorstore import get_client, get_model, open_collection

K=5
QUERIES=[
    "dark matter and galaxy rotation curves",
    "quantum entanglement in many-body systems",
    "black hole entropy",
    "high temperature superconductivity",
    "gamma ray burst observations",
    "lattice QCD simulation",
    "Bose-Einstein condensate in an optical lattice",
    "graph coloring algorithms",
    "prime numbers and number theory",
    "neural network learning algorithm",
]


def main():
    collection=open_collection(get_client())
    model=get_model()
    model.encode("warm up")

    data=collection.get(include=["embeddings"])
    ids=np.array(data["ids"])
    vectors = np.array(data["embeddings"])

    embed_ms, search_ms, recalls=[],[],[]
    for q in QUERIES:
        t0=time.perf_counter()
        qv=model.encode(q, normalize_embeddings=True)
        t1=time.perf_counter()
        res=collection.query(query_embeddings=[qv.tolist()], n_results=K)
        t2=time.perf_counter()

        exact=ids[np.argsort(-(vectors @ qv))[:K]]
        recall=len(set(res["ids"][0]) & set(exact))/K
        embed_ms.append((t1-t0)*1000)
        search_ms.append((t2-t1)*1000)
        recalls.append(recall)

        print(f"\n{q}")
        print(f"  top hit: {res['metadatas'][0][0]['title']} (similarity {1-res['distances'][0][0]:.3f})")
        print(f"  embed {embed_ms[-1]:.1f} ms | search {search_ms[-1]:.1f} ms | recall@{K} vs exact {recall:.2f}")

    print(f"\n{len(QUERIES)} queries over {collection.count()} chunks")
    print(f"embed  ms: mean {np.mean(embed_ms):.1f}, p95 {np.percentile(embed_ms, 95):.1f}")
    print(f"search ms: mean {np.mean(search_ms):.1f}, p95 {np.percentile(search_ms, 95):.1f}")
    print(f"mean recall@{K} vs exact search: {np.mean(recalls):.3f}")


if __name__ == "__main__":
    main()