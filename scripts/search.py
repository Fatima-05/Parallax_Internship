import sys

from src.vectorstore import get_client, get_model, open_collection, search

query=" ".join(sys.argv[1:]) or input("Query: ")
collection=open_collection(get_client())
result=search(collection, get_model(), query)

for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
    print(f"\n[{1 - dist:.3f}] {meta['title']} (doc {meta['doc_id']}, chunk {meta['chunk']})")
    print(doc[:200] + "...")