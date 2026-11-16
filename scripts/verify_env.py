import sys

allow_cpu = "--allow-cpu" in sys.argv
ok = True

def check(name, fn):
    global ok
    try:
        print(f"[ OK ] {name}: {fn()}")
    except Exception as e:
        ok = False
        print(f"[FAIL] {name}: {e}")

def gpu():
    import torch
    if torch.cuda.is_available():
        return f"{torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})"
    if allow_cpu:
        return "no GPU, continuing on CPU"
    raise RuntimeError("CUDA not available")

def pandas_check():
    import pandas as pd
    assert pd.DataFrame({"a": [1, 2]})["a"].sum() == 3
    return pd.__version__

def st_check():
    import sentence_transformers
    return sentence_transformers.__version__

def chroma_check():
    import chromadb
    chromadb.EphemeralClient()
    return chromadb.__version__

def spacy_check():
    import spacy
    spacy.blank("en")("test")
    return spacy.__version__

check("python", lambda: sys.version.split()[0])
check("torch / GPU", gpu)
check("pandas", pandas_check)
check("sentence-transformers", st_check)
check("chromadb", chroma_check)
check("spacy", spacy_check)

print("\nAll checks passed." if ok else "\nSome checks failed.")
sys.exit(0 if ok else 1)