# Semantic Search Project

I'm building a semantic search system during my internship, one week at a time, all in
this one repository. The idea is to clean a big pile of text, turn it into embeddings,
store them in a vector database, and search by meaning instead of by exact words.

## Weekly progress
- **Week 1 (Nov 16 – 22):** environment setup, downloading data, cleaning it. Done.
- **Week 2 (Nov 23 – 29):** chunking, embeddings, ChromaDB and a search benchmark. Done.

## Setup
You need Python 3.10 or newer. I used 3.11.9 on Windows.

```bash
python -m venv venv
venv\Scripts\activate
```
Install PyTorch first using the command for your computer from https://pytorch.org
(or `pip install torch` if you have no GPU). Then:
```bash
pip install -r requirements.txt
python scripts/verify_env.py
```
No GPU? Add `--allow-cpu` to the last command.

## Packages I use
- **torch**: deep learning library, runs on the GPU
- **pandas, pyarrow**: working with tables and saving Parquet files
- **datasets**: downloading data from Hugging Face
- **beautifulsoup4**: removing HTML
- **langdetect**: checking the language of a text
- **pytest**: running tests
- **spacy, sentence-transformers, chromadb**: installed and checked, but not used yet
- **numpy, python-dotenv**: supporting libraries

Exact versions are in `requirements.txt`.

---

## Week 1: Data download and cleaning

**What I did:** I downloaded 6,000 ArXiv abstracts and cleaned them. After cleaning I
had 5,993 documents.

**To run it:**
```bash
python scripts/download_data.py
python -m src.preprocessing --input data/raw/arxiv.jsonl
python -m pytest -v
```
The clean data ends up in `data/processed/clean_corpus.parquet`.

**Why ArXiv abstracts:** they're real text, they suit an ML project, and I could stream
them instead of downloading the whole dataset. I took 6,000 because some get removed
during cleaning and I needed at least 5,000 left.

**How the cleaning works:** each step is its own small function.
1. Remove HTML tags and turn codes like `&amp;` back into `&`.
2. Make characters consistent (for example curly quotes become normal quotes).
3. Remove extra spaces and blank lines.

Then I throw away texts that are too short, not in English, or duplicated. At the end
the script checks the result and refuses to save it if something is wrong.

**Tests:** 7 tests check the cleaning functions, the filters and the final check. They
all pass.

**What came out:**
```text
{'input': 6000, 'after_length_filter': 6000, 'after_language_filter': 5998, 'after_dedupe': 5993}
```
Nothing was too short, 2 weren't English, 5 were duplicates. Final dataset: 5,993
documents with the columns `id`, `title` and `text`.

My environment check:
```text
[ OK ] python: 3.11.9
[ OK ] torch / GPU: NVIDIA GeForce MX450 (CUDA 13.2)
[ OK ] pandas: 2.2.3
[ OK ] sentence-transformers: 3.3.1
[ OK ] chromadb: 0.5.23
[ OK ] spacy: 3.8.2
```

**What could be better:**
- Only exact duplicates are removed.
- Language detection is shaky on very short texts.
- The math in the abstracts (LaTeX) is left as it is.


## Week 2: Chunking, embeddings and ChromaDB

**What I did:** I split the 5,993 cleaned abstracts into 14,088 chunks, turned them into
embeddings, stored them in ChromaDB, and built a search script and a benchmark.

**To run it:**
```bash
python -m scripts.build_index
python -m scripts.search black hole entropy
python -m scripts.benchmark
python -m pytest -v
```
The database is saved in `chroma_db/` and the embedding log in `logs/build_index.log`.

**Chunking:** a recursive splitter that cuts at paragraphs, then lines, then sentences,
then spaces. Chunks are at most 500 characters and overlap by 100 so nothing is lost at
the edges. 6 tests cover it.

**Embeddings:** `all-MiniLM-L6-v2` (384 numbers per chunk). It took about 6 to 8 ms per
chunk on my GPU, 156.7 s for the whole build, with a peak of 403 MB GPU memory and about
1.45 GB RAM.

**Distance metric:** cosine distance (`1 - cosine similarity`), so a value close to 0
means very similar meaning. It has to be set when the collection is created.

**Saving and re-running:** chunk IDs are fixed and stored chunks are skipped, so running
the build twice doesn't create duplicates. `--reset` rebuilds from scratch. The model name
is saved with the collection, and the scripts stop if it doesn't match. An empty or
missing collection gives a clear message.

**Benchmark** (10 queries, 14,088 chunks):
```text
embed  ms: mean 14.4, p95 18.8
search ms: mean 7.0, p95 15.3
mean recall@5 vs exact search: 1.000
```
I compared ChromaDB's results with an exact search. My first run scored 0.900 because one
query missed the true neighbours. Raising `hnsw:search_ef` from 10 to 100 and rebuilding
fixed it. Most top results were clearly on topic, but the prime numbers and neural network
queries matched only loosely.

**What could be better:**
- Chunks are cut by characters, not tokens.
- The data is mostly 2007 physics and math papers, so machine learning queries match weakly.
- I didn't test other chunk sizes.