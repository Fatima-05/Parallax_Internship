# Semantic Search Project

I'm building a semantic search system during my internship, one week at a time, all in
this one repository. The idea is to clean a big pile of text, turn it into embeddings,
store them in a vector database, and search by meaning instead of by exact words.

## Weekly progress
- **Week 1 (Nov 16 – 22):** environment setup, downloading data, cleaning it. Done.

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