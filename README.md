# Semantic Search Project: Week 1

This week I got the project set up and built the first part of the pipeline: downloading
a real text dataset and cleaning it so it's ready for the next steps. I downloaded
6,000 ArXiv paper abstracts and, after cleaning, ended up with 5,993 usable documents.

## Folder layout
```text
semantic-search/
├── README.md
├── requirements.txt          # the packages I used
├── requirements.lock.txt     # exact versions from pip freeze
├── data/
│   ├── raw/                  # downloaded data (not in git)
│   └── processed/            # cleaned data (not in git)
├── scripts/
│   ├── verify_env.py         # checks my setup works
│   └── download_data.py      # downloads the dataset
├── src/
│   └── preprocessing.py      # the cleaning code
└── tests/
    └── test_preprocessing.py # tests for the cleaning code
```

## How to run it
You need Python 3.10 or newer (I used 3.11.9 on Windows). Run everything from the
main project folder.

**1. Set up the environment**
```bash
python -m venv venv
venv\Scripts\activate
```
Install PyTorch first, using the command for your computer from https://pytorch.org
(or just `pip install torch` if you don't have a GPU). Then:
```bash
pip install -r requirements.txt
python scripts/verify_env.py
```
If you don't have a GPU, add `--allow-cpu` to the last command.

**2. Get and clean the data**
```bash
python scripts/download_data.py
python -m src.preprocessing --input data/raw/arxiv.jsonl
```
The cleaned data is saved to `data/processed/clean_corpus.parquet`.

**3. Run the tests**
```bash
python -m pytest -v
```

## What I did and why

**The data.** I used ArXiv abstracts from Hugging Face. They're real text and fit an ML
project. I streamed them instead of downloading the whole dataset, and I took 6,000
because I knew some would be removed during cleaning and I needed at least 5,000 left.

**The cleaning.** I wrote three small functions, one for each problem:
- `strip_html` removes HTML tags and turns things like `&amp;` back into `&`.
- `normalize_unicode` makes characters consistent (for example curly quotes become
  normal quotes) and removes invisible characters.
- `clean_whitespace` gets rid of extra spaces and blank lines.

They run in that order: HTML first so special codes are decoded, and whitespace last
to tidy up whatever the earlier steps left behind.

**The filtering.** After cleaning, I remove documents that are too short, documents that
aren't in English, and duplicates.

**The check at the end.** Before saving, the script checks the result. If there are
fewer than 5,000 documents, leftover HTML, or duplicates, it stops with an error
instead of saving.

**The tests.** There are 7 tests covering each cleaning function, the filters and the
final check.

## Results

My environment check:
```text
[ OK ] python: 3.11.9
[ OK ] torch / GPU: NVIDIA GeForce MX450 (CUDA 13.2)
[ OK ] pandas: 2.2.3
[ OK ] sentence-transformers: 3.3.1
[ OK ] chromadb: 0.5.23
[ OK ] spacy: 3.8.2
```
(ChromaDB shows a telemetry warning when it starts. It's harmless.)

What the cleaning did:
```text
{'input': 6000, 'after_length_filter': 6000, 'after_language_filter': 5998, 'after_dedupe': 5993}
```
Nothing was too short, 2 documents weren't English, and 5 were duplicates. That leaves
**5,993 documents** with the columns `id`, `title` and `text`. All 7 tests pass.

## Things that could be better
- It only removes exact duplicates, so near-duplicates stay in.
- Language detection isn't very reliable on very short texts.
- The abstracts contain math written in LaTeX (like `$(k,\ell)$`). I left it as it is.
- spaCy is installed and checked, but I haven't used it in the cleaning yet.