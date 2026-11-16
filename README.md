# Semantic Search Pipeline: Week 1

This week I set up the project and built the first stage of the pipeline: getting a
real text dataset and cleaning it so it's ready for embedding and search later on.
I downloaded 6,000 ArXiv abstracts and cleaned them down to 5,993 documents, saved
as a Parquet file.

## Project structure
```text
semantic-search/
├── README.md
├── requirements.txt          # packages I installed directly
├── requirements.lock.txt     # exact versions (from pip freeze)
├── data/
│   ├── raw/                  # raw download, git-ignored
│   └── processed/            # cleaned dataset, git-ignored
├── scripts/
│   ├── verify_env.py         # checks GPU and imports
│   └── download_data.py      # downloads the raw data
├── src/
│   ├── __init__.py
│   └── preprocessing.py      # cleaning functions + command line entry point
└── tests/
    ├── __init__.py
    └── test_preprocessing.py # unit tests
```

## Setting it up
You need Python 3.10 or newer. I used 3.11.9 on Windows.

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate        # PowerShell: venv\Scripts\Activate.ps1
   ```
2. Install PyTorch first, because the right version depends on your GPU and CUDA
   version. The install command comes from https://pytorch.org. If you don't have
   a GPU, `pip install torch` is enough.
3. Install everything else:
   ```bash
   pip install -r requirements.txt
   ```
4. Check that it all works (add `--allow-cpu` if you have no GPU):
   ```bash
   python scripts/verify_env.py
   ```

## Running it
Run these from the root folder of the repo:

```bash
python scripts/download_data.py                              # saves data/raw/arxiv.jsonl
python -m src.preprocessing --input data/raw/arxiv.jsonl     # saves data/processed/clean_corpus.parquet
python -m pytest -v                                          # runs the tests
```

## How I approached it

**Choosing the data.** I went with ArXiv abstracts (`gfissore/arxiv-abstracts-2021`
on Hugging Face). They're real text, they fit an ML-focused project, and the dataset
can be streamed, so I didn't have to download the whole thing. I saved 6,000 rather
than 5,000 on purpose, since I expected the cleaning to remove some documents and
I still needed 5,000 or more at the end.

**Cleaning.** I split the cleaning into three small functions so each one can be tested
on its own:

1. `strip_html` removes HTML tags (including the contents of script and style tags)
   and decodes things like `&amp;`.
2. `normalize_unicode` converts text to a standard form (NFKC), turns curly quotes and
   long dashes into plain ones, and removes invisible characters.
3. `clean_whitespace` collapses repeated spaces and extra blank lines.

I put them in this order for a reason: HTML has to go first so entities are decoded
before anything else touches the text, and whitespace goes last because the earlier
steps can leave extra spaces behind.

**Filtering.** After cleaning, I drop documents that are shorter than 100 characters,
documents that aren't English (using `langdetect`, with a fixed seed so the results
are the same every run), and exact duplicates, ignoring upper/lower case.

**Validation.** At the end the script checks the result. If there are fewer than 5,000
documents, any HTML tags left over, or any duplicates, it stops with an error instead
of saving the file. That way I can't accidentally hand in a bad dataset.

**Tests.** There are 7 pytest tests. They cover each cleaning function, the combined
pipeline (including weird input like `None`), the filters and the validator.

## Results

Output of `python scripts/verify_env.py`:
```text
[ OK ] python: 3.11.9
[ OK ] torch / GPU: NVIDIA GeForce MX450 (CUDA 13.2)
[ OK ] pandas: 2.2.3
[ OK ] sentence-transformers: 3.3.1
[ OK ] chromadb: 0.5.23
[ OK ] spacy: 3.8.2

All checks passed.
```
ChromaDB also prints a telemetry warning when it starts. It's harmless and doesn't
affect anything.

Cleaning report:
```text
{'input': 6000, 'after_length_filter': 6000, 'after_language_filter': 5998, 'after_dedupe': 5993}
```
So no documents were too short, 2 were not English, and 5 were duplicates. The final
dataset has **5,993 documents** with the columns `id`, `title` and `text`.

All 7 tests pass.

## Limitations
- Duplicate detection only finds exact matches, so near-duplicates get through.
- `langdetect` can be unreliable on very short text.
- The abstracts contain LaTeX math like `$(k,\ell)$`. I left it in because it's part of
  the original text, but it might need handling later.
- spaCy is installed and checked in the environment test, but the Week 1 cleaning
  doesn't use it yet.