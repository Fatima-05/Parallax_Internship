import argparse
import html
import re
import unicodedata

import pandas as pd
from bs4 import BeautifulSoup
from langdetect import DetectorFactory, detect

DetectorFactory.seed = 0  # make language detection deterministic

QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                        "–": "-", "—": "-", "…": "..."})
INVISIBLE = re.compile("[\u200b-\u200f\u2060\ufeff\u00ad]")
CONTROL = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def strip_html(text):
    if "<" not in text and "&" not in text:
        return text
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return html.unescape(soup.get_text(" "))


def normalize_unicode(text):
    text = unicodedata.normalize("NFKC", text).translate(QUOTES)
    return CONTROL.sub("", INVISIBLE.sub("", text))


def clean_whitespace(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[^\S\n]+", " ", text)      # collapse spaces/tabs
    text = re.sub(r" ?\n ?", "\n", text)       # trim around newlines
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def clean_text(text):
    if not isinstance(text, str):
        return ""
    return clean_whitespace(normalize_unicode(strip_html(text)))


def is_english(text):
    try:
        return detect(text[:1000]) == "en"
    except Exception:  # empty / undetectable text
        return False


def preprocess(df, text_col="text", min_chars=100):
    """Clean text, then drop short, non-English and duplicate docs."""
    report = {"input": len(df)}
    df = df.copy()
    df[text_col] = df[text_col].map(clean_text)

    df = df[df[text_col].str.len() >= min_chars]
    report["after_length_filter"] = len(df)

    df = df[df[text_col].map(is_english).astype(bool)]
    report["after_language_filter"] = len(df)

    df = df[~df[text_col].str.lower().duplicated()]
    report["after_dedupe"] = len(df)

    return df.reset_index(drop=True), report


def validate(df, text_col="text", min_docs=5000):
    problems = []
    if len(df) < min_docs:
        problems.append(f"only {len(df)} docs, need {min_docs}")
    if df[text_col].str.contains(r"<[a-zA-Z/][^>]*>").any():
        problems.append("HTML tags remain")
    if df[text_col].str.lower().duplicated().any():
        problems.append("duplicates remain")
    return problems


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="data/processed/clean_corpus.parquet")
    p.add_argument("--min-chars", type=int, default=100)
    p.add_argument("--min-docs", type=int, default=5000)
    a = p.parse_args()

    clean, report = preprocess(pd.read_json(a.input, lines=True, dtype=False),
                               min_chars=a.min_chars)
    print(report)
    problems = validate(clean, min_docs=a.min_docs)
    if problems:
        raise SystemExit(f"Validation failed: {problems}")
    clean.to_parquet(a.output, index=False)
    print(f"Saved {len(clean)} docs to {a.output}")


if __name__ == "__main__":
    main()