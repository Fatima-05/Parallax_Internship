import pandas as pd
from src.preprocessing import (clean_text, clean_whitespace, is_english,
                               normalize_unicode, preprocess, strip_html, validate)

EN = ("Machine learning models learn patterns from data and are used for many "
      "tasks such as classification, translation and retrieval of documents.")
FR = ("Les modèles d'apprentissage automatique apprennent des motifs à partir "
      "des données et sont utilisés pour de nombreuses tâches de classification.")


def test_strip_html():
    assert " ".join(strip_html("<p>Tom &amp; <b>Jerry</b></p>").split()) == "Tom & Jerry"
    assert strip_html("<style>p{}</style>Hi<script>x()</script>").strip() == "Hi"
    assert strip_html("a < b") == "a < b"


def test_normalize_unicode():
    assert normalize_unicode("ﬁne ＡＢＣ x²") == "fine ABC x2"
    assert normalize_unicode("“don’t” — ok…") == '"don\'t" - ok...'
    assert normalize_unicode("zero\u200bwidth") == "zerowidth"
    assert normalize_unicode("a\x00b\nc") == "ab\nc"


def test_clean_whitespace():
    assert clean_whitespace("  a   b\t c  \n\n\n\n d ") == "a b c\n\nd"
    assert clean_whitespace("a\u00a0\u00a0b") == "a b"


def test_clean_text_full_pipeline_and_bad_input():
    assert clean_text("<p>  “Hi”&nbsp;&amp;   ﬁne </p>") == '"Hi" & fine'
    assert clean_text(None) == "" and clean_text(float("nan")) == ""


def test_is_english():
    assert is_english(EN) and not is_english(FR) and not is_english("12345")


def test_preprocess_filters():
    df = pd.DataFrame({"text": [f"<p>{EN}</p>", EN, "short", FR, None]})
    out, report = preprocess(df, min_chars=50)
    assert len(out) == 1 and "<p>" not in out.loc[0, "text"]
    assert report == {"input": 5, "after_length_filter": 3,
                      "after_language_filter": 2, "after_dedupe": 1}


def test_validate():
    good = pd.DataFrame({"text": [EN + " one", EN + " two"]})
    assert validate(good, min_docs=2) == []
    bad = pd.DataFrame({"text": ["<b>x</b>", "<b>x</b>"]})
    assert len(validate(bad, min_docs=5)) == 3