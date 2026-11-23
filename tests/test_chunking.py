from src.chunking import split_text

TEXT=("Neural networks learn patterns from data. " *4+"\n\n")*6

def test_short_text_is_one_chunk():
    assert split_text("hello world", chunk_size=50)==["hello world"]


def test_empty_text_gives_no_chunks():
    assert split_text("   ")==[]

def test_chunks_respect_size():
    chunks=split_text(TEXT, chunk_size=200, overlap=40)
    assert len(chunks)>1
    assert all(len(c)<=200 for c in chunks)

def test_chunks_overlap():
    words=" ".join(f"word{i}" for i in range(200))
    chunks=split_text(words, chunk_size=100, overlap=30)
    for a, b in zip(chunks, chunks[1:]):
        assert set(a.split()) & set(b.split())

def test_no_words_lost():
    words=" ".join(f"word{i}" for i in range(200))
    chunks=split_text(words, chunk_size=100, overlap=30)
    assert set(" ".join(chunks).split())==set(words.split())

def test_text_without_separators_is_cut_hard():
    chunks=split_text("x"*1000, chunk_size=300, overlap=50)
    assert all(len(c)<=300 for c in chunks)
    assert len(chunks)==4