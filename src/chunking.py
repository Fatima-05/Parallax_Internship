SEPARATORS = ["\n\n", "\n", ". ", " "]

def split_text(text, chunk_size=500, overlap=100, separators=SEPARATORS):
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    if not separators:
        step = chunk_size - overlap
        return [text[i:i + chunk_size] for i in range(0, len(text), step)]

    sep, rest = separators[0], separators[1:]
    if sep not in text:
        return split_text(text, chunk_size, overlap, rest)

    chunks = []
    current = []
    for piece in text.split(sep):
        if len(piece) > chunk_size:
            if current:
                chunks.append(sep.join(current))
                current = []
            chunks.extend(split_text(piece, chunk_size, overlap, rest))
            continue

        if current and len(sep.join(current + [piece])) > chunk_size:
            chunks.append(sep.join(current))
            while current and len(sep.join(current)) > overlap:
                current.pop(0)
            while current and len(sep.join(current + [piece])) > chunk_size:
                current.pop(0)
        current.append(piece)

    if current:
        chunks.append(sep.join(current))
    return chunks