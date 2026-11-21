
import re

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


# --------------------------------------------------
# 1. Split long text into smaller chunks
# --------------------------------------------------

def split_long_text(text, chunk_size, overlap):

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size"
        )

    chunks = []
    start = 0

    while start < len(text):

        end = min(start + chunk_size, len(text))

        # Try to avoid cutting words
        if end < len(text):

            boundary = text.rfind(" ", start, end)

            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


# --------------------------------------------------
# 2. Split a document into chunks
# --------------------------------------------------

def chunk_document(
    document,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):

    text = document.page_content

    # Split text using empty lines between paragraphs
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:

        # Handle paragraphs larger than chunk_size
        if len(paragraph) > chunk_size:

            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""

            chunks.extend(
                split_long_text(
                    paragraph,
                    chunk_size,
                    overlap
                )
            )

            continue

        candidate = (
            current_chunk + "\n\n" + paragraph
            if current_chunk
            else paragraph
        )

        if len(candidate) <= chunk_size:

            current_chunk = candidate

        else:

            if current_chunk:
                chunks.append(current_chunk)

            current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# --------------------------------------------------
# 3. Convert all documents into chunk records
# --------------------------------------------------

def create_chunks(
    documents,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):

    chunk_records = []

    for document in documents:

        chunks = chunk_document(
            document,
            chunk_size=chunk_size,
            overlap=overlap
        )

        for chunk_index, chunk_text in enumerate(chunks):

            chunk_records.append({

                "text": chunk_text,

                "metadata": {
                    **document.metadata,
                    "chunk_index": chunk_index
                }

            })

    return chunk_records
