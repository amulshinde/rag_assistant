import hashlib
import json
import os
import pickle
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
from PyPDF2 import PdfReader

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def _get_cache_paths(pdf_path, cache_dir="cache"):
    pdf_path = Path(pdf_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    base_name = pdf_path.stem
    safe_name = "".join(
        c if c.isalnum() or c in "._-" else "_"
        for c in base_name
    )

    return {
        "chunks": cache_dir / f"{safe_name}.chunks.pkl",
        "embeddings": cache_dir / f"{safe_name}.embeddings.npy",
        "index": cache_dir / f"{safe_name}.faiss",
        "meta": cache_dir / f"{safe_name}.meta.json",
    }


def _file_hash(path):
    hasher = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            hasher.update(block)

    return hasher.hexdigest()


def artifacts_exist(pdf_path, cache_dir="cache"):
    paths = _get_cache_paths(pdf_path, cache_dir)

    if not all(path.exists() for path in paths.values()):
        return False

    try:
        with open(paths["meta"], "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return False

    if meta.get("pdf_path") != str(Path(pdf_path).resolve()):
        return False

    return meta.get("file_hash") == _file_hash(pdf_path)


def save_rag_artifacts(pdf_path, chunks, embeddings, index, cache_dir="cache"):
    paths = _get_cache_paths(pdf_path, cache_dir)

    with open(paths["chunks"], "wb") as f:
        pickle.dump(chunks, f)

    np.save(str(paths["embeddings"]), embeddings)
    faiss.write_index(index, str(paths["index"]))

    metadata = {
        "pdf_path": str(Path(pdf_path).resolve()),
        "file_hash": _file_hash(pdf_path),
        "chunk_count": len(chunks),
        "vector_count": int(index.ntotal),
        "dimension": int(embeddings.shape[1]),
        "saved_at": datetime.utcnow().isoformat() + "Z",
    }

    with open(paths["meta"], "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Artifacts saved.")


def load_rag_artifacts(pdf_path, cache_dir="cache"):
    paths = _get_cache_paths(pdf_path, cache_dir)

    with open(paths["chunks"], "rb") as f:
        chunks = pickle.load(f)

    embeddings = np.load(str(paths["embeddings"]))
    index = faiss.read_index(str(paths["index"]))

    print("Artifacts loaded.")

    return chunks, embeddings, index


# --------------------------------------------------
# 1. Read PDF
# --------------------------------------------------
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def pdf_has_text(pdf_path, min_chars=20):
    """Return True if the PDF contains at least `min_chars` of extractable text.

    This is a lightweight check used to detect image-only (scanned) PDFs
    which do not contain embedded/selectable text.
    """
    text = extract_text_from_pdf(pdf_path)

    if not text:
        return False

    # normalize whitespace and check length
    cleaned = "".join(text.split())

    return len(cleaned) >= min_chars

# --------------------------------------------------
# 2. Split text into chunks
# --------------------------------------------------
def split_into_chunks(text, chunk_size=300, overlap=100):
    words = text.split()

    chunks = []

    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])

        if len(chunk.strip()) > 0:
            chunks.append(chunk)

    return chunks


# --------------------------------------------------
# 3. Build FAISS Index
# --------------------------------------------------
def build_faiss_index(chunks, model):
    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index, embeddings


# --------------------------------------------------
# 4. Search
# --------------------------------------------------
def search(query, model, index, chunks, top_k=3):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    retrieved_indices = indices[0]

    # preserve original document order
    ordered_indices = sorted(retrieved_indices)

    retrieved_chunks = [
        chunks[i]
        for i in ordered_indices
    ]

    context = "\n\n".join(retrieved_chunks)

    return {
        "distances": distances[0],
        "indices": retrieved_indices,
        "context": context,
        "chunks": retrieved_chunks,
    }

