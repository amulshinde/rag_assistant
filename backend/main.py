import os
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from sentence_transformers import SentenceTransformer

# Import your exact local utils and config
from config import EMBEDDING_MODEL
from utils import (
    artifacts_exist,
    extract_text_from_pdf,
    pdf_has_text,
    load_rag_artifacts,
    save_rag_artifacts,
    search,
    split_into_chunks,
    build_faiss_index,
)
from gen_ai import ask_gemini

app = FastAPI()

# Configuration paths inside docker container
UPLOAD_DIR = Path("/tmp/rag_uploads")
CACHE_DIR = Path("/tmp/rag_cache")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

print("Loading local embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL)

# Generate Embeddings Endpoint
@app.post("/generate_embeddings")
async def generate_embeddings(file: UploadFile = File(...)):
    # 1. Save uploaded file to container temp directory
    pdf_file = UPLOAD_DIR / file.filename
    with open(pdf_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # 2. Check Cache or Compute Embeddings using your exact POC logic
        if artifacts_exist(pdf_file, CACHE_DIR):
            chunks, embeddings, index = load_rag_artifacts(pdf_file, CACHE_DIR)
        else:
            text = extract_text_from_pdf(str(pdf_file))

            # Validate that the uploaded PDF contains extractable text.
            # If the PDF is image-based (scanned) then we cannot extract text
            # with the current pipeline and should ask the user to upload
            # a text-based PDF instead.
            if not pdf_has_text(str(pdf_file)):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Uploaded PDF appears to be image-based (scanned) or contains "
                        "no extractable text. Please upload a text-based PDF (not a scan)."
                    ),
                )
            chunks = split_into_chunks(text, chunk_size=500, overlap=100)
            index, embeddings = build_faiss_index(chunks, model)
            save_rag_artifacts(pdf_file, chunks, embeddings, index, CACHE_DIR)

        return {
                    "message": "Embeddings generated successfully!",
                    "filename": file.filename
                }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def process_rag(
    filename: str = Form(...),
    query: str = Form(...)
):

    try:

        pdf_file = UPLOAD_DIR / filename

        if not pdf_file.exists():
            raise HTTPException(
                status_code=404,
                detail="Document not found. Please upload the PDF again."
            )

        if not artifacts_exist(pdf_file, CACHE_DIR):
            raise HTTPException(
                status_code=404,
                detail="Embeddings not found. Please regenerate embeddings."
            )

        chunks, embeddings, index = load_rag_artifacts(
            pdf_file,
            CACHE_DIR
        )

        result = search(
            query=query,
            model=model,
            index=index,
            chunks=chunks,
            top_k=3
        )

        answer = ask_gemini(
            query,
            result["context"]
        )

        return {
            "answer": answer
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )