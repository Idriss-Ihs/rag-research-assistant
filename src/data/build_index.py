"""
build_index.py
---------------
Encodes text chunks into embeddings and stores them in a FAISS vector index.
Input:  data/processed/*_chunks.json
Output: data/processed/faiss_index/
"""

import os
import json
import yaml
from pathlib import Path
from tqdm import tqdm
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from src.utils.logger import setup_logger


# ───────────────────────────────
# Load configuration
# ───────────────────────────────
def load_config(path="src/config/settings.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ───────────────────────────────
# Load text chunks
# ───────────────────────────────
def load_chunks(processed_dir: Path):
    chunks = []
    metadata = []
    for file in processed_dir.glob("*_chunks.json"):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for idx, chunk in enumerate(data):
                chunks.append(chunk)
                metadata.append({
                    "source": file.stem.replace("_chunks", ""),
                    "chunk_id": idx
                })
    return chunks, metadata


# ───────────────────────────────
# Main FAISS building function
# ───────────────────────────────
def build_faiss_index():
    cfg = load_config()
    logger = setup_logger("build_index", f"{cfg['paths']['logs']}/build_index.log")
    logger.info("Starting FAISS index construction...")

    processed_dir = Path(cfg["paths"]["processed"])
    vector_dir = Path(cfg["paths"]["vector_store"])
    vector_dir.mkdir(parents=True, exist_ok=True)

    # Load text chunks
    chunks, metadata = load_chunks(processed_dir)
    if not chunks:
        logger.warning("No chunk files found. Run preprocess_text.py first.")
        return

    # Load embedding model
    model_name = cfg["rag"]["embedding_model"]
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    # Encode chunks
    logger.info(f"Encoding {len(chunks)} text chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    logger.info(f"FAISS index created with {index.ntotal} vectors (dim={dim})")

    # Save index
    faiss.write_index(index, str(vector_dir / "faiss_index.bin"))
    np.save(vector_dir / "metadata.npy", np.array(metadata, dtype=object))
    logger.info("FAISS index and metadata saved successfully.")

    logger.info("FAISS index construction completed ✅")


# ───────────────────────────────
# Entry point
# ───────────────────────────────
if __name__ == "__main__":
    build_faiss_index()
