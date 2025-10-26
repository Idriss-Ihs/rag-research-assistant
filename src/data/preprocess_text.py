"""
preprocess_text.py
------------------
Cleans and chunks text files into manageable segments for embedding.
Input:  data/interim/*.txt
Output: data/processed/<paper_name>_chunks.json
"""

import re
import json
from pathlib import Path
from tqdm import tqdm
import yaml
from src.utils.logger import setup_logger


# ───────────────────────────────
# Load Configuration
# ───────────────────────────────
def load_config(path="src/config/settings.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ───────────────────────────────
# Text Cleaning
# ───────────────────────────────
def clean_text(text: str) -> str:
    """Removes multiple spaces, broken lines, and non-printable chars."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^A-Za-z0-9À-ÿ.,;:?!%()'\"\\-\\s]", "", text)
    return text.strip()


# ───────────────────────────────
# Text Chunking
# ───────────────────────────────
def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Splits text into overlapping chunks of approximately `chunk_size` words.
    Overlap ensures smoother transitions between chunks.
    """
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


# ───────────────────────────────
# Main Preprocessing Function
# ───────────────────────────────
def preprocess_texts():
    cfg = load_config()
    logger = setup_logger("preprocess_text", f"{cfg['paths']['logs']}/preprocess_text.log")
    logger.info("Starting text preprocessing pipeline...")

    interim_dir = Path(cfg["paths"]["interim"])
    processed_dir = Path(cfg["paths"]["processed"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    chunk_size = cfg["rag"]["chunk_size"]
    overlap = cfg["rag"]["chunk_overlap"]

    txt_files = list(interim_dir.glob("*.txt"))
    if not txt_files:
        logger.warning("No text files found in data/interim/. Run load_papers.py first.")
        return

    for txt_file in tqdm(txt_files, desc="Preprocessing text files"):
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                text = f.read()

            cleaned = clean_text(text)
            chunks = chunk_text(cleaned, chunk_size, overlap)

            out_path = processed_dir / f"{txt_file.stem}_chunks.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)

            logger.info(f"Saved {len(chunks)} chunks: {out_path.name}")

        except Exception as e:
            logger.error(f"Failed processing {txt_file.name}: {e}")

    logger.info("Text preprocessing completed successfully.")


# ───────────────────────────────
# Entry Point
# ───────────────────────────────
if __name__ == "__main__":
    preprocess_texts()
