"""
load_papers.py
--------------
Extracts and saves text from PDF research papers.
Input:  data/raw/*.pdf
Output: data/interim/<paper_name>.txt
"""

import os
from pathlib import Path
import pdfplumber
import yaml
from tqdm import tqdm
from src.utils.logger import setup_logger


# ───────────────────────────────
# Load configuration
# ───────────────────────────────
def load_config(path: str = "src/config/settings.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ───────────────────────────────
# PDF → text extraction
# ───────────────────────────────
def extract_text_from_pdf(pdf_path: Path, logger):
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        logger.error(f"Failed to extract {pdf_path.name}: {e}")
        return ""


# ───────────────────────────────
# Main function
# ───────────────────────────────
def load_papers():
    cfg = load_config()
    logger = setup_logger("load_papers", f"{cfg['paths']['logs']}/load_papers.log")
    logger.info("Starting PDF ingestion pipeline...")

    raw_dir = Path(cfg["paths"]["raw"])
    interim_dir = Path(cfg["paths"]["interim"])
    interim_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = list(raw_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in data/raw/. Please add some.")
        return

    for pdf in tqdm(pdf_files, desc="Extracting PDFs"):
        text = extract_text_from_pdf(pdf, logger)
        if text:
            out_path = interim_dir / f"{pdf.stem}.txt"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
            logger.info(f"Saved text: {out_path.name}")

    logger.info("PDF ingestion pipeline completed.")


# ───────────────────────────────
# Entry point
# ───────────────────────────────
if __name__ == "__main__":
    load_papers()
