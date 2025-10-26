"""
download_papers.py
------------------
Downloads a batch of research papers (PDFs) from arXiv by keyword.
Saved to: data/raw/
"""

import os
import requests
import feedparser
from pathlib import Path
import yaml
from tqdm import tqdm
from src.utils.logger import setup_logger


# ───────────────────────────────
# Load configuration
# ───────────────────────────────
def load_config(path="src/config/settings.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ───────────────────────────────
# Download from arXiv
# ───────────────────────────────
def download_from_arxiv(query="climate", max_results=5):
    """Fetch metadata and PDF URLs from arXiv."""
    base_url = "http://export.arxiv.org/api/query?"
    search = f"search_query=all:{query}&start=0&max_results={max_results}"
    response = requests.get(base_url + search)
    feed = feedparser.parse(response.text)
    return [
        {
            "title": entry.title.replace("\n", " ").strip(),
            "pdf_url": next(
                (l.href for l in entry.links if l.type == "application/pdf"), None
            ),
        }
        for entry in feed.entries
    ]


# ───────────────────────────────
# Download PDFs
# ───────────────────────────────
def download_papers():
    cfg = load_config()
    logger = setup_logger("download_papers", f"{cfg['paths']['logs']}/download_papers.log")

    logger.info("Starting paper batch download...")
    out_dir = Path(cfg["paths"]["raw"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load settings
    topic = cfg.get("rag", {}).get("topic", "climate change")
    max_results = cfg.get("rag", {}).get("batch_size", 5)

    logger.info(f"Querying arXiv for: {topic} (limit={max_results})")
    papers = download_from_arxiv(query=topic, max_results=max_results)

    if not papers:
        logger.warning("No papers found for query.")
        return

    for paper in tqdm(papers, desc="Downloading papers"):
        title = paper["title"].replace(" ", "_").replace("/", "_")[:80]
        pdf_url = paper["pdf_url"]
        if not pdf_url:
            logger.warning(f"No PDF found for: {title}")
            continue

        try:
            pdf_path = out_dir / f"{title}.pdf"
            if pdf_path.exists():
                logger.info(f"Already downloaded: {title}")
                continue

            r = requests.get(pdf_url, timeout=30)
            if r.status_code == 200:
                with open(pdf_path, "wb") as f:
                    f.write(r.content)
                logger.info(f"Downloaded: {title}")
            else:
                logger.warning(f"Failed {title}: HTTP {r.status_code}")

        except Exception as e:
            logger.error(f"Error downloading {title}: {e}")

    logger.info("Batch paper download completed.")


# ───────────────────────────────
# Entry point
# ───────────────────────────────
if __name__ == "__main__":
    download_papers()
