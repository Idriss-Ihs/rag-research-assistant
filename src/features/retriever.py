"""
retriever.py
------------
Semantic retriever over a FAISS index with optional MMR diversification.

Inputs:
  - data/processed/vector_store/faiss_index.bin
  - data/processed/vector_store/metadata.npy
  - data/processed/*_chunks.json   (to resolve text by (source, chunk_id))

Config (src/config/settings.yaml):
  - rag.embedding_model
  - rag.top_k
  - paths.vector_store, paths.processed, paths.logs

Usage (CLI):
  python -m src.features.retriever --query "What is transfer learning?" --k 5 --mmr
"""

from __future__ import annotations

import argparse
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

from src.utils.logger import setup_logger


# ───────────────────────────────
# Config helpers
# ───────────────────────────────
def load_config(path: str = "src/config/settings.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ───────────────────────────────
# Metadata & chunk text access
# ───────────────────────────────
def load_metadata(vector_dir: Path) -> np.ndarray:
    meta_path = vector_dir / "metadata.npy"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata: {meta_path}")
    return np.load(meta_path, allow_pickle=True)  # dtype=object (dicts)


@lru_cache(maxsize=128)
def _load_chunks_for_source(processed_dir: Path, source: str) -> List[str]:
    """
    Load all chunks for a given source file (cached).
    source is the base stem used when saving: <source>_chunks.json
    """
    p = processed_dir / f"{source}_chunks.json"
    if not p.exists():
        # backward compatibility: sometimes stems got sanitized; try looser match
        all_json = list(processed_dir.glob("*_chunks.json"))
        candidates = [q for q in all_json if q.stem.replace("_chunks", "").startswith(source)]
        if not candidates:
            raise FileNotFoundError(f"Could not find chunks file for source='{source}'")
        p = candidates[0]

    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_text_from_meta(meta: Dict, processed_dir: Path) -> str:
    """
    Given {'source': <stem>, 'chunk_id': <int>} return the chunk text.
    """
    chunks = _load_chunks_for_source(processed_dir, meta["source"])
    cid = meta["chunk_id"]
    if cid < 0 or cid >= len(chunks):
        return ""
    return chunks[cid]


# ───────────────────────────────
# FAISS wrapper
# ───────────────────────────────
class FaissRetriever:
    def __init__(self, cfg: dict, logger=None):
        self.cfg = cfg
        self.logger = logger or setup_logger("retriever", f"{cfg['paths']['logs']}/retriever.log")

        self.vector_dir = Path(cfg["paths"]["vector_store"])
        self.processed_dir = Path(cfg["paths"]["processed"])

        self.index_path = self.vector_dir / "faiss_index.bin"
        if not self.index_path.exists():
            raise FileNotFoundError(f"Missing FAISS index: {self.index_path}")

        self.logger.info(f"Loading FAISS index from: {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))

        self.logger.info("Loading metadata…")
        self.metadata = load_metadata(self.vector_dir)

        model_name = cfg["rag"]["embedding_model"]
        self.logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    # ---------- utilities ----------
    @staticmethod
    def _dist_to_similarity(d: float) -> float:
        """
        Convert FAISS L2 distance to a [0,1] similarity proxy.
        Works well when vectors are not unit-normalized.
        """
        return 1.0 / (1.0 + float(d))

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def _reconstruct_vectors(self, ids: List[int]) -> np.ndarray:
        """
        Reconstruct (retrieve) vectors from FAISS for given ids.
        """
        dim = self.index.d
        out = np.empty((len(ids), dim), dtype=np.float32)
        for i, idx in enumerate(ids):
            out[i] = self.index.reconstruct(int(idx))
        return out

    # ---------- core retrieval ----------
    def search(self, query: str, k: int = None, mmr: bool = False, fetch_k: int = None, lambda_mult: float = 0.5) -> List[Dict]:
        """
        Perform semantic search with optional MMR re-ranking.
        - query: user question
        - k: number of results to return (default: cfg['rag']['top_k'])
        - mmr: if True, perform Maximal Marginal Relevance re-ranking
        - fetch_k: pool size before MMR (default: max(20, 4k))
        - lambda_mult: tradeoff for MMR (1.0 → relevance only, 0.0 → diversity only)
        """
        if k is None:
            k = int(self.cfg["rag"]["top_k"])
        if fetch_k is None:
            fetch_k = max(20, 4 * k)

        self.logger.info(f"Searching: k={k}, mmr={mmr}, fetch_k={fetch_k}, lambda={lambda_mult}")
        q_emb = self._encode([query]).astype(np.float32)

        # FAISS search
        distances, indices = self.index.search(q_emb, fetch_k)  # shapes: (1, fetch_k)
        distances = distances[0]
        indices = indices[0]

        # build initial candidates
        candidates = []
        for d, idx in zip(distances, indices):
            if idx == -1:
                continue
            meta = self.metadata[idx].item() if hasattr(self.metadata[idx], "item") else self.metadata[idx]
            sim = self._dist_to_similarity(d)
            text = resolve_text_from_meta(meta, self.processed_dir)
            if not text.strip():
                continue
            candidates.append({
                "id": int(idx),
                "score": float(sim),
                "distance": float(d),
                "source": meta["source"],
                "chunk_id": int(meta["chunk_id"]),
                "text": text
            })

        # MMR re-ranking (optional)
        if mmr and candidates:
            # reconstruct vectors for selected pool
            pool_ids = [c["id"] for c in candidates]
            pool_vecs = self._reconstruct_vectors(pool_ids)
            selected = self._mmr_select(q_emb[0], pool_vecs, k=k, lambda_mult=lambda_mult)

            # order by MMR selection
            reranked = [candidates[i] for i in selected]
            results = reranked[:k]
        else:
            # simple top-k by score
            results = sorted(candidates, key=lambda x: x["score"], reverse=True)[:k]

        # post-process (deduplicate near-identical texts)
        results = self._dedupe_by_text(results)

        self.logger.info(f"Returning {len(results)} results.")
        return results

    def _mmr_select(self, query_vec: np.ndarray, doc_vecs: np.ndarray, k: int, lambda_mult: float) -> List[int]:
        """
        Maximal Marginal Relevance selection.
        Returns indices of selected items from doc_vecs.
        """
        n = doc_vecs.shape[0]
        if n <= k:
            return list(range(n))

        # precompute similarities
        q_sims = np.array([self._cosine_similarity(query_vec, dv) for dv in doc_vecs])
        d_sims = np.matmul(doc_vecs, doc_vecs.T)
        # normalize doc-doc sims to cosine range
        norms = np.linalg.norm(doc_vecs, axis=1, keepdims=True)
        denom = np.matmul(norms, norms.T) + 1e-8
        d_sims = d_sims / denom

        selected = []
        candidate_indices = list(range(n))

        # greedily pick
        first = int(np.argmax(q_sims))
        selected.append(first)
        candidate_indices.remove(first)

        while len(selected) < k and candidate_indices:
            mmr_scores = []
            for idx in candidate_indices:
                rel = q_sims[idx]
                div = max(d_sims[idx, selected]) if selected else 0.0
                mmr_score = lambda_mult * rel - (1.0 - lambda_mult) * div
                mmr_scores.append((mmr_score, idx))
            _, next_idx = max(mmr_scores, key=lambda t: t[0])
            selected.append(next_idx)
            candidate_indices.remove(next_idx)

        return selected

    @staticmethod
    def _dedupe_by_text(results: List[Dict], min_diff_chars: int = 30) -> List[Dict]:
        """
        Remove near-duplicate results based on text prefix similarity.
        """
        seen = []
        out = []
        for r in results:
            prefix = r["text"][:200]
            if not any(_levenshtein_close(prefix, s, min_diff_chars) for s in seen):
                seen.append(prefix)
                out.append(r)
        return out


# ───────────────────────────────
# Simple text similarity check (no external deps)
# ───────────────────────────────
def _levenshtein_close(a: str, b: str, min_diff_chars: int) -> bool:
    """
    Very lightweight 'closeness' check: absolute difference in length
    and shared prefix ratio. Not true Levenshtein but good enough here.
    """
    if abs(len(a) - len(b)) < min_diff_chars and a[:80] == b[:80]:
        return True
    return False


# ───────────────────────────────
# CLI for quick testing
# ───────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True, help="User question")
    parser.add_argument("--k", type=int, default=None, help="Top-k results")
    parser.add_argument("--mmr", action="store_true", help="Use MMR re-ranking")
    parser.add_argument("--lambda_mult", type=float, default=0.5, help="MMR lambda (0..1)")
    args = parser.parse_args()

    cfg = load_config()
    retriever = FaissRetriever(cfg)
    hits = retriever.search(args.query, k=args.k, mmr=args.mmr, lambda_mult=args.lambda_mult)

    print("\n=== RETRIEVAL RESULTS ===")
    for i, h in enumerate(hits, 1):
        print(f"\n[{i}] score={h['score']:.4f}  source={h['source']}  chunk={h['chunk_id']}")
        print(h["text"][:600], "..." if len(h["text"]) > 600 else "")


if __name__ == "__main__":
    main()
