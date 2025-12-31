import json
import os
from dataclasses import dataclass
from typing import List

import numpy as np
from rank_bm25 import BM25Okapi
import faiss


from .models import BillDoc
from .slang import normalize_query


def _safe_text(doc: BillDoc) -> str:
    """
    Build a searchable text blob from a bill.
    """
    parts = [
        doc.title or "",
        doc.short_title or "",
        doc.summary or "",
        " ".join(doc.topics or []),
        doc.sponsor or "",
        doc.latest_action or "",
    ]
    return " ".join(p for p in parts if p).strip()


def load_bills_jsonl(path: str) -> List[BillDoc]:
    """
    Load bill documents from a .jsonl file.
    Each line must be one JSON object.
    """
    bills: List[BillDoc] = []

    if not os.path.exists(path):
        return bills

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                bills.append(BillDoc.model_validate(data))
            except Exception:
                # Skip malformed lines
                continue

    return bills


@dataclass
class SearchResult:
    doc: BillDoc
    score: float


class HybridBillIndex:
    """
    Hybrid search index:
    - BM25 keyword matching
    - Semantic vector similarity (FAISS) (lazy-loaded)
    """

    def __init__(
        self,
        bills: List[BillDoc],
        model_name: str = "all-MiniLM-L6-v2",
        enable_embeddings: bool = False,  # <-- important: default OFF for stability
    ):
        self.bills = bills
        self.model_name = model_name
        self.enable_embeddings = enable_embeddings

        # Build text corpus
        self.texts = [_safe_text(b) for b in bills]
        self.tokenized = [t.split() for t in self.texts]

        self.bm25 = BM25Okapi(self.tokenized) if bills else None

        # Embedding/FAISS pieces (only created if enabled AND we have bills)
        self.embedder = None
        self.embeddings = None
        self.faiss_index = None

        if self.enable_embeddings and bills:
            self._build_vector_index()


    def _build_vector_index(self):
    from sentence_transformers import SentenceTransformer
    self.embedder = SentenceTransformer(self.model_name)

    embeddings = self.embedder.encode(
        self.texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        self.faiss_index = index

    def refresh(self, bills: List[BillDoc]):
        """
        Rebuild the index when data changes.
        """
        self.__init__(bills=bills)

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Search bills using a blend of BM25 and vector similarity.
        """
        if not self.bills:
            return []

        normalized_query = normalize_query(query)

        # --- BM25 keyword scores ---
        bm25_scores = np.zeros(len(self.bills), dtype=np.float32)
        if self.bm25 is not None:
            raw_scores = self.bm25.get_scores(
                normalized_query.split()
            )
            bm25_scores = np.array(raw_scores, dtype=np.float32)
            if bm25_scores.max() > 0:
                bm25_scores /= bm25_scores.max()

        # --- Vector similarity scores ---
        vec_scores = np.zeros(len(self.bills), dtype=np.float32)
        if self.faiss_index is not None:
            query_vec = self.embedder.encode(
                [normalized_query],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            query_vec = np.asarray(query_vec, dtype=np.float32)

            distances, indices = self.faiss_index.search(
                query_vec,
                k=min(top_k * 10, len(self.bills)),
            )

            for score, idx in zip(distances[0], indices[0]):
                vec_scores[idx] = float(score)

            if vec_scores.max() > 0:
                vec_scores /= vec_scores.max()

        # --- Blend scores ---
        blended_scores = (0.45 * bm25_scores) + (0.55 * vec_scores)

        top_indices = np.argsort(-blended_scores)[:top_k]

        return [
            SearchResult(self.bills[i], float(blended_scores[i]))
            for i in top_indices
        ]

