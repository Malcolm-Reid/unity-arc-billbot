import json
import os
from dataclasses import dataclass
from typing import List

import numpy as np
from rank_bm25 import BM25Okapi

from .models import BillDoc
from .slang import normalize_query


def _safe_text(doc: BillDoc) -> str:
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
    bills: List[BillDoc] = []

    if not os.path.exists(path):
        return bills

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                bills.append(BillDoc.model_validate(json.loads(line)))
            except Exception:
                continue

    return bills


@dataclass
class SearchResult:
    doc: BillDoc
    score: float


class HybridBillIndex:
    """
    Lightweight search index using BM25 only.
    (Semantic embeddings intentionally disabled for stability.)
    """

    def __init__(self, bills: List[BillDoc]):
        self.bills = bills
        self.texts = [_safe_text(b) for b in bills]
        self.tokenized = [t.split() for t in self.texts]

        self.bm25 = BM25Okapi(self.tokenized) if bills else None

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        if not self.bills or not self.bm25:
            return []

        normalized = normalize_query(query)
        scores = self.bm25.get_scores(normalized.split())

        scored = list(enumerate(scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        results: List[SearchResult] = []
        for idx, score in scored[:top_k]:
            results.append(SearchResult(self.bills[idx], float(score)))

        return results
