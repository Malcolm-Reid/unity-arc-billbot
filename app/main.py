from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- Import your existing utilities (safe ones) ---
from .config import settings
from .slang import normalize_query
from .llm import make_plain_english_answer

# Optional: these exist in your project already (used for refresh)
from .ingest_openstates import fetch_recent_state_bills, append_to_jsonl


# =========================
# Models (keep self-contained)
# =========================

class ChatRequest(BaseModel):
    message: str = Field(..., description="User question")
    jurisdiction: str = Field("unknown", description="federal|state|unknown")
    state: Optional[str] = Field(None, description="Two-letter state code if state")
    top_k: int = Field(5, ge=1, le=20)

class Candidate(BaseModel):
    title: str
    bill_id: Optional[str] = None
    url: Optional[str] = None
    jurisdiction: Optional[str] = None
    state: Optional[str] = None
    score: float = 0.0
    summary: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    candidates: List[Candidate] = []
    # If we need a follow-up question, these fields help the widget
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    options: Optional[List[str]] = None


# =========================
# Simple local index (no embeddings)
# =========================

STOP = set("""
a an the and or but if then else of to in for on with without from by is are was were be been being
this that these those it its you your we our they their i me my
about into over under between among as at
""".split())

def _tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    toks = [t for t in text.split() if t and t not in STOP and len(t) > 2]
    return toks

def _safe_read_jsonl(path: str) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    if not path:
        return docs
    if not os.path.exists(path):
        return docs
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    docs.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return docs
    return docs

def _normalize_doc(d: Dict[str, Any]) -> Dict[str, Any]:
    # Be defensive: different sources use different keys
    title = d.get("title") or d.get("name") or d.get("bill_title") or ""
    summary = d.get("summary") or d.get("description") or d.get("abstract") or ""
    bill_id = d.get("bill_id") or d.get("bill") or d.get("id") or d.get("identifier")
    url = d.get("url") or d.get("link") or d.get("source_url")

    jurisdiction = d.get("jurisdiction")
    state = d.get("state") or d.get("state_code")

    return {
        "title": str(title),
        "summary": str(summary),
        "bill_id": str(bill_id) if bill_id is not None else None,
        "url": str(url) if url is not None else None,
        "jurisdiction": str(jurisdiction) if jurisdiction else None,
        "state": str(state) if state else None,
        "_raw": d,
    }

def _score(query_toks: List[str], doc: Dict[str, Any]) -> float:
    # Simple token overlap scoring across title + summary
    text = f"{doc.get('title','')} {doc.get('summary','')}".lower()
    doc_toks = set(_tokenize(text))
    if not doc_toks:
        return 0.0
    hits = sum(1 for t in query_toks if t in doc_toks)
    return hits / max(6, len(query_toks))

@dataclass
class MultiIndex:
    federal_docs: List[Dict[str, Any]]
    state_docs: List[Dict[str, Any]]
    built_at: float

    def search(self, message: str, jurisdiction: str, state: Optional[str], top_k: int) -> List[Candidate]:
        q = normalize_query(message)
        q_toks = _tokenize(q)

        if jurisdiction == "federal":
            docs = self.federal_docs
        elif jurisdiction == "state":
            docs = self.state_docs
            if state:
                docs = [d for d in docs if (d.get("state") or "").upper() == state.upper()]
        else:
            docs = self.federal_docs + self.state_docs

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for d in docs:
            s = _score(q_toks, d)
            if s > 0:
                scored.append((s, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for s, d in scored[:top_k]:
            results.append(
                Candidate(
                    title=d.get("title") or "Untitled bill",
                    bill_id=d.get("bill_id"),
                    url=d.get("url"),
                    jurisdiction=d.get("jurisdiction"),
                    state=d.get("state"),
                    score=float(round(s, 4)),
                    summary=(d.get("summary") or None),
                )
            )
        return results


# =========================
# App + state
# =========================

app = FastAPI(
    title="Unity Arc Political Education BillBot",
    version="0.4.0",
)

# CORS (so GitHub Pages + Squarespace can call the API)
allowed_origins = [
    "https://malcolm-reid.github.io",
    "https://malcolm-reid.github.io/unity-arc-billbot",
    "https://www.unityarcadvocacy.com",
    "https://unityarcadvocacy.com",
    "https://unity-arc-billbot-production.up.railway.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory index (rebuild anytime we refresh)
MULTI: Optional[MultiIndex] = None


def rebuild_index() -> MultiIndex:
    fed_path = getattr(settings, "data_path_fed", "") or ""
    state_path = getattr(settings, "data_path_state", "") or ""

    fed_raw = _safe_read_jsonl(fed_path)
    st_raw = _safe_read_jsonl(state_path)

    fed_docs = []
    for d in fed_raw:
        nd = _normalize_doc(d)
        nd["jurisdiction"] = "federal"
        fed_docs.append(nd)

    state_docs = []
    for d in st_raw:
        nd = _normalize_doc(d)
        nd["jurisdiction"] = "state"
        state_docs.append(nd)

    return MultiIndex(
        federal_docs=fed_docs,
        state_docs=state_docs,
        built_at=time.time(),
    )


def ensure_index() -> MultiIndex:
    global MULTI
    if MULTI is None:
        MULTI = rebuild_index()
    return MULTI


# =========================
# Startup warm load
# =========================

@app.on_event("startup")
def warm_start():
    # Don’t crash if files are empty — app should still boot.
    global MULTI
    MULTI = rebuild_index()
    print(f"[startup] index built. federal={len(MULTI.federal_docs)} state={len(MULTI.state_docs)}")


# =========================
# Helpers for conversation
# =========================

TOPIC_OPTIONS = ["health care", "immigration", "taxes", "guns", "AI/privacy", "education", "voting"]

def ask_jurisdiction() -> ChatResponse:
    return ChatResponse(
        answer="Are you asking about a **federal** bill (Congress) or a **state** bill?",
        needs_clarification=True,
        clarification_question="Choose one:",
        options=["Federal (Congress)", "My State"],
        candidates=[],
    )

def ask_topic() -> ChatResponse:
    return ChatResponse(
        answer="I didn’t find a match yet. What topic is it closest to?",
        needs_clarification=True,
        clarification_question="Pick a topic:",
        options=TOPIC_OPTIONS,
        candidates=[],
    )

def normalize_jurisdiction(j: str) -> str:
    j = (j or "").strip().lower()
    if j in ("federal", "congress", "us", "national"):
        return "federal"
    if j in ("state", "my state"):
        return "state"
    return "unknown"


# =========================
# Routes
# =========================

@app.get("/health")
def health():
    idx = ensure_index()
    return {
        "status": "ok",
        "federal_loaded": len(idx.federal_docs),
        "state_loaded": len(idx.state_docs),
        "index_built_at": idx.built_at,
        "data_path_fed": getattr(settings, "data_path_fed", None),
        "data_path_state": getattr(settings, "data_path_state", None),
    }


@app.post("/ask", response_model=ChatResponse)
def ask(req: ChatRequest):
    idx = ensure_index()

    msg = (req.message or "").strip()
    if not msg:
        raise HTTPException(status_code=400, detail="message is required")

    jurisdiction = normalize_jurisdiction(req.jurisdiction)

    # If the widget didn’t send jurisdiction yet, ask once.
    if jurisdiction == "unknown":
        return ask_jurisdiction()

    # If they chose state but didn’t provide a state code, ask for it.
    # (We keep this simple; you can later expand to auto-detect from user IP.)
    if jurisdiction == "state" and not req.state:
        return ChatResponse(
            answer="Which state should I use? Reply with a 2-letter code like **GA**, **NY**, **CA**.",
            needs_clarification=True,
            clarification_question="State code?",
            options=None,
            candidates=[],
        )

    # If no data loaded, be honest and tell them.
    if jurisdiction == "federal" and len(idx.federal_docs) == 0:
        return ChatResponse(
            answer="I’m online, but I don’t have any **federal bill data loaded yet**. Please run the federal refresh endpoint, then try again.",
            candidates=[],
        )
    if jurisdiction == "state" and len(idx.state_docs) == 0:
        return ChatResponse(
            answer="I’m online, but I don’t have any **state bill data loaded yet**. Please run the state refresh endpoint, then try again.",
            candidates=[],
        )

    # Search
    cands = idx.search(msg, jurisdiction=jurisdiction, state=req.state, top_k=req.top_k)

    # If nothing found, ask for topic as a fallback
    if not cands:
        return ask_topic()

    # Generate plain-English answer (uses your existing llm.py)
    try:
        answer = make_plain_english_answer(
            user_question=msg,
            candidates=[c.model_dump() for c in cands],
            jurisdiction=jurisdiction,
            state=req.state,
        )
    except TypeError:
        # If your make_plain_english_answer signature differs, fall back to a simple response.
        top = cands[0]
        answer = (
            f"Here’s what I found: **{top.title}**\n\n"
            "If you can add one more clue (a name, sponsor, or what people are arguing about), I can narrow it down."
        )

    return ChatResponse(answer=answer, candidates=cands)


# -------------------------
# Refresh endpoints
# -------------------------

@app.get("/federal/refresh")
def refresh_federal(days: int = 30, limit: int = 200):
    """
    This endpoint is intentionally minimal because your federal ingestion
    method may vary (Congress.gov API vs other). If you already have a federal
    ingest module, call it here.

    For now: it just rebuilds index from whatever is already in the fed JSONL file.
    """
    global MULTI
    MULTI = rebuild_index()
    return {
        "status": "ok",
        "message": "Rebuilt index from current JSONL files.",
        "federal_loaded": len(MULTI.federal_docs),
        "state_loaded": len(MULTI.state_docs),
    }


@app.get("/state/refresh")
def refresh_state(state: str, days: int = 30, limit: int = 100):
    """
    Pull recent state bills using your existing OpenStates ingest function,
    append them to the state JSONL file, then rebuild index.
    """
    state = (state or "").strip().upper()
    if not re.match(r"^[A-Z]{2}$", state):
        raise HTTPException(status_code=400, detail="state must be a 2-letter code like GA")

    # Fetch from OpenStates and append to JSONL
    try:
        bills = fetch_recent_state_bills(state=state, days=days, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenStates fetch failed: {e}")

    try:
        append_to_jsonl(getattr(settings, "data_path_state", ""), bills)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Writing JSONL failed: {e}")

    # Rebuild index
    global MULTI
    MULTI = rebuild_index()

    return {
        "status": "ok",
        "state": state,
        "added": len(bills) if isinstance(bills, list) else None,
        "federal_loaded": len(MULTI.federal_docs),
        "state_loaded": len(MULTI.state_docs),
    }
