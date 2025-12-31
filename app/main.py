from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .models import ChatRequest, ChatResponse, Candidate
from .retrieval_multi import load_multi_index
from .llm import make_plain_english_answer
from .slang import normalize_query
from .ingest_openstates import fetch_recent_state_bills, append_to_jsonl

app = FastAPI(
    title="Unity Arc Political Education BillBot",
    version="0.3.0",
)


# Allow browser-based widgets (GitHub Pages + Squarespace) to call the API
allowed_origins = [
    "https://malcolm-reid.github.io",
    "https://www.unityarcadvocacy.com",
    "https://unityarcadvocacy.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

    title="Unity Arc Political Education BillBot",
    version="0.3.0"
)

# --- Allow Squarespace to talk to this API ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.unityarcadvocacy.com",
        "https://unityarcadvocacy.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load bill indexes on startup ---
multi = load_multi_index(
    settings.data_path_fed,
    settings.data_path_state
)

@app.get("/health")
def health():
    return {
        "ok": True,
        "federal_loaded": len(multi.fed.bills),
        "state_loaded": len(multi.state.bills),
    }

@app.post("/reload")
def reload_data():
    global multi
    multi = load_multi_index(
        settings.data_path_fed,
        settings.data_path_state
    )
    return {
        "ok": True,
        "federal_loaded": len(multi.fed.bills),
        "state_loaded": len(multi.state.bills),
    }

# --- NEW: Refresh state bills on demand ---
@app.post("/state/refresh")
def refresh_state(state: str, days: int = 14, limit: int = 75):
    st = (state or "").upper().strip()
    if len(st) != 2:
        raise HTTPException(
            status_code=400,
            detail="State must be a 2-letter code like GA or NY."
        )

    if not settings.openstates_api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENSTATES_API_KEY is not configured."
        )

    try:
        docs = fetch_recent_state_bills(
            state=st,
            days=days,
            limit=limit
        )
        added = append_to_jsonl(
            settings.data_path_state,
            docs
        )

        global multi
        multi = load_multi_index(
            settings.data_path_fed,
            settings.data_path_state
        )

        return {
            "ok": True,
            "state": st,
            "fetched": len(docs),
            "added": added,
            "state_loaded": len(multi.state.bills),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to refresh state bills: {e}"
        )

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    user_query = req.message
    interpretation = normalize_query(user_query)

    # --- Step 1: Ask jurisdiction if unknown ---
    if req.jurisdiction == "unknown":
        return ChatResponse(
            interpretation=interpretation,
            needs_clarification=True,
            clarification_question=(
                "Are you asking about a **federal** bill (Congress) "
                "or a **state** bill?"
            ),
            clarification_options=[
                "Federal (Congress)",
                "My State"
            ],
        )

    # --- Step 2: Ask for state if needed ---
    if req.jurisdiction == "state" and not req.state:
        return ChatResponse(
            interpretation=interpretation,
            needs_clarification=True,
            clarification_question=(
                "Which state should I use? "
                "(Example: GA, California, New York)"
            ),
            clarification_options=[],
        )

    candidates = []

    def add_candidates(results, max_take):
        for r in results[:max_take]:
            candidates.append(
                Candidate(
                    bill_id=r.doc.bill_id,
                    title=r.doc.title,
                    score=r.score,
                    url=r.doc.url,
                    jurisdiction=r.doc.jurisdiction,
                    state=r.doc.state,
                )
            )

    # --- Search logic ---
    if req.jurisdiction == "federal":
        results = multi.fed.search(
            user_query,
            top_k=req.top_k
        )
        add_candidates(results, req.top_k)

    elif req.jurisdiction == "state":
        st = (req.state or "").upper()
        state_docs = [
            d for d in multi.state.bills
            if (d.state or "").upper() == st
        ]

        if not state_docs:
            return ChatResponse(
                interpretation=interpretation,
                needs_clarification=True,
                clarification_question=(
                    f"I don’t have {st} bills loaded yet. "
                    f"Please try again in a moment."
                ),
            )

        temp_index = multi.state.__class__(state_docs)
        results = temp_index.search(
            user_query,
            top_k=req.top_k
        )
        add_candidates(results, req.top_k)

    if not candidates:
        return ChatResponse(
            interpretation=interpretation,
            needs_clarification=True,
            clarification_question=(
                "I didn’t find a match yet. "
                "What topic is it closest to?"
            ),
        )

    # --- Disambiguation check ---
    sorted_cands = sorted(
        candidates,
        key=lambda c: c.score,
        reverse=True
    )

    top = sorted_cands[0].score
    second = sorted_cands[1].score if len(sorted_cands) > 1 else 0.0

    if top < 0.35 or (top - second) < 0.07:
        return ChatResponse(
            interpretation=interpretation,
            needs_clarification=True,
            clarification_question=(
                "I might have a few candidates — "
                "which one sounds closest?"
            ),
            clarification_options=[
                c.title for c in sorted_cands[:3]
            ],
            candidates=sorted_cands[:req.top_k],
        )

    # --- Generate answer ---
    best_id = sorted_cands[0].bill_id
    best_doc = None

    for idx in [multi.fed, multi.state]:
        for d in idx.bills:
            if d.bill_id == best_id:
                best_doc = d
                break
        if best_doc:
            break

    if not best_doc:
        return ChatResponse(
            interpretation=interpretation,
            needs_clarification=True,
            clarification_question=(
                "I found candidates but couldn’t "
                "load the bill details."
            ),
        )

    answer, citations = make_plain_english_answer(
        user_query,
        best_doc
    )

    return ChatResponse(
        interpretation=interpretation,
        needs_clarification=False,
        candidates=sorted_cands[:req.top_k],
        answer=answer,
        citations=citations,
    )




