# app/ingest_openstates.py
import os
import json
import time
from typing import Any, Dict, List, Optional

import requests


OPENSTATES_API_BASE = "https://v3.openstates.org"


def _write_jsonl(path: str, docs: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def _headers(api_key: str) -> Dict[str, str]:
    return {"X-API-KEY": api_key}


def _safe_str(x: Any) -> str:
    return x if isinstance(x, str) else ""


def fetch_recent_state_bills(
    *,
    api_key: Optional[str] = None,
    state: str = "GA",           # two-letter postal code
    limit: int = 50,
    sleep_s: float = 0.15,
) -> List[Dict[str, Any]]:
    """
    Fetches recent bills from OpenStates for a specific state, then enriches each bill.
    """
    api_key = api_key or os.getenv("OPENSTATES_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENSTATES_API_KEY environment variable.")

    state = state.strip().upper()
    if len(state) != 2:
        raise ValueError("state must be a 2-letter code, e.g., GA, CA, NY.")

    # OpenStates uses "jurisdiction" IDs like "Georgia"
    # but the /bills endpoint allows jurisdiction=state abbreviation in many cases.
    # We handle both by attempting a query and using results.
    params = {
        "jurisdiction": state,
        "sort": "updated_desc",
        "page": 1,
        "per_page": min(limit, 100),
    }

    url = f"{OPENSTATES_API_BASE}/bills"
    r = requests.get(url, headers=_headers(api_key), params=params, timeout=30)
    if r.status_code == 400:
        # Fallback: try "Georgia"-style jurisdiction name if needed
        # You can override by setting OPENSTATES_JURISDICTION_NAME in Railway.
        jname = os.getenv("OPENSTATES_JURISDICTION_NAME", "").strip()
        if jname:
            params["jurisdiction"] = jname
            r = requests.get(url, headers=_headers(api_key), params=params, timeout=30)

    r.raise_for_status()
    payload = r.json()
    results = payload.get("results", []) or []

    docs: List[Dict[str, Any]] = []

    for b in results[:limit]:
        bill_id = b.get("id")  # OpenStates internal ID (stable)
        identifier = _safe_str(b.get("identifier"))  # e.g., HB 123
        title = _safe_str(b.get("title"))
        subject = b.get("subject") or []
        subjects = [s for s in subject if isinstance(s, str)][:10]

        # Enrich from bill detail endpoint: /bills/{bill_id}
        summary = ""
        actions_blurb = ""
        sponsors_blurb = ""
        sources_url = ""

        if bill_id:
            detail_url = f"{OPENSTATES_API_BASE}/bills/{bill_id}"
            dr = requests.get(detail_url, headers=_headers(api_key), timeout=30)
            if dr.status_code == 200:
                d = dr.json() or {}
                abstracts = d.get("abstracts") or []
                if abstracts and isinstance(abstracts, list):
                    summary = _safe_str(abstracts[0].get("abstract")) or ""

                actions = d.get("actions") or []
                if actions and isinstance(actions, list):
                    parts = []
                    for a in actions[:6]:
                        date = _safe_str(a.get("date"))
                        desc = _safe_str(a.get("description"))
                        if date and desc:
                            parts.append(f"{date}: {desc}")
                        elif desc:
                            parts.append(desc)
                    if parts:
                        actions_blurb = "Recent actions: " + " | ".join(parts)

                sponsors = d.get("sponsorships") or []
                if sponsors and isinstance(sponsors, list):
                    sp = []
                    for s in sponsors[:5]:
                        name = _safe_str(s.get("name"))
                        if name:
                            sp.append(name)
                    if sp:
                        sponsors_blurb = "Sponsors: " + ", ".join(sp)

                sources = d.get("sources") or []
                if sources and isinstance(sources, list):
                    sources_url = _safe_str(sources[0].get("url"))

        doc_id = f"{state.lower()}-{bill_id or identifier.replace(' ', '').lower()}"
        combined_parts = [
            f"Title: {title}".strip(),
            f"Bill: {identifier} ({state})".strip() if identifier else f"State: {state}",
            f"Summary: {summary}".strip() if summary else "",
            actions_blurb.strip() if actions_blurb else "",
            sponsors_blurb.strip() if sponsors_blurb else "",
            ("Subjects: " + ", ".join(subjects)) if subjects else "",
        ]
        combined_text = "\n".join([p for p in combined_parts if p])

        docs.append(
            {
                "id": doc_id,
                "jurisdiction": state,
                "scope": "state",
                "identifier": identifier,
                "title": title,
                "text": combined_text,
                "source": "openstates",
                "url": sources_url or "",
                "updated_at": _safe_str(b.get("updated_at")) or "",
            }
        )

        time.sleep(sleep_s)

    return docs


def refresh_state_jsonl(out_path: str, *, state: str = "GA", limit: int = 50) -> int:
    docs = fetch_recent_state_bills(state=state, limit=limit)
    _write_jsonl(out_path, docs)
    return len(docs)
