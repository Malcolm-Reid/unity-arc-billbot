# app/ingest_congress.py
import os
import json
import time
from typing import Any, Dict, List, Optional

import requests


CONGRESS_API_BASE = "https://api.congress.gov/v3"


def _safe_get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _write_jsonl(path: str, docs: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def _summarize_actions(actions: List[Dict[str, Any]], max_items: int = 6) -> str:
    if not actions:
        return ""
    items = []
    for a in actions[:max_items]:
        date = a.get("actionDate") or ""
        text = a.get("text") or ""
        if date and text:
            items.append(f"{date}: {text}")
        elif text:
            items.append(text)
    return "Recent actions: " + " | ".join(items) if items else ""


def fetch_recent_federal_bills(
    *,
    api_key: Optional[str] = None,
    limit: int = 50,
    congress: int = 118,
    bill_type: Optional[str] = None,  # e.g., "hr", "s"
    sleep_s: float = 0.15,
) -> List[Dict[str, Any]]:
    """
    Pulls recent federal bills from api.congress.gov, then enriches each bill
    with details when available.
    """
    api_key = api_key or os.getenv("CONGRESS_API_KEY")
    if not api_key:
        raise RuntimeError("Missing CONGRESS_API_KEY environment variable.")

    params = {
        "api_key": api_key,
        "limit": min(limit, 250),
        "offset": 0,
        "format": "json",
        "congress": congress,
    }
    if bill_type:
        params["billType"] = bill_type

    url = f"{CONGRESS_API_BASE}/bill"
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()

    bills = payload.get("bills", []) or []
    docs: List[Dict[str, Any]] = []

    for b in bills[:limit]:
        bill_type = b.get("type")  # hr, s, hjres, etc.
        bill_number = b.get("number")
        bill_congress = b.get("congress")
        title = b.get("title") or b.get("shortTitle") or ""

        # Build a stable ID
        doc_id = f"us-{bill_congress}-{bill_type}{bill_number}"

        # Try to enrich with bill detail endpoint
        actions = []
        subjects = []
        summary_text = ""
        official_title = ""

        if bill_type and bill_number and bill_congress:
            detail_url = f"{CONGRESS_API_BASE}/bill/{bill_congress}/{bill_type}/{bill_number}"
            dr = requests.get(
                detail_url,
                params={"api_key": api_key, "format": "json"},
                timeout=30,
            )
            if dr.status_code == 200:
                d = dr.json().get("bill", {}) or {}
                official_title = d.get("title") or ""
                title = title or official_title

                actions = _safe_get(d, "actions", "actions", default=[]) or []
                subjects = _safe_get(d, "subjects", "legislativeSubjects", default=[]) or []
                # Congress API sometimes has summaries array; we pick the first text
                summaries = _safe_get(d, "summaries", "summaries", default=[]) or []
                if summaries and isinstance(summaries, list):
                    summary_text = summaries[0].get("text") or ""

        # Normalize subjects into strings
        subj_names = []
        for s in subjects[:10]:
            nm = s.get("name")
            if nm:
                subj_names.append(nm)

        action_blurb = _summarize_actions(actions)

        # Create the "text" that your index will search
        combined_text_parts = [
            f"Title: {title}".strip(),
            f"Bill: {bill_type.upper() if bill_type else ''}{bill_number} (Congress {bill_congress})".strip(),
            f"Summary: {summary_text}".strip() if summary_text else "",
            action_blurb.strip() if action_blurb else "",
            ("Subjects: " + ", ".join(subj_names)) if subj_names else "",
        ]
        combined_text = "\n".join([p for p in combined_text_parts if p])

        # Best-effort public URL
        public_url = b.get("url") or ""
        if not public_url and bill_congress and bill_type and bill_number:
            public_url = f"https://www.congress.gov/bill/{bill_congress}th-congress/{'house' if bill_type=='hr' else 'senate'}-bill/{bill_number}"

        docs.append(
            {
                "id": doc_id,
                "jurisdiction": "US",
                "scope": "federal",
                "bill_type": bill_type,
                "bill_number": bill_number,
                "congress": bill_congress,
                "title": title,
                "text": combined_text,
                "source": "congress_api",
                "url": public_url,
                "updated_at": b.get("updateDate") or "",
            }
        )

        time.sleep(sleep_s)

    return docs


def refresh_federal_jsonl(
    out_path: str,
    *,
    limit: int = 50,
    congress: int = 118,
) -> int:
    docs = fetch_recent_federal_bills(limit=limit, congress=congress)
    _write_jsonl(out_path, docs)
    return len(docs)
