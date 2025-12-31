import json
import os
import requests
from datetime import datetime, timedelta

from .config import settings
from .models import BillDoc

BASE_URL = "https://v3.openstates.org"


def _headers():
    if not settings.openstates_api_key:
        raise RuntimeError("OPENSTATES_API_KEY is not configured.")
    return {
        "X-API-KEY": settings.openstates_api_key
    }


def fetch_recent_state_bills(
    state: str,
    days: int = 14,
    limit: int = 75,
) -> list[BillDoc]:
    """
    Fetch recently updated bills for a given state using Open States API.
    State must be a 2-letter abbreviation (e.g., GA, NY).
    """
    st = state.upper().strip()
    if len(st) != 2:
        raise ValueError("State must be a 2-letter abbreviation.")

    since_date = (
        datetime.utcnow() - timedelta(days=days)
    ).date().isoformat()

    params = {
        "jurisdiction": st,
        "updated_since": since_date,
        "per_page": limit,
        "sort": "updated_desc",
    }

    response = requests.get(
        f"{BASE_URL}/bills",
        headers=_headers(),
        params=params,
        timeout=30,
    )

    response.raise_for_status()
    data = response.json()

    results = data.get("results", []) if isinstance(data, dict) else data
    docs: list[BillDoc] = []

    for item in results:
        bill_id = item.get("id") or f"openstates-{st}-{item.get('identifier','unknown')}"
        title = item.get("title") or item.get("identifier") or "State bill"

        summary = None
        if item.get("abstracts"):
            summary = (item["abstracts"][0] or {}).get("abstract")

        latest_action = None
        latest_action_date = None
        if item.get("actions"):
            last_action = item["actions"][-1]
            latest_action = last_action.get("description")
            latest_action_date = last_action.get("date")

        sponsor = None
        if item.get("sponsorships"):
            sponsor = (item["sponsorships"][0] or {}).get("name")

        url = item.get("openstates_url")
        if not url and item.get("sources"):
            url = (item["sources"][0] or {}).get("url")

        docs.append(
            BillDoc(
                bill_id=bill_id,
                jurisdiction="state",
                state=st,
                title=title,
                short_title=item.get("identifier"),
                summary=summary,
                sponsor=sponsor,
                latest_action=latest_action,
                latest_action_date=latest_action_date,
                topics=[],
                url=url,
                text_url=None,
                extra=item,
            )
        )

    return docs


def append_to_jsonl(path: str, docs: list[BillDoc]) -> int:
    """
    Append new bills to a JSONL file, avoiding duplicates by bill_id.
    Returns number of new bills added.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    existing_ids = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    existing_ids.add(json.loads(line).get("bill_id"))
                except Exception:
                    continue

    new_docs = [d for d in docs if d.bill_id not in existing_ids]

    if not new_docs:
        return 0

    with open(path, "a", encoding="utf-8") as f:
        for doc in new_docs:
            f.write(doc.model_dump_json())
            f.write("\n")

    return len(new_docs)
