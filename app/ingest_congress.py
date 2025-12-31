import requests
from datetime import datetime, timedelta

from .config import settings
from .models import BillDoc

BASE = "https://api.congress.gov/v3"

def fetch_recent_federal_bills(days: int = 14, limit: int = 50) -> list[BillDoc]:
    if not settings.congress_api_key:
        raise RuntimeError("CONGRESS_API_KEY is not configured.")

    since = (datetime.utcnow() - timedelta(days=days)).date().isoformat()

    # NOTE: Congress.gov API endpoints vary a bit; this is the common pattern:
    params = {
        "format": "json",
        "limit": limit,
        "fromDateTime": since,
        "api_key": settings.congress_api_key,
    }

    r = requests.get(f"{BASE}/bill", params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    bills = []
    for item in data.get("bills", []):
        bill_id = item.get("bill", {}).get("billId") or item.get("billId") or ""
        title = item.get("title") or "Federal bill"
        number = str(item.get("number") or "")
        bill_type = item.get("type") or None
        congress = item.get("congress") or None

        # Sponsor may be nested depending on payload
        sponsor = None
        s = item.get("sponsors")
        if isinstance(s, list) and s:
            sponsor = s[0].get("fullName") or s[0].get("name")

        url = item.get("url")

        bills.append(
            BillDoc(
                bill_id=bill_id or f"{bill_type}{number}-{congress}",
                jurisdiction="federal",
                state=None,
                congress=congress,
                bill_type=bill_type,
                number=number,
                title=title,
                short_title=None,
                summary=item.get("summary") or None,
                sponsor=sponsor,
                latest_action=item.get("latestAction", {}).get("text") if isinstance(item.get("latestAction"), dict) else None,
                latest_action_date=item.get("latestAction", {}).get("actionDate") if isinstance(item.get("latestAction"), dict) else None,
                topics=[],
                url=url,
                text_url=None,
                extra=item,
            )
        )

    return bills
