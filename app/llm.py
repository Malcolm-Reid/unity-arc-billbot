from .models import BillDoc


def make_plain_english_answer(user_query: str, bill: BillDoc) -> tuple[str, list[str]]:
    """
    Safe MVP answer generator.
    Uses only bill fields we have stored (no hallucinating).
    """
    citations: list[str] = []

    if bill.url:
        citations.append(bill.url)
    if bill.text_url:
        citations.append(bill.text_url)

    bullets = []

    if bill.summary:
        bullets.append(f"- **Official summary (simplified):** {bill.summary}")
    else:
        bullets.append("- **Summary:** (No summary is stored yet for this bill.)")

    if bill.latest_action:
        date = bill.latest_action_date or "unknown date"
        bullets.append(f"- **Latest action:** {bill.latest_action} ({date})")

    if bill.sponsor:
        bullets.append(f"- **Sponsor:** {bill.sponsor}")

    if bill.topics:
        bullets.append(f"- **Topics:** {', '.join(bill.topics)}")

    answer = (
        "Here’s the bill that best matches what you asked about:\n\n"
        f"**{bill.title}**\n"
        f"(Bill ID: {bill.bill_id})\n\n"
        "What it’s about:\n"
        + "\n".join(bullets)
        + "\n\n"
        "If this doesn’t look like the one you meant, tell me one extra clue "
        "(topic, sponsor, or where you heard it) and I’ll narrow it down."
    )

    return answer, citations
