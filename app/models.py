from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Literal

# Jurisdiction types
Jurisdiction = Literal[
    "federal",
    "state",
    "both",
    "unknown"
]

class BillDoc(BaseModel):
    # Core identifiers
    bill_id: str
    jurisdiction: Jurisdiction = "federal"
    state: Optional[str] = None  # 2-letter code like GA, NY

    # Federal-only fields
    congress: Optional[int] = None
    bill_type: Optional[str] = None
    number: Optional[str] = None

    # Descriptive fields
    title: str
    short_title: Optional[str] = None
    summary: Optional[str] = None
    sponsor: Optional[str] = None

    # Status
    latest_action: Optional[str] = None
    latest_action_date: Optional[str] = None

    # Topics and links
    topics: List[str] = Field(default_factory=list)
    url: Optional[str] = None
    text_url: Optional[str] = None

    # Raw source data (for debugging / future use)
    extra: Dict = Field(default_factory=dict)

class ChatRequest(BaseModel):
    message: str
    top_k: int = 5

    # Qualifier answers
    jurisdiction: Jurisdiction = "unknown"
    state: Optional[str] = None

class Candidate(BaseModel):
    bill_id: str
    title: str
    score: float
    url: Optional[str] = None
    jurisdiction: Jurisdiction = "federal"
    state: Optional[str] = None

class ChatResponse(BaseModel):
    interpretation: str

    needs_clarification: bool
    clarification_question: Optional[str] = None
    clarification_options: List[str] = Field(default_factory=list)

    candidates: List[Candidate] = Field(default_factory=list)
    answer: Optional[str] = None
    citations: List[str] = Field(default_factory=list)
