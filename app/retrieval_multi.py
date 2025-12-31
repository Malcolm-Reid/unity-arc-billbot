from dataclasses import dataclass
from .retrieval import HybridBillIndex, load_bills_jsonl
from .models import BillDoc


@dataclass
class MultiIndex:
    fed: HybridBillIndex
    state: HybridBillIndex


def load_multi_index(path_fed: str, path_state: str) -> MultiIndex:
    """
    Load federal and state bill datasets and build indexes for each.
    """
    fed_docs: list[BillDoc] = load_bills_jsonl(path_fed)
    state_docs: list[BillDoc] = load_bills_jsonl(path_state)

    return MultiIndex(
        fed=HybridBillIndex(fed_docs),
        state=HybridBillIndex(state_docs),
    )
print(f"Loaded {len(fed_docs)} federal bills")
print(f"Loaded {len(state_docs)} state bills")
