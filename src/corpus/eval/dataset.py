from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field

Category = Literal["factual", "multi_hop", "negative", "conversational"]


class Case(BaseModel):
    id: str
    query: str
    category: Category
    expected_sources: list[str] = Field(default_factory=list)
    expected_keyphrases: list[str] = Field(default_factory=list)
    notes: Annotated[str, Field(default="")] = ""


def load_golden(path: str | Path) -> list[Case]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"golden dataset not found at {p} — run from the repo root or pass --golden"
        )
    raw = json.loads(p.read_text())
    return [Case.model_validate(c) for c in raw]
