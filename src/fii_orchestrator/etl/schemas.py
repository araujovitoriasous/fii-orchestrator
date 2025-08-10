from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class NewsItem(BaseModel):
    id: str
    title: str
    link: str
    published_at: datetime
    source: str
    summary: Optional[str] = None
    raw_tickers: list[str] = Field(default_factory=list)  # ex: ["KNRI11", "HGLG11"]
