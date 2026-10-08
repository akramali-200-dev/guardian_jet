from pydantic import BaseModel, Field
from typing import Literal


class SearchRequest(BaseModel):
    query: str = Field(..., description="User search query", min_length=1)
    user_id: str = Field(..., description="User ID")

class SearchResponse(BaseModel):
    success: bool
    query: str
    answer: str
    records: str
    total_results: int

class RouteDecision(BaseModel):
    type: Literal["DATA", "DOCS", "GREETING"] = Field(..., description="DATA, DOCS or GREETING")
    stock_type: Literal["AVAILABLE", "SOLD", "ANY"] = Field(..., description="Available, Sold or Any stock type")
