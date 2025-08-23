from pydantic import BaseModel
from typing import Generic, TypeVar, List

T = TypeVar("T")

class PaginatedResponse(BaseModel, Generic[T]):
    data: List[T]
    total: int
    page: int
    limit: int
    has_next: bool
    has_prev: bool
