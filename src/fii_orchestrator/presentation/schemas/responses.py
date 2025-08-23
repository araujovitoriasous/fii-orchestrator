from pydantic import BaseModel
from typing import List, Any, Optional
from datetime import datetime

class StandardResponse(BaseModel):
    data: Any
    message: Optional[str] = None
    timestamp: datetime = datetime.now()

class FundResponse(BaseModel):
    ticker: str
    cnpj: Optional[str] = None
    razao_social: Optional[str] = None
    fonte: str
