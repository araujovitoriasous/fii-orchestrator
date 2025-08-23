from abc import ABC, abstractmethod
from typing import List, Any
from datetime import datetime

class Event(ABC):
    """Classe base para eventos."""

    def __init__(self, aggregate_id: str, version: int = 1):
        self.aggregate_id = aggregate_id
        self.version = version
        self.timestamp = datetime.utcnow()
