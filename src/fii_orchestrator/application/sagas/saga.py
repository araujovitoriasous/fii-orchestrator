from abc import ABC, abstractmethod
from typing import List, Any, Callable
from enum import Enum

class SagaStep(Enum):
    """Estados dos passos da saga."""
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
