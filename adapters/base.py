from abc import ABC, abstractmethod
from typing import Dict, Any, Iterable

class DatasetAdapter(ABC):
    """Base class all dataset adapters must implement."""

    def __init__(self, path: str):
        self.path = path

    @abstractmethod
    def iter_items(self) -> Iterable[Dict[str, Any]]:
        """Yield standardized items. Each must include 'id'."""
        raise NotImplementedError

    @abstractmethod
    def build_prompt(self, item: Dict[str, Any]) -> str:
        """Return the prompt string to send to the model."""
        raise NotImplementedError

    @abstractmethod
    def score(self, item: Dict[str, Any], model_output: str) -> int:
        """Return 1 if correct, 0 otherwise."""
        raise NotImplementedError
