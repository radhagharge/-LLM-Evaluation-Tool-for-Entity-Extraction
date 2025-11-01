from abc import ABC, abstractmethod

class ModelInterface(ABC):
    """Abstract base class for all LLM model adapters."""

    @abstractmethod
    def extract_entities(self, text: str, entities: list[str]):
        """
        Extract the given entities from the provided text.
        Must return:
        - extracted_entities (dict)
        - latency (float, seconds)
        - token_usage (int)
        - cost (float)
        """
        pass
