from abc import ABC, abstractmethod

class Model(ABC):
    name: str

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Return model's raw completion text."""
        raise NotImplementedError


class EchoModel(Model):
    """
    Dummy model for testing pipeline: just echoes the prompt.
    Obviously gets everything wrong; used to test wiring.
    """
    def __init__(self):
        self.name = "echo"

    def generate(self, prompt: str) -> str:
        return prompt  # terrible model, great for smoke tests


def get_model(name: str) -> Model:
    """
    Extend this to add real models:
      - gemini
      - deepseek
      - llama
      - mistral
    For now supports:
      - "echo"
    """
    if name == "echo":
        return EchoModel()
    raise ValueError(f"Unknown model: {name}")
