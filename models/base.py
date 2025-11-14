from abc import ABC, abstractmethod

# -------------------- IMPORTS --------------------
from models.google_gemini import generate_text as gemini_generate
from models.openrouter_llama import generate_text as llama_generate
from models.openrouter_mixtral import generate_text as mixtral_generate
from models.openrouter_gptoss import generate_text as gptoss_generate  # NEW


# -------------------- BASE CLASS --------------------
class Model(ABC):
    name: str

    @abstractmethod
    def generate(self, prompt: str, **gen_kwargs) -> str:
        """
        Return model's raw completion text.

        gen_kwargs can include things like:
          - max_tokens / max_output_tokens
          - temperature
          - top_p
          - etc.
        """
        raise NotImplementedError


# -------------------- DUMMY MODEL --------------------
class EchoModel(Model):
    """Dummy model for testing pipeline: just echoes the prompt."""
    def __init__(self):
        self.name = "echo"

    def generate(self, prompt: str, **gen_kwargs) -> str:
        # ignore gen_kwargs; just echo
        return prompt


# -------------------- REAL MODELS --------------------
class GeminiModel(Model):
    def __init__(self):
        self.name = "gemini-2.5-flash-lite"

    def generate(self, prompt: str, **gen_kwargs) -> str:
        # google_gemini.generate_text now accepts **gen_kwargs
        return gemini_generate(prompt, **gen_kwargs)


class LlamaModel(Model):
    def __init__(self, name: str = "llama-3.3-70b-instruct"):
        self.name = name

    def generate(self, prompt: str, **gen_kwargs) -> str:
        return llama_generate(prompt, **gen_kwargs)


class MixtralModel(Model):
    def __init__(self, name: str = "mistral-7b-instruct"):
        self.name = name

    def generate(self, prompt: str, **gen_kwargs) -> str:
        return mixtral_generate(prompt, **gen_kwargs)


class GPTOssModel(Model):
    def __init__(self, name: str = "gpt-oss-20b"):
        self.name = name

    def generate(self, prompt: str, **gen_kwargs) -> str:
        return gptoss_generate(prompt, **gen_kwargs)


# -------------------- FACTORY --------------------
def get_model(name: str) -> Model:
    """
    Returns the correct model class based on name.
    Supports:
      - echo
      - gemini-2.5-flash-lite
      - gpt-oss-20b
      - llama-3.3-70b-instruct
      - mistral-7b-instruct
    """
    name = name.lower()

    if name == "echo":
        return EchoModel()
    elif name.startswith("gemini"):
        return GeminiModel()
    elif name.startswith("gpt-oss"):
        return GPTOssModel(name)
    elif name.startswith("llama"):
        return LlamaModel(name)
    elif name.startswith("mistral"):
        return MixtralModel(name)
    else:
        raise ValueError(f"Unknown model: {name}")
