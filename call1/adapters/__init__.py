"""Platform inference boundary. Shared application code imports only this module."""
from __future__ import annotations
import os
from typing import Protocol
from call1.models.schemas import TranscriptTurn

class InferenceAdapter(Protocol):
    def transcribe(self, path: str, channels: int, agent_channel: int = 0) -> list[TranscriptTurn]: ...
    def generate(self, system: str, prompt: str, max_tokens: int, response_schema: dict | None = None, text_model_path: str | None = None) -> str: ...
    def attribute_speakers(self, path: str, turns: list[TranscriptTurn]) -> list[TranscriptTurn]: ...
    def health(self) -> dict: ...

def get_adapter() -> InferenceAdapter:
    backend = os.getenv("CALL1_BACKEND", "mlx")
    if backend != "mlx":
        raise ValueError(f"Unsupported appliance adapter: {backend}")
    from .mlx import MLXAdapter
    return MLXAdapter()
