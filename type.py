"""Core data types for the real-time voice assistant pipeline.

TextToken and ResponseToken are intentionally distinct types to enforce
type safety at stage boundaries (Req 11.1).
"""

from dataclasses import dataclass


@dataclass
class AudioChunk:
    """Raw audio data produced by MicCapture or TTSEngine."""
    data: bytes


@dataclass
class TextToken:
    """Transcription token emitted by STTEngine.

    Distinct from ResponseToken — not interchangeable (Req 11.1, 11.2).
    """
    text: str
    is_final: bool


@dataclass
class ResponseToken:
    """Response token emitted by LLMEngine.

    Distinct from TextToken — not interchangeable (Req 11.1, 11.3).
    """
    text: str
