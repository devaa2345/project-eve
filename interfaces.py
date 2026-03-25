"""Abstract base classes for all pipeline stage components.

Defines the streaming method signatures that concrete implementations must satisfy.
"""

from abc import ABC, abstractmethod
from typing import Callable, Iterator, Optional

from type import AudioChunk, TextToken, ResponseToken


class MicCapture(ABC):
    """Streams raw audio from the microphone in real-time (Req 1.1)."""

    @abstractmethod
    def start(self) -> Iterator[AudioChunk]:
        """Begin capturing audio and yield chunks as they arrive."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop audio capture and release hardware resources."""
        ...


class STTEngine(ABC):
    """Transcribes a streaming audio input into TextTokens (Req 2.1, 11.2)."""

    @abstractmethod
    def stream_transcribe(
        self, audio_stream: Iterator[AudioChunk]
    ) -> Iterator[TextToken]:
        """Consume audio chunks and yield TextTokens incrementally.

        Sets is_final=True on the last token of an utterance.
        Yields nothing if audio_stream is empty (Req 2.4).
        """
        ...


class LLMEngine(ABC):
    """Generates response tokens from a stream of TextTokens (Req 3.1, 11.3)."""

    @abstractmethod
    def stream_complete(
        self, token_stream: Iterator[TextToken]
    ) -> Iterator[ResponseToken]:
        """Accumulate TextTokens; invoke model on is_final=True and yield ResponseTokens.

        Yields nothing if token_stream is empty (Req 3.5).
        """
        ...


class TTSEngine(ABC):
    """Synthesizes audio from a stream of ResponseTokens (Req 4.1, 11.4)."""

    on_synthesis_complete: Optional[Callable[[], None]] = None

    @abstractmethod
    def stream_synthesize(
        self, token_stream: Iterator[ResponseToken]
    ) -> Iterator[AudioChunk]:
        """Consume ResponseTokens and yield AudioChunks incrementally.

        After yielding the last chunk, invokes on_synthesis_complete if set (Req 4.3).
        Yields nothing and does not invoke the callback if token_stream is empty (Req 4.5).
        """
        ...


class LatencyTracker(ABC):
    """Records per-stage timestamps and computes latency metrics."""

    @abstractmethod
    def mark(self, stage: str, timestamp: float) -> None:
        """Record a timestamp for the given stage (first-write-wins, Req 6.2)."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear all stage timestamps (Req 6.4)."""
        ...

    @abstractmethod
    def report(self) -> dict:
        """Return full latency report; raises MissingStageError if any stage is None (Req 8.2)."""
        ...

    @abstractmethod
    def report_partial(self) -> dict:
        """Return partial latency report; never raises (Req 9.1, 9.2)."""
        ...


class Speaker(ABC):
    """Plays audio chunks in real-time (Req 5.1)."""

    def __init__(self, latency_tracker: LatencyTracker) -> None:
        self.latency_tracker = latency_tracker

    @abstractmethod
    def play(self, audio_stream: Iterator[AudioChunk]) -> None:
        """Play audio chunks as they arrive without buffering the full stream (Req 5.1, 5.2)."""
        ...

    @abstractmethod
    def handle_synthesis_complete(self) -> None:
        """Callback registered on TTSEngine; signals Speaker to finalize playback (Req 5.3)."""
        ...
