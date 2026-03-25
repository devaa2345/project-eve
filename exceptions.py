"""Typed exception classes for the real-time voice assistant pipeline.

All stage components and the Pipeline import exceptions from here.
"""

from typing import Sequence


class MissingStageError(Exception):
    """Raised by LatencyTracker.report() when one or more stage timestamps are None.

    The missing stage names are available via the `missing_stages` attribute.
    """

    def __init__(self, missing_stages: Sequence[str]) -> None:
        self.missing_stages = list(missing_stages)
        stages_str = ", ".join(missing_stages)
        super().__init__(f"Missing stage timestamps: {stages_str}")


class MicCaptureError(Exception):
    """Raised when MicCapture fails to produce audio chunks (Req 1.3)."""


class STTError(Exception):
    """Raised when STTEngine encounters a transcription failure (Req 12.1)."""


class LLMError(Exception):
    """Raised when LLMEngine encounters a generation failure (Req 12.2)."""


class TTSError(Exception):
    """Raised when TTSEngine encounters a synthesis failure (Req 12.3)."""
