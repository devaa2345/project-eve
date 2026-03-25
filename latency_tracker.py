"""Concrete LatencyTracker implementation for the voice assistant pipeline.

Records per-stage timestamps and computes latency metrics across the five
pipeline stage boundaries: MIC_CAPTURE → STT_PARTIAL → LLM_FIRST_TOKEN →
TTS_FIRST_AUDIO → TTS_COMPLETE.
"""

from enum import Enum, auto
from typing import Optional

from exceptions import MissingStageError
from interfaces import LatencyTracker as LatencyTrackerABC


class Stage(Enum):
    """Pipeline stage markers in processing order."""
    MIC_CAPTURE = auto()
    STT_PARTIAL = auto()
    LLM_FIRST_TOKEN = auto()
    TTS_FIRST_AUDIO = auto()
    TTS_COMPLETE = auto()


# Ordered list used for iteration and report ordering
_STAGE_ORDER = [
    Stage.MIC_CAPTURE,
    Stage.STT_PARTIAL,
    Stage.LLM_FIRST_TOKEN,
    Stage.TTS_FIRST_AUDIO,
    Stage.TTS_COMPLETE,
]


class LatencyTracker(LatencyTrackerABC):
    """Records timestamps at pipeline stage boundaries and computes latency metrics.

    All timestamps are floats (seconds, e.g. from time.monotonic()).
    Computed metrics are returned in milliseconds.

    Correctness invariants:
    - mark() is first-write-wins: repeated calls for the same stage are ignored (Req 6.2)
    - report() raises MissingStageError if any stage is None (Req 8.2)
    - report_partial() never raises (Req 9.1, 9.2)
    - reset() clears all timestamps (Req 6.4)
    """

    def __init__(self) -> None:
        self._timestamps: dict[Stage, Optional[float]] = {s: None for s in _STAGE_ORDER}

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def mark(self, stage: Stage, timestamp: float) -> None:  # type: ignore[override]
        """Record timestamp for stage; first-write-wins (Req 6.2, 6.3)."""
        if self._timestamps[stage] is None:
            self._timestamps[stage] = timestamp

    def reset(self) -> None:
        """Clear all stage timestamps (Req 6.4)."""
        for stage in _STAGE_ORDER:
            self._timestamps[stage] = None

    def report(self) -> dict:
        """Return full latency report in milliseconds.

        Raises MissingStageError listing all missing stages if any timestamp is None (Req 8.2).
        """
        missing = [s.name for s in _STAGE_ORDER if self._timestamps[s] is None]
        if missing:
            raise MissingStageError(missing)
        return self._compute_metrics()

    def report_partial(self) -> dict:
        """Return partial latency report; uses None for any missing stage (Req 9.1, 9.2).

        Never raises.
        """
        return self._compute_metrics()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_metrics(self) -> dict:
        """Compute all five latency metrics; missing stages produce None values."""
        t = self._timestamps

        mic_capture_ms = self._delta_ms(t[Stage.MIC_CAPTURE], t[Stage.STT_PARTIAL])
        stt_to_llm_ms = self._delta_ms(t[Stage.STT_PARTIAL], t[Stage.LLM_FIRST_TOKEN])
        llm_to_tts_ms = self._delta_ms(t[Stage.LLM_FIRST_TOKEN], t[Stage.TTS_FIRST_AUDIO])
        tts_complete_ms = self._delta_ms(t[Stage.TTS_FIRST_AUDIO], t[Stage.TTS_COMPLETE])

        # time_to_first_audio_ms = sum of the first three deltas (Req 7.5)
        if mic_capture_ms is not None and stt_to_llm_ms is not None and llm_to_tts_ms is not None:
            time_to_first_audio_ms = mic_capture_ms + stt_to_llm_ms + llm_to_tts_ms
        else:
            time_to_first_audio_ms = None

        return {
            "mic_capture_ms": mic_capture_ms,
            "stt_to_llm_ms": stt_to_llm_ms,
            "llm_to_tts_ms": llm_to_tts_ms,
            "tts_complete_ms": tts_complete_ms,
            "time_to_first_audio_ms": time_to_first_audio_ms,
        }

    @staticmethod
    def _delta_ms(start: Optional[float], end: Optional[float]) -> Optional[float]:
        """Return (end - start) * 1000 if both are set, else None."""
        if start is None or end is None:
            return None
        return (end - start) * 1000.0
