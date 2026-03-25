"""Pipeline orchestrator for the real-time voice assistant.

Wires MicCapture → STTEngine → LLMEngine → TTSEngine → Speaker in a
fully streaming chain. Manages latency tracking, timeout guard, error
handling, and interaction reset/interruption.

Requirements: 1.2, 2.2, 3.3, 4.2, 6.1, 10.1–10.4, 12.1–12.4, 13.1–13.4
"""

import itertools
import logging
import threading
import time
from typing import Iterator, Optional

from exceptions import LLMError, MicCaptureError, STTError, TTSError
from latency_tracker import LatencyTracker, Stage
from mic_capture import MicCapture
from stt_engine import STTEngine
from llm_engine import LLMEngine
from tts_engine import TTSEngine
from speaker import Speaker
from type import AudioChunk, ResponseToken, TextToken

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 30


class Pipeline:
    """End-to-end streaming voice assistant pipeline.

    Usage:
        pipeline = Pipeline(mic, stt, llm, tts, speaker, tracker)
        pipeline.start_interaction()   # blocks until TTS_COMPLETE or timeout
        report = tracker.report()
    """

    def __init__(
        self,
        mic: MicCapture,
        stt: STTEngine,
        llm: LLMEngine,
        tts: TTSEngine,
        speaker: Speaker,
        latency_tracker: LatencyTracker,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._mic = mic
        self._stt = stt
        self._llm = llm
        self._tts = tts
        self._speaker = speaker
        self._tracker = latency_tracker
        self._timeout_seconds = timeout_seconds

        # Cancellation flag — set to True to abort the active interaction
        self._cancel_event = threading.Event()
        self._cancel_event.set()  # start in "no active interaction" state
        # Timeout timer handle
        self._timeout_timer: Optional[threading.Timer] = None
        # Lock protecting interaction lifecycle transitions
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_interaction(self) -> None:
        """Begin a new voice interaction.

        If a previous interaction is still active, it is cancelled first
        (Req 13.1). Blocks until TTS_COMPLETE is recorded or the timeout
        fires.
        """
        with self._lock:
            # Cancel any in-flight interaction before starting a new one (Req 13.1)
            self._cancel_interaction_locked()

            # Reset cancellation flag for the new interaction
            self._cancel_event.clear()

            # Register the synthesis-complete callback on TTS (Req 5.3)
            self._tts.on_synthesis_complete = self._speaker.handle_synthesis_complete

            # Arm the timeout guard (Req 10.1, 10.2)
            self._timeout_timer = threading.Timer(
                self._timeout_seconds, self._on_timeout
            )
            self._timeout_timer.daemon = True
            self._timeout_timer.start()

        try:
            self._run_pipeline()
        except MicCaptureError:
            # MicCaptureError: surface to caller and halt — do NOT cancel-and-continue (Req 1.3)
            self._disarm_timeout()
            raise
        except (STTError, LLMError, TTSError) as exc:
            # Stage errors: log, cancel, release resources (Req 12.1–12.3)
            logger.error("Pipeline stage error: %s", exc)
            self.cancel_interaction()
            raise
        except Exception as exc:
            logger.error("Unexpected pipeline error: %s", exc)
            self.cancel_interaction()
            raise
        else:
            self._disarm_timeout()

    def cancel_interaction(self) -> None:
        """Cancel the active interaction: reportPartial() then reset() (Req 13.3).

        Safe to call from any thread. Always calls reportPartial()+reset()
        regardless of whether an interaction is in flight.
        """
        with self._lock:
            self._disarm_timeout()
            self._cancel_event.set()
            self._tracker.report_partial()
            self._tracker.reset()
            self._tts.on_synthesis_complete = None

    # ------------------------------------------------------------------
    # Internal pipeline execution
    # ------------------------------------------------------------------

    def _run_pipeline(self) -> None:
        """Execute the full MicCapture → STT → LLM → TTS → Speaker chain.

        Each stage boundary uses an eager peek (next()) to force execution of
        the upstream generator and record the latency timestamp before passing
        the reconstituted stream downstream. This ensures all marks are recorded
        even when a downstream stage (e.g. Speaker mock) doesn't iterate lazily.
        """

        # --- MicCapture → STT ---
        # Eagerly pull the first audio chunk so MIC_CAPTURE is marked immediately.
        raw_audio = self._mic.start()
        try:
            first_audio = next(iter(raw_audio))
        except StopIteration:
            return  # no audio — nothing to do
        self._tracker.mark(Stage.MIC_CAPTURE, time.monotonic())
        audio_stream = itertools.chain([first_audio], raw_audio)

        # --- STT → LLM ---
        # Pass the audio stream to STT; eagerly pull the first TextToken so
        # STT_PARTIAL is marked before handing off to LLM.
        raw_tokens = self._stt.stream_transcribe(audio_stream)
        try:
            first_token = next(iter(raw_tokens))
        except StopIteration:
            return  # no transcription output
        self._tracker.mark(Stage.STT_PARTIAL, time.monotonic())
        text_token_stream = self._checked_token_stream(
            itertools.chain([first_token], raw_tokens)
        )

        # --- LLM → TTS ---
        raw_response = self._llm.stream_complete(text_token_stream)
        try:
            first_response = next(iter(raw_response))
        except StopIteration:
            return  # no LLM output
        self._tracker.mark(Stage.LLM_FIRST_TOKEN, time.monotonic())
        response_token_stream = itertools.chain([first_response], raw_response)

        # --- TTS → Speaker ---
        raw_tts_audio = self._tts.stream_synthesize(response_token_stream)
        try:
            first_tts_chunk = next(iter(raw_tts_audio))
        except StopIteration:
            return  # no TTS output
        self._tracker.mark(Stage.TTS_FIRST_AUDIO, time.monotonic())
        tts_audio_stream = self._cancellable_stream(
            itertools.chain([first_tts_chunk], raw_tts_audio)
        )

        # --- Speaker ---
        self._speaker.play(tts_audio_stream)

    def _checked_token_stream(
        self, stream: Iterator[TextToken]
    ) -> Iterator[TextToken]:
        """Yield tokens, stopping early on cancellation."""
        for token in stream:
            yield token
            if self._cancel_event.is_set():
                return

    def _cancellable_stream(
        self, stream: Iterator[AudioChunk]
    ) -> Iterator[AudioChunk]:
        """Yield audio chunks, stopping early on cancellation."""
        for chunk in stream:
            yield chunk
            if self._cancel_event.is_set():
                return

    # ------------------------------------------------------------------
    # Timeout handler
    # ------------------------------------------------------------------

    def _on_timeout(self) -> None:
        """Fired when TTS_COMPLETE has not been recorded within timeout_seconds.

        Calls reportPartial() (NOT report()) then resets (Req 10.3, 10.4).
        """
        logger.warning(
            "Pipeline interaction timed out after %ss", self._timeout_seconds
        )
        self._cancel_event.set()
        self._mic.stop()
        # Use cancel_interaction() which always calls reportPartial()+reset()
        self.cancel_interaction()

    def _disarm_timeout(self) -> None:
        """Cancel the pending timeout timer if it hasn't fired yet."""
        if self._timeout_timer is not None:
            self._timeout_timer.cancel()
            self._timeout_timer = None

    # ------------------------------------------------------------------
    # Cancellation (must be called with self._lock held)
    # ------------------------------------------------------------------

    def _cancel_interaction_locked(self) -> None:
        """Core cancellation logic — must be called with self._lock held.

        Ordering: reportPartial() THEN reset() (Req 13.3).
        Deregisters on_synthesis_complete to prevent stale callbacks (Req 13.4).

        Only calls reportPartial()/reset() if an interaction was actually in
        progress (i.e. the cancel_event was not already set from a prior reset).
        """
        self._disarm_timeout()

        # Only report+reset if there was an active interaction to cancel.
        # _cancel_event is cleared at the start of each new interaction, so
        # if it is already set here it means no interaction is in flight.
        interaction_was_active = not self._cancel_event.is_set()
        self._cancel_event.set()

        if interaction_was_active:
            # reportPartial() before reset() — mandatory ordering (Req 13.3)
            self._tracker.report_partial()
            self._tracker.reset()

        # Deregister callback so a late-firing TTSEngine from the cancelled
        # interaction cannot mark TTS_COMPLETE on the next interaction (Req 13.4)
        self._tts.on_synthesis_complete = None
