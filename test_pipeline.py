"""Tests for the Pipeline orchestrator.

Covers:
- 5.5: Timeout safety — cancel_interaction() called, report() never called (Req 10.2–10.4)
- 5.6: reportPartial-before-reset ordering invariant (Req 13.3, 13.4)
- 5.7: Integration tests — happy path, interruption, error propagation (Req 1.1, 3.2, 5.4, 12.4, 13.1)
"""

import threading
import time
from typing import Iterator
from unittest.mock import MagicMock, call, patch, PropertyMock

import pytest

from exceptions import LLMError, MicCaptureError, STTError, TTSError
from latency_tracker import LatencyTracker, Stage
from pipeline import Pipeline
from type import AudioChunk, ResponseToken, TextToken


# ---------------------------------------------------------------------------
# Test helpers / factories
# ---------------------------------------------------------------------------

def _make_pipeline(
    mic=None, stt=None, llm=None, tts=None, speaker=None, tracker=None,
    timeout_seconds=30,
) -> Pipeline:
    """Build a Pipeline with sensible MagicMock defaults for any omitted stage."""
    mic = mic or MagicMock()
    stt = stt or MagicMock()
    llm = llm or MagicMock()
    tts = tts or MagicMock()
    speaker = speaker or MagicMock()
    tracker = tracker or MagicMock()
    return Pipeline(mic, stt, llm, tts, speaker, tracker, timeout_seconds=timeout_seconds)


def _audio_chunks(*data: bytes) -> list[AudioChunk]:
    return [AudioChunk(data=d) for d in data]


def _text_tokens(*pairs: tuple[str, bool]) -> list[TextToken]:
    return [TextToken(text=t, is_final=f) for t, f in pairs]


def _response_tokens(*texts: str) -> list[ResponseToken]:
    return [ResponseToken(text=t) for t in texts]


# ---------------------------------------------------------------------------
# 5.5 — Timeout safety
# ---------------------------------------------------------------------------

class TestTimeoutSafety:
    """Validates Req 10.2, 10.3, 10.4."""

    def test_timeout_calls_cancel_not_report(self):
        """When timeout fires, cancel_interaction() is called; report() is never called."""
        tracker = MagicMock(spec=LatencyTracker)
        pipeline = _make_pipeline(tracker=tracker, timeout_seconds=0.05)

        stall_done = threading.Event()

        # Mic yields one chunk so MIC_CAPTURE is marked, then STT stalls
        pipeline._mic.start.return_value = iter([AudioChunk(data=b"\x00")])

        def stalling_stt(audio_stream):
            list(audio_stream)  # consume mic stream
            stall_done.wait(timeout=2.0)  # stall here until timeout fires
            return iter([])

        pipeline._stt.stream_transcribe.side_effect = stalling_stt
        pipeline._llm.stream_complete.return_value = iter([])
        pipeline._tts.stream_synthesize.return_value = iter([])

        # Patch cancel_interaction to unblock the stall
        original_cancel = pipeline.cancel_interaction
        def cancel_and_unblock():
            stall_done.set()
            original_cancel()
        pipeline.cancel_interaction = cancel_and_unblock

        pipeline.start_interaction()

        tracker.report.assert_not_called()
        tracker.report_partial.assert_called()
        tracker.reset.assert_called()

    def test_timeout_respects_custom_duration(self):
        """Timeout fires at the configured duration, not always at 30s."""
        tracker = MagicMock(spec=LatencyTracker)
        pipeline = _make_pipeline(tracker=tracker, timeout_seconds=0.05)

        stall_done = threading.Event()
        pipeline._mic.start.return_value = iter([AudioChunk(data=b"\x00")])

        def stalling_stt(audio_stream):
            list(audio_stream)
            stall_done.wait(timeout=2.0)
            return iter([])

        pipeline._stt.stream_transcribe.side_effect = stalling_stt

        original_cancel = pipeline.cancel_interaction
        def cancel_and_unblock():
            stall_done.set()
            original_cancel()
        pipeline.cancel_interaction = cancel_and_unblock

        start = time.monotonic()
        pipeline.start_interaction()
        elapsed = time.monotonic() - start

        assert elapsed < 1.0, f"Timeout took {elapsed:.2f}s — custom duration not honoured"

    def test_report_partial_called_before_reset_on_timeout(self):
        """reportPartial() must be called before reset() on the timeout path (Req 13.3)."""
        call_order: list[str] = []
        tracker = MagicMock(spec=LatencyTracker)
        tracker.report_partial.side_effect = lambda: call_order.append("report_partial")
        tracker.reset.side_effect = lambda: call_order.append("reset")

        pipeline = _make_pipeline(tracker=tracker, timeout_seconds=0.05)
        stall_done = threading.Event()
        pipeline._mic.start.return_value = iter([AudioChunk(data=b"\x00")])

        def stalling_stt(audio_stream):
            list(audio_stream)
            stall_done.wait(timeout=2.0)
            return iter([])

        pipeline._stt.stream_transcribe.side_effect = stalling_stt

        original_cancel = pipeline.cancel_interaction
        def cancel_and_unblock():
            stall_done.set()
            original_cancel()
        pipeline.cancel_interaction = cancel_and_unblock

        pipeline.start_interaction()

        assert "report_partial" in call_order
        assert "reset" in call_order
        assert call_order.index("report_partial") < call_order.index("reset")


# ---------------------------------------------------------------------------
# 5.6 — reportPartial-before-reset ordering invariant
# ---------------------------------------------------------------------------

class TestReportPartialBeforeReset:
    """Validates Req 13.3, 13.4."""

    def test_cancel_interaction_ordering(self):
        """cancel_interaction() must call reportPartial() before reset()."""
        call_order: list[str] = []
        tracker = MagicMock(spec=LatencyTracker)
        tracker.report_partial.side_effect = lambda: call_order.append("report_partial")
        tracker.reset.side_effect = lambda: call_order.append("reset")

        pipeline = _make_pipeline(tracker=tracker)
        pipeline.cancel_interaction()

        assert call_order == ["report_partial", "reset"], (
            f"Expected ['report_partial', 'reset'], got {call_order}"
        )

    def test_tracker_reset_before_new_interaction_marks(self):
        """After cancellation, tracker must be in reset state before new marks (Req 13.4)."""
        tracker = LatencyTracker()

        # Seed a timestamp to simulate a mid-flight interaction
        tracker.mark(Stage.MIC_CAPTURE, 1.0)
        tracker.mark(Stage.STT_PARTIAL, 2.0)

        pipeline = _make_pipeline(tracker=tracker)
        pipeline.cancel_interaction()

        # After cancel, all timestamps must be None
        for stage in Stage:
            assert tracker._timestamps[stage] is None, (
                f"Stage {stage} not cleared after cancel_interaction()"
            )

    def test_stale_callback_deregistered_after_cancel(self):
        """on_synthesis_complete must be None after cancellation (Req 13.4)."""
        tts = MagicMock()
        pipeline = _make_pipeline(tts=tts)

        # Simulate a registered callback
        tts.on_synthesis_complete = MagicMock()
        pipeline.cancel_interaction()

        assert tts.on_synthesis_complete is None, (
            "on_synthesis_complete must be deregistered after cancel to prevent stale callbacks"
        )


# ---------------------------------------------------------------------------
# 5.7 — Integration tests
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    """Integration tests for the full pipeline (Req 1.1, 3.2, 5.4, 12.4, 13.1)."""

    def test_happy_path_all_stage_markers_recorded(self):
        """Full happy-path: all five stage markers recorded, report() returns complete metrics."""
        tracker = LatencyTracker()

        mic = MagicMock()
        stt = MagicMock()
        llm = MagicMock()
        tts = MagicMock()
        speaker = MagicMock()

        mic.start.return_value = iter([AudioChunk(data=b"\x00")])
        stt.stream_transcribe.return_value = iter([TextToken(text="hello", is_final=True)])
        llm.stream_complete.return_value = iter([ResponseToken(text="hi")])
        tts.stream_synthesize.return_value = iter([AudioChunk(data=b"\x01")])

        # Speaker.play() must mark TTS_COMPLETE — simulate it
        def fake_play(audio_stream):
            list(audio_stream)  # drain the stream
            tracker.mark(Stage.TTS_COMPLETE, time.monotonic())

        speaker.play.side_effect = fake_play

        pipeline = Pipeline(mic, stt, llm, tts, speaker, tracker)
        pipeline.start_interaction()

        # All five markers must be set
        for stage in Stage:
            assert tracker._timestamps[stage] is not None, f"{stage} not recorded"

        # report() must succeed with all metrics
        report = tracker.report()
        assert report["mic_capture_ms"] is not None
        assert report["stt_to_llm_ms"] is not None
        assert report["llm_to_tts_ms"] is not None
        assert report["tts_complete_ms"] is not None
        assert report["time_to_first_audio_ms"] is not None

    def test_interruption_cancels_previous_and_resets_tracker(self):
        """New interaction cancels previous; tracker resets cleanly (Req 13.1)."""
        tracker = LatencyTracker()
        tracker.mark(Stage.MIC_CAPTURE, 1.0)  # simulate in-flight interaction

        pipeline = _make_pipeline(tracker=tracker)
        # Simulate an active interaction by clearing the cancel event
        pipeline._cancel_event.clear()

        # Starting a new interaction should cancel the previous one
        pipeline._mic.start.return_value = iter([])
        pipeline._stt.stream_transcribe.return_value = iter([])
        pipeline._llm.stream_complete.return_value = iter([])
        pipeline._tts.stream_synthesize.return_value = iter([])

        pipeline.start_interaction()

        # Tracker must have been reset (previous MIC_CAPTURE cleared)
        for stage in Stage:
            assert tracker._timestamps[stage] is None, (
                f"Stage {stage} should be None after reset"
            )

    def test_stt_error_triggers_cancel_and_cleanup(self):
        """STTError causes cancel_interaction() and stream cleanup (Req 12.1, 12.4)."""
        call_order: list[str] = []
        tracker = MagicMock(spec=LatencyTracker)
        tracker.report_partial.side_effect = lambda: call_order.append("report_partial")
        tracker.reset.side_effect = lambda: call_order.append("reset")

        mic = MagicMock()
        stt = MagicMock()

        mic.start.return_value = iter([AudioChunk(data=b"\x00")])
        stt.stream_transcribe.side_effect = STTError("transcription failed")

        pipeline = _make_pipeline(mic=mic, stt=stt, tracker=tracker)

        with pytest.raises(STTError):
            pipeline.start_interaction()

        assert "report_partial" in call_order
        assert "reset" in call_order
        assert call_order.index("report_partial") < call_order.index("reset")

    def test_llm_error_triggers_cancel_and_cleanup(self):
        """LLMError causes cancel_interaction() (Req 12.2, 12.4)."""
        tracker = MagicMock(spec=LatencyTracker)
        mic = MagicMock()
        stt = MagicMock()
        llm = MagicMock()

        mic.start.return_value = iter([AudioChunk(data=b"\x00")])
        stt.stream_transcribe.return_value = iter([TextToken(text="hi", is_final=True)])
        llm.stream_complete.side_effect = LLMError("generation failed")

        pipeline = _make_pipeline(mic=mic, stt=stt, llm=llm, tracker=tracker)

        with pytest.raises(LLMError):
            pipeline.start_interaction()

        tracker.report_partial.assert_called()
        tracker.reset.assert_called()

    def test_tts_error_triggers_cancel_and_cleanup(self):
        """TTSError causes cancel_interaction() (Req 12.3, 12.4)."""
        tracker = MagicMock(spec=LatencyTracker)
        mic = MagicMock()
        stt = MagicMock()
        llm = MagicMock()
        tts = MagicMock()

        mic.start.return_value = iter([AudioChunk(data=b"\x00")])
        stt.stream_transcribe.return_value = iter([TextToken(text="hi", is_final=True)])
        llm.stream_complete.return_value = iter([ResponseToken(text="hello")])
        tts.stream_synthesize.side_effect = TTSError("synthesis failed")

        pipeline = _make_pipeline(mic=mic, stt=stt, llm=llm, tts=tts, tracker=tracker)

        with pytest.raises(TTSError):
            pipeline.start_interaction()

        tracker.report_partial.assert_called()
        tracker.reset.assert_called()

    def test_mic_capture_error_halts_without_cancel(self):
        """MicCaptureError surfaces to caller and halts — does NOT call cancel (Req 1.3)."""
        tracker = MagicMock(spec=LatencyTracker)
        mic = MagicMock()
        mic.start.side_effect = MicCaptureError("no mic")

        pipeline = _make_pipeline(mic=mic, tracker=tracker)

        with pytest.raises(MicCaptureError):
            pipeline.start_interaction()

        # MicCaptureError is a halt — reportPartial/reset are NOT called via cancel path
        # (the error is re-raised before cancel_interaction() is invoked)
        tracker.report_partial.assert_not_called()

    def test_tts_first_audio_marked_via_peek_not_lost(self):
        """TTS_FIRST_AUDIO is marked via peek; the first chunk is not dropped (Req 4.2)."""
        tracker = LatencyTracker()
        mic = MagicMock()
        stt = MagicMock()
        llm = MagicMock()
        tts = MagicMock()
        speaker = MagicMock()

        chunks = [AudioChunk(data=b"\xAA"), AudioChunk(data=b"\xBB")]
        mic.start.return_value = iter([AudioChunk(data=b"\x00")])
        stt.stream_transcribe.return_value = iter([TextToken(text="hi", is_final=True)])
        llm.stream_complete.return_value = iter([ResponseToken(text="hello")])
        tts.stream_synthesize.return_value = iter(chunks)

        played_chunks: list[AudioChunk] = []

        def fake_play(audio_stream):
            for c in audio_stream:
                played_chunks.append(c)
            tracker.mark(Stage.TTS_COMPLETE, time.monotonic())

        speaker.play.side_effect = fake_play

        pipeline = Pipeline(mic, stt, llm, tts, speaker, tracker)
        pipeline.start_interaction()

        # TTS_FIRST_AUDIO must be marked
        assert tracker._timestamps[Stage.TTS_FIRST_AUDIO] is not None
        # Both chunks must have been played — first chunk not dropped by peek
        assert len(played_chunks) == 2
        assert played_chunks[0].data == b"\xAA"
        assert played_chunks[1].data == b"\xBB"
