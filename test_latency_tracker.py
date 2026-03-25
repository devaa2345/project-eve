"""Tests for LatencyTracker: property-based tests (hypothesis) and unit tests.

Property tests:
  - Property 1: mark() first-write-wins idempotency (Req 6.2)
  - Property 2: monotonic timestamps produce non-negative deltas (Req 7.1–7.4)
  - Property 3: time_to_first_audio_ms == sum of first three deltas (Req 7.5)
  - Property 4: report() raises MissingStageError iff any stage is None (Req 8.2)

Unit tests:
  - reset() clears all fields (Req 6.4)
  - report_partial() never raises (Req 9.1, 9.2)
  - report() returns correct values when all stages present (Req 8.1)
  - MissingStageError message identifies specific missing stages (Req 8.2)
"""

import pytest
from hypothesis import given, assume, settings
from hypothesis import strategies as st

from latency_tracker import LatencyTracker, Stage, _STAGE_ORDER
from exceptions import MissingStageError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tracker() -> LatencyTracker:
    return LatencyTracker()


def _fill_all(tracker: LatencyTracker, timestamps: list[float]) -> None:
    """Mark all five stages with the provided timestamps (must be length 5)."""
    for stage, ts in zip(_STAGE_ORDER, timestamps):
        tracker.mark(stage, ts)


# ---------------------------------------------------------------------------
# Property 1: mark() first-write-wins idempotency (Req 6.2)
# ---------------------------------------------------------------------------

@given(
    stage=st.sampled_from(list(Stage)),
    first_ts=st.floats(min_value=0.0, max_value=1e9, allow_nan=False, allow_infinity=False),
    second_ts=st.floats(min_value=0.0, max_value=1e9, allow_nan=False, allow_infinity=False),
)
def test_mark_idempotency(stage, first_ts, second_ts):
    """Calling mark() a second time for the same stage must not overwrite the first value."""
    tracker = _make_tracker()
    tracker.mark(stage, first_ts)
    tracker.mark(stage, second_ts)
    assert tracker._timestamps[stage] == first_ts


# ---------------------------------------------------------------------------
# Property 2: monotonic timestamps → non-negative deltas (Req 7.1–7.4)
# ---------------------------------------------------------------------------

@given(
    offsets=st.lists(
        st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        min_size=5,
        max_size=5,
    )
)
def test_monotonic_timestamps_produce_nonnegative_deltas(offsets):
    """For any monotonically increasing set of timestamps, all computed deltas are >= 0."""
    # Build strictly non-decreasing timestamps from cumulative offsets
    timestamps = []
    t = 0.0
    for offset in offsets:
        t += abs(offset)
        timestamps.append(t)

    tracker = _make_tracker()
    _fill_all(tracker, timestamps)
    result = tracker.report()

    assert result["mic_capture_ms"] >= 0
    assert result["stt_to_llm_ms"] >= 0
    assert result["llm_to_tts_ms"] >= 0
    assert result["tts_complete_ms"] >= 0


# ---------------------------------------------------------------------------
# Property 3: time_to_first_audio_ms == sum of first three deltas (Req 7.5)
# ---------------------------------------------------------------------------

@given(
    offsets=st.lists(
        st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        min_size=5,
        max_size=5,
    )
)
def test_time_to_first_audio_ms_sum_invariant(offsets):
    """time_to_first_audio_ms must always equal mic_capture_ms + stt_to_llm_ms + llm_to_tts_ms."""
    timestamps = []
    t = 0.0
    for offset in offsets:
        t += abs(offset)
        timestamps.append(t)

    tracker = _make_tracker()
    _fill_all(tracker, timestamps)
    result = tracker.report()

    expected = result["mic_capture_ms"] + result["stt_to_llm_ms"] + result["llm_to_tts_ms"]
    assert abs(result["time_to_first_audio_ms"] - expected) < 1e-9


# ---------------------------------------------------------------------------
# Property 4: report() raises MissingStageError iff any stage is None (Req 8.2)
# ---------------------------------------------------------------------------

@given(
    present=st.frozensets(st.sampled_from(list(Stage))),
    offsets=st.lists(
        st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        min_size=5,
        max_size=5,
    ),
)
def test_report_raises_iff_any_stage_missing(present, offsets):
    """report() raises MissingStageError exactly when at least one stage timestamp is None."""
    timestamps = []
    t = 0.0
    for offset in offsets:
        t += abs(offset)
        timestamps.append(t)

    tracker = _make_tracker()
    for stage, ts in zip(_STAGE_ORDER, timestamps):
        if stage in present:
            tracker.mark(stage, ts)

    all_present = len(present) == len(_STAGE_ORDER)

    if all_present:
        # Should not raise
        result = tracker.report()
        assert isinstance(result, dict)
    else:
        with pytest.raises(MissingStageError):
            tracker.report()


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_all_timestamps(self):
        """reset() must return tracker to fully-None state (Req 6.4)."""
        tracker = _make_tracker()
        _fill_all(tracker, [0.0, 1.0, 2.0, 3.0, 4.0])
        tracker.reset()
        for stage in _STAGE_ORDER:
            assert tracker._timestamps[stage] is None

    def test_reset_allows_new_marks(self):
        """After reset(), mark() should record fresh timestamps."""
        tracker = _make_tracker()
        _fill_all(tracker, [0.0, 1.0, 2.0, 3.0, 4.0])
        tracker.reset()
        tracker.mark(Stage.MIC_CAPTURE, 99.0)
        assert tracker._timestamps[Stage.MIC_CAPTURE] == 99.0


class TestReportPartial:
    def test_report_partial_never_raises_on_empty(self):
        """report_partial() must not raise when no stages are set (Req 9.2)."""
        tracker = _make_tracker()
        result = tracker.report_partial()
        assert result["mic_capture_ms"] is None
        assert result["stt_to_llm_ms"] is None
        assert result["llm_to_tts_ms"] is None
        assert result["tts_complete_ms"] is None
        assert result["time_to_first_audio_ms"] is None

    def test_report_partial_never_raises_on_partial(self):
        """report_partial() must not raise when only some stages are set (Req 9.1)."""
        tracker = _make_tracker()
        tracker.mark(Stage.MIC_CAPTURE, 0.0)
        tracker.mark(Stage.STT_PARTIAL, 0.1)
        result = tracker.report_partial()
        assert result["mic_capture_ms"] == pytest.approx(100.0)
        assert result["stt_to_llm_ms"] is None
        assert result["time_to_first_audio_ms"] is None

    def test_report_partial_never_raises_on_full(self):
        """report_partial() must not raise when all stages are set."""
        tracker = _make_tracker()
        _fill_all(tracker, [0.0, 0.1, 0.2, 0.3, 0.4])
        result = tracker.report_partial()
        assert result["mic_capture_ms"] is not None


class TestReport:
    def test_report_returns_correct_values(self):
        """report() must return accurate millisecond metrics when all stages present (Req 8.1)."""
        tracker = _make_tracker()
        _fill_all(tracker, [0.0, 0.1, 0.3, 0.6, 1.0])
        result = tracker.report()

        assert result["mic_capture_ms"] == pytest.approx(100.0)   # 0.1 - 0.0
        assert result["stt_to_llm_ms"] == pytest.approx(200.0)    # 0.3 - 0.1
        assert result["llm_to_tts_ms"] == pytest.approx(300.0)    # 0.6 - 0.3
        assert result["tts_complete_ms"] == pytest.approx(400.0)  # 1.0 - 0.6
        assert result["time_to_first_audio_ms"] == pytest.approx(600.0)  # 100+200+300

    def test_report_raises_missing_stage_error_on_incomplete(self):
        """report() must raise MissingStageError when any stage is missing (Req 8.2)."""
        tracker = _make_tracker()
        tracker.mark(Stage.MIC_CAPTURE, 0.0)
        with pytest.raises(MissingStageError):
            tracker.report()

    def test_missing_stage_error_identifies_missing_stages(self):
        """MissingStageError must list the specific missing stage names (Req 8.2)."""
        tracker = _make_tracker()
        tracker.mark(Stage.MIC_CAPTURE, 0.0)
        tracker.mark(Stage.STT_PARTIAL, 0.1)
        # LLM_FIRST_TOKEN, TTS_FIRST_AUDIO, TTS_COMPLETE are missing

        with pytest.raises(MissingStageError) as exc_info:
            tracker.report()

        missing = exc_info.value.missing_stages
        assert "LLM_FIRST_TOKEN" in missing
        assert "TTS_FIRST_AUDIO" in missing
        assert "TTS_COMPLETE" in missing
        assert "MIC_CAPTURE" not in missing
        assert "STT_PARTIAL" not in missing

    def test_missing_stage_error_message_contains_stage_names(self):
        """MissingStageError message string must include the missing stage names."""
        tracker = _make_tracker()
        with pytest.raises(MissingStageError) as exc_info:
            tracker.report()
        msg = str(exc_info.value)
        for stage in _STAGE_ORDER:
            assert stage.name in msg
