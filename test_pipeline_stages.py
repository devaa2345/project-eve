"""Unit tests for pipeline stage components.

Covers:
- STTEngine yields no tokens on empty input (Req 2.4)
- LLMEngine only invokes model on is_final=True token (Req 3.1, 3.5)
- TTSEngine does not emit on_synthesis_complete on empty input (Req 4.5)
- Speaker calls mark(TTS_COMPLETE) after playback finishes (Req 5.4)
"""

import time
from typing import Iterator
from unittest.mock import MagicMock, call, patch

import pytest

from exceptions import LLMError, STTError, TTSError
from latency_tracker import LatencyTracker, Stage
from llm_engine import LLMEngine
from speaker import Speaker
from stt_engine import STTEngine
from tts_engine import TTSEngine
from type import AudioChunk, ResponseToken, TextToken


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunks(*data: bytes) -> Iterator[AudioChunk]:
    for d in data:
        yield AudioChunk(data=d)


def _text_tokens(*pairs: tuple[str, bool]) -> Iterator[TextToken]:
    for text, is_final in pairs:
        yield TextToken(text=text, is_final=is_final)


def _response_tokens(*texts: str) -> Iterator[ResponseToken]:
    for t in texts:
        yield ResponseToken(text=t)


# ---------------------------------------------------------------------------
# STTEngine tests
# ---------------------------------------------------------------------------

class TestSTTEngine:
    """Tests for STTEngine (Req 2.4, 12.1)."""

    def test_yields_no_tokens_on_empty_input(self):
        """STTEngine must yield nothing when audio_stream is empty (Req 2.4)."""
        engine = STTEngine(api_key="test-key")
        result = list(engine.stream_transcribe(iter([])))
        assert result == []

    def test_raises_stt_error_on_api_failure(self):
        """STTEngine must raise STTError when the API call fails (Req 12.1)."""
        engine = STTEngine(api_key="test-key")
        with patch.object(engine._client.chat.completions, "create", side_effect=RuntimeError("API down")):
            with pytest.raises(STTError):
                list(engine.stream_transcribe(_chunks(b"\x00" * 100)))

    def test_last_token_is_final(self):
        """The last emitted TextToken must have is_final=True."""
        engine = STTEngine(api_key="test-key")

        # Build a mock streaming response with two content chunks
        mock_chunk1 = MagicMock()
        mock_chunk1.choices[0].delta.content = "hello"
        mock_chunk2 = MagicMock()
        mock_chunk2.choices[0].delta.content = " world"

        with patch.object(engine._client.chat.completions, "create", return_value=iter([mock_chunk1, mock_chunk2])):
            tokens = list(engine.stream_transcribe(_chunks(b"\x00" * 100)))

        assert len(tokens) == 2
        assert tokens[0].is_final is False
        assert tokens[1].is_final is True


# ---------------------------------------------------------------------------
# LLMEngine tests
# ---------------------------------------------------------------------------

class TestLLMEngine:
    """Tests for LLMEngine (Req 3.1, 3.5, 12.2)."""

    def test_yields_no_tokens_on_empty_input(self):
        """LLMEngine must yield nothing when token_stream is empty (Req 3.5)."""
        engine = LLMEngine(api_key="test-key")
        result = list(engine.stream_complete(iter([])))
        assert result == []

    def test_does_not_invoke_model_on_non_final_tokens(self):
        """LLMEngine must NOT call the API until is_final=True is received (Req 3.1)."""
        engine = LLMEngine(api_key="test-key")
        with patch.object(engine._client.chat.completions, "create") as mock_create:
            # Consume the stream — no is_final token, so model should never be called
            list(engine.stream_complete(_text_tokens(("hello", False), ("world", False))))
            mock_create.assert_not_called()

    def test_invokes_model_on_is_final_token(self):
        """LLMEngine must invoke the model exactly once when is_final=True is received."""
        engine = LLMEngine(api_key="test-key")

        mock_response_chunk = MagicMock()
        mock_response_chunk.choices[0].delta.content = "response text"

        with patch.object(engine._client.chat.completions, "create", return_value=iter([mock_response_chunk])) as mock_create:
            tokens = list(engine.stream_complete(
                _text_tokens(("hello ", False), ("world", True))
            ))

        mock_create.assert_called_once()
        # Verify the full transcript was sent
        call_kwargs = mock_create.call_args
        messages = call_kwargs[1]["messages"] if call_kwargs[1] else call_kwargs[0][1]
        assert "hello world" in str(call_kwargs)

        assert len(tokens) == 1
        assert tokens[0].text == "response text"

    def test_raises_llm_error_on_api_failure(self):
        """LLMEngine must raise LLMError when the API call fails (Req 12.2)."""
        engine = LLMEngine(api_key="test-key")
        with patch.object(engine._client.chat.completions, "create", side_effect=RuntimeError("API down")):
            with pytest.raises(LLMError):
                list(engine.stream_complete(_text_tokens(("hello", True))))

    def test_accumulates_transcript_before_final(self):
        """LLMEngine must accumulate all tokens into the transcript before invoking model."""
        engine = LLMEngine(api_key="test-key")

        mock_chunk = MagicMock()
        mock_chunk.choices[0].delta.content = "ok"

        with patch.object(engine._client.chat.completions, "create", return_value=iter([mock_chunk])) as mock_create:
            list(engine.stream_complete(
                _text_tokens(("one ", False), ("two ", False), ("three", True))
            ))

        call_args = str(mock_create.call_args)
        assert "one two three" in call_args


# ---------------------------------------------------------------------------
# TTSEngine tests
# ---------------------------------------------------------------------------

class TestTTSEngine:
    """Tests for TTSEngine (Req 4.5, 12.3)."""

    def test_yields_nothing_on_empty_input(self):
        """TTSEngine must yield no audio chunks when token_stream is empty (Req 4.5)."""
        engine = TTSEngine(api_key="test-key")
        result = list(engine.stream_synthesize(iter([])))
        assert result == []

    def test_does_not_invoke_callback_on_empty_input(self):
        """TTSEngine must NOT invoke on_synthesis_complete when no tokens received (Req 4.5)."""
        engine = TTSEngine(api_key="test-key")
        callback = MagicMock()
        engine.on_synthesis_complete = callback

        list(engine.stream_synthesize(iter([])))

        callback.assert_not_called()

    def test_invokes_callback_after_synthesis(self):
        """TTSEngine must invoke on_synthesis_complete after all audio is yielded."""
        engine = TTSEngine(api_key="test-key")
        callback = MagicMock()
        engine.on_synthesis_complete = callback

        with patch.object(engine, "_synthesize_sentence", return_value=iter([AudioChunk(data=b"audio")])):
            list(engine.stream_synthesize(_response_tokens("Hello world.")))

        callback.assert_called_once()

    def test_raises_tts_error_on_api_failure(self):
        """TTSEngine must raise TTSError when synthesis fails (Req 12.3)."""
        engine = TTSEngine(api_key="test-key")
        with patch.object(engine, "_synthesize_sentence", side_effect=TTSError("fail")):
            with pytest.raises(TTSError):
                list(engine.stream_synthesize(_response_tokens("Hello.")))

    def test_flushes_remaining_buffer_after_stream_ends(self):
        """TTSEngine must synthesize any remaining text after the token stream ends."""
        engine = TTSEngine(api_key="test-key")
        synthesized: list[str] = []

        def fake_synthesize(sentence: str) -> Iterator[AudioChunk]:
            synthesized.append(sentence)
            yield AudioChunk(data=b"audio")

        with patch.object(engine, "_synthesize_sentence", side_effect=fake_synthesize):
            # "No boundary" — text without sentence-ending punctuation
            list(engine.stream_synthesize(_response_tokens("Hello there")))

        assert "Hello there" in synthesized


# ---------------------------------------------------------------------------
# Speaker tests
# ---------------------------------------------------------------------------

class TestSpeaker:
    """Tests for Speaker (Req 5.4)."""

    def test_marks_tts_complete_after_playback(self):
        """Speaker must call mark(TTS_COMPLETE) after all audio chunks are played (Req 5.4)."""
        import sys
        tracker = LatencyTracker()

        mock_pyaudio = MagicMock()
        mock_pa_instance = MagicMock()
        mock_stream = MagicMock()
        mock_pyaudio.PyAudio.return_value = mock_pa_instance
        mock_pyaudio.paInt16 = 8
        mock_pa_instance.open.return_value = mock_stream

        with patch.dict(sys.modules, {"pyaudio": mock_pyaudio}):
            spk = Speaker(latency_tracker=tracker)
            spk.play(iter([AudioChunk(data=b"\x00" * 100)]))

        assert tracker._timestamps[Stage.TTS_COMPLETE] is not None

    def test_marks_tts_complete_only_after_all_chunks_played(self):
        """TTS_COMPLETE must not be marked until the audio stream is exhausted."""
        import sys
        tracker = LatencyTracker()
        marks_during_play: list[bool] = []

        mock_pyaudio = MagicMock()
        mock_pa_instance = MagicMock()
        mock_stream = MagicMock()
        mock_pyaudio.PyAudio.return_value = mock_pa_instance
        mock_pyaudio.paInt16 = 8

        def write_side_effect(data):
            marks_during_play.append(tracker._timestamps[Stage.TTS_COMPLETE] is None)

        mock_stream.write.side_effect = write_side_effect
        mock_pa_instance.open.return_value = mock_stream

        with patch.dict(sys.modules, {"pyaudio": mock_pyaudio}):
            spk = Speaker(latency_tracker=tracker)
            spk.play(iter([AudioChunk(data=b"\x00" * 100), AudioChunk(data=b"\x01" * 100)]))

        # During playback, TTS_COMPLETE was not yet marked
        assert all(marks_during_play)
        # After playback, it is marked
        assert tracker._timestamps[Stage.TTS_COMPLETE] is not None

    def test_handle_synthesis_complete_sets_flag(self):
        """handle_synthesis_complete must set the internal synthesis complete flag."""
        tracker = LatencyTracker()
        spk = Speaker(latency_tracker=tracker)
        assert spk._synthesis_complete is False
        spk.handle_synthesis_complete()
        assert spk._synthesis_complete is True
