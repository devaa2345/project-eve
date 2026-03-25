"""TTSEngine using pyttsx3 (offline, no API key needed).

Buffers all ResponseTokens, then synthesizes to a WAV in memory,
yields it as AudioChunks. No network calls, no quota.

pip install pyttsx3

Req 4.1, 4.3, 4.4, 4.5, 11.4, 12.3
"""

import io
import os
import tempfile
import wave
from typing import Callable, Iterator, Optional

from exceptions import TTSError
from interfaces import TTSEngine as TTSEngineABC
from type import AudioChunk, ResponseToken

_CHUNK_SIZE = 4096


class TTSEngine(TTSEngineABC):
    def __init__(self, api_key: str | None = None) -> None:
        # api_key unused — pyttsx3 is offline
        self.on_synthesis_complete: Optional[Callable[[], None]] = None

    def stream_synthesize(self, token_stream: Iterator[ResponseToken]) -> Iterator[AudioChunk]:
        tokens = list(token_stream)
        if not tokens:
            return

        text = "".join(t.text for t in tokens).strip()
        if not text:
            return

        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty("rate", 175)   # words per minute
            engine.setProperty("volume", 1.0)

            # Save to a temp WAV file, then read it back as chunks
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            engine.save_to_file(text, tmp_path)
            engine.runAndWait()
            engine.stop()

            with open(tmp_path, "rb") as f:
                audio_data = f.read()

            os.unlink(tmp_path)

            # Yield in chunks
            for i in range(0, len(audio_data), _CHUNK_SIZE):
                yield AudioChunk(data=audio_data[i: i + _CHUNK_SIZE])

        except Exception as exc:
            raise TTSError(f"TTS synthesis failed: {exc}") from exc

        if self.on_synthesis_complete is not None:
            self.on_synthesis_complete()
