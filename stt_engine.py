"""STTEngine using Groq Whisper API (whisper-large-v3).

Free tier, pure REST, no compilation needed.
Get a free key at https://console.groq.com

Req 2.1, 2.3, 2.4, 11.2, 12.1
"""

import io
import os
import wave
from typing import Iterator

from groq import Groq

from exceptions import STTError
from interfaces import STTEngine as STTEngineABC
from type import AudioChunk, TextToken

_MODEL        = "whisper-large-v3"
_SAMPLE_RATE  = 16000
_CHANNELS     = 1
_SAMPLE_WIDTH = 2  # 16-bit PCM


def _to_wav(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(_CHANNELS)
        wf.setsampwidth(_SAMPLE_WIDTH)
        wf.setframerate(_SAMPLE_RATE)
        wf.writeframes(pcm)
    buf.seek(0)
    return buf.read()


class STTEngine(STTEngineABC):
    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.environ.get("GROQ_API_KEY", "")
        self._client = Groq(api_key=key)

    def stream_transcribe(self, audio_stream: Iterator[AudioChunk]) -> Iterator[TextToken]:
        pcm_chunks = [chunk.data for chunk in audio_stream]
        if not pcm_chunks:
            return

        wav_bytes = _to_wav(b"".join(pcm_chunks))

        try:
            audio_file = io.BytesIO(wav_bytes)
            audio_file.name = "audio.wav"

            response = self._client.audio.transcriptions.create(
                model=_MODEL,
                file=audio_file,
                language="en",
            )

            transcript = response.text.strip()
            if transcript:
                yield TextToken(text=transcript, is_final=True)

        except Exception as exc:
            raise STTError(f"Transcription failed: {exc}") from exc
