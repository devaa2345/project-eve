"""Speaker using sounddevice.

Handles WAV-wrapped audio from pyttsx3 — strips the WAV header on the
first chunk, detects sample rate, plays raw PCM.

Req 5.1, 5.2, 5.3, 5.4
"""

import io
import time
import wave
from typing import Iterator

import numpy as np
import sounddevice as sd

from interfaces import Speaker as SpeakerABC
from latency_tracker import LatencyTracker, Stage
from type import AudioChunk

_DTYPE = "int16"


class Speaker(SpeakerABC):
    def __init__(self, latency_tracker: LatencyTracker) -> None:
        super().__init__(latency_tracker)
        self._synthesis_complete = False

    def play(self, audio_stream: Iterator[AudioChunk]) -> None:
        # Collect all chunks first (pyttsx3 gives us a complete WAV)
        raw = b"".join(chunk.data for chunk in audio_stream)
        if not raw:
            self.latency_tracker.mark(Stage.TTS_COMPLETE, time.monotonic())
            return

        try:
            # Try to parse as WAV
            buf = io.BytesIO(raw)
            with wave.open(buf, "rb") as wf:
                sample_rate = wf.getframerate()
                n_channels  = wf.getnchannels()
                pcm = wf.readframes(wf.getnframes())
        except Exception:
            # Not a WAV — treat as raw PCM at 24kHz
            pcm = raw
            sample_rate = 24000
            n_channels  = 1

        audio = np.frombuffer(pcm, dtype=_DTYPE)
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels)

        sd.play(audio, samplerate=sample_rate, blocking=True)

        self.latency_tracker.mark(Stage.TTS_COMPLETE, time.monotonic())

    def handle_synthesis_complete(self) -> None:
        self._synthesis_complete = True
