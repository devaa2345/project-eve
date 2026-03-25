"""Concrete MicCapture implementation using sounddevice.

Streams raw audio chunks from the microphone in real-time without file I/O.
Raises MicCaptureError on hardware failure (Req 1.1, 1.3).
"""

import queue
from typing import Iterator

import numpy as np
import sounddevice as sd

from exceptions import MicCaptureError
from interfaces import MicCapture as MicCaptureABC
from type import AudioChunk

_SAMPLE_RATE = 16000   # Hz — standard for STT models
_CHANNELS = 1          # Mono
_CHUNK_FRAMES = 1024   # Frames per buffer read
_DTYPE = "int16"


class MicCapture(MicCaptureABC):
    """Streams audio from the default microphone using sounddevice.

    Usage:
        mic = MicCapture()
        for chunk in mic.start():
            process(chunk)
        mic.stop()
    """

    def __init__(self) -> None:
        self._running = False
        self._queue: queue.Queue = queue.Queue()

    def start(self) -> Iterator[AudioChunk]:
        """Open the microphone and yield AudioChunks until stop() is called.

        Raises MicCaptureError if the hardware cannot be opened or read.
        """
        self._running = True
        self._queue = queue.Queue()

        def _callback(indata: np.ndarray, frames: int, time_info, status) -> None:
            if status:
                pass  # ignore overflow warnings
            self._queue.put(indata.copy())

        try:
            stream = sd.InputStream(
                samplerate=_SAMPLE_RATE,
                channels=_CHANNELS,
                dtype=_DTYPE,
                blocksize=_CHUNK_FRAMES,
                callback=_callback,
            )
        except Exception as exc:
            raise MicCaptureError(f"Failed to open microphone: {exc}") from exc

        with stream:
            while self._running:
                try:
                    data: np.ndarray = self._queue.get(timeout=0.5)
                    yield AudioChunk(data=data.tobytes())
                except queue.Empty:
                    continue
                except Exception as exc:
                    raise MicCaptureError(f"Microphone read error: {exc}") from exc

    def stop(self) -> None:
        """Signal the capture loop to stop after the current chunk."""
        self._running = False
