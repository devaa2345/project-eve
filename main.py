"""Entry point for the real-time voice assistant.

Usage:
    python main.py

Requires:
    GROQ_API_KEY in .env or environment  (free at https://console.groq.com)

STT:  Groq Whisper-large-v3  (free)
LLM:  Groq Llama-3.1-8b      (free)
TTS:  pyttsx3                 (offline, no key needed)
"""

import logging
import os
import signal
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from latency_tracker import LatencyTracker
from mic_capture import MicCapture
from stt_engine import STTEngine
from llm_engine import LLMEngine
from tts_engine import TTSEngine
from speaker import Speaker
from pipeline import Pipeline
from exceptions import MicCaptureError

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _print_report(tracker: LatencyTracker) -> None:
    try:
        report = tracker.report()
        print("\n--- Latency Report ---")
        print(f"  mic capture:         {report['mic_capture_ms']:.1f} ms")
        print(f"  stt → llm:           {report['stt_to_llm_ms']:.1f} ms")
        print(f"  llm → tts:           {report['llm_to_tts_ms']:.1f} ms")
        print(f"  tts complete:        {report['tts_complete_ms']:.1f} ms")
        print(f"  time to first audio: {report['time_to_first_audio_ms']:.1f} ms")
        print("-" * 24)
    except Exception:
        partial = tracker.report_partial()
        print("\n--- Partial Latency Report ---")
        for k, v in partial.items():
            print(f"  {k}: {v}")
        print("-" * 30)


def main() -> None:
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        print("Error: GROQ_API_KEY not set. Add it to .env or your environment.")
        print("Get a free key at https://console.groq.com")
        sys.exit(1)

    tracker = LatencyTracker()
    mic     = MicCapture()
    stt     = STTEngine(api_key=groq_key)
    llm     = LLMEngine(api_key=groq_key)
    tts     = TTSEngine()
    speaker = Speaker(latency_tracker=tracker)

    pipeline = Pipeline(
        mic=mic, stt=stt, llm=llm, tts=tts,
        speaker=speaker, latency_tracker=tracker,
        timeout_seconds=30,
    )

    def _handle_interrupt(sig, frame):
        print("\nStopping... (processing your speech)")
        mic.stop()

    signal.signal(signal.SIGINT, _handle_interrupt)

    print("Voice assistant ready. Speak, then press Ctrl+C when done talking.")
    print("-" * 60)

    while True:
        try:
            pipeline.start_interaction()
            _print_report(tracker)
            tracker.reset()
        except MicCaptureError as exc:
            logger.error("Microphone error: %s", exc)
            sys.exit(1)
        except KeyboardInterrupt:
            break
        except Exception as exc:
            logger.error("Error: %s", exc)
            tracker.reset()

        try:
            print("\nListening again... Speak then Ctrl+C when done. Ctrl+C again to quit.")
        except KeyboardInterrupt:
            break

    print("Goodbye.")


if __name__ == "__main__":
    main()
