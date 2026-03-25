# 🎙️ Project Eve — Real-Time Voice Assistant

A low-latency, streaming voice assistant pipeline built in Python. Speak into your microphone, get an intelligent response spoken back to you — all wired together in a clean, fully streaming chain.

```
Microphone → STT (Groq Whisper) → LLM (Groq Llama 3.1) → TTS (pyttsx3) → Speaker
```

---

## ✨ Features

- **Fully Streaming Pipeline** — each stage begins processing as soon as the first chunk arrives from the previous stage, minimising time-to-first-audio
- **Real-Time Microphone Capture** — captures raw PCM audio via `sounddevice` and streams it downstream immediately
- **Speech-to-Text** — powered by **Groq Whisper-large-v3** (free API, best-in-class accuracy)
- **Large Language Model** — powered by **Groq Llama 3.1-8b-instant** (free API, ultra-fast inference)
- **Text-to-Speech** — powered by **pyttsx3** (fully offline, no API key required)
- **Latency Tracker** — measures per-stage timestamps at every pipeline boundary and prints a full latency report after each interaction
- **Timeout Guard** — automatically cancels and resets any interaction that exceeds 30 seconds
- **Graceful Cancellation** — safe mid-flight cancellation from any thread; partial reports are always saved before reset
- **Clean Abstractions** — every component is backed by an Abstract Base Class (`interfaces.py`), making any stage trivially swappable

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        Pipeline                         │
│                                                         │
│  MicCapture ──► STTEngine ──► LLMEngine ──► TTSEngine  │
│       │              │             │             │      │
│   AudioChunk      TextToken   ResponseToken  AudioChunk │
│                                                    │    │
│                                               Speaker   │
└─────────────────────────────────────────────────────────┘
         LatencyTracker records timestamps at each ──►
         stage boundary (MIC_CAPTURE, STT_PARTIAL,
         LLM_FIRST_TOKEN, TTS_FIRST_AUDIO, TTS_COMPLETE)
```

### Component Overview

| File | Role |
|---|---|
| `main.py` | Entry point — wires all components and runs the interaction loop |
| `pipeline.py` | Orchestrator — streams data through all stages with timeout & cancellation |
| `mic_capture.py` | Captures live microphone audio as `AudioChunk` stream |
| `stt_engine.py` | Transcribes audio via Groq Whisper-large-v3 → `TextToken` |
| `llm_engine.py` | Generates response via Groq Llama 3.1-8b-instant → `ResponseToken` |
| `tts_engine.py` | Synthesises speech via pyttsx3 (offline) → `AudioChunk` |
| `speaker.py` | Plays audio chunks in real-time via `sounddevice` |
| `latency_tracker.py` | Records per-stage timestamps; computes ms-level latency metrics |
| `interfaces.py` | Abstract Base Classes for every pipeline component |
| `exceptions.py` | Typed exceptions (`MicCaptureError`, `STTError`, `LLMError`, `TTSError`) |
| `type.py` | Shared data types (`AudioChunk`, `TextToken`, `ResponseToken`) |

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.10+
- A free [Groq API key](https://console.groq.com)

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Or export it directly in your shell:

```bash
# Windows (PowerShell)
$env:GROQ_API_KEY="your_groq_api_key_here"

# Linux / macOS
export GROQ_API_KEY="your_groq_api_key_here"
```

### 4. Run

```bash
python main.py
```

```
Voice assistant ready. Speak, then press Ctrl+C when done talking.
------------------------------------------------------------
```

- **Speak** into your microphone
- Press **Ctrl+C** when you have finished speaking
- Eve transcribes, generates a response, and speaks it back
- A latency report is printed after each interaction
- Press **Ctrl+C again** (while waiting) to quit

---

## 📊 Latency Report

After each interaction, a per-stage latency breakdown is printed:

```
--- Latency Report ---
  mic capture:         120.4 ms
  stt → llm:           830.2 ms
  llm → tts:           210.7 ms
  tts complete:        980.1 ms
  time to first audio: 1161.3 ms
------------------------
```

| Metric | Definition |
|---|---|
| `mic_capture_ms` | Time from mic start to first audio chunk arriving at STT |
| `stt_to_llm_ms` | Time from first STT token to first LLM response token |
| `llm_to_tts_ms` | Time from first LLM token to first TTS audio chunk |
| `tts_complete_ms` | Time from first TTS chunk to synthesis completion |
| `time_to_first_audio` | End-to-end time until the user hears the first audio |

---

## 🧪 Testing

The project includes a comprehensive test suite using `pytest` and `hypothesis`.

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest test_pipeline.py -v
pytest test_latency_tracker.py -v
pytest test_pipeline_stages.py -v
```

| Test File | Coverage |
|---|---|
| `test_pipeline.py` | End-to-end pipeline integration, cancellation, timeout |
| `test_latency_tracker.py` | LatencyTracker correctness, first-write-wins, partial reports |
| `test_pipeline_stages.py` | Individual stage unit tests (STT, LLM, TTS, Speaker) |

---

## 🔧 Extending Eve

Every pipeline component is backed by an Abstract Base Class in `interfaces.py`. To swap out any component, subclass the appropriate ABC and pass your implementation to `Pipeline`:

```python
from interfaces import STTEngine
from type import AudioChunk, TextToken
from typing import Iterator

class MyCustomSTT(STTEngine):
    def stream_transcribe(self, audio_stream: Iterator[AudioChunk]) -> Iterator[TextToken]:
        # your implementation here
        ...
```

Then wire it in `main.py`:

```python
stt = MyCustomSTT()
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `groq` | Groq API client — Whisper STT + Llama LLM (free tier) |
| `pyttsx3` | Offline TTS engine — no API key required |
| `sounddevice` | Microphone capture and audio playback |
| `numpy` | Audio buffer handling |
| `python-dotenv` | `.env` file loading |
| `pytest` | Test runner |
| `hypothesis` | Property-based testing |

---

## 📝 License

This project is released under the [MIT License](LICENSE).

---

> **Get your free Groq API key** at [console.groq.com](https://console.groq.com) — no credit card required.
