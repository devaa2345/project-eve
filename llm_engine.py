"""LLMEngine using Groq Llama 3 (free tier, fast inference).

Get a free key at https://console.groq.com

Req 3.1, 3.2, 3.4, 3.5, 11.3, 12.2
"""

import os
from typing import Iterator

from groq import Groq

from exceptions import LLMError
from interfaces import LLMEngine as LLMEngineABC
from type import ResponseToken, TextToken

_MODEL = "llama-3.1-8b-instant"


class LLMEngine(LLMEngineABC):
    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.environ.get("GROQ_API_KEY", "")
        self._client = Groq(api_key=key)

    def stream_complete(self, token_stream: Iterator[TextToken]) -> Iterator[ResponseToken]:
        transcript_parts: list[str] = []
        received_any = False

        for token in token_stream:
            received_any = True
            transcript_parts.append(token.text)

            if token.is_final:
                transcript = "".join(transcript_parts)
                transcript_parts = []

                try:
                    response = self._client.chat.completions.create(
                        model=_MODEL,
                        messages=[{"role": "user", "content": transcript}],
                        stream=True,
                    )
                    for chunk in response:
                        content = chunk.choices[0].delta.content or ""
                        if content:
                            yield ResponseToken(text=content)
                except Exception as exc:
                    raise LLMError(f"LLM generation failed: {exc}") from exc

        _ = received_any
