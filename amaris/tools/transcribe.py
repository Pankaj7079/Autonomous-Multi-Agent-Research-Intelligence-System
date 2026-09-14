"""Speech to text for the composer's microphone.

Groq serves whisper-large-v3 on the same free key the agents already use, so this needs no
new dependency and no local model download — the `groq` SDK arrives with langchain-groq.
Transcription is I/O at the edge like input validation: the text it produces goes through
exactly the same guardrail and pipeline as a typed question.
"""

from __future__ import annotations

import asyncio
import io
import re
from typing import Any

from amaris.config.settings import get_settings
from amaris.observability.logging import logger

# whisper invents a stock phrase when handed silence or noise — measured against this very
# endpoint, one second of digital silence transcribes as " Thank you.". Researching that
# would cost a full 60-90s pipeline run for a question nobody asked.
SILENCE_ARTIFACTS = frozenset(
    {
        "",
        "you",
        "thank you",
        "thanks",
        "thanks for watching",
        "thank you for watching",
        "bye",
        "okay",
        "so",
        "uh",
        "um",
    }
)

_PUNCTUATION = re.compile(r"[^\w\s]")


class TranscriptionError(Exception):
    """Raised when audio cannot be turned into a usable question. Shown to the user."""


def is_silence_artifact(text: str) -> bool:
    """True when whisper returned one of its stock phrases for an empty recording."""
    normalised = _PUNCTUATION.sub("", text).strip().lower()
    return normalised in SILENCE_ARTIFACTS


async def transcribe(data: bytes, filename: str = "question.wav") -> str:
    """Audio bytes to a question. Raises TranscriptionError with a readable reason."""
    settings = get_settings()

    if len(data) < settings.stt_min_bytes:
        raise TranscriptionError("that recording was too short to hear — hold the mic and speak")
    if len(data) > settings.stt_max_bytes:
        limit_mb = settings.stt_max_bytes / 1_000_000
        raise TranscriptionError(f"the recording is larger than the {limit_mb:.0f}MB limit")

    key = settings.key("groq_api_key")
    if not key:
        raise TranscriptionError("speech input needs GROQ_API_KEY — it serves whisper for free")

    def call() -> Any:
        from groq import Groq

        # the sdk wants a named file object, not raw bytes — the extension picks the decoder
        buffer = io.BytesIO(data)
        buffer.name = filename
        return Groq(api_key=key).audio.transcriptions.create(
            file=buffer, model=settings.stt_model, response_format="json"
        )

    loop = asyncio.get_running_loop()
    try:
        # the groq sdk is synchronous, so it must not run on the event loop
        response = await loop.run_in_executor(None, call)
    except Exception as exc:
        raise TranscriptionError(f"could not transcribe that: {exc}") from exc

    text = str(getattr(response, "text", "")).strip()
    if is_silence_artifact(text):
        raise TranscriptionError("nothing was said in that recording — try again")

    logger.bind(model=settings.stt_model, chars=len(text), bytes=len(data)).info("transcribe.done")
    return text
