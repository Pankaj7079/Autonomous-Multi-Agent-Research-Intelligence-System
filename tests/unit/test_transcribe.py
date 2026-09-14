"""Speech input: the size guards and the silence hallucination whisper is known for."""

from __future__ import annotations

import pytest

from amaris.tools import transcribe as transcribe_module
from amaris.tools.transcribe import TranscriptionError, is_silence_artifact, transcribe


class FakeTranscription:
    def __init__(self, text: str) -> None:
        self.text = text


def _patch_groq(monkeypatch: pytest.MonkeyPatch, text: str) -> dict[str, object]:
    """Stand in for the groq sdk so no test ever makes a real call."""
    seen: dict[str, object] = {}

    class FakeTranscriptions:
        def create(self, **kwargs: object) -> FakeTranscription:
            seen.update(kwargs)
            return FakeTranscription(text)

    class FakeAudio:
        transcriptions = FakeTranscriptions()

    class FakeGroq:
        def __init__(self, api_key: str | None = None) -> None:
            self.audio = FakeAudio()

    monkeypatch.setattr("groq.Groq", FakeGroq)
    return seen


@pytest.mark.parametrize(
    "text", [" Thank you.", "you", "Thanks for watching!", "", "  .  ", "Okay"]
)
def test_whisper_silence_phrases_are_recognised(text: str) -> None:
    """One second of silence transcribes as " Thank you." on this endpoint — measured."""
    assert is_silence_artifact(text)


@pytest.mark.parametrize("text", ["what is the MCP protocol?", "thank you for explaining RAG"])
def test_a_real_question_is_not_mistaken_for_silence(text: str) -> None:
    assert not is_silence_artifact(text)


async def test_a_mis_tap_on_the_mic_is_refused_before_any_api_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = _patch_groq(monkeypatch, "should not be reached")

    with pytest.raises(TranscriptionError, match="too short"):
        await transcribe(b"tiny")

    assert not called, "a sub-threshold recording must not cost an api call"


async def test_audio_over_the_size_cap_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = transcribe_module.get_settings()
    oversized = b"x" * (settings.stt_max_bytes + 1)

    with pytest.raises(TranscriptionError, match="larger than"):
        await transcribe(oversized)


async def test_a_hallucinated_transcript_never_becomes_a_research_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Researching " Thank you." would spend a full pipeline on a question nobody asked."""
    _patch_groq(monkeypatch, " Thank you.")

    with pytest.raises(TranscriptionError, match="nothing was said"):
        await transcribe(b"x" * 5_000)


async def test_a_spoken_question_comes_back_as_text(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _patch_groq(monkeypatch, "  What is the MCP protocol?  ")

    assert await transcribe(b"x" * 5_000, "spoken.wav") == "What is the MCP protocol?"
    # the extension picks the decoder, so the filename has to survive to the sdk
    assert getattr(seen["file"], "name", "") == "spoken.wav"
