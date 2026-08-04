# =============================================================================
# src/jarvis/llm/speech.py - text to speech
# =============================================================================
#
# SENTENCE-CHUNKED STREAMING is the point of this module. Waiting for a
# complete reply before speaking means several seconds of silence while
# the model writes. Instead the text stream is watched, and the moment a
# sentence completes it is synthesised and sent - so JARVIS starts
# talking while he is still thinking about the rest.
#
# That single decision is most of the difference between a voice
# assistant that feels alive and one that feels like a phone tree.
#
# TWO BACKENDS, chosen by config - the same pattern as the embedder,
# for the same reason:
#
#   kokoro - a model running here. Free, private, no network, no
#     per-character cost. Needs ~350MB of model files and enough machine
#     to run them, which a laptop or a Mac mini has and a 1GB VPS does
#     not. The default.
#   groq   - Orpheus, hosted. Fast, ~$22 per million characters, and the
#     right answer if the Core ever lands somewhere that cannot host a
#     model. Note Orpheus caps input at 200 characters per request,
#     which sentence chunking mostly handles and the guard below
#     enforces.
#
# SYNTHESIS HAPPENS ON THE CORE either way. A browser cannot hold an API
# key without handing it to anyone who opens devtools, and it certainly
# cannot hold a 350MB model. The client asks for audio and gets bytes.
#
# TEXT FOR SPEECH IS NOT TEXT FOR READING. Markdown asterisks, bare
# URLs, and code blocks all read aloud as noise, so there is a cleaning
# pass first - MK2's _clean_for_speech, reborn.
# =============================================================================

from __future__ import annotations

import asyncio
import io
import re
from pathlib import Path
from typing import Protocol

import httpx

from jarvis.common.log import get_logger
from jarvis.common.settings import CoreSettings

log = get_logger("llm.speech")

_TIMEOUT_S = 30

# Orpheus rejects anything longer. Sentence chunking keeps us well under
# normally, but one long sentence would otherwise fail the whole clip.
_ORPHEUS_MAX_CHARS = 200

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

# Below this, a "sentence" is probably an abbreviation or a fragment;
# synthesising it separately produces a stutter.
_MIN_CHUNK_CHARS = 12


def clean_for_speech(text: str) -> str:
    """Strip what should not be read aloud.

    Every rule here is something that sounded wrong when spoken:
    asterisks read as "asterisk", URLs read character by character,
    code blocks read as punctuation soup.
    """
    cleaned = text

    cleaned = re.sub(r"```[\s\S]*?```", " (code omitted) ", cleaned)
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)
    cleaned = re.sub(r"^#{1,6}\s+", "", cleaned, flags=re.M)
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"https?://\S+", " a link ", cleaned)
    cleaned = re.sub(r"^\s*[-*+]\s+", "", cleaned, flags=re.M)

    # A 26-character ULID read aloud is unbearable.
    cleaned = re.sub(r"\b[0-9A-HJKMNP-TV-Z]{26}\b", "that job", cleaned)

    return re.sub(r"\s+", " ", cleaned).strip()


class SentenceBuffer:
    """Accumulates streamed text and yields complete sentences."""

    def __init__(self) -> None:
        self._pending = ""

    def feed(self, chunk: str) -> list[str]:
        """Add text; return any sentences that just completed."""
        self._pending += chunk
        parts = _SENTENCE_END.split(self._pending)
        if len(parts) < 2:
            return []
        self._pending = parts[-1]
        return [p.strip() for p in parts[:-1] if len(p.strip()) >= _MIN_CHUNK_CHARS]

    def flush(self) -> str:
        """Whatever is left when the reply ends."""
        remaining = self._pending.strip()
        self._pending = ""
        return remaining


class SpeechBackend(Protocol):
    """What the client connection needs from any synthesiser."""

    @property
    def available(self) -> bool: ...

    async def synthesise(self, text: str) -> bytes: ...


class NullSpeaker:
    """No speech configured. Replies stay text; nothing else changes."""

    @property
    def available(self) -> bool:
        return False

    async def synthesise(self, text: str) -> bytes:
        return b""


class KokoroSpeaker:
    """Kokoro running locally through ONNX.

    THE MODEL LOADS LAZILY, on first use rather than at boot. Loading
    takes a few seconds and ~300MB, and a Core that never speaks should
    not pay for either - which is most of the time, since Telegram is
    text.

    Synthesis is CPU-bound, so it runs in a thread. Doing it on the
    event loop would stall every other task in the Core for a few
    hundred milliseconds per sentence, which is exactly the wrong
    moment to be unresponsive.
    """

    def __init__(self, model_path: str, voices_path: str, voice: str) -> None:
        self._model_path = Path(model_path).expanduser()
        self._voices_path = Path(voices_path).expanduser()
        self._voice = voice
        self._kokoro: object | None = None
        self._load_lock = asyncio.Lock()

    @property
    def available(self) -> bool:
        """Whether the model files are present. Absent means the owner
        has not downloaded them, which is a setup step, not an error."""
        return self._model_path.exists() and self._voices_path.exists()

    async def _ensure_loaded(self) -> object | None:
        if self._kokoro is not None:
            return self._kokoro
        async with self._load_lock:
            if self._kokoro is not None:
                return self._kokoro      # another caller won the race
            try:
                from kokoro_onnx import Kokoro
                self._kokoro = await asyncio.to_thread(
                    Kokoro, str(self._model_path), str(self._voices_path)
                )
                log.info("kokoro loaded", extra={"voice": self._voice})
            except Exception:
                log.error("kokoro failed to load", exc_info=True)
                return None
        return self._kokoro

    async def synthesise(self, text: str) -> bytes:
        spoken = clean_for_speech(text)
        if not spoken:
            return b""

        kokoro = await self._ensure_loaded()
        if kokoro is None:
            return b""

        def _render() -> bytes:
            import soundfile

            samples, sample_rate = kokoro.create(  # type: ignore[attr-defined]
                spoken, voice=self._voice, speed=1.0, lang="en-us"
            )
            buffer = io.BytesIO()
            soundfile.write(buffer, samples, sample_rate, format="WAV")
            return buffer.getvalue()

        try:
            return await asyncio.to_thread(_render)
        except Exception:
            log.error("kokoro synthesis failed", exc_info=True)
            return b""


class GroqSpeaker:
    """Orpheus, hosted by Groq.

    Orpheus supports VOCAL DIRECTIONS in brackets - [dryly], [warm] -
    which suit JARVIS rather well. Left off by default: used on every
    sentence they sound theatrical, and the docs are clear that fewer
    directions produce the more natural cadence.
    """

    def __init__(self, api_key: str, model: str, voice: str) -> None:
        self._key = api_key
        self._model = model
        self._voice = voice

    @property
    def available(self) -> bool:
        return bool(self._key)

    async def synthesise(self, text: str) -> bytes:
        spoken = clean_for_speech(text)
        if not spoken:
            return b""

        # Orpheus rejects input over 200 characters. Sentence chunking
        # usually keeps us well under, but one long sentence would
        # otherwise fail the whole clip - better clipped than silent.
        if len(spoken) > _ORPHEUS_MAX_CHARS:
            spoken = spoken[:_ORPHEUS_MAX_CHARS].rsplit(" ", 1)[0]

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
                response = await client.post(
                    "https://api.groq.com/openai/v1/audio/speech",
                    headers={"Authorization": f"Bearer {self._key}"},
                    json={
                        "model": self._model,
                        "voice": self._voice,
                        "input": spoken,
                        "response_format": "wav",
                    },
                )
                response.raise_for_status()
                return response.content
        except httpx.HTTPStatusError as exc:
            log.error("speech synthesis refused", extra={
                "status": exc.response.status_code,
                "detail": exc.response.text[:300],
            })
            return b""
        except Exception:
            log.error("speech synthesis failed", exc_info=True)
            return b""


def create_speaker(settings: CoreSettings) -> SpeechBackend:
    """Build the configured backend.

    Misconfiguration degrades to silent replies with a warning - never
    a failed boot. Speech is a nicety; the daemon is not.
    """
    provider = settings.tts_provider

    if provider == "kokoro":
        speaker = KokoroSpeaker(
            settings.kokoro_model_path,
            settings.kokoro_voices_path,
            settings.tts_voice,
        )
        if not speaker.available:
            log.warning("kokoro model files not found - speech disabled", extra={
                "model": settings.kokoro_model_path,
                "voices": settings.kokoro_voices_path,
            })
            return NullSpeaker()
        log.info("speech: local kokoro", extra={"voice": settings.tts_voice})
        return speaker

    if provider == "groq":
        key = (
            settings.groq_api_key.get_secret_value().strip()
            if settings.groq_api_key else ""
        )
        if not key:
            log.warning("tts provider is groq but no key set - speech disabled")
            return NullSpeaker()
        log.info("speech: groq orpheus", extra={
            "model": settings.tts_model, "voice": settings.tts_voice,
        })
        return GroqSpeaker(key, settings.tts_model, settings.tts_voice)

    log.info("speech disabled - replies are text only")
    return NullSpeaker()
