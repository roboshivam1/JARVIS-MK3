# =============================================================================
# src/jarvis/llm/transcription.py - speech to text
# =============================================================================
#
# Groq's hosted Whisper rather than a local model, which is the choice
# MK2 made and it holds up: their hardware runs it faster than a laptop
# does, typically a few hundred milliseconds for a few seconds of
# speech, and cheaply enough not to think about.
#
# The real argument is architectural. Keeping transcription on the Core
# means every client stays dumb - record, send, done - which is what the
# RPi wearable will need, since it cannot run Whisper at all. A fat
# client would mean reimplementing this for every surface.
#
# NO AUDIO CONVERSION. Browsers record WebM/Opus; Groq accepts it
# directly. Converting would mean ffmpeg, a system dependency, and a
# processing step in the latency budget - for nothing.
#
# Unavailability is normal, not exceptional: no key means no voice,
# and text keeps working.
# =============================================================================

from __future__ import annotations

import httpx

from jarvis.common.log import get_logger
from jarvis.common.settings import CoreSettings

log = get_logger("llm.transcription")

_GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
_TIMEOUT_S = 60

# Enough for a couple of minutes of speech. A longer recording is
# almost certainly a stuck button rather than a long sentence.
MAX_AUDIO_BYTES = 25_000_000

# Names this system says constantly and Whisper has never heard. Kept
# short: a long prompt biases toward its own vocabulary rather than
# merely permitting it.
_VOCABULARY_PROMPT = (
    "JARVIS, ATHENA, PROTEUS, DAEDALUS, CALLIOPE, MNEMOSYNE, "
    "Shivam, SqOnion, LNMIIT, Jaipur, MK3."
)


class Transcriber:
    """Audio in, text out."""

    def __init__(self, settings: CoreSettings) -> None:
        self._model = settings.model_transcriber
        self._key = (
            settings.groq_api_key.get_secret_value().strip()
            if settings.groq_api_key else ""
        )

    @property
    def available(self) -> bool:
        """False means voice input is off; text is unaffected."""
        return bool(self._key)

    async def transcribe(
        self,
        audio: bytes,
        mime: str = "audio/webm",
        language: str | None = None,
    ) -> str:
        """Transcribe one utterance. Returns empty string on failure.

        Failure is degradation, never an exception: a failed
        transcription should leave the owner able to type, not crash
        the turn he was in the middle of.
        """
        if not self.available:
            return ""
        if not audio:
            return ""
        if len(audio) > MAX_AUDIO_BYTES:
            log.warning("audio too large to transcribe", extra={
                "bytes": len(audio),
            })
            return ""

        extension = "webm" if "webm" in mime else "wav"
        # Whisper accepts a prompt that biases it toward expected
        # vocabulary. Names it has never seen - the subagents, the
        # owner's projects - otherwise come back as whatever ordinary
        # word sounds closest, which is both wrong and confusing to
        # read back.
        form: dict[str, object] = {
            "model": (None, self._model),
            "prompt": (None, _VOCABULARY_PROMPT),
        }
        if language:
            form["language"] = (None, language)

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
                response = await client.post(
                    _GROQ_URL,
                    headers={"Authorization": f"Bearer {self._key}"},
                    files={
                        "file": (f"speech.{extension}", audio, mime),
                        **form,  # type: ignore[dict-item]
                    },
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            log.error("transcription refused", extra={
                "status": exc.response.status_code,
                "detail": exc.response.text[:300],
            })
            return ""
        except Exception:
            log.error("transcription failed", exc_info=True)
            return ""

        text = str(data.get("text", "")).strip()
        log.info("transcribed", extra={
            "bytes": len(audio), "characters": len(text),
        })
        return text
