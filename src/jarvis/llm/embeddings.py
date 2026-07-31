# =============================================================================
# src/jarvis/llm/embeddings.py - turning sentences into comparable numbers
# =============================================================================
#
# An embedding is a list of numbers placing a sentence in a space where
# similar MEANINGS sit close together. That is what lets memory answer
# "what am I studying?" with a fact that never uses the word "studying".
#
# Three backends behind one small protocol:
#
#   OllamaEmbedder - a model running locally. No key, no bill, no rate
#     limit; needs `ollama serve` and a pulled model. The right default
#     on a machine you own, and the shape the design docs intend for a
#     local-llm worker.
#   VoyageEmbedder - hosted. The right choice for a small VPS with no
#     local model, given a card on file.
#   NullEmbedder   - nothing configured. Memory runs keyword-only rather
#     than failing; an explicit no-op object beats None checks scattered
#     through the memory service.
#
# VERSION STRINGS INCLUDE THE PROVIDER ("ollama:nomic-embed-text").
# Vectors from different models are not comparable - different lengths,
# different geometry - so a provider or model change must invalidate old
# vectors rather than silently producing meaningless similarity scores.
# The sleep cycle re-embeds anything whose version is stale, which makes
# switching providers a config edit plus a background catch-up.
#
# Failure is degradation, never an exception: every path returns empty
# and logs. Memory gets worse; conversations and jobs carry on.
# =============================================================================

from __future__ import annotations

from typing import Protocol

import httpx
import voyageai

from jarvis.common.log import get_logger
from jarvis.common.settings import CoreSettings

log = get_logger("llm.embeddings")

_VOYAGE_BATCH_SIZE = 128
_OLLAMA_TIMEOUT_S = 60.0


class Embedder(Protocol):
    """What the memory service needs from any embedding backend."""

    @property
    def available(self) -> bool: ...

    @property
    def version(self) -> str: ...

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float] | None: ...


class NullEmbedder:
    """No embeddings configured: memory runs keyword-only."""

    @property
    def available(self) -> bool:
        return False

    @property
    def version(self) -> str:
        return "none"

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return []

    async def embed_query(self, text: str) -> list[float] | None:
        return None


class OllamaEmbedder:
    """Embeddings from a model running on this machine.

    nomic-embed-text (and most instruction-tuned embedding models) expect
    a task prefix telling them whether text is being stored or searched
    with. Adding the right prefix measurably improves matching, so the
    two entry points differ rather than sharing a flag someone forgets.
    """

    def __init__(self, base_url: str, model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model

    @property
    def available(self) -> bool:
        # A local server can go down between calls, so "available" means
        # configured, not reachable. Unreachability shows up as an empty
        # result from a call, which callers already handle.
        return True

    @property
    def version(self) -> str:
        return f"ollama:{self._model}"

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._embed([f"search_document: {t}" for t in texts])

    async def embed_query(self, text: str) -> list[float] | None:
        vectors = await self._embed([f"search_query: {text}"])
        return vectors[0] if vectors else None

    async def _embed(self, inputs: list[str]) -> list[list[float]]:
        if not inputs:
            return []
        try:
            async with httpx.AsyncClient(timeout=_OLLAMA_TIMEOUT_S) as client:
                response = await client.post(
                    f"{self._base_url}/api/embed",
                    json={"model": self._model, "input": inputs},
                )
                response.raise_for_status()
                data = response.json()
            vectors = data.get("embeddings", [])
            if len(vectors) != len(inputs):
                log.error("ollama returned wrong vector count", extra={
                    "expected": len(inputs), "got": len(vectors),
                })
                return []
            return [[float(x) for x in v] for v in vectors]
        except httpx.ConnectError:
            # The single most likely failure, and worth its own message:
            # the server is not running.
            log.error("ollama unreachable - is `ollama serve` running?", extra={
                "base_url": self._base_url, "model": self._model,
            })
            return []
        except Exception:
            log.error("ollama embedding call failed", exc_info=True, extra={
                "model": self._model,
            })
            return []


class VoyageEmbedder:
    """Embeddings from Voyage AI's hosted API."""

    def __init__(self, api_key: str, model: str) -> None:
        self._client = voyageai.AsyncClient(api_key=api_key)
        self._model = model

    @property
    def available(self) -> bool:
        return True

    @property
    def version(self) -> str:
        return f"voyage:{self._model}"

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._embed(texts, input_type="document")

    async def embed_query(self, text: str) -> list[float] | None:
        vectors = await self._embed([text], input_type="query")
        return vectors[0] if vectors else None

    async def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _VOYAGE_BATCH_SIZE):
            batch = texts[start:start + _VOYAGE_BATCH_SIZE]
            try:
                result = await self._client.embed(
                    batch, model=self._model, input_type=input_type
                )
                vectors.extend(result.embeddings)
            except Exception:
                log.error("voyage embedding call failed", exc_info=True, extra={
                    "model": self._model, "batch_size": len(batch),
                })
                return []
        return vectors


def create_embedder(settings: CoreSettings) -> Embedder:
    """Build the configured backend. Misconfiguration degrades to
    keyword-only memory with a warning - never a failed boot."""
    provider = settings.embedder_provider

    if provider == "ollama":
        log.info("embeddings: local ollama", extra={
            "model": settings.model_embedder,
            "base_url": settings.ollama_base_url,
        })
        return OllamaEmbedder(settings.ollama_base_url, settings.model_embedder)

    if provider == "voyage":
        key = (
            settings.voyage_api_key.get_secret_value().strip()
            if settings.voyage_api_key else ""
        )
        if not key:
            log.warning("embedder_provider is voyage but no key set - "
                        "memory will run keyword-only")
            return NullEmbedder()
        log.info("embeddings: voyage", extra={"model": settings.model_embedder})
        return VoyageEmbedder(key, settings.model_embedder)

    log.info("embeddings disabled - memory runs keyword-only")
    return NullEmbedder()