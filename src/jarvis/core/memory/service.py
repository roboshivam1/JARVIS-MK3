# =============================================================================
# src/jarvis/core/memory/service.py - remembering and recalling
# =============================================================================
#
# The memory system's front door. Two operations matter:
#
#   store()  - remember one fact, checking first whether it restates
#              something already known (near-identical meaning), in which
#              case the new statement SUPERSEDES the old rather than both
#              coexisting. This is what stops memory accumulating twelve
#              versions of the same claim.
#
#   search() - find relevant facts by FUSING two searches:
#                vector  - closeness of MEANING (catches paraphrase)
#                keyword - closeness of WORDS  (catches exact names)
#              Each alone has a blind spot; together they cover for each
#              other. MK2 had only weak keyword matching, which is
#              precisely why it forgot things asked in different words.
#
# Final ranking blends five signals and scales by confidence, so a shaky
# belief ranks below a solid one even when it matches well.
#
# Degradation: with no embedder, search runs keyword-only and store()
# skips dedup. Memory gets worse, never broken. Facts stored during an
# outage carry no vector until the sleep cycle backfills them.
# =============================================================================

from __future__ import annotations

import math
from dataclasses import dataclass

from jarvis.common.facts import Fact, FactCategory
from jarvis.common.ids import utc_now
from jarvis.common.log import get_logger
from jarvis.core.db.repos.facts import FactsRepo
from jarvis.llm.embeddings import Embedder  # now a Protocol; any backend fits

log = get_logger("core.memory")

# Meaning this close means "the same claim, restated" - supersede rather
# than store twice. High on purpose: false merges lose information, while
# a missed merge only costs a duplicate the sleep cycle can clean up.
_DEDUP_THRESHOLD = 0.92

# Embedding spaces are ANISOTROPIC: their vectors occupy a narrow cone,
# so even unrelated sentences score 0.4-0.5 cosine and a perfect match
# rarely exceeds 0.85. Treating raw cosine as a 0-1 relevance score
# therefore wastes most of the range and makes everything look equally
# relevant. These bounds map the band that actually carries signal onto
# a full 0-1 scale.
#
# They are MODEL-SPECIFIC. Swapping the embedding model means checking
# them again: embed a few obviously-related and obviously-unrelated
# pairs and see where the two clouds sit.
_SIM_FLOOR = 0.45   # at or below: treat as unrelated
_SIM_CEIL = 0.80    # at or above: treat as a strong match

# Below this fused relevance, a fact is not worth the owner's context.
_RELEVANCE_FLOOR = 0.05


def rescale_similarity(similarity: float) -> float:
    """Raw cosine -> usable 0-1 relevance."""
    span = _SIM_CEIL - _SIM_FLOOR
    return max(0.0, min(1.0, (similarity - _SIM_FLOOR) / span))


@dataclass(frozen=True)
class RetrievalWeights:
    """How signals combine into a ranking.

    Two tiers, deliberately:

      RELEVANCE (vector + keyword, summing to 1.0) answers the only
      question that matters first - does this fact address the query?

      BOOSTS (importance, recency, access) are MULTIPLIERS on relevance,
      never additions. An earlier version added them, which handed every
      fact in the vault a constant score before relevance was considered
      at all - so an irrelevant but important fact could outrank a
      relevant one. Boosts break ties between things that already match;
      they must never manufacture a match.
    """

    vector: float = 0.7         # closeness of meaning
    keyword: float = 0.3        # closeness of words

    max_importance_boost: float = 0.30
    max_recency_boost: float = 0.15
    max_access_boost: float = 0.10

    recency_halflife_days: float = 90.0


@dataclass(frozen=True)
class ScoredFact:
    fact: Fact
    score: float


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """How aligned two vectors are: 1.0 identical, 0.0 unrelated.

    Computed with magnitudes rather than assuming unit-length vectors -
    Voyage returns normalised embeddings today, but a future local model
    might not, and a silent wrong answer here would be hard to notice.
    """
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


class MemoryService:
    """Store and recall what the system knows about the owner."""

    def __init__(
        self,
        facts: FactsRepo,
        embedder: Embedder,
        weights: RetrievalWeights | None = None,
    ) -> None:
        self._facts = facts
        self._embedder = embedder
        self._w = weights or RetrievalWeights()

    # -- storing --------------------------------------------------------------

    async def store(
        self,
        text: str,
        category: FactCategory = FactCategory.OTHER,
        importance: float = 0.5,
        source_event_ids: list[str] | None = None,
    ) -> Fact:
        """Remember one fact, superseding a near-identical existing one."""
        fact = Fact(
            text=text.strip(),
            category=category,
            importance=importance,
            source_event_ids=source_event_ids or [],
        )

        vector: list[float] | None = None
        if self._embedder.available:
            vectors = await self._embedder.embed_documents([fact.text])
            vector = vectors[0] if vectors else None

        duplicate_of: Fact | None = None
        if vector is not None:
            duplicate_of = await self._find_near_duplicate(vector)
            fact = fact.model_copy(update={"embedder_version": self._embedder.version})

        await self._facts.store(fact, embedding=vector)

        if duplicate_of is not None:
            # The newer statement wins: same claim, possibly corrected.
            # Judging genuine contradictions (rather than restatements)
            # needs a model call and belongs in the sleep cycle.
            await self._facts.supersede(duplicate_of.id, fact.id)
            log.info("fact restated - superseded older version", extra={
                "new_fact": fact.id, "old_fact": duplicate_of.id,
            })
        else:
            log.info("fact stored", extra={
                "fact_id": fact.id, "category": fact.category.value,
            })
        return fact

    async def _find_near_duplicate(self, vector: list[float]) -> Fact | None:
        best: tuple[float, Fact] | None = None
        for fact, stored in await self._facts.active_with_vectors():
            similarity = cosine_similarity(vector, stored)
            if similarity >= _DEDUP_THRESHOLD and (best is None or similarity > best[0]):
                best = (similarity, fact)
        return best[1] if best else None

    # -- recalling ------------------------------------------------------------

    async def search(self, query: str, k: int = 6) -> list[ScoredFact]:
        """The fused search. Returns the k best facts, best first, and
        records that they were used."""
        candidates: dict[str, Fact] = {}
        vector_scores: dict[str, float] = {}
        keyword_scores: dict[str, float] = {}

        # Path 1: meaning.
        if self._embedder.available:
            query_vector = await self._embedder.embed_query(query)
            if query_vector is not None:
                for fact, stored in await self._facts.active_with_vectors():
                    relevance = rescale_similarity(
                        cosine_similarity(query_vector, stored)
                    )
                    if relevance > 0.0:   # below the floor is genuinely unrelated
                        candidates[fact.id] = fact
                        vector_scores[fact.id] = relevance

        # Path 2: words.
        for fact, score in await self._facts.search_keyword(query, limit=20):
            candidates[fact.id] = fact
            keyword_scores[fact.id] = score

        if not candidates:
            return []

        now = utc_now()
        scored: list[ScoredFact] = []
        for fact_id, fact in candidates.items():
            # Tier 1: does this fact address the query at all?
            relevance = (
                self._w.vector * vector_scores.get(fact_id, 0.0)
                + self._w.keyword * keyword_scores.get(fact_id, 0.0)
            )
            if relevance < _RELEVANCE_FLOOR:
                continue

            # Tier 2: among facts that DO address it, prefer the
            # important, the fresh, and the repeatedly useful.
            age_days = (now - fact.created_ts).total_seconds() / 86400
            recency = math.exp(-age_days / self._w.recency_halflife_days)
            access = min(1.0, fact.access_count / 10.0)
            boost = (
                1.0
                + self._w.max_importance_boost * fact.importance
                + self._w.max_recency_boost * recency
                + self._w.max_access_boost * access
            )

            # A shaky belief ranks below a solid one, however well it matches.
            scored.append(ScoredFact(
                fact=fact, score=relevance * boost * fact.confidence
            ))

        scored.sort(key=lambda s: s.score, reverse=True)
        top = scored[:k]

        # Being retrieved is being used: recency and count feed ranking
        # and protect useful facts from decay.
        await self._facts.touch([s.fact.id for s in top])
        return top

    async def backfill_embeddings(self, limit: int = 200) -> int:
        """Give vectors to facts that have none, or whose model is stale.
        Run by the sleep cycle; also useful after adding a key."""
        if not self._embedder.available:
            return 0
        pending = await self._facts.needing_embedding(self._embedder.version, limit)
        if not pending:
            return 0
        vectors = await self._embedder.embed_documents([f.text for f in pending])
        if len(vectors) != len(pending):
            return 0
        for fact, vector in zip(pending, vectors):
            await self._facts.set_embedding(fact.id, vector, self._embedder.version)
        log.info("backfilled embeddings", extra={"count": len(pending)})
        return len(pending)