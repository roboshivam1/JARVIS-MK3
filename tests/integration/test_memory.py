# =============================================================================
# tests/integration/test_memory.py - the fact vault and its ranking
# =============================================================================
#
# The embedder is faked with FIXED vectors, so similarity is exact and
# the tests measure our fusion and ranking logic rather than a model's
# quality. Testing against a real embedding model would be testing
# somebody else's software, slowly.
#
# The load-bearing test here is test_relevance_beats_importance: it locks
# in the scoring fix where boosts became multipliers instead of addends.
# The bug it guards against made every fact score alike and was invisible
# except by staring at the numbers.
# =============================================================================

from __future__ import annotations

import math

import pytest

from jarvis.common.facts import FactCategory
from jarvis.core.db.database import Database
from jarvis.core.db.repos.facts import FactsRepo
from jarvis.core.memory.profile import ProfileStore
from jarvis.core.memory.service import MemoryService, cosine_similarity


def _unit(cosine_to_x_axis: float) -> list[float]:
    """A unit vector at a chosen cosine from the query axis.

    Three dimensions throughout, so tests that need vectors far apart
    FROM EACH OTHER (not just from the query) have room to express that.
    All fake vectors must share this dimensionality: cosine_similarity
    deliberately returns 0.0 for mismatched lengths, so a stray 2D vector
    silently matches nothing.
    """
    return [
        cosine_to_x_axis,
        math.sqrt(max(0.0, 1 - cosine_to_x_axis ** 2)),
        0.0,
    ]


QUERY_VECTOR = [1.0, 0.0, 0.0]


class FakeEmbedder:
    """Returns preset vectors, so similarity is exactly what a test says."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    @property
    def available(self) -> bool:
        return True

    @property
    def version(self) -> str:
        return "fake:v1"

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Unlisted text gets a vector orthogonal to the query: present,
        # but matching nothing. Same dimensionality as everything else.
        return [self._vectors.get(t, [0.0, 1.0, 0.0]) for t in texts]

    async def embed_query(self, text: str) -> list[float] | None:
        return self._vectors.get(text, QUERY_VECTOR)


class NoEmbedder:
    """Stands in for a missing or broken embedding backend."""

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


class TestSimilarity:
    def test_identical_vectors(self) -> None:
        # Floats do not land exactly on 1.0 after a square root; that is
        # arithmetic, not a bug.
        assert cosine_similarity([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_magnitude_does_not_matter(self) -> None:
        # Correctness here protects against a future embedder that does
        # not return unit-length vectors.
        assert cosine_similarity([3.0, 0.0], [7.0, 0.0]) == 1.0

    def test_mismatched_lengths_are_not_comparable(self) -> None:
        assert cosine_similarity([1.0], [1.0, 0.0]) == 0.0


class TestStorage:
    async def test_store_then_retrieve(self, db: Database) -> None:
        text = "beta gamma delta"
        memory = MemoryService(
            FactsRepo(db), FakeEmbedder({text: _unit(0.9)})  # type: ignore[arg-type]
        )
        await memory.store(text, FactCategory.PROJECT, importance=0.5)

        hits = await memory.search("alpha")
        assert len(hits) == 1 and hits[0].fact.text == text

    async def test_near_duplicate_supersedes(self, db: Database) -> None:
        first, second = "epsilon zeta", "epsilon zeta eta"
        # Both point the same way: the same claim, restated.
        memory = MemoryService(
            FactsRepo(db),
            FakeEmbedder({first: _unit(0.95), second: _unit(0.95)}),  # type: ignore[arg-type]
        )
        facts = FactsRepo(db)
        await memory.store(first)
        await memory.store(second)

        active = await facts.all_active()
        assert len(active) == 1 and active[0].text == second
        assert active[0].supersedes is not None

    async def test_distinct_facts_coexist(self, db: Database) -> None:
        a, b = "theta iota", "kappa lambda"
        memory = MemoryService(
            FactsRepo(db),
            FakeEmbedder({a: _unit(0.9), b: _unit(0.1)}),  # type: ignore[arg-type]
        )
        await memory.store(a)
        await memory.store(b)
        assert await FactsRepo(db).count_active() == 2


class TestRanking:
    async def test_relevance_beats_importance(self, db: Database) -> None:
        # REGRESSION: boosts must MULTIPLY relevance, never add to it.
        # When they were addends, an important-but-irrelevant fact could
        # outrank a relevant one, and every fact scored roughly alike.
        #
        # The two vectors must differ in closeness to the QUERY while
        # staying far apart FROM EACH OTHER - otherwise store() sees a
        # restatement and supersedes one, leaving nothing to rank. Two
        # dimensions cannot express that; three can.
        relevant = "mu nu xi"          # close to the query in meaning
        important = "omicron pi rho"   # barely related, but defining
        memory = MemoryService(
            FactsRepo(db),
            FakeEmbedder({                     # type: ignore[arg-type]
                relevant: [0.78, 0.63, 0.0],   # cosine 0.78 to the query
                important: [0.55, 0.0, 0.84],  # cosine 0.55, and only
                                               # 0.43 to the other fact
            }),
        )
        await memory.store(relevant, importance=0.1)
        await memory.store(important, importance=1.0)

        # Both survived: this is a ranking test, not a dedup test.
        assert await FactsRepo(db).count_active() == 2

        hits = await memory.search("alpha", k=5)
        assert hits[0].fact.text == relevant
        assert hits[0].score > hits[-1].score * 1.5   # genuinely separated

    async def test_unrelated_facts_are_excluded(self, db: Database) -> None:
        unrelated = "sigma tau"
        memory = MemoryService(
            FactsRepo(db), FakeEmbedder({unrelated: _unit(0.1)})  # type: ignore[arg-type]
        )
        await memory.store(unrelated)
        assert await memory.search("alpha") == []

    async def test_retrieval_marks_facts_as_used(self, db: Database) -> None:
        text = "upsilon phi"
        facts = FactsRepo(db)
        memory = MemoryService(facts, FakeEmbedder({text: _unit(0.9)}))  # type: ignore[arg-type]
        stored = await memory.store(text)

        await memory.search("alpha")
        after = await facts.get(stored.id)
        assert after is not None and after.access_count == 1


class TestDegradation:
    async def test_keyword_search_works_without_embeddings(
        self, db: Database
    ) -> None:
        # No embedder: memory must still find things by their words.
        memory = MemoryService(FactsRepo(db), NoEmbedder())  # type: ignore[arg-type]
        await memory.store("Shivam is building SqOnion, a wall-mounted purifier")

        hits = await memory.search("SqOnion")
        assert len(hits) == 1

    async def test_keyword_stemming_matches_word_forms(self, db: Database) -> None:
        memory = MemoryService(FactsRepo(db), NoEmbedder())  # type: ignore[arg-type]
        await memory.store("Shivam studies engineering at LNMIIT")

        hits = await memory.search("studying")
        assert len(hits) == 1

    async def test_fts_special_characters_do_not_crash(self, db: Database) -> None:
        # Raw text reaches the FTS query builder; quotes and operators
        # are ordinary English, not syntax errors.
        memory = MemoryService(FactsRepo(db), NoEmbedder())  # type: ignore[arg-type]
        await memory.store("Shivam prefers SQLite (WAL mode) over Postgres")
        assert await memory.search('what about "SQLite" AND Postgres?') is not None


class TestProfile:
    async def test_versions_append_and_newest_wins(self, db: Database) -> None:
        profile = ProfileStore(db)
        assert await profile.current() == ""

        await profile.write("first version", generated_by="seed")
        await profile.write("second version", generated_by="archivist")

        assert await profile.current() == "second version"
        history = await profile.history()
        assert len(history) == 2          # nothing overwritten
        assert history[0].generated_by == "archivist"

    async def test_overlong_profile_is_capped(self, db: Database) -> None:
        profile = ProfileStore(db)
        await profile.write("x" * 20000, generated_by="archivist")
        assert len(await profile.current()) < 20000