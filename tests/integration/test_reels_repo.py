# =============================================================================
# tests/integration/test_reels_repo.py - the reel episode log and contracts
# =============================================================================
#
# The promises that matter:
#   - an episode round-trips whole, brief and lines included
#   - status moves only from the statuses the caller expects
#   - the owner's short ref finds exactly the episode he means
#   - the shared contracts reject what the pipeline must never carry
# =============================================================================

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.common.reels import (
    PHASE_A_SCHEMA,
    EpisodeStatus,
    EpisodeType,
    PhaseAOutput,
    ReelBrief,
    ReelCheckpoint,
    ReelEpisode,
    forbidden_mentions,
    kit_episode_id,
)
from jarvis.core.db.database import Database
from jarvis.core.db.repos.reels import ReelsRepo
from jarvis.core.queue.registry import JobTypeRegistry
from jarvis.jobs.reels import register_reel_job_metadata


def _brief() -> ReelBrief:
    return ReelBrief(
        episode_type=EpisodeType.FIELD_NOTES,
        premise="Sir says five more minutes at 1:12 a.m. and means ninety.",
        beats=["The promise", "The evidence", "The verdict"],
        facts=["It was a Tuesday"],
    )


def _episode(request: str = "five more minutes") -> ReelEpisode:
    episode = ReelEpisode(
        kit_id="placeholder", type=EpisodeType.FIELD_NOTES,
        request=request, brief=_brief(),
    )
    episode.kit_id = kit_episode_id(episode.type, episode.id)
    return episode


@pytest.fixture
def repo(db: Database) -> ReelsRepo:
    return ReelsRepo(db)


async def test_episode_round_trips(repo: ReelsRepo) -> None:
    episode = _episode()
    await repo.create(episode)
    await repo.update(episode.id, lines=["At one twelve a.m.", "Sir said."],
                      title="Five more minutes", round=1)

    loaded = await repo.get(episode.id)
    assert loaded is not None
    assert loaded.brief == episode.brief
    assert loaded.lines == ["At one twelve a.m.", "Sir said."]
    assert loaded.title == "Five more minutes"
    assert loaded.status is EpisodeStatus.BRIEFED
    assert loaded.kit_id.startswith("field-notes-")


async def test_status_moves_only_from_expected(repo: ReelsRepo) -> None:
    episode = _episode()
    await repo.create(episode)

    assert await repo.set_status(
        episode.id, EpisodeStatus.DRAFTING, expected=[EpisodeStatus.BRIEFED]
    )
    # A second mover expecting the old status loses, and changes nothing.
    assert not await repo.set_status(
        episode.id, EpisodeStatus.CANCELLED, expected=[EpisodeStatus.BRIEFED]
    )
    loaded = await repo.get(episode.id)
    assert loaded is not None and loaded.status is EpisodeStatus.DRAFTING


async def test_status_sets_fields_atomically(repo: ReelsRepo) -> None:
    episode = _episode()
    await repo.create(episode)
    moved = await repo.set_status(
        episode.id, EpisodeStatus.FAILED, error="render fell back to CPU"
    )
    loaded = await repo.get(episode.id)
    assert moved and loaded is not None
    assert loaded.error == "render fell back to CPU"


async def test_update_refuses_status_and_unknown_columns(repo: ReelsRepo) -> None:
    episode = _episode()
    await repo.create(episode)
    with pytest.raises(ValueError):
        await repo.update(episode.id, status="posted")
    with pytest.raises(ValueError):
        await repo.update(episode.id, request="something else")


async def test_ref_lookup(repo: ReelsRepo) -> None:
    first, second = _episode("one"), _episode("two")
    await repo.create(first)
    await repo.create(second)

    found = await repo.by_ref(first.ref)
    assert [e.id for e in found] == [first.id]
    assert [e.id for e in await repo.by_ref(second.kit_id)] == [second.id]
    assert await repo.by_ref("ab") == []           # too short to mean anything


async def test_active_excludes_settled(repo: ReelsRepo) -> None:
    live, done = _episode("live"), _episode("done")
    await repo.create(live)
    await repo.create(done)
    await repo.set_status(done.id, EpisodeStatus.DELIVERED)

    assert [e.id for e in await repo.active()] == [live.id]
    assert len(await repo.recent()) == 2


def test_brand_check_catches_variants() -> None:
    assert forbidden_mentions("a plug for The Basement Supply Co.")
    assert forbidden_mentions("basementsupply.in")
    assert forbidden_mentions("BASEMENT  SUPPLY")
    assert forbidden_mentions("the basement workshop, supplying tea") == []


def test_brief_limits() -> None:
    with pytest.raises(ValidationError):
        ReelBrief(episode_type=EpisodeType.SHIFT_LOG, premise="Too few beats.",
                  beats=["one", "two"])
    with pytest.raises(ValidationError):
        ReelBrief(episode_type=EpisodeType.SHIFT_LOG, premise="Bad type here.",
                  beats=["a", "b", "c"], facts=["x" * 301])


def test_checkpoint_and_schema_agree() -> None:
    draft = PhaseAOutput(status="ready_for_review", title="T",
                         lines=["a"], sheet="out/stills/sheet.png")
    checkpoint = ReelCheckpoint(stage="review", round=1, kit_id="k", draft=draft)
    restored = ReelCheckpoint.model_validate(checkpoint.model_dump(mode="json"))
    assert restored.draft == draft
    assert set(PHASE_A_SCHEMA["required"]) <= set(PhaseAOutput.model_fields)


def test_job_type_is_worker_only() -> None:
    registry = JobTypeRegistry()
    register_reel_job_metadata(registry)
    spec = registry.get("reel.make")
    assert spec is not None
    assert spec.handler is None
    assert spec.requires == ["render", "gpu"]
