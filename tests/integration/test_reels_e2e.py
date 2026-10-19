# =============================================================================
# tests/integration/test_reels_e2e.py - one reel, Core to Mac and back
# =============================================================================
#
# The two halves meet here: the Core's real worker endpoint and the
# worker's real job runner, joined by in-memory queues instead of a
# socket, with the fake `claude` standing in for Claude Code. Only
# Telegram and the brief-writing model are faked.
#
# What only this test can show: the gate the Mac raises over the wire
# reaches the owner as a reel review, his notes reach the Mac's next
# attempt through the checkpoint, and the film comes back as a delivery.
# =============================================================================

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Awaitable, Callable

import pytest
from pydantic import BaseModel

from jarvis.common.envelope import Envelope, make_envelope
from jarvis.common.jobs import JobStatus
from jarvis.common.reels import EpisodeStatus, ReviewAction
from jarvis.common.worker_protocol import (
    CORE_JOB_OFFER,
    WORKER_HELLO,
    WORKER_JOB_ACCEPT,
    CoreJobOffer,
    WorkerHello,
    WorkerJobAccept,
)
from jarvis.core.gateway.workers import WorkerConnection
from jarvis.core.queue.registry import JobTypeRegistry
from jarvis.jobs.reels import register_reel_job_metadata
from jarvis.worker.reels import ReelConfig, register_reel_jobs
from jarvis.worker.runner import JobRunner

from tests.integration.test_reels_service import BRIEF, world  # noqa: F401
from tests.integration.test_reels_worker import claude_bin, kit  # noqa: F401


class QueueSocket:
    """The Core's side of a websocket, made of two queues."""

    def __init__(self) -> None:
        self.inbound: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.outbound: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def accept(self) -> None:
        return None

    async def receive_json(self) -> dict[str, Any]:
        return await self.inbound.get()

    async def send_json(self, data: dict[str, Any]) -> None:
        await self.outbound.put(data)


async def _until(check: Callable[[], Awaitable[bool]], what: str, timeout: float = 20) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not await check():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        await asyncio.sleep(0.1)


@pytest.fixture
async def link(world: SimpleNamespace, kit: Path, claude_bin: str):  # type: ignore[no-untyped-def]
    socket = QueueSocket()
    core_types = JobTypeRegistry()
    register_reel_job_metadata(core_types)
    connection = WorkerConnection(
        socket, "secret", world.workers, world.jobs, world.events,  # type: ignore[arg-type]
        world.artifacts, core_types, approvals=world.approval_service,
    )

    worker_types = JobTypeRegistry()
    assert register_reel_jobs(worker_types, ReelConfig(kit_dir=kit, claude_bin=claude_bin),
                              ["macos", "render", "gpu"])

    async def send(kind: str, payload: BaseModel) -> None:
        await socket.inbound.put(make_envelope(kind, payload).model_dump(mode="json"))

    runner = JobRunner(worker_types, send)

    async def worker_loop() -> None:
        await send(WORKER_HELLO, WorkerHello(
            worker_id="macbook", token="secret", capabilities=["macos", "render", "gpu"]))
        while True:
            envelope = Envelope.model_validate(await socket.outbound.get())
            if envelope.kind == CORE_JOB_OFFER:
                offer = envelope.parse_payload()
                assert isinstance(offer, CoreJobOffer)
                await send(WORKER_JOB_ACCEPT, WorkerJobAccept(job_id=offer.job_id))
                runner.start(offer)

    tasks = [asyncio.create_task(connection.run()), asyncio.create_task(worker_loop())]
    yield SimpleNamespace(socket=socket, kit=kit)
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def test_reel_round_trip(world: SimpleNamespace, link: SimpleNamespace) -> None:
    world.llm.replies.append(BRIEF)
    episode = (await world.service.request("five more minutes")).episode

    async def reviews(n: int) -> bool:
        await world.service.tick()
        return len(world.telegram.reviews) >= n

    await _until(lambda: reviews(1), "the first draft")
    first = world.telegram.reviews[0]
    assert first["sheet"].read_bytes() == b"PNG"          # bytes crossed the wire intact
    assert "five more minutes" in first["text"]

    await world.service.decide(episode.id, ReviewAction.NOTES, "Slower ending, please.")
    await _until(lambda: reviews(2), "the revised draft")
    assert "revised" in world.telegram.reviews[1]["text"]
    calls = (link.kit / "calls.jsonl").read_text()
    assert "Slower ending, please." in calls

    await world.service.decide(episode.id, ReviewAction.APPROVE)

    async def delivered() -> bool:
        await world.service.tick()
        return bool(world.telegram.videos)

    await _until(delivered, "the film")
    video, caption = world.telegram.videos[0]
    assert video.read_bytes() == b"MP4DATA"
    assert "-14 LUFS" in caption

    job = await world.jobs.get(episode.job_id)
    assert job.status is JobStatus.SUCCEEDED
    assert job.attempts == 3                              # draft, revise, render
    assert (await world.repo.get(episode.id)).status is EpisodeStatus.DELIVERED
