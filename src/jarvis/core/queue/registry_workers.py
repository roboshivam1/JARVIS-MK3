# =============================================================================
# src/jarvis/core/queue/registry_workers.py - who is connected right now
# =============================================================================
#
# MEMORY ONLY, deliberately. Which workers are connected is a property of
# live sockets, not durable state: if the Core restarts, every worker
# reconnects and re-announces itself within seconds. Persisting this
# would mean waking up with a list of machines that may or may not still
# exist, which is worse than knowing nothing and finding out.
#
# What IS durable lives elsewhere: the jobs a worker holds carry leases
# in the database, so a vanished worker's work is recovered by the same
# reclaim loop that handles a crashed Core-local executor. Remote death
# and local death look identical downstream, which is the dividend of
# having built leases before workers.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from jarvis.common.ids import utc_now
from jarvis.common.log import get_logger

log = get_logger("core.queue.workers")


@dataclass
class ConnectedWorker:
    """One live worker connection."""

    worker_id: str
    capabilities: set[str]
    max_concurrency: int
    version: str
    connected_ts: datetime = field(default_factory=utc_now)
    last_heartbeat_ts: datetime = field(default_factory=utc_now)
    running_job_ids: set[str] = field(default_factory=set)

    @property
    def has_capacity(self) -> bool:
        return len(self.running_job_ids) < self.max_concurrency

    def can_serve(self, requires: list[str]) -> bool:
        return set(requires).issubset(self.capabilities)


class WorkerRegistry:
    """The live fleet. One instance on the Core."""

    def __init__(self) -> None:
        self._workers: dict[str, ConnectedWorker] = {}

    def register(self, worker: ConnectedWorker) -> None:
        """Add or replace a worker. Replacement is normal: a laptop that
        reconnects after a dropped link presents the same id, and the
        newer connection is the real one."""
        if worker.worker_id in self._workers:
            log.info("worker reconnected", extra={"worker_id": worker.worker_id})
        else:
            log.info("worker connected", extra={
                "worker_id": worker.worker_id,
                "capabilities": sorted(worker.capabilities),
                "max_concurrency": worker.max_concurrency,
            })
        self._workers[worker.worker_id] = worker

    def unregister(self, worker_id: str) -> None:
        if self._workers.pop(worker_id, None) is not None:
            log.info("worker disconnected", extra={"worker_id": worker_id})

    def get(self, worker_id: str) -> ConnectedWorker | None:
        return self._workers.get(worker_id)

    def heartbeat(self, worker_id: str, running_job_ids: list[str]) -> None:
        worker = self._workers.get(worker_id)
        if worker is None:
            return
        worker.last_heartbeat_ts = utc_now()
        worker.running_job_ids = set(running_job_ids)

    def eligible_for(self, requires: list[str]) -> list[ConnectedWorker]:
        """Workers that could take this job right now: capable and free."""
        return [
            w for w in self._workers.values()
            if w.can_serve(requires) and w.has_capacity
        ]

    def all_capabilities(self) -> set[str]:
        """Everything the fleet can currently do - used to decide whether
        a job is merely waiting or is actually unservable."""
        capabilities: set[str] = set()
        for worker in self._workers.values():
            capabilities |= worker.capabilities
        return capabilities

    def connected(self) -> list[ConnectedWorker]:
        return list(self._workers.values())

    def count(self) -> int:
        return len(self._workers)