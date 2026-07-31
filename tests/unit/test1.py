import asyncio

from jarvis.common.ids import new_ulid
from jarvis.common.jobs import Job, JobStatus
from jarvis.common.log import setup_logging
from jarvis.common.settings import CoreSettings
from jarvis.core.app import CoreApp


async def main() -> None:
    setup_logging("INFO")
    app = CoreApp(CoreSettings())
    await app.boot()                      # boots once; run() will not re-boot
    assert app.jobs is not None

    job = Job(
        type="research.brief",
        payload={"brief": (
            "Research the current state of small home air purifiers that "
            "mount on walls: the main technologies in use (HEPA, ionic, "
            "photocatalytic), typical CADR ranges for compact units, and "
            "what differentiates premium models. Aimed at someone building "
            "a wall-mounted purifier product."
        )},
        trace_id=new_ulid(),
    )
    await app.jobs.create(job)
    print(f">>> enqueued {job.id}")

    daemon = asyncio.create_task(app.run())

    for _ in range(120):                  # ~10 minute ceiling
        await asyncio.sleep(5)
        if daemon.done():                 # the daemon died - show why
            await daemon
            return
        fresh = await app.jobs.get(job.id)
        assert fresh is not None
        print(f"    status: {fresh.status}  attempts: {fresh.attempts}")
        if fresh.status in (JobStatus.SUCCEEDED, JobStatus.FAILED):
            break

    if fresh.status is JobStatus.SUCCEEDED and fresh.result:
        print("\n=== SUMMARY ===\n" + fresh.result["summary"])
        with open("brief.md", "w") as f:
            f.write(fresh.result["brief_markdown"])
        print("\nfull document written to brief.md")
    else:
        print("failed:", fresh.error)

    app.request_stop()
    await daemon


asyncio.run(main())