import asyncio
from pathlib import Path

from jarvis.common.ids import new_ulid
from jarvis.common.log import setup_logging
from jarvis.common.sessions import Session, Turn, TurnRole
from jarvis.common.settings import CoreSettings
from jarvis.core.db.database import Database
from jarvis.core.db.repos.sessions import SessionsRepo
from jarvis.core.observability.traces import TracesRepo, make_db_trace_sink
from jarvis.llm.layer import LLMLayer, user_message
from jarvis.llm.tiers import Tier


async def main() -> None:
    setup_logging("INFO")
    settings = CoreSettings()
    settings.ensure_data_dirs()

    db = await Database.connect(settings.db_path)
    print("migrations ran:", await db.migrate())   # expect 1 (0002) on first run

    # The swap: LLMLayer now reports to the database, not the log.
    layer = LLMLayer(settings, trace_sink=make_db_trace_sink(db))

    # A conversation, stored properly.
    sessions = SessionsRepo(db)
    session = await sessions.get_or_create_default("scratch")
    print("session:", session.id)

    trace = new_ulid()
    await sessions.append_turn(Turn(
        session_id=session.id, role=TurnRole.USER, content="Report status.",
    ))
    resp = await layer.complete(
        Tier.UTILITY,
        system="You are JARVIS. Dry wit, first person, 'sir'. One sentence.",
        messages=[user_message("Report status.")],
        actor="scratch.batch7",
        trace_id=trace,
    )
    await sessions.append_turn(Turn(
        session_id=session.id, role=TurnRole.ASSISTANT,
        content=resp.text, llm_call_ids=[],  # wired for real in the orchestrator
    ))
    print("JARVIS:", resp.text)

    # Read it all back from the notebook.
    for t in await sessions.recent_turns(session.id):
        print(f"  [{t.role}] {t.content[:60]}")

    traces = TracesRepo(db)
    print("calls in this trace:", len(await traces.calls_for_trace(trace)))
    print("spend today: $", round(await traces.cost_today_usd(), 6))
    for d in await traces.daily_costs():
        print(f"  {d.day}: {d.calls} calls, ${d.cost_usd} / Rs {d.cost_inr}")

    await db.close()


asyncio.run(main())