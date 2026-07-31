import asyncio
import time

import uvicorn

from jarvis.common.log import setup_logging
from jarvis.common.settings import CoreSettings
from jarvis.core.db.database import Database
from jarvis.core.gateway.http import GatewayDeps, create_app
from jarvis.core.observability.traces import TracesRepo


async def main() -> None:
    setup_logging("INFO")
    settings = CoreSettings()
    settings.ensure_data_dirs()
    db = await Database.connect(settings.db_path)
    await db.migrate()

    app = create_app(GatewayDeps(
        settings=settings,
        db=db,
        traces=TracesRepo(db),
        started_monotonic=time.monotonic(),
    ))
    config = uvicorn.Config(
        app, host=settings.gateway_host, port=settings.gateway_port,
        log_config=None,   # our JSON logging, not uvicorn's own format
    )
    await uvicorn.Server(config).serve()


asyncio.run(main())