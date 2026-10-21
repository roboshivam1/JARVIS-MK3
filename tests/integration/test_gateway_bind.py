# =============================================================================
# tests/integration/test_gateway_bind.py - a Core that cannot listen, exits
# =============================================================================
#
# Found on the VPS: a hand-started Core still held port 8321 when the
# service started, uvicorn's sys.exit() escaped its task, and the daemon
# stayed "active (running)" with no gateway at all. It must instead shut
# down cleanly and exit non-zero, so systemd restarts it.
# =============================================================================

from __future__ import annotations

import asyncio
import socket
from pathlib import Path

import pytest

from jarvis.common.settings import CoreSettings
from jarvis.core.app import CoreApp


async def test_a_taken_port_stops_the_core(tmp_path: Path) -> None:
    holder = socket.socket()
    holder.bind(("127.0.0.1", 0))
    holder.listen()
    port = holder.getsockname()[1]
    try:
        app = CoreApp(CoreSettings(
            _env_file=None, data_dir=tmp_path / "data", gateway_port=port,
            gateway_token="test-token",  # type: ignore[arg-type]
            embedder_provider="none", tts_provider="none",
        ))
        with pytest.raises(SystemExit) as stopped:
            await asyncio.wait_for(app.run(), timeout=20)
        assert stopped.value.code == 1
    finally:
        holder.close()
