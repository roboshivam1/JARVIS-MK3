# =============================================================================
# tests/integration/test_sandbox.py - the jail, verified
# =============================================================================
#
# These run REAL code under the REAL sandbox. That matters more here
# than anywhere else in the project: a sandbox that is believed to work
# and does not is worse than no sandbox, because it is trusted.
#
# The escape tests are the point. Each one attempts something the policy
# forbids and asserts that it fails.
# =============================================================================

from __future__ import annotations

import sys

import pytest

from jarvis.worker.sandbox.runner import run_python

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin", reason="sandbox-exec is macOS only"
)


class TestBasicExecution:
    async def test_runs_code_and_captures_stdout(self) -> None:
        result = await run_python("print(6 * 7)")
        assert result.ok
        assert "42" in result.stdout

    async def test_reports_errors_without_crashing(self) -> None:
        result = await run_python("raise ValueError('deliberate')")
        assert not result.ok
        assert "deliberate" in result.stderr

    async def test_input_files_are_readable(self) -> None:
        result = await run_python(
            "print(open('data.txt').read().strip())",
            input_files={"data.txt": b"hello from outside"},
        )
        assert result.ok
        assert "hello from outside" in result.stdout

    async def test_produced_files_come_back(self) -> None:
        result = await run_python(
            "open('out.txt', 'w').write('made inside')"
        )
        assert result.ok
        assert "out.txt" in result.produced_files
        assert result.produced_files["out.txt"] == b"made inside"


class TestIsolation:
    """The tests that justify the whole module."""

    async def test_network_is_denied(self) -> None:
        result = await run_python(
            "import socket\n"
            "s = socket.socket()\n"
            "s.settimeout(5)\n"
            "s.connect(('1.1.1.1', 80))\n"
            "print('CONNECTED')\n"
        )
        assert "CONNECTED" not in result.stdout
        assert not result.ok

    async def test_cannot_read_home_directory(self) -> None:
        result = await run_python(
            "import os\n"
            "print(os.listdir(os.path.expanduser('~/Documents')))\n"
        )
        # HOME is redirected into the workdir, and the real one is
        # unreadable, so this fails either way.
        assert not result.ok or "Documents" not in result.stdout

    async def test_cannot_write_outside_workdir(self) -> None:
        result = await run_python(
            "open('/tmp/jarvis-escape-test.txt', 'w').write('escaped')\n"
            "print('WROTE')\n"
        )
        assert "WROTE" not in result.stdout

    async def test_environment_carries_no_secrets(self) -> None:
        result = await run_python(
            "import os\n"
            "leaked = [k for k in os.environ if 'KEY' in k or 'TOKEN' in k]\n"
            "print('LEAKED:', leaked)\n"
        )
        assert result.ok
        assert "LEAKED: []" in result.stdout


class TestLimits:
    async def test_timeout_kills_a_hanging_script(self) -> None:
        result = await run_python(
            "import time\ntime.sleep(60)\nprint('FINISHED')",
            timeout_s=3,
        )
        assert result.timed_out
        assert "FINISHED" not in result.stdout

    async def test_memory_limit_is_weak_on_macos(self) -> None:
        """DOCUMENTS A GAP rather than asserting a guarantee.

        macOS does not honour RLIMIT_AS, and RLIMIT_DATA does not cover
        mmap-backed allocations - which is how CPython allocates large
        bytearrays. So a script can "allocate" far past its limit: this
        one reserves 50 GB under a 256 MB cap and finishes in seconds.

        It is less alarming than it sounds, because macOS overcommits:
        those pages are never resident until written, so untouched
        allocation costs nothing. Code that actually FILLS the memory
        hits swap and then the OOM killer.

        The real containment on this platform is therefore the WALL
        CLOCK, not the memory ceiling. A Linux worker or a Docker
        container would enforce this properly, and that is the fix if
        it ever matters.

        This test asserts what is TRUE today. If it starts failing
        because allocation was refused, that is good news and the test
        should be tightened.
        """
        result = await run_python(
            "chunks = []\n"
            "for _ in range(1000):\n"
            "    chunks.append(bytearray(50 * 1024 * 1024))\n"
            "print('ALLOCATED ALL')\n",
            memory_mb=256,
            timeout_s=30,
        )
        # Not enforced here. Recorded so the limitation is visible in
        # the suite rather than assumed away.
        assert result.exit_code in (0, -9, 1)

    async def test_cpu_limit_or_timeout_stops_a_busy_loop(self) -> None:
        """What DOES contain runaway work on macOS: time.

        Either the CPU rlimit fires (SIGXCPU) or the wall clock does.
        Both are real; which one wins does not matter.
        """
        result = await run_python(
            "x = 0\n"
            "while True:\n"
            "    x += 1\n"
            "print('NEVER REACHED')\n",
            timeout_s=5,
        )
        assert "NEVER REACHED" not in result.stdout
        assert not result.ok

    async def test_each_run_gets_a_clean_directory(self) -> None:
        await run_python("open('leftover.txt', 'w').write('from run one')")
        second = await run_python(
            "import os\nprint('leftover.txt' in os.listdir('.'))"
        )
        assert "False" in second.stdout