# =============================================================================
# src/jarvis/worker/sandbox/runner.py - running code in the jail
# =============================================================================
#
# One function that matters: run_python(). Give it code and optional
# input files, get back stdout, stderr, exit status, and whatever files
# the code produced.
#
# THE LIFECYCLE, and why each step exists:
#
#   1. Fresh temp directory. Not a reused one - a previous run's files
#      are a channel between two things that should not know about each
#      other, and a source of confusing results.
#   2. Inputs copied IN explicitly. The sandbox sees an empty room plus
#      exactly what it was handed. Analysing a CSV means giving it that
#      CSV, not giving it the owner's Documents folder.
#   3. Execute under sandbox-exec with resource limits.
#   4. Outputs copied OUT: any file the code created.
#   5. Directory destroyed.
#
# RESOURCE LIMITS sit inside the sandbox, not instead of it: the kernel
# policy governs ACCESS, setrlimit governs CONSUMPTION, and a wall-clock
# timeout governs the case where the process is neither accessing nor
# consuming, merely stuck.
# =============================================================================

from __future__ import annotations

import asyncio
import resource
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from jarvis.common.log import get_logger
from jarvis.worker.sandbox.policy import write_policy

log = get_logger("worker.sandbox")

DEFAULT_TIMEOUT_S = 60
DEFAULT_MEMORY_MB = 2048
DEFAULT_CPU_S = 50            # below the wall clock, so CPU-bound loops
                              # die by limit rather than by timeout
MAX_OUTPUT_CHARS = 20_000     # what comes back to the model


@dataclass
class SandboxResult:
    """What one run produced."""

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False
    produced_files: dict[str, bytes] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def summary(self) -> str:
        """The text the agent reads. Truncated from the MIDDLE when
        long: the start says what happened and the end usually holds the
        error, so cutting the middle keeps both."""
        parts: list[str] = []
        if self.timed_out:
            parts.append("[timed out]")
        if self.stdout:
            parts.append(f"stdout:\n{_truncate(self.stdout)}")
        if self.stderr:
            parts.append(f"stderr:\n{_truncate(self.stderr)}")
        if self.produced_files:
            names = ", ".join(sorted(self.produced_files))
            parts.append(f"files produced: {names}")
        if not parts:
            parts.append("(no output)")
        parts.append(f"exit code: {self.exit_code}")
        return "\n\n".join(parts)


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    half = MAX_OUTPUT_CHARS // 2
    return (
        f"{text[:half]}\n\n[... {len(text) - MAX_OUTPUT_CHARS} characters "
        f"cut from the middle ...]\n\n{text[-half:]}"
    )


def _apply_limits(memory_mb: int, cpu_s: int) -> None:
    """Set resource ceilings in the child, just before exec.

    BEST EFFORT, deliberately. This runs inside the forked child, where
    ANY exception aborts the spawn entirely - so a limit the platform
    refuses would take the whole sandbox down rather than merely going
    unenforced. macOS is exactly that case: RLIMIT_AS is not honoured
    there the way it is on Linux, and setting it raises.

    Each limit is attempted independently and a refusal is skipped. The
    kernel sandbox is the real containment; these are a second layer
    against runaway consumption, and a partial second layer beats a
    sandbox that will not start.
    """
    limit_bytes = memory_mb * 1024 * 1024

    for limit_name, value in (
        ("RLIMIT_AS", (limit_bytes, limit_bytes)),     # Linux: address space
        ("RLIMIT_DATA", (limit_bytes, limit_bytes)),   # macOS: heap
        ("RLIMIT_CPU", (cpu_s, cpu_s)),
        ("RLIMIT_CORE", (0, 0)),                       # no gigabyte core dumps
    ):
        limit = getattr(resource, limit_name, None)
        if limit is None:
            continue
        try:
            resource.setrlimit(limit, value)
        except (ValueError, OSError):
            pass


async def run_python(
    code: str,
    input_files: dict[str, bytes] | None = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    memory_mb: int = DEFAULT_MEMORY_MB,
    allow_network: bool = False,
) -> SandboxResult:
    """Execute Python in a kernel-sandboxed, disposable directory."""
    # .resolve() is load-bearing on macOS: mkdtemp returns a path under
    # /var/folders, but /var is a SYMLINK to /private/var. The kernel
    # evaluates real paths, so a policy naming the symlinked form
    # matches nothing - the workdir rules silently apply to no
    # directory, and the sandbox cannot open its own script:
    #
    #     can't open file '.../_run.py': [Errno 1] Operation not permitted
    workdir = Path(tempfile.mkdtemp(prefix="jarvis-sandbox-")).resolve()
    policy_path = write_policy(workdir, allow_network=allow_network)

    try:
        for name, content in (input_files or {}).items():
            safe_name = Path(name).name       # a name, never a path
            (workdir / safe_name).write_bytes(content)

        before = {p.name for p in workdir.iterdir()}
        script = workdir / "_run.py"
        script.write_text(code, encoding="utf-8")
        before.add("_run.py")

        process = await asyncio.create_subprocess_exec(
            "sandbox-exec", "-f", str(policy_path),
            sys.executable, str(script),
            cwd=workdir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            # A scrubbed environment: no API keys, no tokens, no paths
            # into the owner's home. Whatever the code learns, it learns
            # from its inputs.
            env={
                "PATH": "/usr/bin:/bin",
                "HOME": str(workdir),
                "TMPDIR": str(workdir),
                "PYTHONDONTWRITEBYTECODE": "1",
                "MPLBACKEND": "Agg",       # matplotlib without a display
            },
            preexec_fn=lambda: _apply_limits(memory_mb, DEFAULT_CPU_S),
        )

        timed_out = False
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout_s
            )
        except asyncio.TimeoutError:
            timed_out = True
            process.kill()
            stdout, stderr = await process.communicate()

        produced: dict[str, bytes] = {}
        for path in workdir.iterdir():
            if path.name in before or not path.is_file():
                continue
            produced[path.name] = path.read_bytes()

        result = SandboxResult(
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
            exit_code=process.returncode if process.returncode is not None else -1,
            timed_out=timed_out,
            produced_files=produced,
        )
        log.info("sandbox run complete", extra={
            "exit_code": result.exit_code,
            "timed_out": timed_out,
            "files_produced": len(produced),
            "network": allow_network,
        })
        return result

    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        policy_path.unlink(missing_ok=True)