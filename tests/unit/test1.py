import asyncio
import subprocess
import sys
import tempfile
from pathlib import Path

from jarvis.worker.sandbox.policy import build_policy, interpreter_read_paths

workdir = Path(tempfile.mkdtemp(prefix="probe-"))
policy = build_policy(workdir)
policy_path = workdir.parent / "probe.sb"
policy_path.write_text(policy)

script = workdir / "t.py"
script.write_text('print("HELLO")')

print("=== sys.executable ===")
print(sys.executable)
print("\n=== interpreter_read_paths() ===")
for p in interpreter_read_paths():
    print(" ", p)
print("\n=== policy ===")
print(policy)
print("=== running (no capture, so errors show) ===")
r = subprocess.run(
    ["sandbox-exec", "-f", str(policy_path), sys.executable, str(script)],
    cwd=workdir,
)
print("exit code:", r.returncode)