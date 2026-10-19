# =============================================================================
# tests/integration/test_reels_worker.py - the Mac side of reel.make
# =============================================================================
#
# A fake `claude` executable stands in for Claude Code: it writes the files
# the real kit would and prints the same JSON report. Everything else is
# real - the subprocess, the checkpoint state machine, the uploads.
#
# The promises that matter:
#   - a draft pauses for review and never renders on its own
#   - only a decision for THIS round moves the job on; a bare re-offer
#     asks again without redoing any work
#   - notes and the render resume the same Claude session
#   - a usage limit is a retry, a missing login is final
#   - a render the GPU could not do comes back to Sir, not into a retry loop
#   - the kit folder must still hold the episode before it is revised
#   - a draft with no fresh contact sheet is incomplete and cannot render,
#     and commands the allow-list refused are named, not guessed at
#   - frames from another build never reach the render
#   - a new prop holds the render until Sir approves it; approving runs the
#     kit's command as argv, never through Claude Code; notes go back to
#     Claude Code as a revision
# =============================================================================

from __future__ import annotations

import json
import os
import stat
import time
import sys
from pathlib import Path
from typing import Any

import pytest

from jarvis.common.reels import EpisodeType, ReelBrief, ReelJobIn, ReelJobOut
from jarvis.core.queue.registry import JobContext, PausedForApproval, PermanentJobError
from jarvis.worker.reels import ReelConfig, ReelMaker, UsageLimited, register_reel_jobs
from jarvis.core.queue.registry import JobTypeRegistry

FAKE_CLAUDE = r'''#!PYTHON
import json, re, sys
from pathlib import Path

args = sys.argv[1:]
prompt = args[args.index("-p") + 1]
session = args[args.index("--resume") + 1] if "--resume" in args else "sess-1"
kit = Path.cwd()
control = json.loads((kit / "control.json").read_text()) if (kit / "control.json").exists() else {}
with open(kit / "calls.jsonl", "a") as log:
    log.write(json.dumps({"prompt": prompt, "resume": "--resume" in args,
                          "session": session, "args": args}) + "\n")

def report(structured=None, error=None, code=0, denials=None):
    out = {"type": "result", "subtype": "success" if error is None else "error",
           "is_error": error is not None, "result": error or "done",
           "session_id": session, "total_cost_usd": 0.05}
    if structured is not None:
        out["structured_output"] = structured
    if denials:
        out["permission_denials"] = denials
    print(json.dumps(out))
    sys.exit(code)

mode = control.get("mode", "ok")
if mode == "usage_limit":
    report(error="Claude AI usage limit reached|1760000000", code=1)
if mode == "logged_out":
    report(error="Invalid API key - Please run /login", code=1)

if prompt.startswith("Approved."):
    if mode == "cpu":
        report({"status": "failed", "mp4": "", "verification": "", "gpu_fallback": True})
    (kit / "out").mkdir(exist_ok=True)
    (kit / "out" / "film-v1.mp4").write_bytes(b"MP4DATA")
    report({"status": "delivered", "mp4": "out/film-v1.mp4",
            "verification": "30.2s 1080x1920 -14 LUFS ok"})

match = re.search(r'episode id "([^"]+)"', prompt)
if match:
    (kit / "episode.json").write_text(json.dumps({"id": match.group(1)}))
if mode == "no_stills":
    # The second reel's real failure: voice denied, so no build, no stills.
    report({"status": "failed", "title": "Five more minutes",
            "lines": ["At one twelve a.m., Sir said five more minutes."],
            "sheet": "out/stills/sheet.png", "problems": ["voice was blocked"]},
           denials=[{"tool_name": "Bash", "tool_use_id": "t1",
                     "tool_input": {"command": "npm run voice 2>&1 | tail -20"}}])
(kit / "out" / "stills").mkdir(parents=True, exist_ok=True)
(kit / "out" / "stills" / "sheet.png").write_bytes(b"PNG")
(kit / "out" / "voice.wav").write_bytes(b"RIFF")
title = "Five more minutes" if "notes" not in prompt else "Five more minutes (revised)"
new_props = []
if mode == "props":
    (kit / "props" / "bike").mkdir(parents=True, exist_ok=True)
    (kit / "props" / "bike" / "preview.png").write_bytes(b"BIKE")
    new_props = [{"id": "bike", "preview": "props/bike/preview.png",
                  "description": "A red road bike leaning on the workbench."}]
report({"status": "ready_for_review", "title": title,
        "lines": ["At one twelve a.m., Sir said five more minutes."],
        "sheet": "out/stills/sheet.png", "voice": "out/voice.wav", "problems": [],
        "new_props": new_props})
'''


class FakeContext:
    """A JobContext whose effects are recorded instead of sent."""

    def __init__(self, checkpoint: dict[str, Any] | None = None) -> None:
        self.checkpoints: list[dict[str, Any]] = []
        self.artifacts: dict[str, bytes] = {}
        self.notes: list[str] = []
        self.ctx = JobContext(
            job_id="01J00000000000000000000000",
            trace_id="01J00000000000000000000001",
            checkpoint=checkpoint,
            save_checkpoint=self._save,
            progress=self._progress,
            write_artifact=self._write,
            approval_granted=checkpoint is not None,   # what the queue sends
        )

    async def _save(self, state: dict[str, Any]) -> None:
        self.checkpoints.append(json.loads(json.dumps(state)))

    async def _progress(self, note: str) -> None:
        self.notes.append(note)

    async def _write(self, name: str, mime: str, content: bytes) -> str:
        self.artifacts[name] = content
        return name

    @property
    def last(self) -> dict[str, Any]:
        return self.checkpoints[-1]


@pytest.fixture
def kit(tmp_path: Path) -> Path:
    kit = tmp_path / "jarvis-reels"
    kit.mkdir()
    (kit / "package.json").write_text("{}")
    return kit


@pytest.fixture
def claude_bin(tmp_path: Path) -> str:
    # A shell wrapper, not a shebang: a shebang line cannot hold an
    # interpreter path containing a space, and "JARVIS MK3" has one.
    script = tmp_path / "bin" / "fake_claude.py"
    script.parent.mkdir()
    script.write_text(FAKE_CLAUDE)
    path = tmp_path / "bin" / "claude"
    path.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


@pytest.fixture
def maker(kit: Path, claude_bin: str) -> ReelMaker:
    return ReelMaker(ReelConfig(kit_dir=kit, claude_bin=claude_bin))


def _payload() -> ReelJobIn:
    return ReelJobIn(
        episode_id="01J0000000000000000000000E",
        kit_id="field-notes-00000e",
        brief=ReelBrief(
            episode_type=EpisodeType.FIELD_NOTES,
            premise="Sir says five more minutes and means ninety.",
            beats=["The promise", "The evidence", "The verdict"],
        ),
    )


def _calls(kit: Path) -> list[dict[str, Any]]:
    path = kit / "calls.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def _decide(checkpoint: dict[str, Any], action: str, notes: str = "") -> dict[str, Any]:
    """What the Core writes before releasing the gate."""
    return {**checkpoint, "decision": {
        "action": action, "round": checkpoint["round"], "notes": notes}}


async def test_draft_pauses_and_never_renders(maker: ReelMaker, kit: Path) -> None:
    run = FakeContext()
    with pytest.raises(PausedForApproval):
        await maker.handle(_payload(), run.ctx)

    assert run.last["stage"] == "review"
    assert run.last["round"] == 1
    assert run.last["session_id"] == "sess-1"
    assert set(run.artifacts) >= {
        "field-notes-00000e-r1-sheet.png", "field-notes-00000e-r1-episode.json"}
    assert any(name.startswith("field-notes-00000e-r1-voice") for name in run.artifacts)
    assert run.ctx.pending_approval is not None
    assert run.ctx.pending_approval["tool"] == "reel_review"
    assert "five more minutes" in run.ctx.pending_approval["detail"]

    calls = _calls(kit)
    assert len(calls) == 1
    assert "--bare" not in calls[0]["args"]
    assert "Basement Supply" in calls[0]["prompt"]          # the hard rules ride along
    assert not (kit / "out" / "film-v1.mp4").exists()


async def test_reoffer_without_decision_asks_again(maker: ReelMaker, kit: Path) -> None:
    first = FakeContext()
    with pytest.raises(PausedForApproval):
        await maker.handle(_payload(), first.ctx)

    # The queue re-offers with approval_granted=True after a crash. That
    # flag must not be enough to render.
    again = FakeContext(first.last)
    assert again.ctx.approval_granted
    with pytest.raises(PausedForApproval):
        await maker.handle(_payload(), again.ctx)
    assert len(_calls(kit)) == 1                     # no work redone
    assert not (kit / "out" / "film-v1.mp4").exists()


async def test_stale_decision_is_not_an_answer(maker: ReelMaker, kit: Path) -> None:
    first = FakeContext()
    with pytest.raises(PausedForApproval):
        await maker.handle(_payload(), first.ctx)
    stale = {**first.last, "decision": {"action": "approve", "round": 99, "notes": ""}}
    with pytest.raises(PausedForApproval):
        await maker.handle(_payload(), FakeContext(stale).ctx)
    assert len(_calls(kit)) == 1


async def test_notes_then_approve_renders_on_same_session(
    maker: ReelMaker, kit: Path
) -> None:
    first = FakeContext()
    with pytest.raises(PausedForApproval):
        await maker.handle(_payload(), first.ctx)

    second = FakeContext(_decide(first.last, "notes", "Slower on the last line."))
    with pytest.raises(PausedForApproval):
        await maker.handle(_payload(), second.ctx)
    assert second.last["round"] == 2
    assert "field-notes-00000e-r2-sheet.png" in second.artifacts

    third = FakeContext(_decide(second.last, "approve"))
    output = await maker.handle(_payload(), third.ctx)
    assert isinstance(output, ReelJobOut)
    assert output.mp4_name == "field-notes-00000e.mp4"
    assert third.artifacts["field-notes-00000e.mp4"] == b"MP4DATA"
    assert "-14 LUFS" in output.verification

    calls = _calls(kit)
    assert [c["resume"] for c in calls] == [False, True, True]
    assert {c["session"] for c in calls} == {"sess-1"}
    assert "Slower on the last line." in calls[1]["prompt"]
    assert calls[2]["prompt"].startswith("Approved.")


async def test_usage_limit_is_a_retry(maker: ReelMaker, kit: Path) -> None:
    (kit / "control.json").write_text(json.dumps({"mode": "usage_limit"}))
    run = FakeContext()
    with pytest.raises(UsageLimited):
        await maker.handle(_payload(), run.ctx)
    # The session was kept, so the retry resumes rather than restarts.
    assert run.last["session_id"] == "sess-1"
    assert not isinstance(UsageLimited("x"), PermanentJobError)


async def test_logged_out_is_final(maker: ReelMaker, kit: Path) -> None:
    (kit / "control.json").write_text(json.dumps({"mode": "logged_out"}))
    with pytest.raises(PermanentJobError, match="not logged in"):
        await maker.handle(_payload(), FakeContext().ctx)


async def test_cpu_fallback_comes_back_to_sir(maker: ReelMaker, kit: Path) -> None:
    first = FakeContext()
    with pytest.raises(PausedForApproval):
        await maker.handle(_payload(), first.ctx)

    (kit / "control.json").write_text(json.dumps({"mode": "cpu"}))
    second = FakeContext(_decide(first.last, "approve"))
    with pytest.raises(PausedForApproval):
        await maker.handle(_payload(), second.ctx)
    assert second.last["kind"] == "render_failed"
    assert second.last["round"] == 2
    assert "CPU" in second.last["render_error"]
    assert second.ctx.pending_approval is not None
    assert "render failed" in second.ctx.pending_approval["summary"]

    # Approving a failed render runs the render again.
    (kit / "control.json").write_text(json.dumps({"mode": "ok"}))
    third = FakeContext(_decide(second.last, "approve"))
    output = await maker.handle(_payload(), third.ctx)
    assert isinstance(output, ReelJobOut)


async def test_kit_must_still_hold_the_episode(maker: ReelMaker, kit: Path) -> None:
    first = FakeContext()
    with pytest.raises(PausedForApproval):
        await maker.handle(_payload(), first.ctx)

    (kit / "episode.json").write_text(json.dumps({"id": "someone-else"}))
    with pytest.raises(PermanentJobError, match="someone-else"):
        await maker.handle(_payload(), FakeContext(_decide(first.last, "approve")).ctx)


async def test_paths_outside_the_kit_are_ignored(maker: ReelMaker, kit: Path) -> None:
    assert maker._inside_kit("/etc/passwd") is None
    assert maker._inside_kit("../outside.png") is None
    assert maker._inside_kit("out/stills/sheet.png") == (
        kit / "out" / "stills" / "sheet.png").resolve()


def test_registration_needs_kit_and_claude(kit: Path, claude_bin: str, tmp_path: Path) -> None:
    assert not register_reel_jobs(
        JobTypeRegistry(), ReelConfig(kit_dir=tmp_path / "nope", claude_bin=claude_bin),
        ["render", "gpu"])
    assert not register_reel_jobs(
        JobTypeRegistry(), ReelConfig(kit_dir=kit, claude_bin="no-such-claude-bin"),
        ["render", "gpu"])
    registry = JobTypeRegistry()
    assert register_reel_jobs(registry, ReelConfig(kit_dir=kit, claude_bin=claude_bin),
                              ["macos", "render", "gpu"])
    spec = registry.get("reel.make")
    assert spec is not None and spec.handler is not None


async def test_incomplete_draft_cannot_render(maker: ReelMaker, kit: Path) -> None:
    # A contact sheet from an earlier episode is sitting in the kit.
    old = kit / "out" / "stills" / "sheet.png"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"OLD")
    an_hour_ago = time.time() - 3600
    os.utime(old, (an_hour_ago, an_hour_ago))

    (kit / "control.json").write_text(json.dumps({"mode": "no_stills"}))
    first = FakeContext()
    with pytest.raises(PausedForApproval):
        await maker.handle(_payload(), first.ctx)

    assert first.last["kind"] == "incomplete"
    assert not any(name.endswith("-sheet.png") for name in first.artifacts)
    problems = first.last["draft"]["problems"]
    assert any("`npm run voice 2>&1 | tail -20`" in p for p in problems)
    assert any("No new contact sheet" in p for p in problems)
    assert first.ctx.pending_approval is not None
    assert "incomplete" in first.ctx.pending_approval["summary"]

    call = _calls(kit)[0]
    allowed = call["args"][call["args"].index("--allowedTools") + 1]
    assert "Bash(tail *)" in allowed and "Bash(npm run *)" in allowed
    assert "`npm run <script>`" in call["prompt"]

    # Even a forced approve finishes the draft instead of rendering.
    (kit / "control.json").write_text(json.dumps({"mode": "ok"}))
    second = FakeContext(_decide(first.last, "approve"))
    with pytest.raises(PausedForApproval):
        await maker.handle(_payload(), second.ctx)
    assert second.last["kind"] == "draft" and second.last["round"] == 2
    assert "field-notes-00000e-r2-sheet.png" in second.artifacts
    assert "Finish the draft" in _calls(kit)[1]["prompt"]
    assert not (kit / "out" / "film-v1.mp4").exists()


async def test_frames_from_another_build_are_cleared(maker: ReelMaker, kit: Path) -> None:
    out = kit / "out"
    frames = out / "frames"
    frames.mkdir(parents=True)
    (frames / "f0000.png").write_bytes(b"old film")
    (out / "film.js").write_text("this episode")

    maker._prepare_frames()                       # no record of the build: cleared
    assert not frames.exists()

    frames.mkdir()
    (frames / "f0000.png").write_bytes(b"this film")
    maker._prepare_frames()                       # same build: a resume keeps them
    assert (frames / "f0000.png").read_bytes() == b"this film"

    (out / "film.js").write_text("a revised build")
    maker._prepare_frames()                       # rebuilt since: cleared again
    assert not frames.exists()



FAKE_PROP = r"""
import json, sys
from pathlib import Path
kit = Path.cwd()
with open(kit / "prop_calls.jsonl", "a") as log:
    log.write(json.dumps(sys.argv[1:]) + "\n")
if (kit / "prop_fail").exists():
    print("prop bike: preview.png missing")
    sys.exit(1)
print(f"prop {sys.argv[2]}: {sys.argv[1]}d")
"""


@pytest.fixture
def prop_maker(kit: Path, claude_bin: str, tmp_path: Path) -> ReelMaker:
    script = tmp_path / "bin" / "fake_prop.py"
    script.write_text(FAKE_PROP)
    return ReelMaker(ReelConfig(kit_dir=kit, claude_bin=claude_bin,
                                prop_command=(sys.executable, str(script))))


def _prop_calls(kit: Path) -> list[list[str]]:
    path = kit / "prop_calls.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _decide_prop(checkpoint: dict[str, Any], action: str, prop_id: str,
                 notes: str = "") -> dict[str, Any]:
    return {**checkpoint, "decision": {"action": action, "round": checkpoint["round"],
                                       "notes": notes, "prop_id": prop_id}}


async def test_a_new_prop_holds_the_render(prop_maker: ReelMaker, kit: Path) -> None:
    (kit / "control.json").write_text(json.dumps({"mode": "props"}))
    first = FakeContext()
    with pytest.raises(PausedForApproval):
        await prop_maker.handle(_payload(), first.ctx)
    assert first.last["props"] == [{"id": "bike", "status": "draft", "notes": "",
                                    "description": "A red road bike leaning on the workbench."}]
    assert first.artifacts["field-notes-00000e-r1-prop-bike.png"] == b"BIKE"
    args = _calls(kit)[0]["args"]
    denied = args[args.index("--disallowedTools") + 1]
    assert "Bash(npm run prop -- approve*)" in denied
    assert "Bash(cp *)" in args[args.index("--allowedTools") + 1]
    assert "Do not approve or reject props" in _calls(kit)[0]["prompt"]

    # Approving the draft first changes nothing but a notice.
    held = FakeContext(_decide(first.last, "approve"))
    with pytest.raises(PausedForApproval):
        await prop_maker.handle(_payload(), held.ctx)
    assert "waiting for your approval: bike" in held.last["notice"]
    assert held.last["round"] == 1 and len(_calls(kit)) == 1

    approved = FakeContext(_decide_prop(held.last, "prop_approve", "bike"))
    with pytest.raises(PausedForApproval):
        await prop_maker.handle(_payload(), approved.ctx)
    assert _prop_calls(kit) == [["approve", "bike"]]
    assert approved.last["props"][0]["status"] == "approved"
    assert "Every prop is approved" in approved.last["notice"]
    assert approved.last["round"] == 1                 # no new draft
    assert len(_calls(kit)) == 1                       # Claude Code was not involved

    output = await prop_maker.handle(_payload(), FakeContext(_decide(approved.last, "approve")).ctx)
    assert isinstance(output, ReelJobOut)


async def test_prop_notes_go_back_to_claude(prop_maker: ReelMaker, kit: Path) -> None:
    (kit / "control.json").write_text(json.dumps({"mode": "props"}))
    first = FakeContext()
    with pytest.raises(PausedForApproval):
        await prop_maker.handle(_payload(), first.ctx)

    notes = 'Make it red; it is "too" shiny && rm -rf /'
    second = FakeContext(_decide_prop(first.last, "prop_notes", "bike", notes))
    with pytest.raises(PausedForApproval):
        await prop_maker.handle(_payload(), second.ctx)
    assert _prop_calls(kit) == [["reject", "bike", notes]]      # one argv element
    assert "Sir rejected prop bike" in _calls(kit)[1]["prompt"]
    assert second.last["round"] == 2
    assert second.last["props"][0]["status"] == "draft"          # listed again: back to draft


async def test_bad_prop_ids_and_failures_never_run_blind(prop_maker: ReelMaker, kit: Path) -> None:
    (kit / "control.json").write_text(json.dumps({"mode": "props"}))
    first = FakeContext()
    with pytest.raises(PausedForApproval):
        await prop_maker.handle(_payload(), first.ctx)

    sneaky = FakeContext(_decide_prop(first.last, "prop_approve", "../bike"))
    with pytest.raises(PausedForApproval):
        await prop_maker.handle(_payload(), sneaky.ctx)
    assert _prop_calls(kit) == []
    assert "no prop" in sneaky.last["notice"]

    (kit / "prop_fail").write_text("")
    failed = FakeContext(_decide_prop(sneaky.last, "prop_approve", "bike"))
    with pytest.raises(PausedForApproval):
        await prop_maker.handle(_payload(), failed.ctx)
    assert "failed: prop bike: preview.png missing" in failed.last["notice"]
    assert failed.last["props"][0]["status"] == "draft"
