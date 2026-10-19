# =============================================================================
# src/jarvis/worker/reels.py - the reel.make handler, on the Mac
# =============================================================================
#
# Drives the reel kit (~/Development/jarvis-reels) through Claude Code in
# headless mode, on the owner's Claude subscription. Claude Code runs
# inside the kit folder, so it loads the kit's CLAUDE.md and the
# jarvis-reel skill on its own; this file only says WHAT to do next and
# reads back what was done.
#
# The job pauses for the owner between drafting and rendering:
#
#   drafting  -> claude -p (new session): episode.json, voice, build, stills
#                upload sheet, voice, episode.json      -> pause for review
#   revising  -> claude -p --resume: apply Sir's notes  -> pause for review
#   rendering -> claude -p --resume: render, mux        -> upload the mp4, done
#
# A draft that stops short - no fresh contact sheet - goes to review as
# INCOMPLETE: there is nothing to render, so the owner gets Notes and
# Cancel but no Approve. Old frames are cleared before every render that
# is not a resume of the same build, because the renderer skips frames
# that already exist and would splice a previous film into this one.
#
# WHAT THIS FILE TRUSTS: only the checkpoint. The queue marks a re-offer
# as approved whenever a checkpoint exists, crash or not, so the
# approval_granted flag is ignored here. The Core writes the owner's real
# answer into checkpoint.decision, tagged with the round it answers, and
# anything else means "nobody has answered yet - ask again".
#
# PROPS (kit v4). When a script needs an object the Basement lacks, Claude
# Code builds it as a draft prop, and the kit refuses to render until Sir
# approves it. Approval is a command only the worker runs, as a plain
# subprocess, when Sir taps the button - Claude Code is denied it. Each
# approval re-pauses the job without a new draft; a rejection goes back
# to Claude Code as notes.
#
# WHAT IT NEVER DOES: creative work. Script, staging and fixing stills
# all happen inside Claude Code, which has the bible and the skill.
# =============================================================================

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import signal
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from jarvis.common.capabilities import Gate
from jarvis.common.log import get_logger
from jarvis.common.reels import (
    HARD_RULES,
    PHASE_A_SCHEMA,
    PHASE_B_SCHEMA,
    PROP_ID,
    REEL_JOB_TYPE,
    REEL_REQUIRES,
    REEL_REVIEW_TOOL,
    PhaseAOutput,
    PhaseBOutput,
    PropState,
    ReelCheckpoint,
    ReelJobIn,
    ReelJobOut,
)
from jarvis.core.queue.registry import (
    JobContext,
    JobTypeRegistry,
    JobTypeSpec,
    PausedForApproval,
    PermanentJobError,
)
from jarvis.jobs.reels import reel_job_spec

log = get_logger("worker.reels")

ACTOR = "worker.reels"

# What Claude Code may run inside the kit. Headless, it cannot ask, so
# anything not listed is denied and the run carries on without it.
# Pipes and && are checked command by command: the read-only helpers are
# what lets `npm run voice 2>&1 | tail -20` through.
_READ_HELPERS = [
    "Bash(ls *)", "Bash(cat *)", "Bash(head *)", "Bash(tail *)",
    "Bash(wc *)", "Bash(grep *)", "Bash(ffprobe *)",
]
_DRAFT_TOOLS = ",".join([
    "Read", "Edit", "Write", "Glob", "Grep",
    "Bash(npm run *)", "Bash(node *)", "Bash(cp *)", *_READ_HELPERS,
])
_RENDER_TOOLS = ",".join([
    "Read", "Glob", "Grep",
    "Bash(npm run *)", "Bash(node *)", "Bash(rm -rf out/frames)", *_READ_HELPERS,
])

# Prop approval is Sir's alone. Deny rules win over allow rules, so the
# broad `npm run *` above can stay. A guardrail against a confused
# model, not a lock: anything that can write files could edit a prop's
# meta.json. The prompt says so too.
_DENIED_TOOLS = ",".join(
    f"Bash({form} {verb}*)"
    for form in ("npm run prop --", "npm run prop", "node tools/prop.mjs",
                 "node ./tools/prop.mjs")
    for verb in ("approve", "reject")
)

# A long script takes the kit's spaced-out voice requests longer: about
# seven seconds a line on top of the base timeout.
_VOICE_S_PER_LINE = 7
_LINES_GUESS = 15

# Said in every prompt, because a denied command is silent: the plain
# forms are the ones the allow-list matches.
_COMMANDS = (
    "You are running unattended and cannot ask for permission. Run scripts "
    "in their plain form - `npm run <script>` or `node <file>` - with no "
    "`cd`, no `&&` chains and no environment variables in front; piping "
    "into head or tail is fine. Anything else is denied. If a command is "
    "denied anyway, report the exact command as a problem."
)

_PROPS = (
    "If the script needs an object that is not in the Basement or the props "
    "library, make it as a prop following PROPS.md, and list it in new_props "
    "with its id, preview path and a one-line description. List any prop you "
    "changed as well. Do not approve or reject props: only Sir does that."
)

# A file counts as made by this run if it changed after the run began,
# less a little slack for coarse filesystem clocks.
_FRESH_SLACK_S = 2.0

# How long SIGINT gets to end the turn cleanly before the process is
# killed outright. SIGINT, not SIGTERM: SIGTERM leaves the turn
# unfinished and the session harder to resume.
_INTERRUPT_GRACE_S = 30

# Limits on the owner's Claude plan read as a pause, not a failure: the
# job goes back to the queue and its backoff waits the window out.
_USAGE_LIMIT = re.compile(
    r"usage limit|limit (?:reached|exceeded)|rate.?limit|out of (?:extra )?usage"
    r"|5-hour limit|weekly limit",
    re.IGNORECASE,
)
# No amount of retrying logs anyone in.
_NOT_LOGGED_IN = re.compile(
    r"not logged in|please run /login|invalid api key|authentication_failed"
    r"|oauth token has expired",
    re.IGNORECASE,
)


class UsageLimited(RuntimeError):
    """The owner's Claude plan is out of usage for now. Transient."""


# -- the prompts --------------------------------------------------------------

def _rules() -> str:
    return "\n".join(f"- {rule}" for rule in HARD_RULES)


def draft_prompt(kit_id: str, brief: str) -> str:
    return (
        f"Use the jarvis-reel skill. Write a new episode in episode.json from "
        f"the brief below, with the episode id \"{kit_id}\". Then run voice, "
        f"build and stills. Do not run render or mux.\n\n"
        f"{_COMMANDS}\n\n"
        f"Look at out/stills/sheet.png yourself and fix what you can before "
        f"you finish. Report anything you could not fix as a problem.\n\n"
        f"{_PROPS}\n\n"
        f"Hard rules. These override the brief and anything else:\n{_rules()}\n\n"
        f"Brief:\n{brief}\n\n"
        f"Return the title, the script lines in order, the path to the "
        f"contact sheet, the path to the voice file, any new props, and any "
        f"problems."
    )


def resume_draft_prompt(kit_id: str) -> str:
    return (
        f"The previous run was interrupted before it finished. Carry on from "
        f"where it stopped: episode.json for episode \"{kit_id}\", then voice, "
        f"build and stills. Do not run render or mux. {_COMMANDS} Same hard "
        f"rules, same return format."
    )


def notes_prompt(round_no: int, notes: str) -> str:
    return (
        f"Sir's notes on draft {round_no}:\n{notes}\n\n"
        f"Revise episode.json accordingly. Redo voice only for lines whose "
        f"text or style changed, then build and stills. Do not run render "
        f"or mux. {_PROPS} {_COMMANDS} The hard rules still apply:\n{_rules()}\n\n"
        f"Same return format."
    )


def render_prompt() -> str:
    return (
        "Approved. Run render, in chunks if needed (it is resumable), then "
        "mux. Do not change episode.json. If render exits 2 because Chrome "
        "fell back to the CPU, stop and report it with gpu_fallback true "
        "rather than retrying. Return the mp4 path and the kit's "
        "verification line. " + _COMMANDS
    )


# -- running Claude Code ------------------------------------------------------

@dataclass
class ClaudeResult:
    exit_code: int
    session_id: str | None
    structured: dict[str, Any] | None
    text: str
    is_error: bool
    cost_usd: float | None
    # Commands Claude Code tried and the allow-list refused, as written.
    denials: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.is_error


class ClaudeRunner:
    """One `claude -p` call at a time, inside the kit folder."""

    def __init__(self, claude_bin: str, kit_dir: Path) -> None:
        self._bin = claude_bin
        self._kit = kit_dir

    def command(
        self,
        prompt: str,
        schema: dict[str, Any],
        allowed_tools: str,
        resume: str | None,
    ) -> list[str]:
        # NOT --bare: bare mode skips the subscription login, the kit's
        # CLAUDE.md and the jarvis-reel skill - everything that matters.
        args = [
            self._bin, "-p", prompt,
            "--output-format", "json",
            "--json-schema", json.dumps(schema),
            "--permission-mode", "acceptEdits",
            "--permission-prompts", "none",
            "--allowedTools", allowed_tools,
            "--disallowedTools", _DENIED_TOOLS,
        ]
        if resume:
            args += ["--resume", resume]
        return args

    async def run(
        self,
        prompt: str,
        schema: dict[str, Any],
        allowed_tools: str,
        resume: str | None,
        timeout_s: int,
    ) -> ClaudeResult:
        process = await asyncio.create_subprocess_exec(
            *self.command(prompt, schema, allowed_tools, resume),
            cwd=self._kit,
            env=_subscription_env(),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout_s
            )
        except asyncio.CancelledError:
            await _interrupt(process)
            raise
        except asyncio.TimeoutError:
            await _interrupt(process)
            # Transient: the checkpoint holds the session, so the retry
            # resumes the same conversation instead of starting over.
            raise RuntimeError(
                f"Claude Code took longer than {timeout_s}s and was interrupted"
            ) from None
        return _parse_result(process.returncode or 0, stdout, stderr)


def _subscription_env() -> dict[str, str]:
    """The environment for Claude Code, minus anything that would switch
    it from the owner's subscription to pay-per-token API billing."""
    env = dict(os.environ)
    for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        env.pop(key, None)
    return env


async def _interrupt(process: asyncio.subprocess.Process) -> None:
    """End the turn cleanly, so the session can be resumed later."""
    if process.returncode is not None:
        return
    try:
        process.send_signal(signal.SIGINT)
        await asyncio.wait_for(process.wait(), timeout=_INTERRUPT_GRACE_S)
    except (asyncio.TimeoutError, ProcessLookupError):
        try:
            process.kill()
        except ProcessLookupError:
            pass


def _parse_result(exit_code: int, stdout: bytes, stderr: bytes) -> ClaudeResult:
    """Read Claude Code's JSON report. Tolerates stray lines before it."""
    text_out = stdout.decode(errors="replace").strip()
    data: dict[str, Any] = {}
    for line in reversed(text_out.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                data = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    if not data and text_out.startswith("{"):
        try:
            data = json.loads(text_out)
        except json.JSONDecodeError:
            data = {}

    result_text = str(data.get("result") or "")
    if not result_text:
        result_text = (text_out or stderr.decode(errors="replace")).strip()
    structured = data.get("structured_output")
    cost = data.get("total_cost_usd")
    return ClaudeResult(
        exit_code=exit_code,
        session_id=data.get("session_id"),
        structured=structured if isinstance(structured, dict) else None,
        text=result_text[:4000],
        is_error=bool(data.get("is_error", exit_code != 0)),
        cost_usd=float(cost) if isinstance(cost, (int, float)) else None,
        denials=_denials(data.get("permission_denials")),
    )


def _denials(raw: Any) -> list[str]:
    """Claude Code's permission_denials, as the commands it tried."""
    found: list[str] = []
    if not isinstance(raw, list):
        return found
    for item in raw:
        if not isinstance(item, dict):
            continue
        tool = str(item.get("tool_name") or "tool")
        args = item.get("tool_input")
        if tool == "Bash" and isinstance(args, dict) and args.get("command"):
            text = str(args["command"])
        else:
            text = f"{tool} {json.dumps(args)[:120]}" if args else tool
        text = " ".join(text.split())[:160]
        if text not in found:
            found.append(text)
    return found[:6]


def _denied_problem(denials: list[str]) -> str:
    return "Denied by the allow-list: " + "; ".join(f"`{d}`" for d in denials)


def _raise_for(result: ClaudeResult, phase: str) -> None:
    """Turn a failed run into the right kind of failure."""
    if result.ok and result.structured is not None:
        return
    detail = result.text or f"exit code {result.exit_code}"
    if result.denials:
        detail += f" ({_denied_problem(result.denials)})"
    if _USAGE_LIMIT.search(detail):
        raise UsageLimited(
            f"Claude usage limit reached during {phase} - the job will retry "
            f"after its backoff. {detail[:200]}"
        )
    if _NOT_LOGGED_IN.search(detail):
        raise PermanentJobError(
            f"Claude Code is not logged in on this Mac. Run `claude` once in a "
            f"terminal and log in, then ask again. ({detail[:200]})"
        )
    if result.ok:
        raise RuntimeError(f"{phase}: Claude Code returned no structured output")
    raise RuntimeError(f"{phase} failed: {detail[:500]}")


# -- the handler --------------------------------------------------------------

@dataclass
class ReelConfig:
    kit_dir: Path
    claude_bin: str = "claude"
    draft_timeout_s: int = 1500
    render_timeout_s: int = 2700
    # How the worker runs the kit's prop command. Overridable for tests.
    prop_command: tuple[str, ...] = ("npm", "run", "prop", "--")


class ReelMaker:
    """The state machine. One instance per worker; one reel at a time,
    because the kit has a single episode.json and a single out/."""

    def __init__(self, config: ReelConfig, runner: ClaudeRunner | None = None) -> None:
        self._config = config
        self._kit = config.kit_dir
        self._claude = runner or ClaudeRunner(config.claude_bin, config.kit_dir)
        self._lock = asyncio.Lock()

    async def handle(self, payload: BaseModel, ctx: JobContext) -> BaseModel:
        assert isinstance(payload, ReelJobIn)
        async with self._lock:
            return await self._step(payload, ctx)

    async def _step(self, payload: ReelJobIn, ctx: JobContext) -> ReelJobOut:
        try:
            checkpoint = (
                ReelCheckpoint.model_validate(ctx.checkpoint)
                if ctx.checkpoint else
                ReelCheckpoint(stage="drafting", kit_id=payload.kit_id)
            )
        except ValidationError as exc:
            raise PermanentJobError(f"unreadable reel checkpoint: {exc}") from exc
        if checkpoint.kit_id != payload.kit_id:
            raise PermanentJobError("checkpoint belongs to a different episode")

        if checkpoint.stage == "review":
            decision = checkpoint.decision
            if decision is None or decision.round != checkpoint.round:
                # Re-offered without an answer for THIS round - a crash, a
                # tap on a generic button, a stale decision. Ask again
                # rather than guess; redo nothing.
                raise await self._pause(ctx, checkpoint)
            checkpoint = checkpoint.model_copy(update={"notice": ""})
            if decision.action == "prop_approve":
                raise await self._pause(ctx, await self._approve_prop(checkpoint, decision.prop_id))
            if decision.action == "prop_notes":
                checkpoint = await self._reject_prop(checkpoint, decision.prop_id, decision.notes)
                if checkpoint.stage == "review":
                    raise await self._pause(ctx, checkpoint)
            elif decision.action == "notes":
                checkpoint = checkpoint.model_copy(update={
                    "stage": "revising", "notes": decision.notes, "decision": None,
                })
            elif checkpoint.kind == "incomplete":
                # Nothing to render. The Core never sends this; if it
                # arrives anyway, finishing the draft is the only sane
                # reading of "go ahead".
                checkpoint = checkpoint.model_copy(update={
                    "stage": "revising", "decision": None,
                    "notes": "Finish the draft: voice, build and stills, then "
                             "check the contact sheet.",
                })
            elif checkpoint.pending_props:
                # The Core refuses this; the kit would too. Say so.
                raise await self._pause(ctx, checkpoint.model_copy(update={
                    "decision": None,
                    "notice": "Props still waiting for your approval: "
                              + ", ".join(checkpoint.pending_props),
                }))
            else:
                checkpoint = checkpoint.model_copy(update={
                    "stage": "rendering", "decision": None, "render_error": "",
                })
            await ctx.save_checkpoint(checkpoint.model_dump(mode="json"))

        if checkpoint.stage in ("drafting", "revising"):
            checkpoint = await self._draft(payload, ctx, checkpoint)
            raise await self._pause(ctx, checkpoint)

        return await self._render(payload, ctx, checkpoint)

    # -- drafting -------------------------------------------------------------

    async def _draft(
        self, payload: ReelJobIn, ctx: JobContext, checkpoint: ReelCheckpoint
    ) -> ReelCheckpoint:
        if checkpoint.stage == "revising":
            self._check_kit_holds(payload.kit_id)
            prompt = notes_prompt(checkpoint.round, checkpoint.notes)
            await ctx.progress(f"revising draft {checkpoint.round} on Sir's notes")
        elif checkpoint.session_id:
            prompt = resume_draft_prompt(payload.kit_id)
            await ctx.progress("resuming the interrupted draft")
        else:
            prompt = draft_prompt(payload.kit_id, payload.brief.as_text())
            await ctx.progress("drafting: script, voice and stills")

        started = time.time()
        expected_lines = len(checkpoint.draft.lines) if checkpoint.draft else 0
        result = await self._claude.run(
            prompt, PHASE_A_SCHEMA, _DRAFT_TOOLS,
            resume=checkpoint.session_id,
            timeout_s=self._config.draft_timeout_s
            + _VOICE_S_PER_LINE * (expected_lines or _LINES_GUESS),
        )
        if result.session_id and result.session_id != checkpoint.session_id:
            # Saved at once: a retry must resume this session, not start
            # a second one that knows nothing of the first.
            checkpoint = checkpoint.model_copy(update={"session_id": result.session_id})
            await ctx.save_checkpoint(checkpoint.model_dump(mode="json"))
        _raise_for(result, "drafting")
        assert result.structured is not None

        try:
            draft = PhaseAOutput.model_validate(result.structured)
        except ValidationError as exc:
            raise RuntimeError(f"drafting: malformed result: {exc}") from exc

        round_no = checkpoint.round + 1
        problems = list(draft.problems)
        if result.denials:
            problems.append(_denied_problem(result.denials))
        actual_id = self._kit_episode_id()
        if actual_id is not None and actual_id != payload.kit_id:
            problems.append(
                f"episode.json carries id {actual_id!r}, expected {payload.kit_id!r}"
            )

        props = {prop.id: prop for prop in checkpoint.props}
        for new in draft.new_props:
            if not PROP_ID.match(new.id):
                problems.append(f"Claude Code named a prop {new.id!r}, which is not a valid id; ignored.")
                continue
            # New or changed: either way the kit wants it approved again.
            props[new.id] = PropState(id=new.id, description=new.description[:300])

        await ctx.progress(f"draft {round_no} ready - sending for review")
        has_sheet = await self._upload_draft(ctx, payload.kit_id, round_no, draft, started)
        for new in draft.new_props:
            if new.id in props:
                await self._upload_prop(ctx, f"{payload.kit_id}-r{round_no}", new.id, new.preview)
        complete = has_sheet and draft.status == "ready_for_review"
        if not has_sheet:
            problems.append(
                "No new contact sheet was made this round, so there is "
                "nothing to render yet."
            )

        return checkpoint.model_copy(update={
            "stage": "review", "round": round_no,
            "kind": "draft" if complete else "incomplete",
            "draft": draft.model_copy(update={"problems": problems}),
            "decision": None, "notes": "", "render_error": "",
            "props": list(props.values()), "notice": "",
        })

    async def _upload_prop(self, ctx: JobContext, stem: str, prop_id: str, reported: str) -> None:
        preview = self._inside_kit(reported) or self._kit / "props" / prop_id / "preview.png"
        if preview.is_file():
            await ctx.write_artifact(f"{stem}-prop-{prop_id}.png", "image/png",
                                     preview.read_bytes())

    # -- props ----------------------------------------------------------------

    async def _approve_prop(self, checkpoint: ReelCheckpoint, prop_id: str) -> ReelCheckpoint:
        """Run the kit's approval for one prop. Same round, no new draft:
        the notice is what Sir sees."""
        if not PROP_ID.match(prop_id) or prop_id not in {p.id for p in checkpoint.props}:
            return checkpoint.model_copy(update={
                "decision": None, "notice": f"There is no prop {prop_id!r} on this draft.",
            })
        ok, output = await self._prop("approve", prop_id)
        if not ok:
            return checkpoint.model_copy(update={
                "decision": None,
                "notice": f"Approving prop {prop_id} failed: {output}",
            })
        props = [p.model_copy(update={"status": "approved"}) if p.id == prop_id else p
                 for p in checkpoint.props]
        updated = checkpoint.model_copy(update={"props": props, "decision": None})
        notice = f"Prop {prop_id} approved."
        if not updated.pending_props:
            notice += " Every prop is approved - approve the draft to render."
        return updated.model_copy(update={"notice": notice})

    async def _reject_prop(
        self, checkpoint: ReelCheckpoint, prop_id: str, notes: str
    ) -> ReelCheckpoint:
        """Record the rejection in the kit, then send Claude Code back to
        fix the prop. Stays in review if the kit would not take it."""
        if not PROP_ID.match(prop_id) or prop_id not in {p.id for p in checkpoint.props}:
            return checkpoint.model_copy(update={
                "decision": None, "notice": f"There is no prop {prop_id!r} on this draft.",
            })
        ok, output = await self._prop("reject", prop_id, notes)
        if not ok:
            return checkpoint.model_copy(update={
                "decision": None,
                "notice": f"Rejecting prop {prop_id} failed: {output}",
            })
        props = [p.model_copy(update={"status": "rejected", "notes": notes})
                 if p.id == prop_id else p for p in checkpoint.props]
        return checkpoint.model_copy(update={
            "stage": "revising", "props": props, "decision": None,
            "notes": (f"Sir rejected prop {prop_id}: {notes}\n"
                      f"Fix it, preview it again, rebuild and redo stills, and "
                      f"list it in new_props again."),
        })

    async def _prop(self, verb: str, prop_id: str, *extra: str) -> tuple[bool, str]:
        """The kit's prop command, as an argv list: the id is checked and
        Sir's notes are their own argument, never part of a shell string."""
        if not PROP_ID.match(prop_id):
            return False, f"{prop_id!r} is not a valid prop id"
        try:
            process = await asyncio.create_subprocess_exec(
                *self._config.prop_command, verb, prop_id, *extra,
                cwd=self._kit,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=120)
        except (OSError, asyncio.TimeoutError) as exc:
            return False, f"could not run the prop command ({type(exc).__name__})"
        text = " ".join(stdout.decode(errors="replace").split())[-300:]
        return process.returncode == 0, text

    async def _upload_draft(
        self, ctx: JobContext, kit_id: str, round_no: int, draft: PhaseAOutput,
        started: float,
    ) -> bool:
        """Upload what this run made. Only files written since it began
        count: a sheet left over from an earlier episode is not this
        draft's. Returns whether a fresh contact sheet went up."""
        stem = f"{kit_id}-r{round_no}"
        sheet = self._inside_kit(draft.sheet) or self._kit / "out" / "stills" / "sheet.png"
        has_sheet = _fresh(sheet, started)
        if has_sheet:
            await ctx.write_artifact(f"{stem}-sheet.png", "image/png", sheet.read_bytes())

        voice = self._inside_kit(draft.voice) or self._kit / "out" / "voice.wav"
        if _fresh(voice, started):
            name, mime, data = await _listenable(voice, stem)
            await ctx.write_artifact(name, mime, data)

        episode = self._kit / "episode.json"
        if episode.is_file():
            await ctx.write_artifact(
                f"{stem}-episode.json", "application/json", episode.read_bytes()
            )
        return has_sheet

    # -- rendering ------------------------------------------------------------

    async def _render(
        self, payload: ReelJobIn, ctx: JobContext, checkpoint: ReelCheckpoint
    ) -> ReelJobOut:
        self._check_kit_holds(payload.kit_id)
        self._prepare_frames()
        await ctx.progress("rendering frames and muxing")

        result = await self._claude.run(
            render_prompt(), PHASE_B_SCHEMA, _RENDER_TOOLS,
            resume=checkpoint.session_id,
            timeout_s=self._config.render_timeout_s,
        )
        if result.session_id and result.session_id != checkpoint.session_id:
            checkpoint = checkpoint.model_copy(update={"session_id": result.session_id})
            await ctx.save_checkpoint(checkpoint.model_dump(mode="json"))
        _raise_for(result, "rendering")
        assert result.structured is not None

        try:
            output = PhaseBOutput.model_validate(result.structured)
        except ValidationError as exc:
            raise RuntimeError(f"rendering: malformed result: {exc}") from exc

        mp4 = self._inside_kit(output.mp4)
        failure = ""
        if output.gpu_fallback:
            failure = "Chrome fell back to the CPU, so the render stopped."
        elif output.status != "delivered":
            failure = "; ".join(output.problems) or "the render reported failure"
        elif mp4 is None or not mp4.is_file():
            failure = f"the render finished but {output.mp4!r} does not exist"
        if failure and result.denials:
            failure += f" {_denied_problem(result.denials)}"
        if failure and re.search(r"props? (?:are |is )?not approved", failure, re.IGNORECASE):
            failure = f"Waiting for prop approval - the kit refused to render: {failure}"

        if failure:
            # Surface it to Sir instead of retrying blind: approve runs the
            # render again, notes go back to drafting, cancel ends it.
            failed = checkpoint.model_copy(update={
                "stage": "review", "round": checkpoint.round + 1,
                "kind": "render_failed", "render_error": failure, "decision": None,
            })
            raise await self._pause(ctx, failed)

        assert mp4 is not None
        name = f"{payload.kit_id}.mp4"
        await ctx.progress("uploading the film")
        await ctx.write_artifact(name, "video/mp4", mp4.read_bytes())
        title = checkpoint.draft.title if checkpoint.draft else payload.kit_id
        return ReelJobOut(
            episode_id=payload.episode_id,
            summary=f"{title or payload.kit_id}: {output.verification}".strip(": "),
            mp4_name=name,
            verification=output.verification,
        )

    # -- pausing --------------------------------------------------------------

    async def _pause(self, ctx: JobContext, checkpoint: ReelCheckpoint) -> PausedForApproval:
        """Save where we are and ask the owner. Returned, not raised, so
        call sites read `raise await self._pause(...)`."""
        await ctx.save_checkpoint(checkpoint.model_dump(mode="json"))
        draft = checkpoint.draft
        title = (draft.title if draft else "") or checkpoint.kit_id
        if checkpoint.kind == "incomplete":
            summary = f"Reel draft {checkpoint.round} is incomplete: {title}"
            problems = draft.problems if draft else []
            detail = "\n".join(f"- {p}" for p in problems) or "(no detail returned)"
            detail += "\n\nSend notes to have it finished or changed, or cancel."
        elif checkpoint.kind == "render_failed":
            summary = f"Reel render failed: {title}"
            detail = (
                f"{checkpoint.render_error}\n\n"
                f"Approve to run the render again, send notes to revise the "
                f"draft, or cancel."
            )
        else:
            summary = f"Reel draft {checkpoint.round} ready for review: {title}"
            lines = "\n".join(draft.lines) if draft else ""
            detail = lines or "(no script returned)"
            if draft and draft.problems:
                detail += "\n\nProblems:\n" + "\n".join(f"- {p}" for p in draft.problems)
            if checkpoint.pending_props:
                detail += "\n\nProps awaiting approval: " + ", ".join(checkpoint.pending_props)
        if checkpoint.notice:
            detail = f"{checkpoint.notice}\n\n{detail}"
        ctx.pending_approval = {
            "gate": Gate.SPEND.value,
            "actor": ACTOR,
            "tool": REEL_REVIEW_TOOL,
            "summary": summary,
            "detail": detail,
            "risk_note": "Approving spends GPU time and Claude usage on the render.",
        }
        return PausedForApproval(summary)

    # -- the kit folder -------------------------------------------------------

    def _inside_kit(self, path: str) -> Path | None:
        """A path Claude Code reported, if it really is inside the kit."""
        if not path:
            return None
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self._kit / candidate
        try:
            resolved = candidate.resolve()
            resolved.relative_to(self._kit.resolve())
        except (OSError, ValueError):
            return None
        return resolved

    def _kit_episode_id(self) -> str | None:
        try:
            data = json.loads((self._kit / "episode.json").read_text())
        except (OSError, json.JSONDecodeError):
            return None
        value = data.get("id") if isinstance(data, dict) else None
        return str(value) if value else None

    def _prepare_frames(self) -> None:
        """Clear frames that belong to a different build.

        The renderer skips frames that already exist, so leftovers from
        another episode - or an earlier draft of this one - would end up
        in the film. Frames are tied to the build they came from by a
        hash of out/film.js, kept beside the folder rather than in it.
        A retry of the same build keeps its frames and resumes.
        """
        out = self._kit / "out"
        frames = out / "frames"
        stamp = out / ".frames-build"
        film = out / "film.js"
        build = _digest(film) if film.is_file() else ""
        try:
            recorded = stamp.read_text().strip()
        except OSError:
            recorded = ""
        if frames.exists() and (not build or recorded != build):
            log.info("clearing frames from another build", extra={"kit": str(self._kit)})
            shutil.rmtree(frames, ignore_errors=True)
        if build:
            out.mkdir(parents=True, exist_ok=True)
            stamp.write_text(build + "\n")

    def _check_kit_holds(self, kit_id: str) -> None:
        """Revising or rendering the wrong episode would be worse than
        failing: the kit folder must still hold this one."""
        actual = self._kit_episode_id()
        if actual != kit_id:
            raise PermanentJobError(
                f"the kit folder now holds episode {actual!r}, not {kit_id!r} - "
                f"something else used it in between. Start this reel again."
            )


def _fresh(path: Path, since: float) -> bool:
    """Whether a file exists and was written after `since`."""
    try:
        return path.is_file() and path.stat().st_mtime >= since - _FRESH_SLACK_S
    except OSError:
        return False


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def _listenable(voice: Path, stem: str) -> tuple[str, str, bytes]:
    """The voice track as m4a, which Telegram plays inline. Falls back to
    the WAV itself when ffmpeg is missing or fails."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "voice.m4a"
            process = await asyncio.create_subprocess_exec(
                ffmpeg, "-y", "-loglevel", "error", "-i", str(voice),
                "-c:a", "aac", "-b:a", "128k", str(target),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            if await process.wait() == 0 and target.is_file():
                return f"{stem}-voice.m4a", "audio/mp4", target.read_bytes()
    return f"{stem}-voice.wav", "audio/wav", voice.read_bytes()


# -- registration -------------------------------------------------------------

def register_reel_jobs(
    registry: JobTypeRegistry,
    config: ReelConfig,
    capabilities: list[str],
) -> bool:
    """Register reel.make on this worker if it can actually make reels.

    Returns whether it registered. A worker that cannot make reels simply
    lacks the handler, and declines reel offers - costing them nothing.
    """
    kit = config.kit_dir
    if not (kit / "package.json").is_file():
        log.warning("reel kit not found - reels disabled", extra={"kit_dir": str(kit)})
        return False
    if shutil.which(config.claude_bin) is None:
        log.warning("claude CLI not on PATH - reels disabled", extra={
            "claude_bin": config.claude_bin,
        })
        return False
    missing = [tag for tag in REEL_REQUIRES if tag not in capabilities]
    if missing:
        log.warning("reels registered, but this worker does not advertise "
                    "every capability they need - add them to "
                    "JARVIS_CAPABILITIES_RAW or no reel will be offered here",
                    extra={"missing": missing})

    maker = ReelMaker(config)
    spec = reel_job_spec()
    registry.register(JobTypeSpec(
        type=REEL_JOB_TYPE,
        input_model=spec.input_model,
        output_model=spec.output_model,
        execution=spec.execution,
        requires=spec.requires,
        default_priority=spec.default_priority,
        timeout_s=spec.timeout_s,
        handler=maker.handle,
    ))
    log.info("reel job type registered", extra={"kit_dir": str(kit)})
    return True
