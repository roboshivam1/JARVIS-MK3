# JARVIS MK-3

> *"Sometimes you gotta run before you can walk."*

An always-on personal AI assistant. Not a chat wrapper — a daemon that keeps
running when you close the laptop, remembers what you told it three weeks ago,
does long-running work on its own machines, and asks before it does anything
it shouldn't do unasked.

MK3 is a ground-up rewrite of [JARVIS MK2](#mk2--mk3-why-the-rewrite). Same
ambition, completely different foundation: one persistent Core process, typed
contracts at every boundary, a real job queue, SQLite as the single source of
truth, and a capability/approval system that decides what JARVIS is allowed
to do before the model gets a vote.

---

## Contents

- [What it actually does](#what-it-actually-does)
- [Design principles](#design-principles)
- [Architecture](#architecture)
- [Repository layout](#repository-layout)
- [The subsystems](#the-subsystems)
  - [Gateway](#gateway--the-http-door)
  - [Orchestrator](#orchestrator--jarvis-himself)
  - [Agent loop and the Guard](#agent-loop-and-the-guard)
  - [Subagents](#subagents)
  - [Job queue and workers](#job-queue-and-workers)
  - [Memory](#memory)
  - [The people database](#the-people-database)
  - [Initiative, notifications, approvals](#initiative-notifications-approvals)
  - [Schedules and watchers](#schedules-and-watchers)
  - [Artifacts and backups](#artifacts-and-backups)
- [Getting started](#getting-started)
- [Voice](#voice)
- [Configuration](#configuration)
- [Data and storage](#data-and-storage)
- [Tests and type checking](#tests-and-type-checking)
- [Deployment](#deployment)
- [Security notes](#security-notes)
- [MK2 → MK3: why the rewrite](#mk2--mk3-why-the-rewrite)

---

## What it actually does

- **Holds a conversation** over the web UI or Telegram, with full session
  history and memory retrieval injected automatically.
- **Remembers.** Facts are stored with confidence, category, provenance and
  an embedding. Recall fuses vector similarity with keyword search, so a
  paraphrased question still finds the fact. Restated facts *supersede*
  older ones instead of piling up.
- **Knows the people in your life.** A separate, structured people database
  with facts, commitments, key dates, aliases and groups — every write of
  which passes through a human review gate.
- **Runs long work in the background** through a durable job queue: research
  briefs, document writing, code tasks, browser automation, backups.
- **Delegates to specialist subagents** — each one a distinct persona, tool
  allowlist and model tier on the shared agent loop.
- **Farms work out to worker machines** that connect over a websocket and
  advertise capabilities (`macos`, `browser`, `sandbox-exec`, `local-llm`…).
  The Core offers a job only to a machine that can actually do it.
- **Acts on its own schedule** — nightly memory maintenance, nightly backups,
  cron schedules you create in conversation, watchers that check a condition
  and notify you when it changes.
- **Asks permission** before anything that spends money, touches credentials,
  publishes something, or deletes outside its sandbox.

---

## Design principles

These are load-bearing. Most of the structure below falls out of them.

**Validation at the boundary.** Every envelope, job payload, tool argument
and config value is a Pydantic model. Bad data dies at the border with a
precise error, never three layers in.

**Fail closed.** An empty gateway token refuses all authed requests. A tool
missing from an actor's allowlist is denied, not permitted. A Telegram bot
token with no owner id refuses to start.

**Degrade, don't die.** No embedder means keyword-only memory, not broken
memory. No Anthropic key means model calls fail with a clear error while the
daemon keeps running. Optional residents (Telegram, voice) restart with
backoff; essential ones (queue, notifier, approvals) take the daemon down so
systemd can restart it cleanly through recovery.

**Supersede, never overwrite.** Facts and people-facts are never deleted by
the system. New information marks the old superseded and keeps the history.

**Tiers, not model names.** Code asks for `Tier.REASONER` or `Tier.UTILITY`.
What those mean is a config line and a restart.

**One door per dangerous thing.** All tool execution goes through the
toolset; all tool execution goes through the Guard; all writes to the people
database go through the review gate. There is exactly one place to audit.

---

## Architecture

Two process roles, one protocol.

```
                   ┌──────────────────────────────────────────┐
   browser ──HTTP──┤                                          │
   Telegram ───────┤              CORE  (always on)           │
                   │                                          │
                   │  gateway ── orchestrator ── agent loop   │
                   │     │            │              │        │
                   │     │        sessionmgr      guard       │
                   │     │            │                       │
                   │  clients     memory ─ facts + embeddings │
                   │              people ─ review gate        │
                   │              queue  ─ dispatcher/reclaim │
                   │              initiative ─ notifier       │
                   │                                          │
                   │            SQLite: data/core.db          │
                   └────────────────┬─────────────────────────┘
                                    │  ws://…/ws/worker
                   ┌────────────────┴─────────────────────────┐
                   │        WORKER  (intermittent)            │
                   │                                          │
                   │  runner ─ sandbox ─ git ─ MCP host       │
                   │  capabilities: macos, browser,           │
                   │                sandbox-exec, local-llm   │
                   └──────────────────────────────────────────┘
```

**The Core** (`python -m jarvis.core`) owns all state. It boots storage, runs
migrations, reclaims in-flight jobs orphaned by a restart, then brings up the
component graph and its residents. It never executes work that needs a
capability it doesn't have.

**Workers** (`python -m jarvis.worker`) are your laptop, a VPS, anything.
They dial the Core, authenticate with the shared token, advertise what they
can do, and accept offers for matching jobs. Disconnecting is normal —
the queue reclaims and re-offers.

**The boundary between them is typed.** `common/worker_protocol.py` and
`common/client_protocol.py` register envelope kinds; nothing crosses the wire
unvalidated.

---

## Repository layout

```
src/jarvis/
  common/          shared contracts — no I/O side effects on import
    settings.py      typed config for every process role
    envelope.py      the wire format
    facts.py         memory fact model
    people.py        people models: proposals, owner edits, enums
    capabilities.py  worker capabilities, risk classes, approval modes
    ids.py           ULIDs and UTC timestamps
    jobs.py, schedules.py, watchers.py, approvals.py, artifacts.py,
    events.py, notifications.py, sessions.py, log.py

  core/            the always-on daemon
    app.py           boot / supervise / shutdown
    sessionmgr.py    conversation sessions, context assembly
    orchestrator/    JARVIS himself: persona + tools
    gateway/         FastAPI app, auth, websockets, static UI
    db/              Database, numbered migrations, one repo per table
    memory/          fact storage, hybrid search, profile document
    people/          people service: debrief → proposals → review
    queue/           job registry, dispatcher, reclaim, core-side worker
    initiative/      when to speak first, notification policy, delivery
    approvals/       approval requests and resolution
    observability/   LLM call traces
    clients/         Telegram bridge

  agentloop/       the shared agent loop
    loop.py          the tool-calling loop itself
    toolset.py       typed tools: name, description, args model, handler
    guard.py         ActorPolicy + GateRule + Verdict
    policies.py      *the* answer to "what is JARVIS allowed to do?"
    mcp_client.py    MCP servers as tools

  subagents/       personas configured onto the agent loop
    researcher.py    ATHENA
    writer.py        CALLIOPE
    engineer.py      DAEDALUS
    operator.py      PROTEUS
    archivist.py     MNEMOSYNE

  jobs/            job type registrations and handlers
    research.py, writing.py, code.py, browser.py, watch.py,
    maintenance.py, backup.py, worker_types.py

  llm/             provider layer
    layer.py         tiers, ToolSpec, message helpers
    anthropic.py     the provider
    tiers.py, pricing.py, embeddings.py, transcription.py, speech.py

  worker/          the intermittent process
    runner.py, workspace.py, sandbox/, git/, tools/, mcp_host.py

tests/
  unit/            models, guard
  integration/     boot, conversation, memory, queue, schedules,
                   approvals, sandbox, worker protocol, people
deploy/
  systemd/jarvis-core.service, vps-setup.md, restore.md
```

---

## The subsystems

### Gateway — the HTTP door

FastAPI, bound to loopback by default, every authed route behind a shared
bearer token.

| Route | What it is |
|---|---|
| `GET /health` | unauthenticated liveness |
| `GET /status` | boot time, residents, queue depth, worker fleet |
| `GET /app` | the web UI |
| `GET /app/people` | the people review + directory UI |
| `WS /ws/client` | the browser's live conversation channel |
| `WS /ws/worker` | worker fleet link |
| `POST /upload` | file ingestion |
| `GET /artifact/{id}` | fetch a produced document |
| `/people/*` | review gate + directory API (see below) |

The UI is deliberately dependency-free: two static HTML files and one JS
file, served straight from `core/gateway/static/`. No build step, no npm, no
framework. It looks like a dark newspaper on purpose.

### Orchestrator — JARVIS himself

`core/orchestrator/agent.py` is the top-level agent: the persona, the
conversation, and a deliberately small set of tools.

```
get_time_context      memory_store        memory_search
run_subagent          enqueue_job         list_jobs / get_job / cancel_job
create_schedule       list_schedules      cancel_schedule
create_watcher        list_watchers       remove_watcher
people_lookup         people_upcoming     people_debrief
```

Note what is *absent*: no shell, no browser, no filesystem. The orchestrator
talks and delegates. Anything with teeth happens inside a subagent or a job,
each with its own narrower allowlist.

### Agent loop and the Guard

`agentloop/loop.py` is a single tool-calling loop shared by the orchestrator
and every subagent. A tool is a name, a description the model reads, a
Pydantic args model, and an async handler. The args model works in both
directions: its JSON schema is the contract shown to the model, and the
model's arguments are validated through it before the handler runs.

**Execution never raises into the loop.** Unknown tool, invalid arguments,
crashing handler — all come back as an error result the model can read and
route around. Failures become information, not stack traces.

`agentloop/policies.py` is the security surface in one readable file:

- **ActorPolicy** — per-actor tool allowlist and model tier. A tool that
  appears without a matching allowlist entry is *denied*. (Worth knowing
  when adding a tool: forget the allowlist entry and it fails silently.)
- **GateRule** — argument-level rules that escalate to an approval request:
  `git_push`, `git_create_repo`, browsing to anything matching `*login*`,
  writing to an absolute path outside the workspace, `sandbox_exec` with
  `sudo` in the command.

### Subagents

Each subagent is a *configuration* of the shared loop — its own persona
prompt, tool allowlist, model tier and output contract.

| Name | Role | Shape of its output |
|---|---|---|
| **ATHENA** | research | summary, findings, sources — a fixed document shape |
| **CALLIOPE** | writing | a finished document, saved as an artifact |
| **DAEDALUS** | engineering | code in a real git workspace, plus a report |
| **PROTEUS** | browser operation | drives a real browser via Playwright MCP |
| **MNEMOSYNE** | archivist | extracted facts; distils the profile document |

### Job queue and workers

Durable, SQLite-backed, with leases and reclaim. Job types are *registered*
with an input model, an output model, an execution class
(`idempotent` / `at-most-once`), a capability requirement, a timeout, and a
handler — or `None` if the Core can't run it.

Registered types:

```
research.brief      write.document      code.task
browser.task        watch.check         memory.sleep_cycle
system.backup       worker.echo
```

`worker.echo`, `code.task` and `browser.task` are registered on the Core for
their *metadata only* — the Core validates payloads and constructs offers,
but the handler lives on a machine that can actually do the thing.

On boot, `_recover()` reclaims every leased or running job orphaned by the
restart. A worker vanishing mid-job is a normal Tuesday, not an incident.

### Memory

Two operations matter.

**`store()`** checks whether the new statement restates something already
known. If it does, the new one *supersedes* the old rather than both
coexisting — this is what stops the vault accumulating twelve versions of
the same claim.

**`search()`** fuses two searches: vector similarity (catches paraphrase)
and SQLite FTS with stemming (catches exact names). Each alone has a blind
spot. Final ranking blends five signals scaled by confidence, so a shaky
belief ranks below a solid one even when it matches the words better.

Facts record *which model* embedded them, because vectors from different
models aren't comparable; stale ones are re-embedded during the nightly
sleep cycle.

Above the fact vault sits the **profile document** — MNEMOSYNE's distilled
standing page about the owner, placed in front of the system on every turn.

With no embedder configured, search falls back to keyword-only and `store()`
skips dedup. Worse memory, still a working daemon.

### The people database

JARVIS's model of the people in your life. Structurally separate from the
fact vault, because a person is not a sentence.

**Tables:** `people`, `person_aliases`, `person_facts`,
`person_commitments`, `person_dates`, `people_groups`,
`person_group_members`, `people_debriefs`, `people_proposals`.

**The review gate is the whole point.** The extraction model can only ever
write to `people_proposals`. Nothing reaches a real table except through an
explicit `approve()`. The flow:

```
you talk  ─┐
           ├─→  debrief()  ─→  extraction model  ─→  people_proposals
archivist ─┘                                              │
                                                    you review
                                              (web UI, inline edits)
                                                     │        │
                                                 approve   reject
                                                     │
                                            one transaction, writes
                                            to people / facts / dates
```

Design details worth knowing if you touch this code:

- **Handles, not ULIDs.** The extraction model sees `R1, R2…` for people and
  `F1, F2…` for existing facts, never 26-character ULIDs. Dangling handles
  are dropped at translation time, not trusted.
- **Decide-then-write.** `approve()` does every read and check first,
  building a write list, then runs the list in one transaction — because the
  `Database.transaction()` helper supports `execute` only, not `query`.
- **Supersession, not deletion.** Retiring a fact marks it superseded and
  records what replaced it.
- **Groups** (`family`, `school`, `college`, `locality`, `project`, `work`,
  `other`) are many-to-many with cascade, names compared `COLLATE NOCASE`.

The UI at `/app/people` has two tabs: **Review** (the pending queue, with
inline editing before approval) and **Directory** (browse by group, open a
person, edit facts, resolve commitments, merge duplicates). The home page
carries a compact panel with the pending count and the next key dates.

In conversation, `people_lookup` and `people_upcoming` read from it, and
`SessionManager` injects a "People mentioned" block into retrieved context
when someone in the database comes up. `memory_store` explicitly redirects
person-facts here instead of into the general vault.

### Initiative, notifications, approvals

The **initiative engine** decides when JARVIS gets to speak first. The
**notification policy** decides whether something is worth interrupting you
for, and the **notifier** delivers it — to the web client, to Telegram, or
it waits for a better hour.

**Approvals** are their own table and their own service. A gate rule fires,
an approval request is created, the job parks, you answer, the job resumes.
Risk classes: `spend`, `outbound`, `publish`, `destructive-file`,
`credential`, `system`.

### Schedules and watchers

**Schedules** are cron expressions in your timezone, stored in SQLite,
created either by seed (the sleep cycle, the nightly backup) or by you in
conversation. Seeded rows are left alone after first creation — your edits
survive every restart.

**Watchers** check a condition on a cadence and notify you when the answer
changes. A `watch.check` job runs; the notifier decides whether it's worth
your attention.

### Artifacts and backups

Documents produced by subagents are written to `data/artifacts/` and indexed
in the `artifacts` table, retrievable at `/artifact/{id}`.

Backups run nightly (default 04:00, after the sleep cycle so the day's
consolidated memory is actually in them), keep `JARVIS_BACKUP_RETENTION_DAYS`
of local copies, and optionally run `JARVIS_BACKUP_COMMAND` with `{path}`
substituted for off-site copy. Local backups protect against mistakes; only
the off-site command protects against a dead disk. `deploy/restore.md` has
the restore procedure.

---

## Getting started

### Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- An **Anthropic API key** ([console.anthropic.com](https://console.anthropic.com))
- Optional: **Ollama** for local embeddings, **Node** for the Playwright/
  filesystem MCP servers, a **Groq key** for voice.

### Install

```bash
git clone https://github.com/roboshivam1/JARVIS-MK3.git
cd JARVIS-MK3
uv sync --all-extras
```

> **Use `--all-extras`, not `--extra dev`.** The `dev` extra alone will
> uninstall the worker extras, and `pytest`/`mypy` live only in `dev` — a
> bare `uv sync` leaves you with `Failed to spawn: pytest`.

### Configure

```bash
cp .env.example .env
$EDITOR .env
```

At minimum set `JARVIS_ANTHROPIC_API_KEY` and generate a gateway token:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Put that value in **both** `JARVIS_GATEWAY_TOKEN` and `JARVIS_WORKER_TOKEN`.
An empty gateway token means every authed route is refused — safe, not open.

### Run the Core

```bash
uv run python -m jarvis.core
```

Migrations run automatically on boot. Open **http://127.0.0.1:8321/app** and
paste your gateway token when prompted. The people UI is at
**/app/people**.

### Run a worker

In a second terminal, on any machine that can reach the Core:

```bash
uv run python -m jarvis.worker
```

Set `JARVIS_WORKER_ID` to something stable — reconnecting with the same id
replaces the old registration instead of leaving a ghost in the fleet. Set
`JARVIS_CAPABILITIES_RAW` honestly: it's a claim, not a detection. Don't
advertise `browser` on a machine with no browser.

---

## Voice

Speech **in** is Groq-hosted Whisper (`JARVIS_GROQ_API_KEY`). Empty key means
voice input is off; typing is unaffected.

Speech **out** has three providers:

| `JARVIS_TTS_PROVIDER` | Notes |
|---|---|
| `kokoro` | local, free, private — needs the model files below |
| `groq` | hosted Orpheus, same key as transcription, 200-char cap per request |
| `none` | replies stay text |

Kokoro needs its model files downloaded once:

```bash
mkdir -p ~/.cache/jarvis && cd ~/.cache/jarvis
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

> **Known gotcha.** The `voice` extra in `pyproject.toml` is currently
> empty. `llm/speech.py` imports `kokoro_onnx` lazily and *logs* rather than
> crashes, so a missing package shows up as JARVIS silently not speaking
> rather than an error. If you hand-install `kokoro-onnx` and `soundfile`,
> the next `uv sync` will remove them again. Fix it properly by declaring
> them:
>
> ```toml
> voice = [
>     "kokoro-onnx>=0.4",
>     "soundfile>=0.12",
> ]
> ```
>
> then re-run `uv sync --all-extras`.

---

## Configuration

Everything enters through `common/settings.py`, from environment variables or
`.env`, all prefixed `JARVIS_`. Secrets are `SecretStr`, so printing or
logging settings can never leak them.

**`.env.example` is the reference** — every variable the system reads is
documented there. The essentials:

| Variable | Default | Notes |
|---|---|---|
| `JARVIS_DATA_DIR` | `./data` | root of *all* persistent state |
| `JARVIS_OWNER_TIMEZONE` | `Asia/Kolkata` | storage is UTC; this is for display and cron |
| `JARVIS_ANTHROPIC_API_KEY` | — | missing = daemon runs, model calls fail clearly |
| `JARVIS_MODEL_REASONER` | `claude-sonnet-4-6` | the `REASONER` tier |
| `JARVIS_MODEL_UTILITY` | `claude-haiku-4-5` | the `UTILITY` tier |
| `JARVIS_EMBEDDER_PROVIDER` | `ollama` | `ollama` / `voyage` / `none` |
| `JARVIS_GATEWAY_HOST` / `_PORT` | `127.0.0.1` / `8321` | never expose bare to the world |
| `JARVIS_GATEWAY_TOKEN` | *(empty)* | empty = all authed routes refused |
| `JARVIS_TELEGRAM_BOT_TOKEN` | *(empty)* | empty = bridge doesn't start, Core runs on |
| `JARVIS_TELEGRAM_OWNER_ID` | `0` | the only user the bot answers |
| `JARVIS_SLEEP_CYCLE_CRON` | `30 3 * * *` | nightly memory maintenance |
| `JARVIS_BACKUP_CRON` | `0 4 * * *` | after the sleep cycle, deliberately |
| `JARVIS_WORKSPACE_DIR` | `~/Desktop/jarvismk3-sandbox` | DAEDALUS's git projects |

Derived paths (`data/core.db`, `data/artifacts/`, `data/backups/`) are
*properties*, not configurable fields — the data layout is a contract the
backup and restore scripts depend on, not a per-deployment choice.

MCP servers are configured in `mcp.json` (filesystem + Playwright by
default).

---

## Data and storage

```
data/
  core.db        every table: sessions, turns, facts, people, jobs,
                 schedules, approvals, notifications, artifacts index,
                 llm_calls, events, watchers, profile_doc
  artifacts/     documents produced by subagents
  backups/       nightly snapshots
```

One SQLite file. Migrations are numbered SQL in
`core/db/migrations/` and run in order on boot, tracked in
`schema_migrations`. To add a table, add `00NN_thing.sql` and a repo in
`core/db/repos/` — never reach into the database from a service.

> **`data/` must never be committed.** It contains your entire conversation
> history, every fact JARVIS knows about you, and the people database —
> which holds facts about real people who did not consent to being in a
> public git repository. It's in `.gitignore`. If it was ever committed,
> `.gitignore` does not remove it from history; that needs
> `git filter-repo --path data --invert-paths` and a force-push, or making
> the repository private.

---

## Tests and type checking

```bash
uv run pytest tests -q          # unit + integration
uv run mypy                     # strict, on the jarvis package
```

Integration tests boot a real `CoreApp` against a temporary database without
binding ports — boot is deliberately separated from starting servers so
tests can inspect the whole component graph.

---

## Deployment

`deploy/vps-setup.md` walks through a small VPS. `deploy/systemd/jarvis-core.service`
is the unit file — it expects the Core to exit loudly on essential-resident
death so systemd restarts it through recovery, which is exactly what
`app.py` does.

On a VPS, keep the gateway on loopback behind a reverse proxy or a
Tailscale interface. Workers connect with `wss://`.

`deploy/restore.md` is the restore procedure. Read it *before* you need it.

---

## Security notes

- The gateway fails closed on an empty token.
- The Telegram bridge refuses to start with a token but no owner id.
- The orchestrator has no shell, no browser and no filesystem access.
- Sandboxed code execution has no network — deliberately. That's why heavy
  libraries (`pandas`, `numpy`, `matplotlib`, `scipy`, `pypdf`, `openpyxl`)
  are pre-installed in the `worker` extra rather than pip-installed on
  demand.
- Gate rules escalate `git_push`, repo creation, login pages, absolute-path
  writes and `sudo` to an approval request.
- Use a **fine-grained** GitHub token scoped to specific repositories.
  Empty token = git capability disabled.

---

## MK2 → MK3: why the rewrite

MK2 worked, and taught the lessons MK3 is built from:

| MK2 | MK3 |
|---|---|
| A script you ran; it died when the terminal closed | A daemon supervised by systemd |
| JSON files for memory, weak keyword matching | SQLite with hybrid vector + FTS search and supersession |
| Keyword dispatcher that misrouted constantly | Typed job registry, capability matching, explicit `run_subagent` |
| Everything on one machine, in one process | Core + worker fleet over a typed protocol |
| Tools were loose functions; anything could call anything | Every tool behind a Pydantic contract and a per-actor allowlist |
| No record of what it did or what it cost | `events`, `llm_calls`, traces, job history |
| Nothing stopped it doing something expensive | Risk classes and approval gates |

The Greek names survived the rewrite, because they were the good part.

---

*Built by [Shivam Kapoor](https://github.com/roboshivam1).*