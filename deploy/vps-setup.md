# VPS Setup Runbook - jarvis-core

Goal: the Core daemon running 24/7 on a small Linux VPS under systemd,
with history carried over from the laptop, and only SSH exposed to the
internet.

Assumptions: Ubuntu 24.04 LTS, 1 vCPU / 1 GB RAM, you have root or sudo.
Any cheap provider works (Hetzner, DigitalOcean, Lightsail, Oracle free
tier). Region: latency barely matters (Telegram polling), so pick close
and cheap.

Every command block states WHERE it runs: [laptop] or [vps].

---

## 1. First contact and a non-root user

SSH in as root once, create the service user, and never use root again.

[vps]

    adduser --disabled-password --gecos "" jarvis
    mkdir -p /home/jarvis/.ssh
    cp ~/.ssh/authorized_keys /home/jarvis/.ssh/   # reuse the key you SSHed with
    chown -R jarvis:jarvis /home/jarvis/.ssh
    chmod 700 /home/jarvis/.ssh
    usermod -aG sudo jarvis

Confirm from a NEW terminal that `ssh jarvis@<vps-ip>` works BEFORE
continuing. Then harden sshd: in /etc/ssh/sshd_config set

    PasswordAuthentication no
    PermitRootLogin no

and `systemctl restart ssh`.

## 2. Firewall: one door

[vps]

    sudo ufw allow OpenSSH
    sudo ufw enable
    sudo ufw status    # expect: only OpenSSH allowed

The gateway binds 127.0.0.1 and is NOT exposed. When you want /status
from the laptop, open a tunnel instead of a port:

[laptop]

    ssh -L 8321:127.0.0.1:8321 jarvis@<vps-ip>
    # then, in another laptop terminal:
    curl -H "Authorization: Bearer <token>" http://127.0.0.1:8321/status

## 3. Install runtime

Ubuntu 24.04 ships Python 3.12 - exactly our floor.

[vps]

    sudo apt update && sudo apt install -y git python3 python3-venv sqlite3
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.profile      # puts uv on PATH
    python3 --version      # expect 3.12.x
    uv --version

## 4. Directory layout

Two trees, on purpose: the CODE (replaceable, comes from git) and the
DATA (sacred, backed up, survives redeploys). Never nest one in the
other.

[vps]

    sudo mkdir -p /opt/jarvis/app /opt/jarvis/data
    sudo chown -R jarvis:jarvis /opt/jarvis

## 5. Code

Clone and install. (If the repo is private, add the VPS's SSH key as a
read-only deploy key on GitHub first: `ssh-keygen -t ed25519` on the vps,
paste ~/.ssh/id_ed25519.pub into repo Settings -> Deploy keys.)

[vps]

    git clone <your-repo-url> /opt/jarvis/app
    cd /opt/jarvis/app
    uv venv
    uv pip install -e .

Note: `-e .` only - no [dev] extras. Tests and mypy live on the laptop;
the server runs lean.

## 6. Configuration

Copy the laptop's .env as the starting point, then fix the paths.

[laptop]

    scp .env jarvis@<vps-ip>:/opt/jarvis/app/.env

[vps] - edit /opt/jarvis/app/.env:

    JARVIS_DATA_DIR=/opt/jarvis/data     # THE line that must change
    # everything else (tokens, keys, models) carries over unchanged

    chmod 600 /opt/jarvis/app/.env       # owner-readable only: it holds secrets

## 7. Carry the history over (optional but worth it)

Your laptop's core.db holds every conversation and rupee so far. Move a
CLEAN copy - stop the laptop Core first so the WAL is settled, then use
sqlite's backup command rather than copying the raw file (it produces a
consistent single-file snapshot even if something is still open):

[laptop]

    sqlite3 data/core.db ".backup data/core-for-vps.db"
    scp data/core-for-vps.db jarvis@<vps-ip>:/opt/jarvis/data/core.db

Skipping this step is also fine - the Core builds a fresh db on first
boot. You lose history, not function.

## 8. IMPORTANT: one bot, one poller

Telegram permits ONE process polling a bot token. From this point on,
the laptop's `python -m jarvis.core` must stay OFF, or both instances
will fight over messages with Conflict errors. The laptop's future role
is worker, not Core.

## 9. Trial run by hand

Always prove it works manually before involving systemd - the error
messages are right in front of you here.

[vps]

    cd /opt/jarvis/app
    .venv/bin/python -m jarvis.core

Expect the same JSON boot lines as on the laptop, ending in
"core running". Message the bot from your phone - it should answer FROM
THE VPS. Ctrl-C to stop.

## 10. Install the service

[vps]

    sudo cp /opt/jarvis/app/deploy/systemd/jarvis-core.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable --now jarvis-core
    systemctl status jarvis-core        # expect: active (running)

Watch the logs live:

    journalctl -u jarvis-core -f

## 11. Acceptance checks

1. Phone: message the bot -> streamed reply. /status shows fresh uptime.
2. Kill test: `sudo systemctl kill -s KILL jarvis-core` mid-conversation
   -> systemd restarts it within seconds (watch journalctl) -> history
   intact from your phone.
3. Reboot test: `sudo reboot`, wait a minute -> bot answers again with no
   human action. This is the moment JARVIS stops depending on you.

## 12. Updating the Core later (the redeploy loop)

[vps]

    cd /opt/jarvis/app
    git pull
    uv pip install -e .            # in case deps changed
    sudo systemctl restart jarvis-core

Migrations run automatically at boot; data is untouched in /opt/jarvis/data.

## 13. Until real backups arrive (a later phase)

The single point of failure is this VPS's disk. Until the nightly backup
job exists, do this manually every few days, laptop-side:

[laptop]

    ssh jarvis@<vps-ip> "sqlite3 /opt/jarvis/data/core.db '.backup /tmp/jarvis-backup.db'"
    scp jarvis@<vps-ip>:/tmp/jarvis-backup.db ./backups/jarvis-$(date +%F).db

Two minutes of paranoia; buys you your assistant's memory back.