# =============================================================================
# src/jarvis/worker/git/config.py - which repositories DAEDALUS may touch
# =============================================================================
#
# A GitHub token typically grants access to EVERY repository the owner
# can reach. That is far more than any agent should have, so the
# narrowing happens here: an explicit allowlist, per repo, with a
# read-or-write flag.
#
# Configuration is a file rather than code, for the same reason MCP
# servers are: granting access to a repo should be an edit and a
# restart, not a code change.
#
#   {
#     "author_name": "JARVIS",
#     "author_email": "jarvis@example.com",
#     "coauthor_trailer": true,
#     "repos": {
#       "roboshivam1/some-project": {"write": true},
#       "roboshivam1/reference-repo": {"write": false}
#     }
#   }
#
# A repo absent from this file does not exist as far as the agent is
# concerned - not "denied with an error", simply invisible.
# =============================================================================

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from jarvis.common.log import get_logger

log = get_logger("worker.git.config")

DEFAULT_CONFIG_PATH = Path("git.json")


@dataclass(frozen=True)
class RepoGrant:
    """What the agent may do with one repository."""

    full_name: str          # "owner/repo"
    write: bool = False     # False = clone and read only


@dataclass
class GitConfig:
    """The whole git capability's configuration."""

    author_name: str = "JARVIS"
    author_email: str = "jarvis@localhost"

    # When true, commits carry a Co-Authored-By trailer naming JARVIS.
    # This makes GitHub attribute the work to both parties, which is
    # honest about how the code was made. Applies to repos the agent
    # creates; existing repos follow the same setting unless the owner
    # wants otherwise.
    coauthor_trailer: bool = True

    # Whether the agent may create NEW repositories. Separate from the
    # allowlist because creation is not about any existing repo.
    allow_create: bool = False

    repos: dict[str, RepoGrant] = field(default_factory=dict)

    def grant_for(self, full_name: str) -> RepoGrant | None:
        """What is permitted for this repo, or None if it is not listed.

        Repos the agent CREATES are granted write implicitly - it made
        them, and requiring a config edit before it could commit would
        make creation useless.
        """
        return self.repos.get(full_name)

    def may_read(self, full_name: str) -> bool:
        return self.grant_for(full_name) is not None

    def may_write(self, full_name: str) -> bool:
        grant = self.grant_for(full_name)
        return grant is not None and grant.write

    def allow_repo(self, full_name: str, write: bool = True) -> None:
        """Grant access at runtime - used after creating a repo, so the
        agent can immediately commit to what it just made. NOT persisted:
        a restart forgets it unless the owner adds it to git.json, which
        is the right default for something granted automatically."""
        self.repos[full_name] = RepoGrant(full_name=full_name, write=write)


def load_git_config(path: Path | None = None) -> GitConfig:
    """Read the configuration. Absent or malformed yields an EMPTY
    config - no repos, no creation - rather than a failed startup. A
    worker with no git configuration is a worker without git, not a
    broken one."""
    config_path = path or DEFAULT_CONFIG_PATH
    if not config_path.exists():
        log.info("no git config - git capability disabled", extra={
            "path": str(config_path),
        })
        return GitConfig()

    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        log.error("git config is not valid json - ignoring it",
                  exc_info=True, extra={"path": str(config_path)})
        return GitConfig()

    repos: dict[str, RepoGrant] = {}
    for full_name, entry in raw.get("repos", {}).items():
        repos[full_name] = RepoGrant(
            full_name=full_name,
            write=bool(entry.get("write", False)),
        )

    config = GitConfig(
        author_name=str(raw.get("author_name", "JARVIS")),
        author_email=str(raw.get("author_email", "jarvis@localhost")),
        coauthor_trailer=bool(raw.get("coauthor_trailer", True)),
        allow_create=bool(raw.get("allow_create", False)),
        repos=repos,
    )
    log.info("git config loaded", extra={
        "repos": sorted(repos),
        "writable": sorted(r for r, g in repos.items() if g.write),
        "allow_create": config.allow_create,
    })
    return config
