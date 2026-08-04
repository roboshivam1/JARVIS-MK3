# =============================================================================
# src/jarvis/worker/tools/files.py - reading and writing project files
# =============================================================================
#
# The capability DAEDALUS was missing. With only "run this script" and
# "save this file", the agent could express a project only as one
# enormous script - which is why prose ended up inside .py files and why
# multi-file work produced run-1.py through run-10.py.
#
# EDIT_FILE EARNS ITS PLACE. Rewriting a three-hundred-line file to
# change one line costs three hundred lines of output tokens, and invites
# the model to "improve" something else while it is in there. A surgical
# replace costs two strings.
#
# Its safety property is the exactly-once match: zero matches means the
# model's assumption about the file was wrong, several means the edit is
# ambiguous about which one it meant. Both are refusals rather than
# guesses, and both hand back a message specific enough to fix.
#
# Every path goes through Workspace.resolve_safe(). These tools run
# OUTSIDE the kernel sandbox - they have to, since the sandbox has no
# persistent filesystem - so the scoping here is the whole boundary.
# =============================================================================

from __future__ import annotations

from pathlib import Path

from jarvis.common.log import get_logger
from jarvis.worker.workspace import PathEscape, Workspace

log = get_logger("worker.tools.files")

# Enough to read a substantial source file whole; short of blowing the
# context window on a vendored dependency someone checked in.
MAX_READ_CHARS = 30_000

# Never read or write these as text, whatever is asked.
_BINARY_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".tar", ".gz",
    ".woff", ".woff2", ".ttf", ".so", ".dylib", ".pyc", ".db",
}


class FileTools:
    """Scoped file operations over the workspace."""

    def __init__(self, workspace: Workspace) -> None:
        self._ws = workspace

    def read(self, path: str) -> str:
        """Read a file. Truncated if very long, with the size stated so
        the agent knows it is seeing part of something."""
        try:
            target = self._ws.resolve_safe(path)
        except PathEscape as exc:
            return f"error: {exc}"

        if not target.exists():
            return f"error: {path} does not exist."
        if target.is_dir():
            return f"error: {path} is a directory. Use list_files or tree."
        if target.suffix.lower() in _BINARY_SUFFIXES:
            return (
                f"error: {path} is binary ({target.suffix}). Read it in "
                f"code instead - pypdf for PDFs, PIL for images."
            )

        try:
            content = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"error: {path} is not valid UTF-8 text."
        except OSError as exc:
            return f"error reading {path}: {exc}"

        if len(content) > MAX_READ_CHARS:
            return (
                content[:MAX_READ_CHARS]
                + f"\n\n[truncated - the file is {len(content)} characters]"
            )
        return content

    def write(self, path: str, content: str) -> str:
        """Create or overwrite a file, making parent directories as
        needed. Overwriting is allowed: code gets rewritten constantly,
        and requiring a separate call for every change would make the
        debug loop needlessly slow."""
        try:
            target = self._ws.resolve_safe(path)
        except PathEscape as exc:
            return f"error: {exc}"

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except OSError as exc:
            return f"error writing {path}: {exc}"

        lines = content.count("\n") + 1
        return f"Wrote {path} ({lines} lines, {len(content)} characters)."

    def edit(self, path: str, old: str, new: str) -> str:
        """Replace an exact string, which must appear exactly once.

        Zero matches means the file does not say what the agent thought
        it said - usually because it is working from memory rather than
        a fresh read. Several matches means the edit is ambiguous. In
        both cases the right answer is to refuse and say which, so the
        agent reads the file and tries again with more context.
        """
        try:
            target = self._ws.resolve_safe(path)
        except PathEscape as exc:
            return f"error: {exc}"

        if not target.exists():
            return f"error: {path} does not exist."

        try:
            content = target.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as exc:
            return f"error reading {path}: {exc}"

        occurrences = content.count(old)
        if occurrences == 0:
            return (
                f"error: that text is not in {path}. Read the file and "
                f"match what is actually there."
            )
        if occurrences > 1:
            return (
                f"error: that text appears {occurrences} times in {path}, "
                f"so the edit is ambiguous. Include surrounding lines to "
                f"make it unique."
            )

        try:
            target.write_text(content.replace(old, new, 1), encoding="utf-8")
        except OSError as exc:
            return f"error writing {path}: {exc}"
        return f"Edited {path}."

    def list_files(self, path: str = "projects") -> str:
        """Names and sizes in one directory. Use tree for structure."""
        try:
            target = self._ws.resolve_safe(path)
        except PathEscape as exc:
            return f"error: {exc}"

        if not target.exists():
            return f"error: {path} does not exist."
        if not target.is_dir():
            return f"error: {path} is a file."

        entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name))
        if not entries:
            return f"{path} is empty."

        lines = []
        for entry in entries:
            if entry.is_dir():
                lines.append(f"{entry.name}/")
            else:
                lines.append(f"{entry.name}  ({entry.stat().st_size}b)")
        return "\n".join(lines)

    def tree(self, path: str = "projects") -> str:
        """The shape of a directory. The single most useful orientation
        tool - an agent that can see a project's structure stops guessing
        at filenames."""
        return self._ws.tree(path)

    def delete(self, path: str) -> str:
        """Remove a file. Directories are refused: recursive deletion by
        an agent is a much larger mistake than a wrong file, and there is
        no undo here."""
        try:
            target = self._ws.resolve_safe(path)
        except PathEscape as exc:
            return f"error: {exc}"

        if not target.exists():
            return f"error: {path} does not exist."
        if target.is_dir():
            return (
                f"error: {path} is a directory. Deleting directories is "
                f"not available - remove files individually, or ask the "
                f"owner."
            )

        try:
            target.unlink()
        except OSError as exc:
            return f"error deleting {path}: {exc}"
        log.info("file deleted", extra={"path": path})
        return f"Deleted {path}."
