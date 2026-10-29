# =============================================================================
# src/jarvis/core/money/mailbox.py - reading JARVIS's own inbox
# =============================================================================
#
# The owner's main Gmail forwards bank alerts (and later, statements) to a
# mailbox that exists only for JARVIS. This reads it over IMAP with an app
# password. His real inbox is never touched.
#
# NEW MAIL ONLY. The first run notes where the inbox stands and imports
# nothing before it - he asked for no history. After that, every message
# with a higher UID is read once. IMAP UIDs only grow within a mailbox;
# if Gmail ever resets them (UIDVALIDITY changes) we start again from the
# current end rather than re-reading old mail.
#
# imaplib is blocking, so each poll runs in a worker thread.
# =============================================================================

from __future__ import annotations

import asyncio
import email
import imaplib
from dataclasses import dataclass
from email.message import Message
from typing import Protocol


@dataclass(frozen=True)
class Fetched:
    uid: int
    message: Message


class MailSource(Protocol):
    """What the money service needs from a mailbox. Faked in tests."""

    async def fetch_since(self, last_uid: int | None, validity: str | None
                          ) -> tuple[list[Fetched], int, str]: ...


class ImapMailbox:
    def __init__(self, host: str, user: str, password: str, folder: str = "INBOX",
                 limit: int = 50) -> None:
        self._host = host
        self._user = user
        self._password = password
        self._folder = folder
        self._limit = limit

    async def fetch_since(self, last_uid: int | None, validity: str | None
                          ) -> tuple[list[Fetched], int, str]:
        """Messages after last_uid, the new high-water mark, and UIDVALIDITY."""
        return await asyncio.to_thread(self._fetch, last_uid, validity)

    def _fetch(self, last_uid: int | None, validity: str | None
               ) -> tuple[list[Fetched], int, str]:
        with imaplib.IMAP4_SSL(self._host, timeout=30) as imap:
            imap.login(self._user, self._password)
            status, data = imap.select(self._folder, readonly=True)
            if status != "OK":
                raise RuntimeError(f"cannot open {self._folder}")
            current_validity = self._validity(imap)
            highest = self._highest_uid(imap)
            if last_uid is None or validity != current_validity:
                return [], highest, current_validity      # start from here

            status, data = imap.uid("SEARCH", f"UID {last_uid + 1}:*")
            uids = [int(u) for u in (data[0] or b"").split() if int(u) > last_uid]
            uids = sorted(uids)[: self._limit]
            fetched: list[Fetched] = []
            for uid in uids:
                status, parts = imap.uid("FETCH", str(uid), "(RFC822)")
                raw = next((p[1] for p in parts if isinstance(p, tuple)), None)
                if status == "OK" and isinstance(raw, bytes):
                    fetched.append(Fetched(uid, email.message_from_bytes(raw)))
            mark = max(uids) if uids else last_uid
            return fetched, mark, current_validity

    @staticmethod
    def _validity(imap: imaplib.IMAP4_SSL) -> str:
        _, data = imap.response("UIDVALIDITY")
        return (data[0].decode() if data and data[0] else "")

    @staticmethod
    def _highest_uid(imap: imaplib.IMAP4_SSL) -> int:
        status, data = imap.uid("SEARCH", "ALL")
        uids = [int(u) for u in (data[0] or b"").split()] if status == "OK" else []
        return max(uids) if uids else 0
