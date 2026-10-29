# =============================================================================
# src/jarvis/core/money/parsers.py - bank alerts into transactions
# =============================================================================
#
# One parser per format, written against the real message. Union Bank of
# India's UPI debit email looks like this (body, after the greeting):
#
#     Your fund transfer request through UPI has been processed successfully.
#     Transaction Details
#     1. Payee Name : Campus P
#     2. Amount : Rs. 65.00
#     3. Channel : UPI
#     4. Transaction ID/RRN : 216155719815
#     5. Transaction Status : Success
#     6. Transaction Date and Time : 29-10-2026 11:19:38
#     7. Debit Account Number : *9572
#
# A message that is not a debit alert in a known format returns None and
# is skipped. Credits are deliberately not parsed: they are not tracked.
#
# Only the fields above leave this function. The rest of the email - the
# greeting with his name, the fraud warnings - is never stored.
# =============================================================================

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from email.message import Message
from html.parser import HTMLParser


@dataclass(frozen=True)
class ParsedDebit:
    ref: str | None
    occurred_at: datetime          # naive, the bank's local time
    amount_paise: int
    payee: str
    channel: str | None
    status: str                    # success | failed | pending
    account_last4: str | None


# -- message bodies as plain text ---------------------------------------------

class _Text(HTMLParser):
    """HTML -> text with a line break wherever a block ends."""

    _BLOCKS = {"p", "div", "br", "li", "tr", "td", "th", "h1", "h2", "h3", "h4",
               "table", "ul", "ol", "section"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("style", "script", "head"):
            self._skip += 1
        if tag in self._BLOCKS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("style", "script", "head") and self._skip:
            self._skip -= 1
        if tag in self._BLOCKS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    parser = _Text()
    parser.feed(html)
    text = "".join(parser.parts).replace("\xa0", " ")
    lines = (re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def message_text(msg: Message) -> str:
    """The readable body of an email, preferring HTML (which UBI sends)."""
    html, plain = "", ""
    parts = msg.walk() if msg.is_multipart() else [msg]
    for part in parts:
        kind = part.get_content_type()
        if kind not in ("text/html", "text/plain") or part.get_filename():
            continue
        payload = part.get_payload(decode=True)
        if not isinstance(payload, bytes):
            continue
        body = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
        if kind == "text/html" and not html:
            html = body
        elif kind == "text/plain" and not plain:
            plain = body
    return html_to_text(html) if html else plain


# -- Union Bank of India --------------------------------------------------------

def _field(text: str, label: str) -> str | None:
    match = re.search(rf"{label}\s*:\s*([^\n]+)", text, re.IGNORECASE)
    return match.group(1).strip() if match else None


def parse_ubi_debit(text: str) -> ParsedDebit | None:
    """A UBI debit alert, or None if this is not one."""
    if "Payee Name" not in text or not re.search(r"Debit Account Number", text, re.I):
        return None
    payee = _field(text, r"Payee Name")
    amount = _field(text, r"Amount")
    when = _field(text, r"Transaction Date and Time")
    if not payee or not amount or not when:
        return None
    money = re.search(r"([\d,]+(?:\.\d{1,2})?)", amount)
    stamp = re.search(r"(\d{2}-\d{2}-\d{4})\s+(\d{2}:\d{2}(?::\d{2})?)", when)
    if not money or not stamp:
        return None
    clock = stamp.group(2) if stamp.group(2).count(":") == 2 else stamp.group(2) + ":00"
    try:
        occurred = datetime.strptime(f"{stamp.group(1)} {clock}", "%d-%m-%Y %H:%M:%S")
    except ValueError:
        return None
    rupees = money.group(1).replace(",", "")
    whole, _, frac = rupees.partition(".")
    paise = int(whole) * 100 + int((frac + "00")[:2])
    if paise <= 0:
        return None

    status_raw = (_field(text, r"Transaction Status") or "success").lower()
    status = ("success" if "success" in status_raw
              else "failed" if any(w in status_raw for w in ("fail", "declin", "reject"))
              else "pending")
    ref = _field(text, r"Transaction ID/RRN")
    ref_clean = re.sub(r"\W", "", ref) if ref else None
    account = _field(text, r"Debit Account Number") or ""
    last4 = re.search(r"(\d{4})\s*$", account)
    return ParsedDebit(
        ref=ref_clean or None,
        occurred_at=occurred,
        amount_paise=paise,
        payee=payee[:120],
        channel=(_field(text, r"Channel") or None),
        status=status,
        account_last4=last4.group(1) if last4 else None,
    )


PARSERS = (parse_ubi_debit,)


def parse_alert(text: str) -> ParsedDebit | None:
    for parser in PARSERS:
        parsed = parser(text)
        if parsed is not None:
            return parsed
    return None
