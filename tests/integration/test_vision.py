# =============================================================================
# tests/integration/test_vision.py - JARVIS sees pictures
# =============================================================================
#
# The promises: an attached photo reaches the model as a picture, in its
# own turn and never again; anything the model cannot look at (HEIC, too
# large) is described instead of breaking the call; the bytes decide the
# image type, not the file name; Telegram photos become turns, and an
# album becomes ONE turn.
# =============================================================================

from __future__ import annotations

import asyncio
import base64
import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from jarvis.agentloop.loop import LoopResult
from jarvis.core.clients import telegram as tg
from jarvis.core.db.database import Database
from jarvis.core.db.repos.artifacts import ArtifactsRepo
from jarvis.core.db.repos.events import EventsRepo
from jarvis.core.db.repos.sessions import SessionsRepo
from jarvis.core.sessionmgr import SessionManager, sniff_image

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 200
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 200
HEIC = b"\x00\x00\x00\x18ftypheic" + b"\x00" * 200


class Eyes:
    """A stand-in orchestrator that records exactly what it was shown."""

    def __init__(self) -> None:
        self.calls: list[list[dict[str, Any]]] = []

    async def respond(self, messages: list[dict[str, Any]], **_: Any) -> LoopResult:
        self.calls.append(messages)
        return LoopResult(text="Seen, sir.", llm_call_ids=[], iterations=1)


@pytest.fixture
def setup(db: Database, tmp_path: Path) -> SimpleNamespace:
    artifacts = ArtifactsRepo(db, tmp_path / "artifacts")
    eyes = Eyes()
    mgr = SessionManager(SessionsRepo(db), EventsRepo(db), eyes,  # type: ignore[arg-type]
                         artifacts=artifacts)
    return SimpleNamespace(artifacts=artifacts, eyes=eyes, mgr=mgr)


async def test_a_photo_is_shown_once_then_remembered_as_text(setup: SimpleNamespace) -> None:
    photo = await setup.artifacts.write("lunch.jpg", "image/jpeg", JPEG, "client.upload")
    await setup.mgr.handle_user_message("web", "log this lunch", attachments=[photo.id])

    content = setup.eyes.calls[-1][-1]["content"]
    assert [b["type"] for b in content] == ["image", "text"]
    assert content[0]["source"] == {"type": "base64", "media_type": "image/jpeg",
                                    "data": base64.b64encode(JPEG).decode()}
    assert content[1]["text"].startswith("log this lunch")
    assert f"[Attached image 1: lunch.jpg, artifact id {photo.id} - shown above.]" \
        in content[1]["text"]

    # The next message: the photo is NOT sent again, only marked.
    await setup.mgr.handle_user_message("web", "how much protein was that?")
    history = setup.eyes.calls[-1]
    assert history[0]["content"] == "log this lunch\n\n[sent with 1 attachment(s)]"
    assert history[-1]["content"] == "how much protein was that?"
    assert not any(isinstance(m["content"], list) for m in history)


async def test_what_cannot_be_shown_is_described(setup: SimpleNamespace) -> None:
    heic = await setup.artifacts.write("IMG_0001.HEIC", "image/heic", HEIC, "client.upload")
    huge = await setup.artifacts.write("scan.png", "image/png", PNG + b"\x00" * 7_000_000,
                                       "client.upload")
    notes = await setup.artifacts.write("notes.txt", "text/plain", b"buy eggs", "client.upload")
    await setup.mgr.handle_user_message("web", "look", attachments=[heic.id, huge.id, notes.id])

    content = setup.eyes.calls[-1][-1]["content"]
    assert isinstance(content, str)            # nothing viewable: plain text, as before
    assert "format you cannot view" in content
    assert "too large to view (7 MB)" in content
    assert "buy eggs" in content


async def test_at_most_five_pictures_a_message(setup: SimpleNamespace) -> None:
    ids = [(await setup.artifacts.write(f"p{i}.png", "image/png", PNG, "x")).id
           for i in range(6)]
    await setup.mgr.handle_user_message("web", "all of these", attachments=ids)
    content = setup.eyes.calls[-1][-1]["content"]
    assert sum(b["type"] == "image" for b in content) == 5
    assert "only the first 5 images" in content[-1]["text"]


def test_the_bytes_decide_the_type() -> None:
    assert sniff_image(PNG) == "image/png"          # whatever the name says
    assert sniff_image(JPEG) == "image/jpeg"
    assert sniff_image(b"GIF89a....") == "image/gif"
    assert sniff_image(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "image/webp"
    assert sniff_image(HEIC) is None
    assert tg._safe_name("../../etc/pass wd?.jpg") == "pass wd_.jpg"


# -- telegram -----------------------------------------------------------------

class FakeBot:
    def __init__(self) -> None:
        self.answers: list[str] = []

    async def get_file(self, file_id: str) -> Any:
        return SimpleNamespace(file_path=f"photos/{file_id}.jpg")

    async def download_file(self, path: str) -> io.BytesIO:
        await asyncio.sleep(0.01)
        return io.BytesIO(JPEG)


def photo_message(mid: int, caption: str | None = None, group: str | None = None,
                  size: int = 50_000) -> Any:
    replies: list[str] = []

    async def answer(text: str) -> None:
        replies.append(text)

    return SimpleNamespace(
        message_id=mid, caption=caption, media_group_id=group, document=None,
        photo=[SimpleNamespace(file_id=f"small{mid}", file_size=1000),
               SimpleNamespace(file_id=f"big{mid}", file_size=size)],
        chat=SimpleNamespace(id=1), answer=answer, replies=replies)


@pytest.fixture
def bridge(setup: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> Any:
    b = tg.TelegramBridge("123456:TEST", 42, setup.mgr, None,  # type: ignore[arg-type]
                          status_provider=None, artifacts=setup.artifacts)  # type: ignore[arg-type]
    b._bot = FakeBot()  # type: ignore[assignment]
    b.turns = []  # type: ignore[attr-defined]

    async def turn(message: Any, text: str, attachments: list[str] | None = None) -> None:
        b.turns.append((text, attachments))  # type: ignore[attr-defined]

    monkeypatch.setattr(b, "_turn", turn)
    monkeypatch.setattr(tg, "_ALBUM_WAIT_S", 0.05)
    return b


async def test_a_telegram_photo_is_a_turn(bridge: Any, setup: SimpleNamespace) -> None:
    await bridge._handle_file(photo_message(7, caption="dinner at the mess"))
    [(text, [artifact_id])] = bridge.turns
    assert text == "dinner at the mess"
    stored = await setup.artifacts.get(artifact_id)
    assert stored.name == "photo-7.jpg" and stored.created_by == "client.telegram"

    await bridge._handle_file(photo_message(8))
    assert bridge.turns[-1][0] == "(sent without a message)"

    too_big = photo_message(9, size=25_000_000)
    await bridge._handle_file(too_big)
    assert "over 20 MB" in too_big.replies[0] and len(bridge.turns) == 2


async def test_an_album_is_one_turn(bridge: Any) -> None:
    await asyncio.gather(*(bridge._handle_file(photo_message(i, group="g1",
                                                             caption="thali" if i == 2 else None))
                           for i in range(1, 4)))
    assert bridge.turns == []                     # still gathering
    await asyncio.sleep(0.3)
    [(text, ids)] = bridge.turns
    assert text == "thali" and len(ids) == 3
    assert bridge._albums == {}
