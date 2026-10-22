# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Audio is ingested by transcribing it with the stack's own STT.

The gateway's ``POST /api/voice/transcribe/detailed`` proxies the STT
provider the operator configured (noo-whisper by default). Ingest sends the
file there with trellion's service credential and mTLS, the same as every
other gateway call, and ingests the transcript. When the gateway cannot
transcribe (voice off, credential refused, file too big) the ingest of that
file FAILS and says why; it never falls back to anything else, and markitdown
(whose AudioConverter uploads to Google) never sees audio, not even inside a
zip.

Every test injects an ``httpx.MockTransport`` under the real client, so the
real constructor, auth flow and multipart encoding run.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import httpx
import pytest

from trellion.live import gateway_auth
from trellion.live import pageindex_ingestion as pi

GATEWAY = "https://gw.test:8080"


class _Store:
    def __init__(self) -> None:
        self.docs: list[dict] = []
        self.pages: list[dict] = []
        self.chunks: list = []

    async def insert_doc(self, **kw):
        self.docs.append(kw)
        return "doc-1", True

    async def insert_page(self, **kw):
        self.pages.append(kw)
        return f"page-{len(self.pages)}"

    async def insert_chunks(self, page_id, chunks):
        self.chunks.append((page_id, chunks))


class _Gateway:
    """A fake gateway: the token door plus the transcribe route."""

    def __init__(self, status: int = 200, body: dict | None = None) -> None:
        self.status = status
        self.body = body if body is not None else {"text": "hello from the meeting", "language": "en"}
        self.exchanges: list[httpx.Request] = []
        self.transcribes: list[httpx.Request] = []

    def token_door(self, request: httpx.Request) -> httpx.Response:
        self.exchanges.append(request)
        return httpx.Response(200, json={"token": "short-lived", "expires_in": 600})

    def route(self, request: httpx.Request) -> httpx.Response:
        request.read()
        self.transcribes.append(request)
        return httpx.Response(self.status, json=self.body)


@pytest.fixture
def gw(monkeypatch: pytest.MonkeyPatch) -> _Gateway:
    g = _Gateway()
    monkeypatch.setenv("NOO_GATEWAY_URL", GATEWAY)
    monkeypatch.setenv("NOO_GATEWAY_KEY", "noo-sk-the-key")
    monkeypatch.setattr(gateway_auth, "_token", None, raising=False)
    monkeypatch.setattr(gateway_auth, "_next_attempt", 0.0, raising=False)
    monkeypatch.setattr(gateway_auth, "_circuit", {}, raising=False)
    monkeypatch.setattr(gateway_auth, "_transport", httpx.MockTransport(g.token_door), raising=False)
    monkeypatch.setattr(pi, "_transcribe_transport", httpx.MockTransport(g.route))
    return g


def _ingestor(store: _Store) -> pi.DocumentIngestor:
    return pi.DocumentIngestor(store=store)


def _audio(tmp_path: Path, name: str = "standup.wav", size: int = 2048) -> Path:
    p = tmp_path / name
    p.write_bytes(b"RIFF" + b"\x00" * (size - 4))
    return p


async def test_audio_goes_to_the_gateway_transcribe_route_with_the_service_credential(
    gw: _Gateway, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ca, cert = object(), ("/certs/svc.crt", "/certs/svc.key")
    monkeypatch.setattr(pi, "ssl_verify", lambda: ca)
    monkeypatch.setattr(pi, "client_cert", lambda: cert)
    built: list[dict] = []
    real = httpx.AsyncClient

    class _Spy(real):  # the real client; only its kwargs are recorded
        def __init__(self, *a, **kw):
            built.append(kw)
            super().__init__(*a, **kw)

    monkeypatch.setattr(pi.httpx, "AsyncClient", _Spy)

    store = _Store()
    f = _audio(tmp_path)
    doc = await _ingestor(store).ingest_file(f, metadata={"folder_path": "/x"})

    assert doc == "doc-1"
    assert len(gw.transcribes) == 1, "exactly one transcribe request, no retries"
    req = gw.transcribes[0]
    assert req.method == "POST"
    assert str(req.url) == f"{GATEWAY}/api/voice/transcribe/detailed"
    assert req.headers["Authorization"] == "Bearer short-lived", "the exchanged service token travels"
    assert gw.exchanges and gw.exchanges[0].headers["Authorization"] == "Bearer noo-sk-the-key"
    body = req.content
    assert req.headers["Content-Type"].startswith("multipart/form-data")
    assert b'name="file"; filename="standup.wav"' in body
    assert b"Content-Type: audio/wav" in body
    assert f.read_bytes() in body
    assert built and built[0]["verify"] is ca and built[0]["cert"] == cert
    assert isinstance(built[0]["auth"], gateway_auth.GatewayHmacAuth)

    assert store.docs[0]["title"] == "standup"
    assert store.docs[0]["source_uri"] == str(f)
    assert store.docs[0]["metadata"] == {"folder_path": "/x"}
    assert [p["text_content"] for p in store.pages] == ["hello from the meeting"]


async def test_segments_become_time_headed_pages(gw: _Gateway, tmp_path: Path) -> None:
    line = "word " * 60  # ~300 chars a segment
    segs = [{"start": i * 10.0, "end": i * 10.0 + 9.5, "text": line.strip()} for i in range(20)]
    gw.body = {"text": " ".join(s["text"] for s in segs), "segments": segs}
    store = _Store()
    await _ingestor(store).ingest_file(_audio(tmp_path))

    pages = [p["text_content"] for p in store.pages]
    assert len(pages) > 1, "a long recording is more than one page"
    assert pages[0].startswith("[00:00–")
    assert pages[1].startswith("[01:"), pages[1][:20]
    for p in pages:
        assert len(p.split("\n", 1)[1]) <= pi.PAGE_SPLIT_CHARS
    joined = " ".join(p.split("\n", 1)[1] for p in pages)
    assert joined.split() == gw.body["text"].split(), "no speech lost or duplicated across pages"


@pytest.mark.parametrize("ext", sorted(pi.AUDIO_MIME_TYPES))
async def test_every_audio_extension_is_transcribed_not_markitdown(
    gw: _Gateway, tmp_path: Path, ext: str
) -> None:
    assert ext not in pi.DocumentIngestor._MARKITDOWN_EXTENSIONS
    store = _Store()
    await _ingestor(store).ingest_file(_audio(tmp_path, f"clip{ext}"))
    assert len(gw.transcribes) == 1
    assert f"Content-Type: {pi.AUDIO_MIME_TYPES[ext]}".encode() in gw.transcribes[0].content


async def test_a_file_over_the_gateway_limit_is_refused_before_sending(
    gw: _Gateway, tmp_path: Path
) -> None:
    f = _audio(tmp_path, size=pi.MAX_AUDIO_BYTES + 1)
    with pytest.raises(pi.AudioTranscriptionError, match="MiB audio limit") as ei:
        await _ingestor(_Store()).ingest_file(f)
    assert "10 MiB" in str(ei.value)
    assert gw.transcribes == [], "an oversized file was uploaded anyway"
    assert pi.MAX_AUDIO_BYTES < pi.GATEWAY_TRANSCRIBE_BODY_LIMIT == 10 * 1024 * 1024


@pytest.mark.parametrize(
    "status,error",
    [
        (502, "STT provider unreachable: error sending request for url (https://noo-whisper:8000)"),
        (503, "STT provider refused: plaintext URL"),
        (404, ""),
    ],
)
async def test_voice_off_fails_the_ingest_and_says_so(
    gw: _Gateway, tmp_path: Path, status: int, error: str
) -> None:
    gw.status, gw.body = status, ({"error": error} if error else {})
    store = _Store()
    with pytest.raises(pi.AudioTranscriptionError) as ei:
        await _ingestor(store).ingest_file(_audio(tmp_path))
    msg = str(ei.value)
    assert "voice transcription is off on this stack; enable voice to ingest audio" in msg
    assert f"HTTP {status}" in msg
    if error:
        assert error in msg, "the gateway's own reason is carried"
    assert len(gw.transcribes) == 1, "no retry storm"
    assert store.docs == [], "nothing was ingested"


async def test_a_refused_credential_names_the_cause(gw: _Gateway, tmp_path: Path) -> None:
    gw.status, gw.body = 403, {}
    with pytest.raises(pi.AudioTranscriptionError, match="refused trellion's service credential"):
        await _ingestor(_Store()).ingest_file(_audio(tmp_path))
    # GatewayHmacAuth retries a refused exchanged token ONCE with the key;
    # that is the whole retry budget.
    assert len(gw.transcribes) <= 2


async def test_a_413_from_the_gateway_names_the_limit(gw: _Gateway, tmp_path: Path) -> None:
    gw.status, gw.body = 413, {"error": "Failed to read audio: length limit exceeded"}
    with pytest.raises(pi.AudioTranscriptionError, match="too large .*10 MiB"):
        await _ingestor(_Store()).ingest_file(_audio(tmp_path))


async def test_no_credential_means_no_request(gw: _Gateway, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("NOO_GATEWAY_KEY")
    with pytest.raises(pi.AudioTranscriptionError, match="service-key provision"):
        await _ingestor(_Store()).ingest_file(_audio(tmp_path))
    assert gw.transcribes == []


async def test_an_empty_transcript_is_not_ingested(gw: _Gateway, tmp_path: Path) -> None:
    gw.body = {"text": "  ", "segments": []}
    store = _Store()
    with pytest.raises(pi.AudioTranscriptionError, match="transcript is empty"):
        await _ingestor(store).ingest_file(_audio(tmp_path))
    assert store.docs == []


async def test_audio_inside_a_zip_never_reaches_audio_converter(
    gw: _Gateway, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    markitdown = pytest.importorskip("markitdown")
    from markitdown.converters import _audio_converter, _transcribe_audio

    hits: list[str] = []

    def _boom(*a, **kw):
        hits.append("called")
        raise AssertionError("markitdown audio transcription was reached")

    monkeypatch.setattr(_audio_converter.AudioConverter, "convert", _boom)
    monkeypatch.setattr(_audio_converter.AudioConverter, "accepts", lambda *a, **kw: hits.append("accepts") or True)
    monkeypatch.setattr(_transcribe_audio, "transcribe_audio", _boom)
    monkeypatch.setattr(_audio_converter, "transcribe_audio", _boom)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("notes.txt", "the agenda for the standup")
        z.writestr("recording.wav", b"RIFF" + b"\x00" * 2044)
        z.writestr("voice.mp3", b"ID3" + b"\x00" * 2045)
    archive = tmp_path / "bundle.zip"
    archive.write_bytes(buf.getvalue())

    ing = _ingestor(_Store())
    assert ing._markitdown is not None
    try:
        await ing.ingest_file(archive)
    except ValueError:
        pass  # what the zip yields is not the point; reaching audio is
    assert hits == [], f"AudioConverter was consulted: {hits}"
    assert gw.transcribes == [], "zip members are not sent for transcription either"
    assert markitdown  # imported for the skip
