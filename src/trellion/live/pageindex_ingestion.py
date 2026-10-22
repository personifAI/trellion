# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Document ingestion agent for PageIndex.

Accepts PDF, Markdown, HTML, Office (DOCX/PPTX/XLSX), plain text and audio
(transcribed by the stack's own STT through the gateway) documents.  When ``markitdown`` is installed, all supported formats
are first converted to Markdown via MarkItDown (with optional local-LLM
image descriptions), then split and chunked.  Falls back to built-in
parsers when MarkItDown is unavailable.
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
from pathlib import Path
from typing import Any, Optional

import httpx
import structlog

from ._tls import ssl_verify, client_cert
from .pageindex_store import PageIndexStore

from trellion.live.llm_throttle import LLMUnavailable

log = structlog.get_logger(__name__)

# ── Optional: MarkItDown (Microsoft) ──────────────────────────────────
try:
    from markitdown import MarkItDown
    MARKITDOWN_AVAILABLE = True
except ImportError:
    MarkItDown = None  # type: ignore[assignment,misc]
    MARKITDOWN_AVAILABLE = False

# Chunking constants — a 3 chars/token estimate (dense/technical text tokenises
# at ~2.5-3 chars/token), so 1500 chars ≈ 500 tokens. ~500 tokens is a
# retrieval-precision choice (one chunk ≈ one focused passage), NOT a limit of
# the embedding model: the default embedder (harrier) has a far larger native
# window than that. The window actually in force is the ``num_ctx`` the gateway
# sends with each embed request (``EMBED_NUM_CTX`` = 1024 tokens in
# gateway-rust/src/embedding_client.rs, sized as 2x this chunk size); input
# beyond it is truncated by the backend.
CHUNK_SIZE_CHARS = 1500  # ~500 tokens, sized for retrieval precision
CHUNK_OVERLAP_CHARS = 200  # ~65 tokens
PAGE_SPLIT_CHARS = 2000  # plain-text page boundary


def _content_hash(text: str) -> str:
    """SHA-256 hex digest of the full document text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks.

    Each chunk is up to CHUNK_SIZE_CHARS characters, with
    CHUNK_OVERLAP_CHARS overlap between consecutive chunks.
    """
    if len(text) <= CHUNK_SIZE_CHARS:
        return [text] if text.strip() else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE_CHARS
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - CHUNK_OVERLAP_CHARS
    return chunks


def _detect_tables(text: str) -> bool:
    """Heuristic: detect markdown/ASCII tables via pipe-delimited rows."""
    return bool(re.search(r"\|.*\|.*\|", text))


def _detect_figures(text: str) -> bool:
    """Heuristic: detect image references in markdown or HTML."""
    return bool(re.search(r"!\[.*?\]\(.*?\)|<img\s", text, re.IGNORECASE))


def _extract_headers(text: str) -> list[str]:
    """Extract markdown-style headers (# through ###)."""
    return re.findall(r"^#{1,3}\s+(.+)$", text, re.MULTILINE)


def _page_meta(text: str, chunks: list[str]) -> dict:
    """Build meta_json for a single page."""
    return {
        "headers": _extract_headers(text),
        "has_table": _detect_tables(text),
        "has_figure": _detect_figures(text),
        "char_count": len(text),
        "chunk_count": len(chunks),
    }


# Markdown offers six heading levels; splitting honoured exactly one. Any
# document that organises itself with `#` or `###` collapsed into one page per
# `##` section — transcripts, reports, anything with sub-structure. Measured on
# a live corpus (2026-08-16): 43% of pages exceeded PAGE_SPLIT_CHARS, 11% by
# 5-25x, the worst a single 51,619-char page holding an entire meeting
# transcript under eleven `###` timestamp headings.
#
# The embedder sends the whole page to the model, but the input window in force
# is the gateway's embed ``num_ctx`` (1024 tokens, roughly 3000 characters; the
# model's native window is larger, the stack caps it) and the backend TRUNCATES
# silently past that. (When this was measured the embedder was a 512-token
# model, which truncated at ~1500 characters; the failure shape is the same,
# only the cut point moved.) So a 51k page's embedding does not represent a
# diluted version of the page; it represents its first few thousand characters
# and nothing else. For that transcript: a header and a few minutes of small talk.
# Everything said in the meeting was absent from page-level search, not merely
# out-ranked. Retrieval then searches chunks only WITHIN pages that already
# won, so the transcript was unreachable.
#
# PAGE_SPLIT_CHARS (2000 chars, ~670 tokens) is what keeps a page inside that
# window. Page-first
# retrieval is built on the assumption that a page embedding represents its
# page — this restores it rather than working around it.
_MD_HEADING_LEVELS = (2, 3, 4, 5, 6)


def _split_markdown(text: str) -> list[str]:
    """Split markdown on headings, descending levels until pages fit the
    size policy.

    Starts at ``## `` (unchanged for documents that already fit). A section
    over PAGE_SPLIT_CHARS is re-split on the next heading level down, and so
    on; whatever is still oversized with no headings left falls through to
    `_split_plain`, which already merges and splits to the same constant.

    So the size policy is not restated here — `PAGE_SPLIT_CHARS` remains the
    single definition of how big a page should be, and this makes the markdown
    path obey the one the plain-text path always has. Size decides WHETHER to
    descend; the document's own headings decide WHERE to cut.

    Short sections are never descended, so a small document does not shatter
    into fragments.
    """
    sections = _split_on_heading(text, 2)
    return sections if sections else [text]


def _split_on_heading(text: str, level: int) -> list[str]:
    """Split on ``#`` * level, recursing into oversized sections."""
    marker = "#" * level
    parts = re.split(rf"(?=^{marker} )", text, flags=re.MULTILINE)
    sections = [p for p in parts if p.strip()]
    if not sections:
        sections = [text] if text.strip() else []

    out: list[str] = []
    for sec in sections:
        if len(sec) <= PAGE_SPLIT_CHARS:
            out.append(sec)
            continue
        deeper = level + 1
        if deeper in _MD_HEADING_LEVELS and re.search(
            rf"^{'#' * deeper} ", sec, flags=re.MULTILINE
        ):
            out.extend(_split_on_heading(sec, deeper))
        else:
            # No structure left to exploit. `_split_plain` is the existing
            # answer for "text with no headings", and it enforces the same
            # constant — reusing it keeps one implementation of the policy.
            out.extend(_split_plain(sec))
    return out


def _split_plain(text: str) -> list[str]:
    """Split plain text on double newlines, merging small blocks up to PAGE_SPLIT_CHARS."""
    raw_blocks = re.split(r"\n\s*\n", text)
    pages: list[str] = []
    current = ""
    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue
        if current and len(current) + len(block) + 2 > PAGE_SPLIT_CHARS:
            pages.append(current)
            current = block
        else:
            current = f"{current}\n\n{block}" if current else block
    if current.strip():
        pages.append(current)
    return pages if pages else [text]


def _split_html(html: str) -> list[str]:
    """Split HTML on semantic block tags.

    Produces plain-text sections by stripping tags and splitting on
    block boundaries (h1-h6, hr, section, article).
    """
    # Insert double newlines before block elements to create split points
    block_re = re.compile(
        r"<(?:h[1-6]|hr|section|article|div)\b[^>]*>",
        re.IGNORECASE,
    )
    text = block_re.sub(lambda m: "\n\n" + m.group(0), html)
    # Strip all tags
    text = re.sub(r"<[^>]+>", "", text)
    # Collapse whitespace runs (but preserve double-newline splits)
    text = re.sub(r"[ \t]+", " ", text)
    return _split_plain(text)


def _render_pdf_page_to_png(file_path: Path, page_no: int, dpi: int = 150) -> bytes:
    """Render a single PDF page to PNG bytes using pymupdf.

    ``page_no`` is 0-indexed.  Returns raw PNG bytes.
    """
    import fitz  # pymupdf

    scale = dpi / 72.0
    with fitz.open(str(file_path)) as doc:
        page = doc[page_no]
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
        return pix.tobytes("png")


def _read_pdf(file_path: Path) -> list[str]:
    """Read PDF pages as a list of per-page text strings.

    Uses pymupdf (fitz) if available. Raises ImportError otherwise.
    """
    import fitz  # pymupdf

    pages: list[str] = []
    with fitz.open(str(file_path)) as doc:
        for page in doc:
            pages.append(page.get_text())
    return pages



def _without_audio_converter(md: Any) -> Any:
    """Remove markitdown's AudioConverter, or refuse markitdown entirely.

    markitdown's AudioConverter transcribes speech with SpeechRecognition's
    ``recognize_google``: the recording is uploaded to Google's web speech API
    (no consent, no opt-out, against noomesh's no-vendor-telemetry rule). The
    zip converter converts archive members with the same converter list, so
    removing it there also covers audio inside a .zip. If this markitdown has
    no ``_converters`` list to filter, fail closed: return None so documents
    fall back to the built-in parsers rather than risk an upload.
    """
    regs = getattr(md, "_converters", None)
    if not isinstance(regs, list):
        log.error("markitdown_audio_guard_unavailable", reason="no _converters list; markitdown disabled")
        return None
    md._converters = [
        r for r in regs
        if type(getattr(r, "converter", r)).__name__ != "AudioConverter"
    ]
    return md


# ── Audio: transcribed by the stack's own STT, through the gateway ────────
#
# markitdown never sees audio (see `_without_audio_converter`). Audio files go
# to the gateway's `POST /api/voice/transcribe/detailed`, which proxies the
# STT provider the operator configured (the voice overlay's noo-whisper by
# default). There is no other path: if the gateway cannot transcribe, the
# ingest of that file fails and says why.

#: The extensions the stack's whisper decodes (the `noo voice transcribe`
#: MIME table, noo-cli/src/commands/voice.rs). The MIME is advisory — the
#: service sniffs the container from the bytes and the filename suffix.
AUDIO_MIME_TYPES: dict[str, str] = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".webm": "audio/webm",
}

#: The gateway's body limit on the transcribe routes
#: (`TRANSCRIBE_BODY_LIMIT`, gateway-rust/src/routes/communication.rs).
GATEWAY_TRANSCRIBE_BODY_LIMIT = 10 * 1024 * 1024
#: The limit applies to the whole multipart body, so the file itself must
#: leave room for the part headers and boundaries.
_MULTIPART_HEADROOM = 16 * 1024
MAX_AUDIO_BYTES = GATEWAY_TRANSCRIBE_BODY_LIMIT - _MULTIPART_HEADROOM

#: The gateway waits 120 s on the STT provider; a little longer here so the
#: gateway's own answer (a 502 naming the provider) arrives instead of a
#: client-side timeout. One request per file, no retries.
TRANSCRIBE_TIMEOUT_SECS = 150.0

TRANSCRIBE_PATH = "/api/voice/transcribe/detailed"

#: Test seam: an ``httpx.MockTransport`` here exercises the real client
#: construction, auth flow and multipart encoding (gateway_auth has the same
#: seam for the same reason).
_transcribe_transport: httpx.AsyncBaseTransport | None = None


class AudioTranscriptionError(ValueError):
    """An audio file could not be transcribed by this stack, so it was not
    ingested. The message names the cause and the fix; nothing falls back to
    any other transcription service."""


_VOICE_OFF = (
    "voice transcription is off on this stack; enable voice to ingest audio"
)


def _gateway_error_detail(resp: httpx.Response) -> str:
    try:
        body = resp.json()
        if isinstance(body, dict) and body.get("error"):
            return str(body["error"])[:300]
    except Exception:  # noqa: BLE001 — a non-JSON error body is still a body
        pass
    return (resp.text or "").strip()[:300]


def _transcription_failure(name: str, resp: httpx.Response) -> AudioTranscriptionError:
    status = resp.status_code
    detail = _gateway_error_detail(resp)
    suffix = f" (gateway: {detail})" if detail else ""
    if status in (401, 403):
        return AudioTranscriptionError(
            f"cannot ingest {name}: the gateway refused trellion's service credential on "
            f"{TRANSCRIBE_PATH} (HTTP {status}), so the audio was not transcribed{suffix}"
        )
    if status == 413:
        return AudioTranscriptionError(
            f"cannot ingest {name}: the gateway refused the upload as too large (HTTP 413); "
            f"its transcribe limit is {GATEWAY_TRANSCRIBE_BODY_LIMIT // (1024 * 1024)} MiB"
        )
    if status == 404 or status in (502, 503, 504):
        return AudioTranscriptionError(
            f"cannot ingest {name}: {_VOICE_OFF} (HTTP {status} from {TRANSCRIBE_PATH}){suffix}"
        )
    if status == 400:
        return AudioTranscriptionError(
            f"cannot ingest {name}: the gateway could not read the audio (HTTP 400){suffix}"
        )
    return AudioTranscriptionError(
        f"cannot ingest {name}: transcription failed (HTTP {status} from {TRANSCRIBE_PATH}){suffix}"
    )


def _fmt_ts(secs: float) -> str:
    secs = max(int(secs), 0)
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _transcript_pages(body: dict) -> tuple[list[str], str]:
    """Pages and full text from a transcribe reply.

    With segments, consecutive segments are packed into pages of at most
    ``PAGE_SPLIT_CHARS``, each headed by its time range, so a long recording
    is many retrievable pages (and a hit says where in the recording it is)
    rather than one page whose embedding sees only its first minutes. Without
    segments, the text is split like any plain text.
    """
    text = str(body.get("text") or "").strip()
    segments = body.get("segments")
    if not isinstance(segments, list) or not segments:
        return _split_plain(text), text

    pages: list[str] = []
    buf: list[str] = []
    buf_len = 0
    start = end = 0.0

    def flush() -> None:
        if buf:
            pages.append(f"[{_fmt_ts(start)}–{_fmt_ts(end)}]\n" + " ".join(buf))

    for seg in segments:
        if not isinstance(seg, dict):
            continue
        seg_text = str(seg.get("text") or "").strip()
        if not seg_text:
            continue
        seg_start = float(seg.get("start") or 0.0)
        seg_end = float(seg.get("end") or seg_start)
        if buf and buf_len + len(seg_text) + 1 > PAGE_SPLIT_CHARS:
            flush()
            buf, buf_len = [], 0
        if not buf:
            start = seg_start
        buf.append(seg_text)
        buf_len += len(seg_text) + 1
        end = seg_end
    flush()

    if not pages:
        return _split_plain(text), text
    return pages, text or "\n".join(pages)


class DocumentIngestor:
    """Ingests documents into PageIndex via PageIndexStore.

    Supports PDF, Markdown, HTML, Office documents (DOCX/PPTX/XLSX/CSV),
    and plain text.  When ``markitdown`` is installed, non-text formats
    are first converted to Markdown.  After a successful ingestion it
    publishes an ``ingest.doc.parsed`` NATS event when a
    ``ClusterMessaging`` instance is provided.
    """

    # Extensions that MarkItDown can handle but our built-in parsers cannot.
    _MARKITDOWN_EXTENSIONS: set[str] = {
        ".docx", ".pptx", ".xlsx", ".xls", ".csv",
        ".json", ".xml", ".zip",
        ".jpg", ".jpeg", ".png",
        # No audio: markitdown transcribes .wav/.mp3 with SpeechRecognition's
        # recognize_google, i.e. it uploads the recording to Google's web
        # speech API. Audio goes to the stack's own STT instead
        # (AUDIO_MIME_TYPES, `_ingest_audio_file`) and markitdown's
        # AudioConverter is removed below (zip members reuse the same
        # converter list, so audio inside a .zip is never transcribed).
    }

    def __init__(
        self,
        store: PageIndexStore,
        messaging: Optional[object] = None,
        llm_url: Optional[str] = None,
        llm_model: Optional[str] = None,
        ocr_config: Optional[Any] = None,
    ) -> None:
        self.store = store
        self.messaging = messaging
        self._markitdown: Any = None

        # OCR settings (from config.OCRConfig or None)
        self._ocr_enabled = getattr(ocr_config, "enabled", False) if ocr_config else False
        self._ocr_model = getattr(ocr_config, "model", "deepseek-ai/DeepSeek-OCR") if ocr_config else "deepseek-ai/DeepSeek-OCR"
        self._ocr_min_text = getattr(ocr_config, "min_text_length", 50) if ocr_config else 50
        self._ocr_dpi = getattr(ocr_config, "dpi", 150) if ocr_config else 150
        self._ocr_prompt = getattr(ocr_config, "prompt", "<image>\nFree OCR.") if ocr_config else "<image>\nFree OCR."

        if MARKITDOWN_AVAILABLE:
            kwargs: dict[str, Any] = {}
            # Wire local LLM for image descriptions in PPTX/images
            if llm_url and llm_model:
                try:
                    from openai import OpenAI
                    from trellion.live.gateway_auth import GatewayHmacAuth
                    # The gateway authenticates its /v1 surface and presents
                    # the cluster CA; the SDK's default client carried
                    # neither, so image descriptions only ever worked
                    # against a bare http backend.
                    kwargs["llm_client"] = OpenAI(
                        base_url=llm_url.rstrip("/") + "/v1",
                        api_key="unused",
                        http_client=httpx.Client(
                            timeout=300.0,
                            verify=ssl_verify(),
                            cert=client_cert(),
                            auth=GatewayHmacAuth(),
                        ),
                    )
                    kwargs["llm_model"] = llm_model
                except ImportError:
                    pass  # openai package not installed — skip LLM descriptions
            self._markitdown = _without_audio_converter(MarkItDown(**kwargs))
            log.info(
                "markitdown_available",
                llm_enabled=bool(kwargs.get("llm_client")),
            )

    # ------------------------------------------------------------------
    # OCR helpers
    # ------------------------------------------------------------------

    async def _ocr_image(self, image_bytes: bytes) -> str:
        """Send an image to the OCR model via the gateway inference API.

        Returns extracted text (Markdown).  On failure returns an empty
        string so ingestion can continue without blocking.
        """
        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:image/png;base64,{b64}"

        payload = {
            "model": self._ocr_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._ocr_prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "max_tokens": 4096,
        }

        try:
            from trellion.live.gateway_auth import GatewayHmacAuth
            from trellion.live.llm_throttle import get_llm_throttle

            # NOO_GATEWAY_URL, never localhost: this code ran inside the
            # gateway once, and `localhost:8080` has meant trellion itself
            # since the worker pool moved to its own container.
            gateway = os.environ.get("NOO_GATEWAY_URL", "https://gateway:8080").rstrip("/")
            async def _post() -> dict:
                async with httpx.AsyncClient(
                    timeout=120.0, verify=ssl_verify(), cert=client_cert(), auth=GatewayHmacAuth(),
                ) as client:
                    resp = await client.post(f"{gateway}/api/inference", json=payload)
                    resp.raise_for_status()
                    return resp.json()

            # Per-page OCR in a loop: same throttle + per-model breaker as
            # every other LLM caller, so a missing OCR model is not retried
            # for every page of every document.
            result = await get_llm_throttle().call(self._ocr_model, _post, what="ocr")

            # Standard OpenAI chat completion response
            choices = result.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")

            # Flat response from gateway proxy
            return result.get("response", "")
        except LLMUnavailable as exc:
            log.debug("ocr_unavailable", model=exc.model, reason=exc.reason)
            return ""
        except Exception:
            log.warning("ocr_failed", model=self._ocr_model, exc_info=True)
            return ""

    async def _ingest_image_file(
        self,
        file_path: Path,
        title: Optional[str] = None,
        source_uri: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """Ingest a standalone image file (.jpg/.png).

        Tries MarkItDown first.  If the result is too short (likely just
        EXIF metadata) and OCR is enabled, falls back to OCR.
        """
        title = title or file_path.stem
        md_text = ""

        # Try MarkItDown (may extract EXIF or embedded text)
        if self._markitdown is not None:
            try:
                import asyncio
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None, self._markitdown.convert, str(file_path),
                )
                md_text = (result.text_content or "").strip()
            except Exception:
                log.debug("markitdown_image_fallback", file=str(file_path))

        # If MarkItDown gave us little/nothing, try OCR
        if len(md_text) < self._ocr_min_text and self._ocr_enabled:
            image_bytes = file_path.read_bytes()
            ocr_text = await self._ocr_image(image_bytes)
            if ocr_text.strip():
                log.info("ocr_applied", file=str(file_path), chars=len(ocr_text))
                md_text = ocr_text

        if not md_text.strip():
            raise ValueError(f"No text extracted from image {file_path}")

        return await self.ingest_markdown(
            md_text, title=title, source_uri=source_uri or str(file_path),
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------

    async def _transcribe(self, file_path: Path) -> dict:
        """POST one audio file to the gateway's transcribe route; return the
        reply JSON. Raises :class:`AudioTranscriptionError` on any failure.

        One request, bounded size and time. The service credential and mTLS
        are the same as every other trellion → gateway call
        (``GatewayHmacAuth``, ``ssl_verify()``, ``client_cert()``).
        """
        from trellion.live.gateway_auth import GatewayCredentialMissing, GatewayHmacAuth

        name = file_path.name
        size = file_path.stat().st_size
        if size == 0:
            raise AudioTranscriptionError(f"cannot ingest {name}: the audio file is empty")
        if size > MAX_AUDIO_BYTES:
            raise AudioTranscriptionError(
                f"cannot ingest {name}: {size / (1024 * 1024):.1f} MiB is over the "
                f"{MAX_AUDIO_BYTES / (1024 * 1024):.2f} MiB audio limit (the gateway's "
                f"transcribe routes accept at most "
                f"{GATEWAY_TRANSCRIBE_BODY_LIMIT // (1024 * 1024)} MiB per upload); "
                f"split the recording or re-encode it at a lower bitrate"
            )
        audio = file_path.read_bytes()
        mime = AUDIO_MIME_TYPES.get(file_path.suffix.lower(), "application/octet-stream")
        gateway = os.environ.get("NOO_GATEWAY_URL", "https://gateway:8080").rstrip("/")
        url = f"{gateway}{TRANSCRIBE_PATH}"

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(TRANSCRIBE_TIMEOUT_SECS, connect=10.0),
                verify=ssl_verify(),
                cert=client_cert(),
                auth=GatewayHmacAuth(),
                transport=_transcribe_transport,
            ) as client:
                resp = await client.post(url, files={"file": (name, audio, mime)})
        except GatewayCredentialMissing as exc:
            raise AudioTranscriptionError(f"cannot ingest {name}: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            # The local refusal circuit (GatewayRefused) answers with the
            # status the gateway gave moments ago, without a request.
            raise _transcription_failure(name, exc.response) from exc
        except httpx.TimeoutException as exc:
            raise AudioTranscriptionError(
                f"cannot ingest {name}: the gateway did not finish transcribing within "
                f"{TRANSCRIBE_TIMEOUT_SECS:.0f}s"
            ) from exc
        except httpx.RequestError as exc:
            raise AudioTranscriptionError(
                f"cannot ingest {name}: the gateway is unreachable at {gateway} "
                f"({type(exc).__name__}: {exc})"
            ) from exc

        if resp.status_code != 200:
            err = _transcription_failure(name, resp)
            log.warning("audio_transcription_failed", file=str(file_path), status=resp.status_code, reason=str(err))
            raise err
        try:
            body = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise AudioTranscriptionError(
                f"cannot ingest {name}: the gateway's transcribe reply was not JSON"
            ) from exc
        if not isinstance(body, dict):
            raise AudioTranscriptionError(
                f"cannot ingest {name}: the gateway's transcribe reply was not an object"
            )
        return body

    async def _ingest_audio_file(
        self,
        file_path: Path,
        title: Optional[str] = None,
        source_uri: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """Transcribe an audio file with the stack's STT and ingest the
        transcript, one page per ~``PAGE_SPLIT_CHARS`` of speech, each headed
        by its time range."""
        title = title or file_path.stem
        body = await self._transcribe(file_path)
        pages, full_text = _transcript_pages(body)
        if not full_text.strip():
            raise AudioTranscriptionError(
                f"cannot ingest {file_path.name}: the transcript is empty (no speech recognised)"
            )
        log.info(
            "audio_transcribed",
            file=str(file_path),
            chars=len(full_text),
            pages=len(pages),
            language=body.get("language"),
            duration=body.get("duration"),
        )
        return await self._ingest_pages(
            pages=pages,
            full_text=full_text,
            title=title,
            source_uri=source_uri or str(file_path),
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ingest_file(
        self,
        file_path: Path,
        title: Optional[str] = None,
        source_uri: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """Ingest a file (auto-detect type from extension).

        Conversion priority:
        1. Audio (AUDIO_MIME_TYPES) — transcribed by the stack's own STT
           through the gateway's ``/api/voice/transcribe/detailed``; never
           markitdown, never any other service.
        2. MarkItDown (if installed) — handles Office, images, etc.
        3. Built-in parsers — PDF (pymupdf), Markdown, HTML, plain text.

        *source_uri* defaults to *file_path* but callers should pass the
        **original** file path when the file has been copied to a working
        directory, so that watched-folder dedup can match it later.

        *metadata* is an optional structured dict (folder_path, drive_id,
        tags, etc.) stored on the docs row for future narrowers to query.

        Returns the doc_id (new or existing if duplicate hash).
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        title = title or file_path.stem
        source_uri = source_uri or str(file_path)

        # --- Audio: the stack's own STT via the gateway, never markitdown ---
        if suffix in AUDIO_MIME_TYPES:
            return await self._ingest_audio_file(file_path, title=title, source_uri=source_uri, metadata=metadata)

        # --- MarkItDown path for formats we can't handle natively ---
        if suffix in self._MARKITDOWN_EXTENSIONS:
            # For images: try MarkItDown first, fall back to OCR
            if suffix in (".jpg", ".jpeg", ".png"):
                return await self._ingest_image_file(file_path, title=title, source_uri=source_uri, metadata=metadata)
            if self._markitdown is not None:
                return await self._ingest_via_markitdown(file_path, title=title, source_uri=source_uri, metadata=metadata)
            log.warning(
                "markitdown_not_available",
                file=str(file_path),
                hint="pip install 'markitdown[all]' to ingest Office/media files",
            )
            raise ValueError(
                f"Cannot ingest {suffix} files without markitdown. "
                f"Install with: pip install 'markitdown[all]'"
            )

        # --- MarkItDown path for formats we *can* handle natively ---
        # Prefer MarkItDown for PDF/HTML too when available (better structure)
        if self._markitdown is not None and suffix in (".pdf", ".html", ".htm"):
            try:
                return await self._ingest_via_markitdown(file_path, title=title, source_uri=source_uri, metadata=metadata)
            except Exception:
                log.debug("markitdown_fallback", file=str(file_path), suffix=suffix)
                # Fall through to built-in parsers

        # --- Built-in parsers ---
        if suffix == ".pdf":
            return await self.ingest_pdf(file_path, title=title, source_uri=source_uri, metadata=metadata)

        text = file_path.read_text(encoding="utf-8")

        if suffix in (".md", ".markdown"):
            return await self.ingest_markdown(text, title=title, source_uri=source_uri, metadata=metadata)
        if suffix in (".html", ".htm"):
            return await self._ingest_pages(
                pages=_split_html(text),
                full_text=text,
                title=title,
                source_uri=source_uri,
                metadata=metadata,
            )
        # Default: plain text
        return await self.ingest_text(text, title=title, source_uri=source_uri, metadata=metadata)

    async def ingest_text(
        self,
        text: str,
        title: str,
        source_uri: str = "",
        metadata: Optional[dict] = None,
    ) -> str:
        """Ingest a plain-text document."""
        pages = _split_plain(text)
        return await self._ingest_pages(
            pages=pages,
            full_text=text,
            title=title,
            source_uri=source_uri,
            metadata=metadata,
        )

    async def ingest_markdown(
        self,
        text: str,
        title: str,
        source_uri: str = "",
        metadata: Optional[dict] = None,
    ) -> str:
        """Ingest a Markdown document (split on ``## `` headers)."""
        pages = _split_markdown(text)
        return await self._ingest_pages(
            pages=pages,
            full_text=text,
            title=title,
            source_uri=source_uri,
            metadata=metadata,
        )

    async def ingest_pdf(
        self,
        file_path: Path,
        title: Optional[str] = None,
        source_uri: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """Ingest a PDF document.

        Requires pymupdf (``pip install pymupdf``).  Raises
        ``ImportError`` if the library is not available.

        When OCR is enabled, pages with fewer than ``min_text_length``
        characters are rendered to PNG and sent to the OCR model.
        """
        file_path = Path(file_path)
        title = title or file_path.stem

        page_texts = _read_pdf(file_path)

        # OCR sparse/scanned pages
        if self._ocr_enabled:
            for idx, text in enumerate(page_texts):
                if len(text.strip()) < self._ocr_min_text:
                    try:
                        png_bytes = _render_pdf_page_to_png(
                            file_path, page_no=idx, dpi=self._ocr_dpi,
                        )
                        ocr_text = await self._ocr_image(png_bytes)
                        if ocr_text.strip():
                            log.info(
                                "ocr_applied",
                                page=idx + 1,
                                chars=len(ocr_text),
                            )
                            page_texts[idx] = ocr_text
                    except Exception:
                        log.warning(
                            "ocr_page_failed",
                            page=idx + 1,
                            file=str(file_path),
                            exc_info=True,
                        )

        full_text = "\n".join(page_texts)

        return await self._ingest_pages(
            pages=page_texts,
            full_text=full_text,
            title=title,
            source_uri=source_uri or str(file_path),
            metadata=metadata,
        )

    async def _ingest_via_markitdown(
        self,
        file_path: Path,
        title: Optional[str] = None,
        source_uri: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """Convert any file to Markdown via MarkItDown, then ingest.

        MarkItDown runs synchronously so we call it in a thread to avoid
        blocking the event loop.
        """
        import asyncio

        file_path = Path(file_path)
        title = title or file_path.stem

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, self._markitdown.convert, str(file_path),
        )
        md_text = result.text_content
        if not md_text or not md_text.strip():
            raise ValueError(f"MarkItDown returned empty content for {file_path}")

        log.info(
            "markitdown_converted",
            file=str(file_path),
            chars=len(md_text),
        )
        return await self.ingest_markdown(
            md_text, title=title, source_uri=source_uri or str(file_path),
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _ingest_pages(
        self,
        pages: list[str],
        full_text: str,
        title: str,
        source_uri: str,
        metadata: Optional[dict] = None,
    ) -> str:
        """Core ingestion loop shared by all format-specific methods."""
        content_hash = _content_hash(full_text)

        # insert_doc returns (doc_id, is_new).  When is_new is False the
        # content hash already exists — skip page/chunk creation to avoid
        # duplicating data in the DB. Metadata is merged into the existing
        # row on dedup hits (see insert_doc) so a re-ingest from a different
        # source can enrich without duplicating content.
        doc_id, is_new = await self.store.insert_doc(
            source_uri=source_uri,
            title=title,
            content_hash=content_hash,
            metadata=metadata,
        )

        if not is_new:
            log.info(
                "document_already_ingested",
                doc_id=doc_id,
                title=title,
                source_uri=source_uri,
            )
            return doc_id

        page_count = 0
        for page_no, page_text in enumerate(pages, start=1):
            page_text = page_text.strip()
            if not page_text:
                continue

            chunks = _chunk_text(page_text)
            meta = _page_meta(page_text, chunks)

            page_id = await self.store.insert_page(
                doc_id=doc_id,
                page_no=page_no,
                text_content=page_text,
                meta=meta,
            )

            if chunks:
                await self.store.insert_chunks(page_id, chunks)

            page_count += 1

        log.info(
            "document_ingested",
            doc_id=doc_id,
            title=title,
            page_count=page_count,
            source_uri=source_uri,
        )

        await self._publish_event(doc_id, title, page_count, source_uri)
        return doc_id

    async def _publish_event(
        self,
        doc_id: str,
        title: str,
        page_count: int,
        source_uri: str,
    ) -> None:
        """Publish ``ingest.doc.parsed`` NATS event if messaging is available."""
        if self.messaging is None:
            return
        try:
            await self.messaging.publish_event(
                "ingest.doc.parsed",
                {
                    "doc_id": doc_id,
                    "title": title,
                    "page_count": page_count,
                    "source_uri": source_uri,
                },
            )
        except Exception:
            log.warning("nats_publish_failed", doc_id=doc_id, exc_info=True)
