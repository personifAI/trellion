# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""RAG ingest never sends audio to Google.

markitdown's AudioConverter transcribes .wav/.mp3 with SpeechRecognition's
``recognize_google`` (an upload to Google's web speech API). Ingest refuses
audio extensions and strips the converter, which also covers audio inside a
.zip (the zip converter reuses the same converter list).
"""
from __future__ import annotations

import pytest

from trellion.live import pageindex_ingestion as pi


class AudioConverter:  # same class name markitdown uses
    pass


class DocxConverter:
    pass


class _Reg:
    def __init__(self, converter):
        self.converter = converter


class _FakeMd:
    def __init__(self):
        self._converters = [_Reg(DocxConverter()), _Reg(AudioConverter()), _Reg(DocxConverter())]


def test_audio_extensions_are_not_ingestible() -> None:
    exts = pi.PageIndexIngestor._MARKITDOWN_EXTENSIONS if hasattr(pi, "PageIndexIngestor") else next(
        v for v in vars(pi).values() if isinstance(v, type) and hasattr(v, "_MARKITDOWN_EXTENSIONS")
    )._MARKITDOWN_EXTENSIONS
    assert exts, "found no _MARKITDOWN_EXTENSIONS: this test parsed the wrong module"
    assert not {".wav", ".mp3", ".m4a", ".flac", ".ogg"} & set(exts)


def test_the_audio_converter_is_removed_and_the_rest_kept() -> None:
    md = pi._without_audio_converter(_FakeMd())
    names = [type(r.converter).__name__ for r in md._converters]
    assert names == ["DocxConverter", "DocxConverter"], names


def test_an_unfilterable_markitdown_is_refused() -> None:
    class _NoList:
        pass
    assert pi._without_audio_converter(_NoList()) is None


def test_real_markitdown_has_no_audio_converter_after_the_guard() -> None:
    markitdown = pytest.importorskip("markitdown")
    md = pi._without_audio_converter(markitdown.MarkItDown())
    names = {type(r.converter).__name__ for r in md._converters}
    assert "AudioConverter" not in names
    assert "DocxConverter" in names, f"guard removed too much: {sorted(names)}"
