# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""One data directory for video memory: NOO_VIDEO_MEMORY_PATH decides it for every
component, and a pre-unification hyphen directory is folded in without losing data."""
from __future__ import annotations

import os
import re
import threading
import types
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "trellion"
HYPHEN = "/data/video" + "-memory"  # spelled in two halves so this file passes its own guard


@pytest.fixture
def vm_path(monkeypatch, tmp_path):
    p = tmp_path / "vm_underscore"
    monkeypatch.setenv("NOO_VIDEO_MEMORY_PATH", str(p))
    return p


# ---------------------------------------------------------------- one source of truth


def test_every_component_resolves_the_env_dir(vm_path):
    from trellion.live.recall.manager import EpisodicRecallManager
    from trellion.live.video.config import VideoMemoryConfig
    from trellion.live.video.manager import VideoMemoryManager
    from trellion.live.video.traversal_archiver import TraversalArchiver

    eng = types.SimpleNamespace()
    assert VideoMemoryConfig().storage_dir == str(vm_path)

    vm = VideoMemoryManager(config=VideoMemoryConfig(), embedding_url="http://x",
                            gateway_url="http://x")
    assert vm.storage_dir == vm_path
    assert vm.traversal_archiver._faiss_base_path == str(vm_path)

    em = EpisodicRecallManager(eng)
    assert em._faiss_base_path == str(vm_path)
    assert em.lifecycle._faiss_base_path == str(vm_path)

    ta = TraversalArchiver(embedding_url="http://x")
    assert ta._faiss_base_path in (None, str(vm_path))  # resolved lazily below
    from trellion.live.video.paths import video_memory_dir
    assert video_memory_dir(ta._faiss_base_path) == str(vm_path)


def test_manager_and_recall_share_one_dir_by_default(monkeypatch):
    """No env set: still ONE default, the same for the facade and the recall manager."""
    monkeypatch.delenv("NOO_VIDEO_MEMORY_PATH", raising=False)
    from trellion.live.recall.manager import EpisodicRecallManager
    from trellion.live.video.config import VideoMemoryConfig

    em = EpisodicRecallManager(types.SimpleNamespace())
    assert VideoMemoryConfig().storage_dir == em._faiss_base_path == "/data/video_memory"


def test_pack_unpack_cli_use_the_env_dir(vm_path, monkeypatch):
    import inspect

    import trellion.cli as cli

    for fn in (cli.pack, cli.unpack):
        assert "video_memory_dir(faiss_path)" in inspect.getsource(fn)


async def test_wired_components_use_the_env_dir(vm_path, monkeypatch):
    import trellion.db.engine as eng
    from trellion.live.video.config import VideoMemoryConfig
    from trellion.live.video.manager import VideoMemoryManager

    monkeypatch.setattr(eng, "_async_engine", types.SimpleNamespace())
    vm = VideoMemoryManager(config=VideoMemoryConfig(), embedding_url="http://x",
                            gateway_url="http://x")
    await vm.startup()
    assert vm._v2_manager._faiss_base_path == str(vm_path)
    assert vm.lifecycle._faiss_base_path == str(vm_path)
    assert vm.status()["storage_dir"] == str(vm_path)


def test_no_second_default_literal_in_source():
    """The hyphen spelling and the underscore default each live in exactly one file."""
    offenders = []
    for py in SRC.rglob("*.py"):
        if py.name == "paths.py" and py.parent.name == "video":
            continue
        text = py.read_text()
        if re.search(r"/data/video[-_]memory", text):
            offenders.append(str(py.relative_to(SRC)))
    assert not offenders, f"second video-memory path literal in: {offenders}"


# ---------------------------------------------------------------- migration


def _w(p: Path, text: str, mtime: float | None = None) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    if mtime is not None:
        os.utime(p, (mtime, mtime))
    return p


def _tree(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): p.read_text() for p in root.rglob("*")
            if p.is_file() and not p.name.startswith(".legacy-m")}


def _all_contents(*roots: Path) -> list[str]:
    return sorted(p.read_text() for r in roots for p in r.rglob("*")
                  if p.is_file() and not p.name.startswith(".legacy-m"))


def test_migration_moves_files_missing_at_target(tmp_path):
    from trellion.live.video.paths import MARKER, migrate_legacy_dir

    legacy, target = tmp_path / "legacy", tmp_path / "target"
    _w(legacy / "p1/faiss/conv.index", "idx")
    _w(legacy / "p1/notes.json", "n")
    res = migrate_legacy_dir(target, legacy)
    assert res.status == "migrated" and res.moved == 2
    assert (target / "p1/faiss/conv.index").read_text() == "idx"
    assert (target / "p1/notes.json").read_text() == "n"
    assert (target / MARKER).exists()
    assert not any(legacy.rglob("*.*"))  # files moved, none left


def test_migration_parks_conflicts_and_keeps_newer(tmp_path):
    from trellion.live.video.paths import migrate_legacy_dir

    legacy, target = tmp_path / "legacy", tmp_path / "target"
    _w(legacy / "a.json", "legacy-newer", 2000)
    _w(target / "a.json", "target-older", 1000)
    _w(legacy / "b.json", "legacy-older", 1000)
    _w(target / "b.json", "target-newer", 2000)
    before = _all_contents(legacy, target)
    res = migrate_legacy_dir(target, legacy)
    assert (target / "a.json").read_text() == "legacy-newer"
    assert (target / "b.json").read_text() == "target-newer"
    assert res.parked_target == 1 and res.parked_legacy == 1 and res.park_dir
    park = Path(res.park_dir)
    assert park.parent == target and park.name.startswith(".legacy-2")
    assert (park / "from-target/a.json").read_text() == "target-older"
    assert (park / "from-legacy/b.json").read_text() == "legacy-older"
    assert _all_contents(target) == before  # every byte of both trees still exists


def test_two_faiss_indexes_are_not_mixed(tmp_path):
    from trellion.live.video.paths import migrate_legacy_dir

    legacy, target = tmp_path / "legacy", tmp_path / "target"
    _w(target / "p1/faiss/conv.index", "T-index", 1000)
    _w(target / "p1/faiss/conv_meta.json", "T-meta", 1000)
    _w(legacy / "p1/faiss/conv.index", "L-index", 5000)  # newer, but must NOT win
    _w(legacy / "p1/faiss/conv_meta.json", "L-meta", 5000)
    _w(legacy / "p1/faiss/trav.index", "L-trav", 5000)  # no conflict, same dir: parked with set
    _w(legacy / "p2/faiss/conv.index", "L-p2", 5000)  # other persona, no conflict: moves
    res = migrate_legacy_dir(target, legacy)
    assert (target / "p1/faiss/conv.index").read_text() == "T-index"
    assert (target / "p1/faiss/conv_meta.json").read_text() == "T-meta"
    assert not (target / "p1/faiss/trav.index").exists()
    assert (target / "p2/faiss/conv.index").read_text() == "L-p2"
    assert res.faiss_units_parked == 1 and res.parked_faiss == ["p1/faiss"]
    park = Path(res.park_dir) / "from-legacy"
    assert (park / "p1/faiss/conv.index").read_text() == "L-index"
    assert (park / "p1/faiss/trav.index").read_text() == "L-trav"


def test_migration_is_idempotent(tmp_path):
    from trellion.live.video.paths import migrate_legacy_dir

    legacy, target = tmp_path / "legacy", tmp_path / "target"
    _w(legacy / "x.json", "1")
    assert migrate_legacy_dir(target, legacy).status == "migrated"
    snap = _tree(target)
    _w(legacy / "late.json", "late")  # old writer still around after the marker
    again = migrate_legacy_dir(target, legacy)
    assert again.status == "already_done" and again.moved == 0
    assert _tree(target) == snap and (legacy / "late.json").read_text() == "late"


def test_nothing_to_do_without_legacy_or_when_same_dir(tmp_path):
    from trellion.live.video.paths import migrate_legacy_dir

    assert migrate_legacy_dir(tmp_path / "t", tmp_path / "absent").status == "nothing_to_do"
    d = tmp_path / "same"
    _w(d / "f", "1")
    assert migrate_legacy_dir(d, d).status == "nothing_to_do"
    assert (d / "f").read_text() == "1"


def test_only_one_concurrent_starter_migrates(tmp_path, monkeypatch):
    import trellion.live.video.paths as paths

    legacy, target = tmp_path / "legacy", tmp_path / "target"
    for i in range(20):
        _w(legacy / f"p{i}/data.json", str(i))
    before = _all_contents(legacy)

    # Slow the winner so the others genuinely overlap with it.
    real = paths._move
    gate = threading.Event()

    def slow_move(src, dst):
        gate.wait(0.5)
        real(src, dst)

    monkeypatch.setattr(paths, "_move", slow_move)
    n = 6
    barrier = threading.Barrier(n)
    results: list = []

    def starter():
        barrier.wait()
        results.append(paths.migrate_legacy_dir(target, legacy))

    ts = [threading.Thread(target=starter) for _ in range(n)]
    for t in ts:
        t.start()
    gate.set()
    for t in ts:
        t.join()
    statuses = sorted(r.status for r in results)
    assert statuses.count("migrated") == 1, statuses
    assert all(s in ("busy", "already_done") for s in statuses if s != "migrated")
    assert _all_contents(target) == before  # nothing lost, nothing duplicated


async def test_manager_startup_runs_the_migration(tmp_path, monkeypatch):
    import trellion.live.video.paths as paths
    from trellion.live.video.config import VideoMemoryConfig
    from trellion.live.video.manager import VideoMemoryManager

    legacy, target = tmp_path / "legacy", tmp_path / "target"
    _w(legacy / "p/faiss/c.index", "x")
    monkeypatch.setattr(paths, "LEGACY_DIR", str(legacy))
    cfg = VideoMemoryConfig(storage_dir=str(target))
    vm = VideoMemoryManager(config=cfg, embedding_url="http://x", gateway_url="http://x")
    await vm.startup()
    assert (target / "p/faiss/c.index").read_text() == "x"
