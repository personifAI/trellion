# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""One data directory for video memory / episodic recall.

``NOO_VIDEO_MEMORY_PATH`` decides it for every reader and writer: the
``VideoMemoryManager`` config default, ``EpisodicRecallManager``, the
lifecycle/replay/archiver components, and the ``pack``/``unpack`` CLI. This
module is the only place the default literal lives.

Before this module existed, ``VideoMemoryConfig.storage_dir`` defaulted to a
different spelling (``/data/video-memory``, hyphen) than the env var the stack
sets (``/data/video_memory``, underscore), so the manager wrote to one
directory while ``pack``/``unpack`` read the other. ``migrate_legacy_dir``
folds an old directory into the configured one, once, without deleting data.
"""
from __future__ import annotations

import errno
import fcntl
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

ENV_VAR = "NOO_VIDEO_MEMORY_PATH"
DEFAULT_DIR = "/data/video_memory"
#: The previous ``VideoMemoryConfig.storage_dir`` default. Read only by the
#: migration; nothing writes here any more.
LEGACY_DIR = "/data/video-memory"

MARKER = ".legacy-migrated"
LOCK = ".legacy-migrate.lock"
PARK_PREFIX = ".legacy-"


def video_memory_dir(explicit: str | None = None) -> str:
    """The configured directory: explicit arg, else the env var, else the default."""
    return explicit or os.environ.get(ENV_VAR) or DEFAULT_DIR


@dataclass
class MigrationResult:
    status: str  # migrated | nothing_to_do | already_done | busy | partial
    moved: int = 0
    parked_legacy: int = 0  # legacy copy kept aside because the target's was kept
    parked_target: int = 0  # target copy kept aside because the legacy one was newer
    faiss_units_parked: int = 0  # whole legacy faiss dirs parked (two indexes)
    errors: int = 0
    park_dir: str | None = None
    parked_faiss: list[str] = field(default_factory=list)


def _same(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return False


def _move(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def _faiss_unit(rel: Path) -> Path | None:
    """The ``.../faiss`` directory a relative path lives under, if any."""
    parts = rel.parts
    if "faiss" in parts[:-1]:
        return Path(*parts[: parts.index("faiss") + 1])
    return None


def migrate_legacy_dir(
    target: str | os.PathLike[str],
    legacy: str | os.PathLike[str] | None = None,
) -> MigrationResult:
    """Fold ``legacy`` into ``target`` once. Never deletes data.

    - A file missing at the target is moved there.
    - A file present in both: the newer wins the live path, the other is parked
      under ``<target>/.legacy-<timestamp>/``.
    - FAISS artifacts (anything under a ``faiss`` directory) are an index plus a
      sidecar that must stay consistent, so they are not mixed file by file:
      if any file of a persona's ``faiss`` directory exists in both places, the
      target's index stays live and the legacy files of that directory are
      parked whole. There is no vector-level merge.
    - ``<target>/.legacy-migrated`` records completion; later runs do nothing.
    - A non-blocking ``flock`` on ``<target>/.legacy-migrate.lock`` lets exactly
      one of several containers sharing the volume do the work.
    """
    target_p, legacy_p = Path(target), Path(legacy or LEGACY_DIR)
    if not legacy_p.is_dir() or _same(legacy_p, target_p):
        return MigrationResult("nothing_to_do")
    try:
        target_p.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("video_memory.legacy_migration_skipped", target=str(target_p),
                       error=str(exc)[:120])
        return MigrationResult("busy", errors=1)
    if (target_p / MARKER).exists():
        return MigrationResult("already_done")

    try:
        lock_fd = os.open(target_p / LOCK, os.O_CREAT | os.O_RDWR, 0o644)
    except OSError as exc:
        logger.warning("video_memory.legacy_migration_skipped", target=str(target_p),
                       error=str(exc)[:120])
        return MigrationResult("busy", errors=1)
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EAGAIN, errno.EACCES):
                return MigrationResult("busy")
            raise
        # Re-check under the lock: the winner may have finished while we waited.
        if (target_p / MARKER).exists():
            return MigrationResult("already_done")
        return _migrate_locked(target_p, legacy_p)
    finally:
        os.close(lock_fd)  # closing releases the flock


def _migrate_locked(target: Path, legacy: Path) -> MigrationResult:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    park = target / f"{PARK_PREFIX}{stamp}"
    res = MigrationResult("migrated")

    files: list[Path] = []
    for root, _dirs, names in os.walk(legacy):
        for n in names:
            files.append(Path(root, n).relative_to(legacy))
    files.sort()

    # Faiss directories with any name collision are parked whole.
    conflicted_units: set[Path] = set()
    for rel in files:
        unit = _faiss_unit(rel)
        if unit is not None and (target / rel).exists():
            conflicted_units.add(unit)

    for rel in files:
        src, dst = legacy / rel, target / rel
        try:
            unit = _faiss_unit(rel)
            if unit in conflicted_units:
                _move(src, park / "from-legacy" / rel)
                res.parked_legacy += 1
                if str(unit) not in res.parked_faiss:
                    res.parked_faiss.append(str(unit))
            elif not dst.exists() and not dst.is_symlink():
                _move(src, dst)
                res.moved += 1
            elif src.stat().st_mtime > dst.stat().st_mtime:
                _move(dst, park / "from-target" / rel)
                _move(src, dst)
                res.parked_target += 1
                res.moved += 1
            else:
                _move(src, park / "from-legacy" / rel)
                res.parked_legacy += 1
        except OSError as exc:
            res.errors += 1
            logger.warning("video_memory.legacy_migration_file_failed", path=str(rel),
                           error=str(exc)[:120])
    res.faiss_units_parked = len(res.parked_faiss)
    if park.exists():
        res.park_dir = str(park)

    # Prune only the now-empty legacy directories (rmdir refuses non-empty).
    for root, _dirs, _names in os.walk(legacy, topdown=False):
        if Path(root) != legacy:
            try:
                os.rmdir(root)
            except OSError:
                pass

    if res.errors:
        res.status = "partial"  # no marker: the next start retries what is left
    else:
        try:
            (target / MARKER).write_text(
                f"migrated {stamp} from {legacy}: moved={res.moved} "
                f"parked_legacy={res.parked_legacy} parked_target={res.parked_target}\n"
            )
        except OSError as exc:
            logger.warning("video_memory.legacy_marker_failed", error=str(exc)[:120])
    logger.info(
        "video_memory.legacy_dir_migrated",
        legacy=str(legacy), target=str(target), status=res.status, moved=res.moved,
        parked_legacy=res.parked_legacy, parked_target=res.parked_target,
        faiss_units_parked=res.faiss_units_parked, parked_faiss=res.parked_faiss,
        park_dir=res.park_dir, errors=res.errors,
    )
    return res
