#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Add (or check) SPDX licence headers on every source file in a repo.

    python spdx.py <repo-root> AGPL-3.0-only            # add where missing
    python spdx.py <repo-root> AGPL-3.0-only --check    # exit 1 if any missing

Under AGPL a lifted file must carry the signal; the header is what makes a
single file self-describing once it leaves the repository. Idempotent: a file
that already carries an SPDX line is left alone.

What it stamps: Python, shell, TOML, YAML, Rust, Go, TS/JS (and JSX/TSX),
CSS and HTML. Everything else (JSON, Markdown, SQL, Dockerfiles, lockfiles,
images, PEM, ...) has no safe comment syntax or is data, and is not touched.

What it will not break, by construction:
  * a shebang (`#!`, but not Rust's `#![inner_attr]`) or `#cloud-config`
    stays line 1;
  * a Python PEP 263 encoding cookie stays within the first two lines;
  * `@charset` (CSS), `<!doctype>` and `<?xml` (HTML) stay first;
  * a UTF-8 BOM stays the first bytes of the file;
  * CRLF files get CRLF headers;
  * the header is followed by a blank line, so in Go it can never fuse with
    the package doc comment and a `//go:build` line below it stays valid
    (a constraint may follow only blank lines and `//` comments), and in
    Rust `//` is not a doc comment so `#![..]` / `//!` below it are valid.

What it skips:
  * directories: VCS/vendor trees (SKIP_DIRS; plus target/dist/build when
    not in a git checkout, see SKIP_DIRS_NO_GIT), fixture/golden data
    (FIXTURE_DIRS: byte-compared by tests), and any subtree that carries its
    own LICENSE (a different licence, e.g. an Apache-2.0 SDK in an AGPL repo).
    `.sql` is never stamped, so sqlx migrations (checksummed once applied)
    are safe; Alembic `.py` revisions are stamped (no checksum, ids are in
    the file body);
  * files: lockfiles, `schema.d.ts` and every `*.d.ts`, `*.gen.*`, and any
    file whose first 1 KiB says it is generated (`@generated`, `DO NOT
    EDIT`, `Code generated`, `auto-generated`), symlinks, non-UTF-8 files
    and empty files;
  * when the root is a git checkout, only files git knows (tracked or
    untracked-but-not-ignored), so build output never gets stamped.

A file that already has a DIFFERENT SPDX id is reported (and fails --check)
but never rewritten.
"""
import re
import subprocess
import sys
from pathlib import Path

COMMENT = {
    '.py': '#', '.sh': '#', '.bash': '#', '.toml': '#', '.yml': '#', '.yaml': '#',
    '.rs': '//', '.go': '//', '.ts': '//', '.tsx': '//', '.js': '//', '.mjs': '//', '.cjs': '//', '.jsx': '//',
    '.css': None, '.html': None,  # block comments
}
# Outside a git checkout we have no ignore rules, so build output dirs are
# skipped by name. Inside one, git already excludes untracked output, and a
# TRACKED `build/` (trellion's src/trellion/build) is source, not output.
SKIP_DIRS = {'.git', 'node_modules', '.venv', 'venv', '__pycache__', '.worktrees', 'vendor'}
SKIP_DIRS_NO_GIT = SKIP_DIRS | {'target', 'dist', 'build'}
FIXTURE_DIRS = {'fixtures', 'testdata', 'golden', 'goldens'}
SKIP_FILES = {'schema.d.ts', 'package-lock.json', 'Cargo.lock'}
OWN_LICENSE_NAMES = {'LICENSE', 'LICENSE.md', 'LICENSE.txt', 'COPYING', 'COPYING.md'}
GENERATED = re.compile(r'@generated|DO NOT EDIT|Code generated|auto-?generated', re.I)
SPDX_LINE = re.compile(r'^[ \t]*(?:#|//|/\*|<!--)[ \t]*SPDX-License-Identifier:[ \t]*([^\s*>]+)', re.M)
ENCODING = re.compile(r'^[ \t\f]*#.*?coding[:=][ \t]*[-\w.]+')
HOLDER = 'PersonifAI LLC'
YEAR = '2026'


def header_for(ext: str, spdx: str) -> str:
    c = COMMENT[ext]
    if c is None:
        if ext == '.css':
            return f'/* SPDX-License-Identifier: {spdx} */\n/* Copyright {YEAR} {HOLDER} */\n'
        return f'<!-- SPDX-License-Identifier: {spdx} -->\n<!-- Copyright {YEAR} {HOLDER} -->\n'
    return f'{c} SPDX-License-Identifier: {spdx}\n{c} Copyright {YEAR} {HOLDER}\n'


def _preserved_prefix(ext: str, lines: list[str]) -> int:
    """How many leading lines must stay above the header."""
    n = 0
    if lines and lines[0].startswith('#!') and not lines[0].startswith('#!['):
        n = 1
    elif lines and lines[0].startswith('#cloud-config'):
        n = 1
    if ext == '.py':
        # PEP 263: the coding cookie must stay on line 1 or 2, so the header
        # goes below whichever of the first two lines carries it.
        for i in (0, 1):
            if i < len(lines) and ENCODING.match(lines[i]):
                n = max(n, i + 1)
    if ext == '.css' and lines and lines[0].lstrip().startswith('@charset'):
        n = 1
    if ext == '.html' and lines:
        low = lines[0].lower().lstrip()
        if low.startswith('<!doctype') or low.startswith('<?xml'):
            n = 1
    return n


def stamp(ext: str, text: str, spdx: str) -> str:
    """Return `text` with the header inserted at the right place."""
    bom = ''
    if text.startswith('﻿'):
        bom, text = '﻿', text[1:]
    nl = '\r\n' if '\r\n' in text[:2000] else '\n'
    lines = text.split(nl)
    n = _preserved_prefix(ext, lines)
    hdr = header_for(ext, spdx).replace('\n', nl) + nl  # + trailing blank line
    head = nl.join(lines[:n]) + nl if n else ''
    rest = nl.join(lines[n:])
    return bom + head + hdr + rest


def candidates(root: Path):
    """Yield (path, path-relative-to-root) for files that could carry a header."""
    paths = None
    git_mode = False
    if (root / '.git').exists():
        try:
            out = subprocess.run(
                ['git', '-C', str(root), 'ls-files', '-z', '--cached', '--others', '--exclude-standard'],
                capture_output=True, check=True).stdout
            paths = [root / s for s in out.decode('utf-8', 'surrogateescape').split('\0') if s]
            git_mode = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            paths = None
    if paths is None:
        paths = sorted(root.rglob('*'))
    skip_dirs = SKIP_DIRS if git_mode else SKIP_DIRS_NO_GIT
    own_license: dict[Path, bool] = {}

    def has_own_license(d: Path) -> bool:
        if d == root:
            return False
        if d not in own_license:
            own_license[d] = any((d / n).is_file() for n in OWN_LICENSE_NAMES) or has_own_license(d.parent)
        return own_license[d]

    for p in paths:
        rel = p.relative_to(root)
        if p.suffix not in COMMENT or p.name in SKIP_FILES:
            continue
        if p.name.endswith('.d.ts') or '.gen.' in p.name:
            continue
        if any(part in skip_dirs or part in FIXTURE_DIRS for part in rel.parts[:-1]):
            continue
        if p.is_symlink() or not p.is_file():
            continue
        if has_own_license(p.parent):
            continue
        yield p, rel


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = [a for a in argv if not a.startswith('--')]
    if len(args) < 2:
        print(__doc__)
        return 2
    root = Path(args[0]).resolve()
    spdx = args[1]
    check = '--check' in argv
    missing: list[Path] = []
    wrong: list[tuple[Path, str]] = []
    for p, rel in candidates(root):
        raw = p.read_bytes()
        try:
            text = raw.decode('utf-8')
        except UnicodeDecodeError:
            continue
        if not text.strip() or GENERATED.search(text[:1024]):
            continue
        m = SPDX_LINE.search(text.lstrip('\ufeff')[:600])
        if m:
            if m.group(1) != spdx:
                wrong.append((rel, m.group(1)))
            continue
        missing.append(rel)
        if not check:
            p.write_bytes(stamp(p.suffix, text, spdx).encode('utf-8'))
    for rel, found in wrong:
        print(f'wrong SPDX id ({found}, expected {spdx}): {rel}')
    if check:
        for rel in missing:
            print(f'missing SPDX header: {rel}')
        return 1 if (missing or wrong) else 0
    print(f'added headers to {len(missing)} files')
    return 1 if wrong else 0


if __name__ == '__main__':
    raise SystemExit(main())
