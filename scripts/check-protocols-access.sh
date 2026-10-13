#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC
#
# CI: prove the trellion-protocols pin resolves (the repository is public).
# The repository is taken from the pin in pyproject.toml, the one place that
# names it, so re-pinning to a new home needs no change here.
set -euo pipefail
cd "$(dirname "$0")/.."
url="$(grep -oE 'https://github\.com/[A-Za-z0-9._-]+/trellion-protocols\.git' pyproject.toml | head -n1)"
if [ -z "$url" ]; then
  echo "::error::no trellion-protocols git pin found in pyproject.toml"; exit 1
fi
git ls-remote "$url" HEAD >/dev/null || {
  echo "::error::cannot read ${url#https://github.com/} (is the repository public and the pin current?)"; exit 1; }
