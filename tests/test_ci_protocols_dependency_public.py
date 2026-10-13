# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""CI reads the trellion-protocols dependency with no credential.

The dependency is public, so no job may carry a token for it, and each job that
installs it first proves the pin resolves anonymously, so a broken pin fails
loudly at that step rather than as a confusing install error.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / ".github" / "workflows" / "ci.yml"
REACH_STEP = "The trellion-protocols pin is reachable"


def test_no_job_carries_a_token_for_the_dependency():
    text = CI.read_text()
    assert not re.search(r"secrets\.\w*PAT\w*", text)
    assert "x-access-token" not in text
    assert "insteadOf" not in text


def test_every_installing_job_checks_the_pin_first():
    jobs = yaml.safe_load(CI.read_text())["jobs"]
    checked = 0
    for name, job in jobs.items():
        steps = job.get("steps", [])
        if not any(s.get("name") == REACH_STEP for s in steps):
            continue
        reach = next(i for i, s in enumerate(steps) if s.get("name") == REACH_STEP)
        installs = [i for i, s in enumerate(steps) if "pip install -e" in str(s.get("run", ""))]
        assert installs, f"{name}: checks the pin but installs nothing"
        assert reach < installs[0], f"{name}: check the pin before installing"
        checked += 1
    assert checked >= 3, f"expected the test/db-test/import-smoke jobs, found {checked}"


def test_the_pin_is_the_public_home_at_a_full_sha():
    deps = (ROOT / "pyproject.toml").read_text()
    pins = re.findall(r"trellion-protocols @ git\+https://github\.com/([^/]+)/trellion-protocols\.git@([0-9a-f]+)", deps)
    assert len(pins) == 1, pins
    owner, sha = pins[0]
    assert owner == "personifAI", owner
    assert len(sha) == 40, "pin by full commit SHA, never a tag or short SHA"
