# SPDX-License-Identifier: AGPL-3.0-only
"""The CLA workflow runs only in the public repo (personifAI/trellion).

The job-level `if` makes GitHub report the job as skipped anywhere else. The
kit master lives in the stack repo, whose
tests/test_cla_workflow_declarations.py checks this file against the master.
"""
import re
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/cla.yml"


def test_job_is_guarded_to_the_public_repo():
    text = WORKFLOW.read_text()
    head = text[text.index("\njobs:\n"):].split("steps:", 1)[0]
    assert re.findall(r"^    if: github\.repository == '([^']+)'$", head, re.M) == ["personifai/trellion"]


def test_workflow_shape():
    text = WORKFLOW.read_text()
    assert text.startswith("# SPDX-License-Identifier: AGPL-3.0-only\n")
    assert "pull_request_target:" in text and "secrets.CLA_PAT" in text
    assert "branch: 'cla-signatures'" in text
    assert "path-to-document: 'https://github.com/${{ github.repository }}/blob/cla-v1.0/CLA.md'" in text
    assert (WORKFLOW.parents[2] / "CLA.md").is_file()


def test_the_action_is_pinned_and_the_allowlist_is_exactly_the_verified_accounts():
    """A tag can move; a SHA cannot. An allowlisted handle that is not ours
    skips the CLA for whoever holds it."""
    text = WORKFLOW.read_text()
    assert re.search(r"uses: contributor-assistant/github-action@[0-9a-f]{40}\b", text)
    allow = re.search(r"allowlist: '([^']*)'", text).group(1).split(",")
    assert sorted(allow) == sorted(["personifai-llc", "richard-raithel", "dependabot[bot]", "github-actions[bot]"])
