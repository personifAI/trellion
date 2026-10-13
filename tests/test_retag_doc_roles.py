# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Retagging is a relabel, not a merge, and it never runs blind.

`kb.default_roles` shipping empty meant every Drive-ingested document was
stored untagged, and an untagged document overlaps no audience — so it is
reachable only by a caller running unrestricted. Nothing in the tree could
fix that after ingest: roles were written once, at ingest, and never again.

These pin the SQL's shape rather than its execution. The methods are two
statements against a live Postgres; what can silently go wrong is not "does
UPDATE work" but which rows are selected, whether the rest of `metadata`
survives, and whether clearing is possible at all.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import re

import pytest

from trellion.live.pageindex_store import PageIndexStore

SRC = pathlib.Path(inspect.getfile(PageIndexStore))


def _method_source(name: str) -> str:
    tree = ast.parse(SRC.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            body = ast.get_source_segment(SRC.read_text(), node) or ""
            # Strip the docstring: it describes the SQL in prose, and a guard
            # that matches prose guards nothing.
            doc = ast.get_docstring(node) or ""
            return body.replace(doc, "") if doc else body
    raise AssertionError(f"{name} is gone; this guard is stale")


def test_both_halves_exist():
    assert hasattr(PageIndexStore, "select_docs_for_retag")
    assert hasattr(PageIndexStore, "set_doc_roles")


def test_the_preview_does_not_write():
    src = _method_source("select_docs_for_retag")
    for verb in ("UPDATE", "INSERT", "DELETE", "commit("):
        assert verb not in src, f"the preview half performs a {verb}"


def test_untagged_means_all_three_empty_states():
    """Absent key, JSON null and empty array. Testing only the absent key
    would skip exactly the rows a half-finished tagging run left behind."""
    src = _method_source("select_docs_for_retag")
    assert "IS NULL" in src
    assert "jsonb_typeof" in src and "'null'" in src
    assert "jsonb_array_length" in src


def test_only_untagged_defaults_true():
    """Backfilling something nobody classified is a different act from
    overwriting a classification somebody chose."""
    sig = inspect.signature(PageIndexStore.select_docs_for_retag)
    assert sig.parameters["only_untagged"].default is True


def test_the_write_replaces_rather_than_merges():
    """Merging would make the operation impossible to undo: you could add an
    audience but never remove one, and removal is the half that matters."""
    src = _method_source("set_doc_roles")
    assert "jsonb_build_object('roles'" in src
    # A merge would union the two arrays; nothing here may do that.
    assert "||  jsonb_build_array" not in src.replace("\n", " ")


def test_the_rest_of_metadata_survives():
    """`aliased_uris` and every ingest-time field must outlive a retag."""
    src = _method_source("set_doc_roles")
    assert "COALESCE(metadata, '{}'::jsonb)" in src, (
        "the update no longer merges into the existing metadata — it is "
        "replacing the whole object, which drops aliased_uris"
    )


def test_an_empty_role_list_clears_rather_than_storing_empty():
    """Untagged and tagged-with-nothing are two states this codebase already
    distinguishes; a retag must be able to return a document to the first."""
    src = _method_source("set_doc_roles")
    assert "- 'roles'" in src, "clearing must remove the key, not store []"


def test_no_doc_ids_is_a_no_op_not_a_whole_table_update():
    """The shape of the accident this guards: `WHERE doc_id = ANY('{}')`
    is harmless, but an early `if not doc_ids` that was ever removed would
    make the next caller's empty list a fleet-wide relabel."""
    src = _method_source("set_doc_roles")
    assert re.search(r"if not doc_ids:\s*\n\s*return 0", src), (
        "set_doc_roles no longer refuses an empty id list"
    )


def test_the_limit_is_bounded_both_ways():
    src = _method_source("select_docs_for_retag")
    assert "max(1, min(" in src, "an unbounded or zero limit reaches the whole table"


@pytest.mark.parametrize("name", ["select_docs_for_retag", "set_doc_roles"])
def test_ids_are_bound_not_interpolated(name):
    """Document ids come from an operator's --doc-id flag."""
    src = _method_source(name)
    assert ":ids" in src
    assert "f\"SELECT" not in src or "doc_id" not in src.split('f"SELECT')[-1][:200], (
        "document ids must never be formatted into the statement"
    )
