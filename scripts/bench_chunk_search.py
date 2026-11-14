# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC
"""Measure PageIndexStore.search_chunks at knowledge-base scale, on SEEDED data.

noomesh-internal i785 (the `kb_chunk_scale` alert fired at ~285k chunks).
This never touches a real database: it refuses any DSN that is not localhost,
and it refuses a database that already holds documents.

It seeds a throw-away database that trellion's own migrations built, then runs
the REAL ``search_chunks`` (the method, not a copy of its SQL) for the two
shapes production uses:

  page    ``retrieve_chunks``: scoped to the pages page-search picked, no roles
  global  chunk-first recall (``_merge_chunk_first_pages``): the whole corpus,
          ``roles`` applied, ``top_k`` 20

under ``hnsw.iterative_scan`` off / relaxed_order / strict_order (set per
database, so the unchanged code picks it up), and reports for each:

  plan        what drives the chunk_embeddings access (EXPLAIN ANALYZE, BUFFERS)
  cold        the first pass after a postgres restart and an OS cache drop
  warm        the third pass
  recall@k    against the exact answer, computed here in numpy over the seeded
              vectors with the same filters (never by the database under test)
  underfill   queries that returned fewer rows than the exact answer has
  audience    rows returned from a document the caller may not read (must be 0;
              the probes are seeded so restricted chunks are the true nearest)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import subprocess
import sys
import time
import uuid
from urllib.parse import urlparse

import numpy as np

DIM = 1024
MODEL = "bench-model-a"
OTHER_MODEL = "bench-model-b"
# The audiences: most documents are untagged (readable with the generic `user`
# role); a few carry a role. `finance` is the restricted audience the probes aim at.
AUDIENCES = [(None, 0.85), (["staff"], 0.10), (["finance"], 0.03), (["legal"], 0.02)]
# (name, GUCs on every connection, fallback GUCs or None). A fallback runs the
# first set; only when that comes back SHORT (fewer than top_k rows) does it run
# the query again with the fallback set, and the latency is the sum. Set per
# engine (libpq startup options), so the code under test is unchanged.
RELAXED = {"hnsw.iterative_scan": "relaxed_order"}
SETTINGS = [
    ("off (today)", {}, None),
    ("relaxed", dict(RELAXED), None),
    ("relaxed, ef_search 100", {**RELAXED, "hnsw.ef_search": "100"}, None),
    ("relaxed, max_scan_tuples 5000", {**RELAXED, "hnsw.max_scan_tuples": "5000"}, None),
    ("off, then relaxed when short", {}, dict(RELAXED)),
    ("off, then relaxed (max_scan_tuples 5000) when short", {}, {**RELAXED, "hnsw.max_scan_tuples": "5000"}),
]
TIE = 0.005  # a returned chunk within this similarity of the k-th exact one counts as a hit


class Vec(list):
    """A query vector carrying its model, as an embed response's does
    (embedding_space.model_of reads `.model`)."""

    model = MODEL


def unit(a: np.ndarray) -> np.ndarray:
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def refuse_unless_local(dsn: str) -> None:
    host = urlparse(dsn.replace("+asyncpg", "")).hostname
    if host not in ("localhost", "127.0.0.1", "::1"):
        sys.exit(f"bench: refusing {host!r}: this benchmark seeds and drops data, so it only runs on localhost")


def build_world(n_chunks: int, n_pages: int, n_docs: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    # Overlapping topics: 2,000 topics around 100 shared super-topics (topic to
    # super cosine ~0.7), chunks around their topic. The first run's 2,000
    # independent clusters stranded HNSW's greedy search (recall 0.68 with no
    # restrictive filter); real embeddings are not islands.
    supers = unit(rng.standard_normal((100, DIM)).astype(np.float32))
    centroids = unit(supers[rng.integers(0, 100, 2000)] + rng.standard_normal((2000, DIM)).astype(np.float32) / np.sqrt(DIM))
    doc_topic = rng.integers(0, len(centroids), n_docs)
    roll = rng.random(n_docs)
    edges = np.cumsum([w for _, w in AUDIENCES])
    doc_aud = np.searchsorted(edges, roll)
    doc_held = rng.random(n_docs) < 0.01
    # Every audience gets at least two readable (not held) documents, so a
    # small run still has restricted chunks to aim the probes at.
    for k in range(len(AUDIENCES)):
        have = np.flatnonzero((doc_aud == k) & ~doc_held)
        for d in rng.choice(n_docs, max(0, 2 - len(have)), replace=False):
            doc_aud[d], doc_held[d] = k, False
    page_doc = np.sort(rng.integers(0, n_docs, n_pages))
    chunk_page = np.sort(rng.integers(0, n_pages, n_chunks))
    chunk_doc = page_doc[chunk_page]
    vecs = np.empty((n_chunks, DIM), dtype=np.float32)
    step = 20000
    for i in range(0, n_chunks, step):
        j = min(i + step, n_chunks)
        # noise of norm ~0.8 around a unit centroid: clustered, not separable by topic alone
        vecs[i:j] = unit(centroids[doc_topic[chunk_doc[i:j]]] + 0.6 * rng.standard_normal((j - i, DIM)).astype(np.float32) / np.sqrt(DIM))
    chunk_model = np.where(rng.random(n_chunks) < 0.05, 1, 0)  # 5% still on the old model
    return {
        "doc_ids": [uuid.uuid4() for _ in range(n_docs)],
        "page_ids": [uuid.uuid4() for _ in range(n_pages)],
        "chunk_ids": [uuid.uuid4() for _ in range(n_chunks)],
        "doc_aud": doc_aud, "doc_held": doc_held, "page_doc": page_doc,
        "chunk_page": chunk_page, "chunk_doc": chunk_doc, "chunk_model": chunk_model, "vecs": vecs,
    }


def visible(w: dict, roles: list[str]) -> np.ndarray:
    """Per document: readable by `roles` (ROLES_WHERE), not held."""
    ok = np.zeros(len(w["doc_ids"]), dtype=bool)
    for k, (aud, _) in enumerate(AUDIENCES):
        sel = w["doc_aud"] == k
        ok[sel] = ("user" in roles) if aud is None else bool(set(aud) & set(roles))
    return ok & ~w["doc_held"]


async def seed(dsn: str, w: dict) -> None:
    import asyncpg
    from pgvector.asyncpg import register_vector

    conn = await asyncpg.connect(dsn.replace("postgresql+asyncpg", "postgresql"))
    try:
        await register_vector(conn)
        if await conn.fetchval("SELECT count(*) FROM trellion.docs"):
            sys.exit("bench: trellion.docs is not empty; this only runs on a database it seeded itself")
        idx = await conn.fetch("SELECT indexname, indexdef FROM pg_indexes WHERE schemaname='trellion' "
                               "AND tablename='chunk_embeddings' AND indexdef ILIKE '%USING hnsw%'")
        for r in idx:  # an HNSW index is built once over the seeded table, as the migration defines it
            await conn.execute(f'DROP INDEX trellion."{r["indexname"]}"')
        text_body = "lorem ipsum " * 60
        now = time.time()
        await conn.copy_records_to_table("docs", schema_name="trellion", columns=["doc_id", "source_uri", "title", "hash", "metadata", "held_at"], records=[
            (d, f"bench://{i}", f"doc {i}", uuid.uuid4().hex,
             json.dumps({} if AUDIENCES[w["doc_aud"][i]][0] is None else {"roles": AUDIENCES[w["doc_aud"][i]][0]}),
             __import__("datetime").datetime.now(__import__("datetime").timezone.utc) if w["doc_held"][i] else None)
            for i, d in enumerate(w["doc_ids"])])
        await conn.copy_records_to_table("pages", schema_name="trellion", columns=["page_id", "doc_id", "page_no", "text"], records=[
            (p, w["doc_ids"][w["page_doc"][i]], i, text_body) for i, p in enumerate(w["page_ids"])])
        await conn.copy_records_to_table("page_chunks", schema_name="trellion", columns=["chunk_id", "page_id", "chunk_no", "text"], records=[
            (c, w["page_ids"][w["chunk_page"][i]], i, text_body) for i, c in enumerate(w["chunk_ids"])])
        models = [MODEL, OTHER_MODEL]
        step = 20000
        for i in range(0, len(w["chunk_ids"]), step):
            await conn.copy_records_to_table("chunk_embeddings", schema_name="trellion",
                                             columns=["chunk_id", "embedding", "embedding_model", "embedding_dim"], records=[
                (w["chunk_ids"][k], w["vecs"][k], models[w["chunk_model"][k]], DIM)
                for k in range(i, min(i + step, len(w["chunk_ids"])))])
        print(f"bench: seeded in {time.time() - now:.0f}s", flush=True)
        t = time.time()
        # A parallel HNSW build allocates maintenance_work_mem in dynamic shared
        # memory, which the stack caps at shm_size 1g: 2GB failed with "could not
        # resize shared memory segment". Stay under it; the build is slower, and
        # it is setup, not measured.
        await conn.execute("SET maintenance_work_mem = '800MB'")
        await conn.execute("SET max_parallel_maintenance_workers = 2")
        for r in idx:
            await conn.execute(r["indexdef"])
        print(f"bench: HNSW index(es) rebuilt in {time.time() - t:.0f}s: {[r['indexname'] for r in idx]}", flush=True)
        await conn.execute("VACUUM ANALYZE trellion.docs, trellion.pages, trellion.page_chunks, trellion.chunk_embeddings")
    finally:
        await conn.close()


def make_queries(w: dict, n: int, seed: int) -> list[dict]:
    """Half near public chunks, half near `finance` chunks (the audience probes:
    the restricted chunks ARE the nearest, so a leak cannot hide)."""
    rng = np.random.default_rng(seed + 1)
    fin = AUDIENCES.index((["finance"], 0.03))
    ok_model = w["chunk_model"] == 0
    public = np.flatnonzero((w["doc_aud"][w["chunk_doc"]] == 0) & ~w["doc_held"][w["chunk_doc"]] & ok_model)
    finance = np.flatnonzero((w["doc_aud"][w["chunk_doc"]] == fin) & ~w["doc_held"][w["chunk_doc"]] & ok_model)
    out = []
    for i in range(n):
        src = rng.choice(public if i % 2 == 0 else finance)
        q = unit(w["vecs"][src] + 0.1 * rng.standard_normal(DIM).astype(np.float32) / np.sqrt(DIM))
        out.append({"vec": q, "kind": "public" if i % 2 == 0 else "finance-probe"})
    return out


def exact(w: dict, q: np.ndarray, k: int, roles: list[str] | None, page_ids: set | None) -> list:
    ok = w["chunk_model"] == 0
    ok &= ~w["doc_held"][w["chunk_doc"]]
    if roles is not None:
        ok &= visible(w, roles)[w["chunk_doc"]]
    if page_ids is not None:
        ok &= np.isin(w["chunk_page"], list(page_ids))
    idx = np.flatnonzero(ok)
    if not len(idx):
        return []
    s = w["vecs"][idx] @ q
    top = idx[np.argsort(-s)[:k]]
    return [str(w["chunk_ids"][t]) for t in top]


SHAPES = [
    # name, roles, top_k, page-scoped?
    ("page (retrieve_chunks)", None, 10, True),
    ("global, roles=[user]", ["user"], 20, False),
    ("global, roles=[user,finance]", ["user", "finance"], 20, False),
    ("global, roles=[finance] (selective)", ["finance"], 20, False),
]


def page_scope(w: dict, q: np.ndarray, n_pages: int = 40) -> set:
    """The ~40 pages page-search would pick (retrieve_chunks' scope): pages of the nearest chunks."""
    order = np.argsort(-(w["vecs"] @ q))
    seen: list[int] = []
    for c in order:
        p = int(w["chunk_page"][c])
        if p not in seen:
            seen.append(p)
        if len(seen) == n_pages:
            break
    return set(seen)


def restart_cold(container: str) -> None:
    subprocess.run(["docker", "restart", container], check=True, capture_output=True)
    subprocess.run(["sudo", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"], check=False, capture_output=True)
    for _ in range(60):
        if subprocess.run(["docker", "exec", container, "pg_isready", "-U", "noo"], capture_output=True).returncode == 0:
            return
        time.sleep(1)
    sys.exit("bench: postgres did not come back after the restart")


async def measure(dsn: str, w: dict, queries: list[dict], container: str, db: str) -> list[dict]:
    from sqlalchemy import event
    from sqlalchemy.ext.asyncio import create_async_engine

    from trellion.db.engine import TRELLION_SCHEMA, _async_dsn
    from trellion.live.pageindex_store import PageIndexStore

    doc_of_chunk = {str(w["chunk_ids"][i]): int(w["chunk_doc"][i]) for i in range(len(w["chunk_ids"]))}
    page_id_str = [str(p) for p in w["page_ids"]]
    scopes = [page_scope(w, q["vec"]) for q in queries]
    rows = []
    index_of = {str(c): i for i, c in enumerate(w["chunk_ids"])}

    def engine(gucs: dict):
        opts = f"-csearch_path={TRELLION_SCHEMA},public" + "".join(f" -c {k}={v}" for k, v in gucs.items())
        return create_async_engine(_async_dsn(dsn), connect_args={"options": opts})

    for setting, gucs, fallback in SETTINGS:
        for shape, roles, k, scoped in SHAPES:
            restart_cold(container)
            eng = engine(gucs)
            eng_fb = engine(fallback) if fallback is not None else None
            store_fb = PageIndexStore(async_engine=eng_fb, embedding_dim=DIM) if eng_fb else None
            fell_back = 0
            seen_sql: list = []

            @event.listens_for(eng.sync_engine, "before_cursor_execute")
            def grab(conn, cursor, statement, params, context, executemany):  # noqa: ANN001
                if "chunk_embeddings" in statement and "ORDER BY" in statement:
                    seen_sql[:] = [(statement, params)]

            store = PageIndexStore(async_engine=eng, embedding_dim=DIM)
            passes = []
            results = []
            for p in range(3):  # pass 0 is cold (just restarted), pass 2 is warm
                lat = []
                for qi, q in enumerate(queries):
                    pids = [page_id_str[i] for i in scopes[qi]] if scoped else None
                    t = time.perf_counter()
                    got = await store.search_chunks(Vec(q["vec"].tolist()), top_k=k, page_ids=pids, roles=roles)
                    if store_fb is not None and len(got) < k:
                        got = await store_fb.search_chunks(Vec(q["vec"].tolist()), top_k=k, page_ids=pids, roles=roles)
                        if p == 2:
                            fell_back += 1
                    lat.append((time.perf_counter() - t) * 1000)
                    if p == 2:
                        results.append(got)
                passes.append(lat)
            # The plan of the last statement, with the same parameters.
            plan_txt = ""
            if seen_sql:
                stmt, params = seen_sql[0]
                async with eng.connect() as c:
                    r = await c.exec_driver_sql("EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) " + stmt, params)
                    plan_txt = "\n".join(x[0] for x in r.fetchall())
            await eng.dispose()
            if eng_fb is not None:
                await eng_fb.dispose()
            recalls, strict, under, leaks = [], [], 0, 0
            vis = visible(w, roles) if roles is not None else None
            for qi, q in enumerate(queries):
                want = exact(w, q["vec"], k, roles, scopes[qi] if scoped else None)
                got = [g["chunk_id"] for g in results[qi]]
                if want:
                    strict.append(len(set(got) & set(want)) / len(want))
                    kth = float(w["vecs"][index_of[want[-1]]] @ q["vec"])
                    hits = sum(1 for g in got if g in want or float(w["vecs"][index_of[g]] @ q["vec"]) >= kth - TIE)
                    recalls.append(min(hits, len(want)) / len(want))
                if len(got) < len(want):
                    under += 1
                if vis is not None:
                    leaks += sum(1 for g in got if not vis[doc_of_chunk[g]])
            drive = next((l.strip() for l in plan_txt.splitlines()
                          if "chunk_embeddings" in l and ("Scan" in l)), "?")
            rows.append({
                "setting": setting, "shape": shape, "k": k,
                "plan": drive, "exec_ms": next((l.split(":")[1].strip() for l in plan_txt.splitlines() if "Execution Time" in l), "?"),
                "cold_first_ms": round(passes[0][0], 1),
                "cold_p50": round(statistics.median(passes[0]), 1),
                "cold_p95": round(np.percentile(passes[0], 95), 1),
                "warm_p50": round(statistics.median(passes[2]), 1),
                "warm_p95": round(np.percentile(passes[2], 95), 1),
                "recall": round(float(np.mean(recalls)) if recalls else float("nan"), 3),
                "min_recall": round(float(np.min(recalls)) if recalls else float("nan"), 3),
                "strict_recall": round(float(np.mean(strict)) if strict else float("nan"), 3),
                "fell_back": fell_back if fallback is not None else None,
                "underfill": under, "leaks": leaks, "plan_text": plan_txt,
            })
            print(json.dumps({k2: v for k2, v in rows[-1].items() if k2 != "plan_text"}), flush=True)
    return rows


# --mode share (i785 step 2): is an EXACT search over the caller's visible set
# cheaper than the relaxed HNSW walk, and below which visible share? One
# audience per share; the caller holds only that role, and the queries sit
# near untagged chunks, so the nearest are never its own (HNSW's worst case).
SHARE_AUDIENCES = [(None, 0.56), (["a1"], 0.01), (["a3"], 0.03), (["a10"], 0.10), (["a30"], 0.30)]


def exact_sql() -> str:
    """search_chunks' statement with the distance ordering made unusable by the
    HNSW index (`+ 0`), so the plan is an exact scan of the rows that pass the
    filter. Same filter, in the same statement."""
    from trellion_protocols.schema import ROLES_WHERE

    # Driven FROM the visible set: the readable chunks are materialised first
    # (docs -> pages -> chunks, the audience filter in this statement), then
    # their embeddings are fetched by primary key. The first share run joined
    # the other way round and computed all 300k distances whatever the share
    # (~3.3 s flat): the planner cannot estimate the JSONB audience predicate.
    return f"""
        WITH vis AS MATERIALIZED (
            SELECT c.chunk_id, c.page_id, c.chunk_no, c.text
            FROM docs d
            JOIN pages p ON p.doc_id = d.doc_id
            JOIN page_chunks c ON c.page_id = p.page_id
            WHERE d.held_at IS NULL AND {ROLES_WHERE}
        )
        SELECT v.chunk_id, v.page_id, v.chunk_no, v.text,
               ce.embedding <=> CAST(:vec AS vector) AS distance
        FROM vis v
        JOIN chunk_embeddings ce ON ce.chunk_id = v.chunk_id
        WHERE ce.embedding_model = :emb_model
        ORDER BY (ce.embedding <=> CAST(:vec AS vector)) + 0
        LIMIT :top_k"""


async def measure_share(dsn: str, w: dict, queries: list[dict], container: str) -> list[dict]:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    from trellion.db.engine import TRELLION_SCHEMA, _async_dsn
    from trellion.live.pageindex_store import PageIndexStore

    index_of = {str(c): i for i, c in enumerate(w["chunk_ids"])}
    doc_of_chunk = {str(w["chunk_ids"][i]): int(w["chunk_doc"][i]) for i in range(len(w["chunk_ids"]))}
    k = 20
    sql = text(exact_sql())
    rows = []
    for aud, share in SHARE_AUDIENCES[1:]:
        roles = aud
        vis = visible(w, roles)
        for path in ("relaxed HNSW (search_chunks)", "exact over the visible set"):
            restart_cold(container)
            # The exact path joins by primary key (nested loops): what a
            # production version would set with SET LOCAL for this statement.
            opts = f"-csearch_path={TRELLION_SCHEMA},public" + (
                " -c hnsw.iterative_scan=relaxed_order" if path.startswith("relaxed")
                else " -c enable_hashjoin=off -c enable_mergejoin=off")
            eng = create_async_engine(_async_dsn(dsn), connect_args={"options": opts})
            store = PageIndexStore(async_engine=eng, embedding_dim=DIM)
            passes, results = [], []
            for p in range(3):
                lat = []
                for q in queries:
                    t = time.perf_counter()
                    if path.startswith("relaxed"):
                        got = [g["chunk_id"] for g in await store.search_chunks(Vec(q["vec"].tolist()), top_k=k, roles=roles)]
                    else:
                        async with eng.connect() as c:
                            r = await c.execute(sql, {"vec": "[" + ",".join(f"{x:.6f}" for x in q["vec"]) + "]",
                                                      "roles": roles, "emb_model": MODEL, "top_k": k})
                            got = [str(x[0]) for x in r.fetchall()]
                    lat.append((time.perf_counter() - t) * 1000)
                    if p == 2:
                        results.append(got)
                passes.append(lat)
            await eng.dispose()
            recalls, under, leaks = [], 0, 0
            for qi, q in enumerate(queries):
                want = exact(w, q["vec"], k, roles, None)
                got = results[qi]
                if want:
                    kth = float(w["vecs"][index_of[want[-1]]] @ q["vec"])
                    hits = sum(1 for g in got if g in want or float(w["vecs"][index_of[g]] @ q["vec"]) >= kth - TIE)
                    recalls.append(min(hits, len(want)) / len(want))
                under += len(got) < len(want)
                leaks += sum(1 for g in got if not vis[doc_of_chunk[g]])
            rows.append({
                "share": f"{share:.0%}", "visible_chunks": int(vis[w["chunk_doc"]].sum()), "path": path,
                "cold_first_ms": round(passes[0][0], 1),
                "cold_p50": round(statistics.median(passes[0]), 1), "cold_p95": round(np.percentile(passes[0], 95), 1),
                "warm_p50": round(statistics.median(passes[2]), 1), "warm_p95": round(np.percentile(passes[2], 95), 1),
                "recall": round(float(np.mean(recalls)) if recalls else float("nan"), 3),
                "underfill": under, "leaks": leaks,
            })
            print(json.dumps(rows[-1]), flush=True)
    return rows


def report_share(rows: list[dict], n: int, nq: int) -> str:
    out = [f"### visible share: exact vs relaxed HNSW at {n:,} seeded chunks ({nq} queries near untagged chunks, k=20)", "",
           "| visible share | visible chunks | path | cold first ms | cold p50/p95 ms | warm p50/p95 ms | recall@20 tie-aware | underfill | leaks |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['share']} | {r['visible_chunks']:,} | {r['path']} | {r['cold_first_ms']} | {r['cold_p50']} / {r['cold_p95']} | "
                   f"{r['warm_p50']} / {r['warm_p95']} | {r['recall']} | {r['underfill']} | {r['leaks']} |")
    return "\n".join(out) + "\n"


def report(rows: list[dict], n: int, nq: int) -> str:
    out = [f"### search_chunks at {n:,} seeded chunks ({nq} queries per cell; recall vs exact numpy)", "",
           "| setting | shape | k | chunk_embeddings access | cold first ms | cold p50/p95 ms | warm p50/p95 ms | recall@k tie-aware (min) | strict recall | fell back | underfill | audience leaks |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['setting']} | {r['shape']} | {r['k']} | `{r['plan'][:80]}` | {r['cold_first_ms']} | "
                   f"{r['cold_p50']} / {r['cold_p95']} | {r['warm_p50']} / {r['warm_p95']} | {r['recall']} ({r['min_recall']}) | "
                   f"{r['strict_recall']} | {'-' if r['fell_back'] is None else r['fell_back']} | {r['underfill']} | {r['leaks']} |")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--db", default="noomesh")
    ap.add_argument("--container", required=True, help="the throw-away postgres container (restarted for cold passes)")
    ap.add_argument("--chunks", type=int, default=300_000)
    ap.add_argument("--queries", type=int, default=50)
    ap.add_argument("--seed", type=int, default=785)
    ap.add_argument("--out", default="bench-out")
    ap.add_argument("--mode", choices=["settings", "share"], default="settings",
                    help="settings: iterative-scan settings per shape; share: exact vs relaxed by visible share")
    a = ap.parse_args()
    refuse_unless_local(a.dsn)
    random.seed(a.seed)
    global AUDIENCES
    if a.mode == "share":
        AUDIENCES = SHARE_AUDIENCES
    w = build_world(a.chunks, a.chunks // 5, a.chunks // 60, a.seed)
    asyncio.run(seed(a.dsn, w))
    if a.mode == "share":
        rng = np.random.default_rng(a.seed + 2)
        public = np.flatnonzero((w["doc_aud"][w["chunk_doc"]] == 0) & ~w["doc_held"][w["chunk_doc"]] & (w["chunk_model"] == 0))
        queries = [{"vec": unit(w["vecs"][rng.choice(public)] + 0.1 * rng.standard_normal(DIM).astype(np.float32) / np.sqrt(DIM)),
                    "kind": "near-untagged"} for _ in range(a.queries)]
        rows = asyncio.run(measure_share(a.dsn, w, queries, a.container))
        md = report_share(rows, a.chunks, a.queries)
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "results.md"), "w") as f:
            f.write(md)
        with open(os.path.join(a.out, "results.json"), "w") as f:
            json.dump(rows, f, indent=1)
        print(md)
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
                f.write(md)
        return 1 if any(r["leaks"] for r in rows) else 0
    queries = make_queries(w, a.queries, a.seed)
    rows = asyncio.run(measure(a.dsn, w, queries, a.container, a.db))
    os.makedirs(a.out, exist_ok=True)
    md = report(rows, a.chunks, a.queries)
    with open(os.path.join(a.out, "results.md"), "w") as f:
        f.write(md)
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print(md)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(md)
    return 1 if any(r["leaks"] for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
