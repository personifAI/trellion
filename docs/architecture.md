# Architecture

How the live path of `trellion` fits together. The code is in
`src/trellion/live/`; when run inside the noomesh stack, the stack's worker
pool calls into it. File names below are relative to `src/trellion/live/`
unless a directory is given.

## Live retrieval path (page-first)

Ingestion turns a document into **pages**, and each page into overlapping
**chunks**. Both are embedded. Retrieval searches pages first, then chunks
inside the winning pages.

### Splitting and chunking (`pageindex_ingestion.py`)

| Constant | Value | Meaning |
|---|---|---|
| `PAGE_SPLIT_CHARS` | 2000 | Target maximum size of a page (about 670 tokens). |
| `CHUNK_SIZE_CHARS` | 1500 | Size of a chunk (about 500 tokens). |
| `CHUNK_OVERLAP_CHARS` | 200 | Overlap between consecutive chunks. |

- **Pages.** Markdown (`ingest_markdown`, `_split_markdown`) splits on `## `
  headings. A section over `PAGE_SPLIT_CHARS` is re-split on the next heading
  level down (`###` through `######`), and what is still too big with no
  headings left falls to the plain-text splitter. Small sections are never
  descended, so a short document does not shatter. Plain text
  (`ingest_text`, `_split_plain`) splits on blank lines and merges blocks up
  to `PAGE_SPLIT_CHARS`. HTML (`_split_html`) is tag-stripped at block
  boundaries and then split like plain text. PDF pages are the PDF's own pages. When MarkItDown is installed, PDF and
  HTML are converted to markdown first and take the markdown path, falling back
  to the built-in parsers if that fails.
- **Chunks.** `_chunk_text` cuts a page into windows of `CHUNK_SIZE_CHARS`
  stepping by `CHUNK_SIZE_CHARS - CHUNK_OVERLAP_CHARS`. A page that fits in one
  chunk is one chunk.
- **Why these sizes.** The ~500-token chunk is a retrieval-precision choice
  (one chunk, one focused passage), not a limit of the embedding model.
  The default embedder (harrier) has a far larger native window. The window
  that actually applies in the stack is the `num_ctx` the gateway sends with
  every embed request: `EMBED_NUM_CTX = 1024` tokens, declared once in
  `gateway-rust/src/embedding_client.rs` and sized as twice the chunk size.
  Input past that is truncated by the backend. That is why pages are capped at
  `PAGE_SPLIT_CHARS`: a page embedding only represents the first
  `num_ctx` tokens of the page, so a page must fit inside it for the embedding
  to describe the whole page. Measured on one corpus, before the heading-descent
  split existed, 43% of pages were over the cap and one meeting transcript was a
  single 51,619-character page that was unreachable by page search.

### Query path (`pageindex_retriever.py::PageIndexRetriever.retrieve`)

1. **Embed the query.** Through `PageIndexEmbedder.embed_text` (gateway), with a
   1000-entry LRU cache keyed on the query hash. Query embeds are not queued
   behind a backfill (see the embed throttle below).
2. **Seed expansion** (add-on only; skipped when no add-on supplies it).
   Graph traversal from `seed_entity_ids`.
3. **Page search.** Three scopings: `candidate_page_ids` (from the graph
   navigator), `candidate_doc_ids` (from the peer narrowers in
   `narrowers/`), or neither (unrestricted). `top_k * 2` pages are fetched.
   `roles` (the caller's audience) is applied in every branch.
4. **Chunk-first recall** (`_merge_chunk_first_pages`). Chunks are searched
   across the corpus (`max(top_k * 4, 20)` of them; within `candidate_page_ids`
   when the graph narrowed the search) and folded into the candidate set: a
   page's score becomes the maximum of its page score and its best chunk's
   score, and pages found only through a chunk are added. It is additive by
   design, so it can add reachability but never demote a page. A failure here
   is logged and the page-level results stand.
5. **Graph boost** (add-on only). Pages mentioning graph-expanded
   entities are re-scored.
6. **Source priors.** For each page with a learned prior other than the
   neutral 0.5, `score = 0.8 * score + 0.2 * prior`; the list is then
   re-sorted. See the feedback loop below.
7. **Chunks inside the winners.** The top `top_k` pages are kept and
   `top_k * 3` chunks are searched within them.
8. **Evidence packs.** `build_evidence` groups chunks by page into
   `EvidencePack`s with the full page text.

With no add-on installed, steps 2 and 5 do nothing and the path is
vector-only.

## Entity-quality gate (`entity_quality.py`)

`check_entity(name, etype)` is the one function that decides whether a string
may be an entity. It returns `None` for acceptable, or a reason string for
junk (math symbols like `x_t` and `theta`, bare numbers, money amounts, LaTeX,
stopwords, one- and two-letter fragments, plus configured patterns). The design
rule is that a false rejection is worse than leftover junk, so the rules are
narrow and the ambiguous ones (a spelled-out `alpha`, `s2`) fire only when the
entity type says math symbol. Configuration and the kill switch are in
[configuration.md](configuration.md).

Everything that creates an entity, or an edge endpoint, asks it:

| Caller | What it gates |
|---|---|
| `live/graph_extractor.py` | entities parsed from the model's reply, before any row is written |
| `live/graph_builder.py` | co-occurrence edge endpoints (also keeps already-stored junk out of new edges) |
| `build/extractors/llm.py` | offline build extraction |
| `build/extractors/classical.py` | spaCy-derived names |
| `live/junk_entity_cleanup.py` | entities already stored |
| add-on extractors, when installed | typed relation endpoints and temporal subjects, through this same module so there is one rule set |

A caller that rejects names logs one `DropTally` line per pass (counts by
reason, a few examples), not one line per name.

## Junk-entity cleanup (`junk_entity_cleanup.py`)

The gate stops new junk; cleanup removes what is already in the graph, using
the same `check_entity`.

- **Dry run by default.** `run(engine)` only SELECTs and reports candidates
  with their footprint (edges, mentions, cluster memberships). Apply is
  explicit (`apply=True`).
- **Stages, then deletes.** Each candidate is snapshotted into the gateway's
  `graph_prune_staging` table, the same table and payload shape that the
  gateway's own `graph_prune` job uses (entity row, its edges, its mentions,
  plus its cluster memberships for human reference), then the entity is
  deleted. Edges, mentions, cluster members, embeddings and signals go with it
  by `ON DELETE CASCADE`. Staging rows are tagged `reason = 'junk_entity'` and
  a run id `entq-<8 hex>`. Staging insert and delete commit together per batch
  of 200.
- **Restore is the gateway's.** `POST /api/v1/graph/prune/restore` restores a
  staged entity. Trellion never restores. Retention of staging rows (30 days
  per the module docstring) is the gateway's `graph_prune` purge.
- **Owner stays on the payload.** Every payload carries a top-level `user_id`
  (the owning account name; `null` for shared) because the gateway's restore
  refuses a row without it.
- **Bounded and serialized.** At most `max_delete` (default 5000) per run,
  the entities with fewest edges first so a cap never leaves cheap orphans
  while taking the connected ones. It takes the same Postgres advisory lock as
  `graph_prune` (`hashtext('noo:graph_prune')`); if that is held the run
  changes nothing and reports `lock_busy`.
- **Where the staging table is.** The trellion engine's `search_path` is
  `trellion,public`, but the table lives in the gateway's `mesh` schema, so the
  caller passes `staging_schema`. noomesh's `worker.rag.entity_cleanup` looks it
  up and passes `mesh`; standalone use (`staging_schema=None`) resolves through
  the search path.
- **How to run it.** `trellion entity-cleanup` (preview) and
  `trellion entity-cleanup --apply` from the CLI (it passes no
  `staging_schema`, so on the cluster layout, where the staging table is in
  `mesh`, an apply fails with 42P01 and stages nothing; use the stack route
  there); through the stack,
  `POST /api/v1/graph/entity-cleanup` (admin only, dry run unless `apply` is the
  boolean `true`) and the gateway's scheduled `entity_cleanup` action.

## Video memory (`video/`, `recall/`)

Episodic ("System 1") memory of proven graph walks and conversation frames.
It is Postgres plus pgvector now: frame embeddings are a column
(`101_frame_embeddings`), searched in SQL; no MP4 files are written.

`video/manager.py::VideoMemoryManager` is the facade the worker pool constructs.
Its engine wiring is **lazy**, because the worker pool starts the manager before
trellion's shared SQLAlchemy engine exists:

- `startup()` creates the storage directory and tries `_ensure_engine()`. If
  `trellion.db.engine.get_async_engine()` has no engine yet, it logs
  `video_memory.engine_deferred` and continues; that is not a fault.
- Every method that needs the engine (`search`, `search_traversals`,
  `run_lifecycle`, `status`, ...) calls `_ensure_engine()` first. The first call
  after the engine exists builds the `TraversalReplayEngine`,
  `RecallLifecycleManager` and `EpisodicRecallManager` and logs
  `video_memory.engine_wired`. Until then `search` and its siblings return empty results
  instead of failing, and `status()["engine_wired"]` says which state it is in.
- The facade keeps the old method names so callers do not change;
  conversation archiving is a no-op (conversations already live in
  `noo_entries` with embeddings).

`recall/` holds the building blocks: `archiver.py` (writes proven traversals),
`replay.py` (finds similar past walks), `lifecycle.py` (prune/promote),
`pg_index.py` (the pgvector index), `walk_store.py`, and `faiss_index.py` for the
FAISS artifacts that remain under `NOO_VIDEO_MEMORY_PATH`, the single directory
every component uses (see "Video memory and recall" in
[configuration.md](configuration.md), including the one-time migration of the old
hyphenated directory).

## Source priors

Pages that helped answer a question can be ranked slightly higher next time.
This package only reads the result: `source_priors.py` loads `source_rewards`
from the JSON file named by `RETRIEVAL_TRACKER_PATH` (mtime-checked at most
every 5 s, last good values kept on a torn file, neutral 0.5 when the file or
key is absent) and the retriever applies it in step 6 of the query path. The
module has no dependency on any add-on, so the base install has nothing to fail
and nothing to log per query. Something else (for example a commercial add-on)
has to write the file; with no writer every prior is neutral and ranking is
unchanged. The reader and the writer must see the same file.

## The embed throttle (`embed_throttle.py`)

Embedding backfills share one GPU embedder behind the gateway. The gateway
admits at most 4 requests per embedder target; one that cannot get a slot
within about 1.5 s is shed with 503 and a `Retry-After` of 2 to 10 seconds
(`gateway-rust/src/embedding_client.rs`). Without a client-side brake, several
loops started together by a stack restart kept the embedder saturated and about three quarters of calls failed.
`EmbedThrottle` is the one place that decides how hard a process may push:

- **Bounded concurrency**: at most `TRELLION_EMBED_MAX_INFLIGHT` (default 1)
  requests per process, shared by the page embedder, chunk embedder, graph
  entity embedder and classifier templates. Interactive query embeds skip the
  semaphore (`bounded=False`, two attempts) so they never wait behind a backlog.
- **Retry with backoff and jitter** on 429, 502, 503, 504, timeouts and
  transport errors, honouring `Retry-After`. After the last attempt it raises
  `EmbedUnavailable`; the rows stay unembedded and a later pass picks them up.
- **AIMD batch size**: a failed call halves the allowed batch (floor
  `BATCH_MIN`), a success adds `BATCH_STEP` (ceiling `BATCH_MAX`).
- **Cool-down and pacing**: after consecutive failures a loop rests a growing,
  jittered interval; after a success it rests `PACE_S`.
- **Start-up jitter**: each loop waits `uniform(0, STARTUP_JITTER_S)` before its
  first pass.

`get_throttle()` returns the process-wide instance. The variables and defaults
are in [configuration.md](configuration.md).
