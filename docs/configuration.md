# Configuration

Every environment variable that `src/trellion` reads, with its default, its
effect, and what the noomesh stack sets. "Stack" below means the
`noo-trellion` container from the noomesh stack's `docker-compose.trellion.yml`
overlay (the `x-trellion-env` block). Outside the stack, set the variables
yourself.

Trellion has no settings file. The few knobs that are structured (the
entity-quality rules) are a JSON file named by an env var.

How to read the "Stack sets" column: **yes** means the compose overlay sets the
variable (value shown); **no** means it is not set, so the code default applies
unless the operator adds it through the compose override file.

## Gateway access

| Variable | Default | Effect | Stack sets |
|---|---|---|---|
| `NOO_GATEWAY_URL` | `https://gateway:8080` (trailing `/` stripped) | Base URL for the add-on token exchange (`POST /api/addon/token`) and for the image-description call in `pageindex_ingestion.py`. Read in `live/gateway_auth.py` and `live/pageindex_ingestion.py`. | yes: `${NOO_TRELLION_GATEWAY_URL:-https://gateway:8080}` |
| `NOO_GATEWAY_KEY_FILE` | unset | Path to a file holding this container's gateway service key. Wins over every other source. A named file that is unreadable or empty raises `SecretFileError`; it never falls through to the env. Read through `secret()` in `live/secret_env.py`. | yes: `/run/secrets/noo_gateway_key` when `NOO_SVC_KEY_TRELLION` is set in the launcher env |
| `NOO_SECRET_NOO_GATEWAY_KEY` | unset | The key as a plain value; second in precedence. | no |
| `NOO_GATEWAY_KEY` | unset | The key as a plain value; third in precedence. With none of the three set, gateway calls raise `GatewayCredentialMissing` and are not sent. | no (the stack delivers the key as a file) |
| `NOO_WORKER_SURFACES` | `""` | Only used to pick which service identity the missing-credential error names. The surface selection itself is noomesh's. Read in `live/gateway_auth.py`. | yes: `ml` (`${NOO_WORKER_SURFACES:-ml}`) |
| `NOO_TLS_CA` | `""` | CA bundle for outbound gateway/embedding calls. When it names an existing file, trellion verifies the chain against it with hostname checking off. Otherwise default verification. `live/_tls.py`. | yes: `/app/pki/ca.crt` when set in the launcher env |
| `NOO_TLS_CERT`, `NOO_TLS_KEY` | `""` | Client certificate and key for mTLS. Both must name existing files, otherwise no client cert is presented. `live/_tls.py`. | yes: `/app/pki/leaf/service.crt`, `/app/pki/leaf/service.key` when set in the launcher env |

Secret precedence in full (`live/secret_env.py`): `NAME_FILE`, then
`NOO_SECRET_NAME`, then `NAME`. Having both `NAME` and `NAME_FILE` non-empty is
a deployment skew; `refuse_on_conflicts()` is the boot check for it. In the
stack, noomesh's `worker_pool` runs its own check over a manifest that includes
`NOO_GATEWAY_KEY`; trellion's check covers the `trellion` CLI and any other
embedder of the package. The only name in `FILE_BACKED_SECRETS` is
`NOO_GATEWAY_KEY`.

## Embedding and LLM endpoints (`PAGEINDEX_*`)

These are read by the noomesh stack's worker pool and passed into `PageIndexRAG(...)` in
`live/rag_integration.py`. Trellion's own code reads one of them directly
(`PAGEINDEX_EMBEDDING_URL`, in the `bench-live` CLI command).

| Variable | Default (worker pool) | Effect | Stack sets |
|---|---|---|---|
| `PAGEINDEX_EMBEDDING_URL` | `https://gateway:8080` (or `services.embedding_url` in `noomesh.yaml`) | Where embedding calls go. **Must resolve to the gateway container, not `localhost`**: inside `noo-trellion`, `localhost:8080` is trellion itself and has no embeddings endpoint. The `bench-live` CLI reads it with default `""`. | yes: `${NOO_TRELLION_EMBEDDING_URL:-https://gateway:8080}` |
| `PAGEINDEX_EMBEDDING_BACKEND` | `gateway` (or `services.embedding_backend`) | Accepted for compatibility and **ignored**: `PageIndexEmbedder` always talks to the gateway, and a value other than `gateway` only logs `pageindex_embedder.backend_ignored`. The engine is chosen by the `role=embed` placement. | yes, but empty unless the launcher env sets it |
| `PAGEINDEX_EMBEDDING_MODEL` | `""` (or `services.embedding_model`) | Not used to select the model. The `role=embed` placement declares what the fleet's embedder serves; pinning a model in config is what broke the harrier cutover (see the comment in `live/pageindex_embedder.py`). | yes, but empty unless the launcher env sets it |
| `PAGEINDEX_LLM_URL` | n/a | **Retired in noomesh 0.6.8 (proposal i145)**: the worker pool no longer reads it and always hands trellion the gateway address (`NOO_GATEWAY_URL`, default `https://gateway:8080`), so ingestion, graph extraction and answers are mesh-routed with failover. A leftover `.env` line only draws a launcher warning. | no |
| `PAGEINDEX_LLM_MODEL` | n/a | **Retired in noomesh 0.6.8 (proposal i145)**: the worker pool always sends the purpose `knowledge` as the model. Which model serves it is a preference in the noomesh DB (`noo cluster class set <class> <node>/<backend> --alias knowledge`); undeclared, any live chat node answers. The constructor default `llama3` still applies only when trellion is used outside a noomesh worker. | no |
| `PAGEINDEX_DSN` | none | Postgres DSN. The worker pool prefers it over `DATABASE_URL`; the `trellion` CLI reads `--dsn`, then `DATABASE_URL`, then `PAGEINDEX_DSN`; the migration env reads it after `NOO_MIGRATIONS_DSN`. | yes: runtime role `noo_app`, `search_path=mesh,public,trellion`, verify-full TLS |
| `PAGEINDEX_EMBEDDING_EXTRA_URLS` | none | Gateway-side only (`gateway-rust/src/embedding_client.rs`); trellion does not read it. Listed because it sits next to the others. | no |

## Embedding space, provenance and re-embed (`TRELLION_EMBEDDING_SPACE`)

| Variable | Default | Effect | Stack sets |
|---|---|---|---|
| `TRELLION_EMBEDDING_SPACE` | `knowledge` | The gateway embedding **space** trellion's vectors live in. Every embed call (page/chunk embedder, entry worker, episode embedder) sends it as `model` on `POST /api/external/embeddings`. Read only by `embedding_space.py`. | no |

The gateway maps a space to the one model that serves it (`noo_embedding_spaces`);
trellion never names a model. What it does record is the model the **response**
named, per vector:

- Every vector column has `embedding_model` and `embedding_dim` beside it
  (`chunk_embeddings`, `page_embeddings`, `entity_embeddings`, `entity_clusters`
  centroids, `noo_entries`, `trellion_concepts` centroids,
  `trellion_traversal_records.query_embedding`, `trellion_episodes`; migration
  `106_embedding_provenance` added the pair to `trellion_episodes`). A cluster
  centroid takes the model of the vectors it was averaged from; a caller that
  knows it passes the centroid as `embedding_space.Vector(values, model)`,
  otherwise the space's current model is recorded.
- **Every similarity query filters `embedding_model = <current model>`**, so a
  table holding vectors from two models never ranks one against the other. The
  current model is the query vector's own (it came from the embed response),
  else the space's record from `GET /api/external/capabilities`
  (`embeddings.spaces`), cached for 30 seconds. If neither is known (gateway
  unreachable, no cached record) the filter is **dropped** and logged once
  (`embedding_space.lookup_failed`): a failed lookup never empties a result.
- Vectors written before migration 106 have a NULL model. The embedder loop
  labels them with the space's current model at start (then every five
  minutes): every pre-provenance vector came from the default space. It is a
  batched `UPDATE ... WHERE embedding_model IS NULL AND <vector> IS NOT NULL`,
  idempotent and restart-safe, and is skipped during a cutover.

### Re-embedding a space into a new model

When the operator starts a migration (`noo cluster embedding-space migrate`),
the space's capabilities entry carries `migration.state`. The embedder loop
(`live/reembed.py`) follows it:

| State | What trellion does |
|---|---|
| none (cancelled or finished) | Clears leftover `embedding_next` / `embedding_next_model` staging columns in batches. |
| `filling` | Per table, in claimed batches (`FOR NO KEY UPDATE SKIP LOCKED`), re-embeds rows whose `embedding_next_model` is not the migration's `next_model`, with `model = "<space>@next"`, into `embedding_next`. The live column keeps serving. Rows written during the migration are picked up by the same scan. Progress goes to `PUT /api/external/embedding-spaces/{space}/migration` about every 30 seconds. |
| `cutover` | Ordinary page/chunk embedding pauses. Per table, one transaction swaps `embedding_next` into the live column and sets `embedding_model`/`embedding_dim`; then the progress PUT carries `cutover_done: true` and the gateway makes the new model the space's model. |

Where the re-embed text comes from, per table: `page_embeddings` (`pages.text`),
`chunk_embeddings` (`page_chunks.text`), `noo_entries` (`content`, clipped to
1500 characters as the entry worker does), `trellion_traversal_records`
(`query_text`), `entity_embeddings` (the neighbourhood description trellion-intel
embeds, rebuilt from `entities` and `edges`). Derived, not embedded:
`entity_clusters` (a leaf centroid is the mean of its members' staged vectors, a
parent's the member-weighted mean of its children's) and `trellion_episodes`
(copied from the traversal record it was archived with; an episode with no such
record cannot be re-embedded). **Unsourced rows.** `trellion_concepts.centroid` (nothing in trellion,
trellion-intel or trellion-evolve writes it) and episodes with no traversal
record have neither stored text nor a derivation, and a row the embedder rejects
even when sent alone (retried one at a time, then logged `reembed.row_skipped`
and skipped for this migration) cannot be re-embedded. None is invented. They
are left out of `done`/`total` and reported per table as
`"unsourced": N` (`{done, total, unsourced}`). At cutover their vector and
`embedding_model` are set to NULL in the same transaction (a skipped
`entity_embeddings` row is deleted, its column being NOT NULL), so they drop out
of search instead of keeping an old-model vector under the new model. Whatever
owns them regenerates them on its next run.

**Stale sweep.** With no migration running, rows whose `embedding_model` is set
but is not the space's current model, and that have a source, are re-embedded
into the live column with the plain space model (episodes copy from their
traversal record). Same cadence as the legacy claim: at start, on a state
change, then every five minutes. This heals the brief cutover race and anything
else written under an old model. Cluster centroids follow their members on the
next clustering run.

Any ordinary write of a vector clears that row's staged `embedding_next` (it
was computed from content that has just changed). The staging columns are
`vector(1024)`, the same width as the live ones: a model of another width needs
the columns retyped, which is not built.

## Database

| Variable | Default | Effect | Stack sets |
|---|---|---|---|
| `DATABASE_URL` | none | CLI DSN fallback (`cli.py`); last-resort source for the migration env; the worker pool's fallback when `PAGEINDEX_DSN` is unset. | yes (same runtime DSN as `PAGEINDEX_DSN`) |
| `PGVECTOR_URL` | none | Migration env source, between `PAGEINDEX_DSN` and `DATABASE_URL`. | yes |
| `NOO_TRELLION_DDL_DATABASE_URL` | none | **First** choice for migrations: the DDL role (`noo_trellion_ddl`). The runtime role never holds CREATE. `db/migrations/env.py`. | yes |
| `NOO_MIGRATIONS_DSN` | none | Second choice for migrations; the legacy privileged path, kept so an older container keeps migrating. | no |
| `NOO_TRELLION_SCHEMA` | `trellion` | Postgres schema the memory plane lives in (`db/engine.py`, `db/migrations/env.py`). Also the schema of the `alembic_version_memory` table. | no |
| `EMBEDDING_DIM` | `768` | Dimension of the `Vector(...)` columns declared in `db/models.py`, the dimension `PageIndexStore` and the pgvector store validate incoming vectors against, the video-memory config default, and the FAISS index dimension. **The migrations create `vector(1024)` columns** (`db/migrations/versions/100_trellion_squash.py`, `101_frame_embeddings.py`), which is what harrier produces. A deployment on the default embedder must set `EMBEDDING_DIM=1024`; the compose default of 768 is only right for a 768-dim embedder. | yes: `${EMBEDDING_DIM:-768}` |

The `trellion` CLI (`cli.py`) resolves its DSN as `--dsn`, then `DATABASE_URL`,
then `PAGEINDEX_DSN`.

## Retrieval feedback (priors)

| Variable | Default | Effect | Stack sets |
|---|---|---|---|
| `RETRIEVAL_TRACKER_PATH` | unset | The add-on's state file, from which the learned **source priors** are read (`live/source_priors.py::default_path`). This package only reads it (an installed add-on is what writes it); the retriever re-checks the file's mtime at most every 5 s. Unset variable, absent file, missing `source_rewards` key, or a torn write means every prior stays neutral (0.5). The reader and any writer must see the same file, so set the same path for both. | yes: a file under `/data` |

See "Source priors" in [architecture.md](architecture.md) for
how priors change scores. If the reader and the writer do not share the file,
priors stay neutral.

## Background loops (`TRELLION_BACKGROUND_WORKERS`)

`PageIndexRAG.startup()` (`live/rag_integration.py`) starts the backlog loops:
page and chunk embedding, entity extraction, graph building and, when an
add-on provides them, enrichment and graph metrics. Set
`TRELLION_BACKGROUND_WORKERS=0` (also `false`, `off`, `no`) on a process that
must not run them because another process on the same database does. That
process still serves queries and ingests; the other one picks up its new rows
on its next poll. Unset or any other value: the loops run. Every loop claims
its rows, so two runners stay correct, but each polls Postgres on its own.

## Embed throttle (`TRELLION_EMBED_*`)

Read once per process when the shared `EmbedThrottle` is built
(`live/embed_throttle.py`, `get_throttle()`). A value that does not parse as a
number falls back to the default. Bounds are applied after parsing. None of
these are set by the stack, so the defaults below are what runs.

| Variable | Default | Bound | Effect |
|---|---|---|---|
| `TRELLION_EMBED_MAX_INFLIGHT` | `1` | min 1 | Concurrent embedding requests per process, shared by every loop in it (page embedder, chunk embedder, graph entity embedder, classifier templates). Interactive query embeds bypass this bound but keep retry. |
| `TRELLION_EMBED_BATCH_MAX` | `32` | min 1 | Largest batch a loop may send; also the starting batch size. |
| `TRELLION_EMBED_BATCH_MIN` | `4` | 1 to `BATCH_MAX` | Floor the batch shrinks to on failure. |
| `TRELLION_EMBED_BATCH_STEP` | `4` | min 1 | Growth per successful call, up to `BATCH_MAX`. |
| `TRELLION_EMBED_MAX_ATTEMPTS` | `4` | min 1 | Attempts per call before `EmbedUnavailable` is raised and the rows are left unembedded for a later pass. |
| `TRELLION_EMBED_BACKOFF_BASE_S` | `2.0` | none | Base of the exponential backoff; the delay before retry n is uniform in `[c/2, c]` with `c = min(cap, base * 2^(n-1))`, raised to `Retry-After` (capped) when the gateway sends one. |
| `TRELLION_EMBED_BACKOFF_CAP_S` | `60.0` | none | Ceiling for backoff and for honoured `Retry-After`. |
| `TRELLION_EMBED_PACE_S` | `0.5` | none | Rest between batches after a success. After consecutive failures the rest is a backoff delay instead. |
| `TRELLION_EMBED_STARTUP_JITTER_S` | `30.0` | `0` disables | Each loop waits `uniform(0, value)` before its first pass, so containers restarted together do not begin together. |
| `TRELLION_EMBED_IDLE_MAX_S` | `10.0` | none | Longest wait between backlog polls while there is nothing to embed (the wait starts at 5 s and doubles up to this). It fallback-bounds how long a newly ingested document waits before it is embedded and citable: the normal wake is Postgres LISTEN/NOTIFY (ingest NOTIFYs `trellion_doc_parsed`, the embedder LISTENs on a direct, non-transaction-pooled connection), so the first embed starts in about a second, and this idle cap is only the fallback. |

Retried statuses: 429, 502, 503, 504, plus timeouts and transport errors.
Every failed call halves the batch size (floor `BATCH_MIN`); every success adds
`BATCH_STEP`.

## Bulk embedding (`NOO_EMBED_BULK`)

Background embedding (the re-embed, the page/chunk backlog, the graph entity
sweep) can hand its batches to the gateway's job route
(`POST /api/external/embeddings/jobs`) instead of holding a synchronous embed.
Query-time embeds never do. Read at call time in `live/embed_bulk.py`.

| Variable | Default | Effect |
|---|---|---|
| `NOO_EMBED_BULK` | off | `1`/`true`/`yes`/`on` sends background embeds as jobs of at most 64 texts, two at a time, polled with backoff (0.5 s growing to 5 s). Off is exactly the synchronous call. A gateway answering 404/405 on submit is treated as old: trellion falls back to the synchronous call and logs once. |
| `NOO_EMBED_BULK_TIMEOUT_S` | `900` | Overall wait for one `embed_bulk` call. A job still pending then raises `EmbedUnavailable` (`bulk_timeout`). |

A 429/503 follows the throttle above (Retry-After honoured, then
`EmbedUnavailable`); a job that ends `dead` is an ordinary error, so the
re-embed retries those rows alone.

## LLM throttle and circuit breaker (`TRELLION_LLM_*`)

Read once per process when the shared `LLMThrottle` is built
(`live/llm_throttle.py`, `get_llm_throttle()`). A value that does not parse as a
number falls back to the default. Bounds are applied after parsing. None of
these are set by the stack, so the defaults below are what runs.

Every looping or batch LLM caller goes through it: live entity extraction
(`live/graph_extractor.py`), per-page OCR (`live/pageindex_ingestion.py`), the
offline build's `LLMClient` (`common/llm_client.py`), the answer generator
(`live/rag_generator.py`, interactive: no retry, no in-flight bound, breaker
honoured), and any add-on that makes LLM calls.

| Variable | Default | Bound | Effect |
|---|---|---|---|
| `TRELLION_LLM_MAX_INFLIGHT` | `2` | min 1 | Concurrent LLM requests per process, shared by every loop in it. Interactive single calls (`bounded=False`) bypass it; the offline build uses its own `llm.max_concurrent`. |
| `TRELLION_LLM_MAX_ATTEMPTS` | `3` | min 1 | Attempts per call before `LLMUnavailable` is raised and the item is left queued. A probe is always one attempt. |
| `TRELLION_LLM_BACKOFF_BASE_S` | `2.0` | none | Same shape as the embed throttle: delay before retry n is uniform in `[c/2, c]`, `c = min(cap, base * 2^(n-1))`, raised to `Retry-After` (capped). |
| `TRELLION_LLM_BACKOFF_CAP_S` | `60.0` | none | Ceiling for backoff and for honoured `Retry-After`. |
| `TRELLION_LLM_BREAKER_THRESHOLD` | `5` | min 1 | Consecutive 502/503/504, timeouts or transport errors for ONE model that open that model's breaker. 429 backs off but never counts. |
| `TRELLION_LLM_BREAKER_COOLDOWN_S` | `30.0` | min 0 | How long an open breaker waits before letting a single probe call through. A failed probe doubles the wait. |
| `TRELLION_LLM_BREAKER_COOLDOWN_CAP_S` | `600.0` | at least `COOLDOWN_S` | Ceiling for the doubling. |
| `TRELLION_LLM_STARTUP_JITTER_S` | `30.0` | `0` disables | The extraction loop waits `uniform(0, value)` before its first pass. |
| `TRELLION_LLM_BUILD_WAIT_S` | `1800` | none | Offline build only: how long a chunk waits for an open breaker to close before the build fails. A build cannot skip a chunk and call the graph done. |

The breaker is per model. A 404 whose body carries `model_not_found` (the
gateway says nobody serves that model) opens it at once; a bare 404 is a wrong
URL and is not treated as absence. While open, no request is sent for that
model: loops leave their pages unclaimed and unextracted (never marked done,
never dropped) and the next pass after the cool-down picks them up. The first
success closes it. Logging is one `llm_breaker_open` warning and one
`llm_breaker_closed` info line (model, seconds open, items deferred), plus
`llm_backpressure` per retry. Expected unavailability logs no traceback;
unexpected exceptions (400, parse bugs) keep theirs and do not trip the breaker.

## Entity quality (`TRELLION_ENTITY_QUALITY*`)

Read by `live/entity_quality.py::load_config`, once per process (the result is
cached in `get_config()`; tests call `reset_config_cache()`). Not set by the
stack.

| Variable | Default | Effect |
|---|---|---|
| `TRELLION_ENTITY_QUALITY` | `on` | Kill switch. `off`, `0`, `false` or `no` (case-insensitive, whitespace trimmed) disables the gate everywhere: `check_entity` returns `None` for every name, so nothing is rejected, and junk-entity cleanup finds no candidates. Any other value, including empty, leaves it on. |
| `TRELLION_ENTITY_QUALITY_CONFIG` | unset | Path to a JSON file that adds to the built-in rules. |

The JSON overlay. Every key is optional; unknown keys are ignored:

```json
{
  "enabled": true,
  "allow": ["ml", "qa"],
  "stopwords": ["whereas"],
  "reject_patterns": ["^tmp_\\d+$"]
}
```

| Key | Type | Effect |
|---|---|---|
| `enabled` | bool, default `true` | ANDed with the env switch. `false` disables the gate even when `TRELLION_ENTITY_QUALITY` is on; `true` cannot re-enable it when the env says off. |
| `allow` | list of strings | Added to the built-in allow-list of short names that are real (`ai`, `ml`, `qa`, `go`, `R`, `C`, ...). Matched case-insensitively, except single-character entries, which are case-sensitive (`R` the language is real, `r` the radius is not). Allow wins over every other rule. |
| `stopwords` | list of strings | Added to the built-in function-word list, lowercased. A name that matches is rejected with reason `stopword`. |
| `reject_patterns` | list of regex strings | Extra patterns, compiled case-insensitive and tested with `re.search` against the whitespace-collapsed name. A match is rejected with reason `configured_pattern`. They run after the built-in symbol and number rules and before the length rule. |

A missing or broken file (unreadable, invalid JSON, bad regex) logs
`entity_quality.config_unreadable` at warning and leaves the defaults in force.
A broken override never takes extraction down and never switches the gate off.
Extra `allow` entries are compared as written, so write multi-character
entries in lowercase (the name is lowercased before the lookup).
Because a bad regex discards the whole overlay, fix the warning rather than
assuming the other keys applied.

Rejection reasons (the strings `check_entity` returns): `not_a_string`,
`empty`, `stopword`, `punctuation`, `numeric`, `latex`, `greek_symbol`,
`math_symbol`, `configured_pattern`, `too_short`.

## Video memory and recall

| Variable | Default | Effect | Stack sets |
|---|---|---|---|
| `NOO_VIDEO_MEMORY_PATH` | `/data/video_memory` | The one directory for video-memory / episodic-recall data (FAISS artifacts, replay data). It decides the directory for every reader and writer: `VideoMemoryConfig.storage_dir` defaults to it, so `VideoMemoryManager` and the lifecycle, replay and archiver components it builds use it; `EpisodicRecallManager` uses it when no `faiss_base_path` is passed; the `pack` and `unpack` CLI commands use it when `--faiss-path` is not given. The default literal lives only in `live/video/paths.py`. | yes: `/data/video_memory` |

Earlier releases had two spellings: `VideoMemoryConfig.storage_dir` defaulted to
`/data/video-memory` (hyphen) while the stack set `NOO_VIDEO_MEMORY_PATH` to
`/data/video_memory` (underscore), so the facade wrote to one directory and
`pack`/`unpack` read the other. That is fixed. On startup the manager folds a
leftover `/data/video-memory` into the configured directory, once:

- files missing at the target are moved; a file present in both keeps the newer
  copy and the other is parked under `<dir>/.legacy-<timestamp>/` (`from-legacy/`
  or `from-target/`). Nothing is deleted.
- FAISS files are not mixed: if a persona's `faiss/` directory exists in both
  places, the configured directory's index stays live and the legacy `faiss/`
  files are parked whole. There is no vector-level merge; re-run the archiver
  if you want the parked vectors back in the live index.
- `<dir>/.legacy-migrated` records completion; a lock file
  (`<dir>/.legacy-migrate.lock`) makes exactly one of several containers on the
  shared volume do the work. One `video_memory.legacy_dir_migrated` log line
  reports the counts.

Frame search itself is pgvector now (see [architecture.md](architecture.md));
the directory holds the FAISS artifacts that remain.

## Tests

| Variable | Default | Effect |
|---|---|---|
| `TRELLION_TEST_DSN` | unset | Enables the database-backed tests (`test_video_pgvector.py`, `test_traversal_user_id.py`, `test_entity_quality.py`, `test_batch_snapshot.py`, ...). Use a `postgresql+asyncpg://` DSN against a database migrated with `alembic -c src/trellion/db/alembic.ini upgrade heads`. Without it those tests skip. See the README. |

## Set by compose, not read by this package

For completeness, so nobody greps trellion for them: `NOO_TRELLION_DATABASE_URL`
(the runtime DSN noomesh's worker pool uses), `NOO_MIGRATIONS_ENVS` (`memory`),
`NOO_WORKER_HEALTH_PORT`, `NOO_WORKER_RSS_LIMIT_MB`,
`NOO_WORKER_GLOBAL_CONCURRENCY`, `NATS_*`, `REDIS_*`, `PGPASSFILE` and the
channel/cloud-provider variables are consumed by noomesh's worker pool, not by
`src/trellion`. They are declared in `docker-compose.trellion.yml` in the
noomesh stack, which is the authority for them.
