# trellion

Trellion is a memory library for AI agents: a Postgres-backed knowledge base
(RAG over pages and chunks), conversation memory, a bi-temporal knowledge
graph, episodic and video recall, and snapshot consolidation. It runs as a
Python library inside your own process. It is also the memory plane of the
[noomesh](https://noomesh.ai) stack, where noomesh's worker pool serves it
over a message bus; nothing here requires that stack.

An offline CLI is included for building a concept graph from documents and
querying it deterministically.

- [docs/architecture.md](docs/architecture.md) - the live retrieval path, the
  entity-quality gate, junk-entity cleanup, video memory, source priors and the
  embed throttle.
- [docs/configuration.md](docs/configuration.md) - every environment variable
  the package reads, with defaults and effects.

## Install

Python 3.11 or newer.

```bash
git clone https://github.com/personifAI/trellion.git
cd trellion
pip install -e .                 # core
pip install -e ".[all]"          # plus every optional extra below
```

Optional extras (see `pyproject.toml`):

| Extra | Adds |
|---|---|
| `clustering` | hierarchical Leiden clustering for the offline build (`leidenalg`, `igraph`) |
| `rag` | PDF and document conversion (`pymupdf`, `markitdown`) |
| `video` | fast video-memory recall (`faiss-cpu`) |
| `graph-igraph` | `python-igraph` graph backend |
| `transport` | Parquet shard packaging (`pyarrow`) |
| `dev` | pytest, asyncpg, alembic, ruff, mypy |

The live memory subsystem stores data in **PostgreSQL with the `pgvector`
extension** (and `pg_trgm`). The migrations verify these extensions but do not
create them, since creating them needs superuser; install them as your
database owner first, then migrate:

```bash
psql "$DATABASE_URL" -c "CREATE EXTENSION IF NOT EXISTS vector" \
                     -c "CREATE EXTENSION IF NOT EXISTS pg_trgm"
alembic -c src/trellion/db/alembic.ini upgrade heads
```

The `pgvector/pgvector:pg16` Docker image has the extension preinstalled. The
migration environment reads the DSN from `NOO_TRELLION_DDL_DATABASE_URL`,
`NOO_MIGRATIONS_DSN`, `PAGEINDEX_DSN`, `PGVECTOR_URL` or `DATABASE_URL`, in that
order. Embedding and LLM calls go through an OpenAI-compatible gateway; see
[docs/configuration.md](docs/configuration.md).

## Library usage

Session memory with the in-process store (no database needed):

```python
from trellion import MemoryConfig, MemorySystem

mem = MemorySystem(MemoryConfig(enabled=True))
session = mem.new_session()
mem.record(session, "user", "I prefer pgvector for embeddings")
print(mem.retrieve("pgvector embeddings", session))
# Relevant memory:
# - (session, user, 0.40) I prefer pgvector for embeddings
```

Pass a `ConversationStore` (Postgres) and an embedder to `MemorySystem` for
persistent, semantic memory. Querying a concept graph from a CSV:

```python
from pathlib import Path
from trellion.query.graph import CKG
from trellion.query.answer import answer_query

ckg = CKG.load(Path("data/sample/learning-graph.csv"))
answer = answer_query(ckg, "What are the prerequisites for Composite Function?")
print(answer.query_type, answer.content)   # T2 Function; Function Notation
```

## CLI

```bash
trellion --help
trellion query --graph data/sample/learning-graph.csv "What are the prerequisites for Composite Function?"

# Build a graph from your own documents. The config names the input directory
# and an OpenAI-compatible LLM and embedding endpoint (configs/build.default.yaml).
trellion build --config configs/build.default.yaml --dsn "$DATABASE_URL"   # prints a snapshot id
trellion query --snapshot-id <id> --dsn "$DATABASE_URL" "What depends on Function?"
```

`build` stores the graph as a snapshot in Postgres. Without a DSN (or with
`--csv-only`) it writes only the debug CSV, which `query --graph` can read.
Other commands: `eval`, `inspect`, `consolidate`, `entity-cleanup`,
`vidmem-backfill`, `bench-live`, `pack` and `unpack`; run
`trellion <command> --help`.

## Running the tests

No database or noomesh stack is needed for most of the suite:

```bash
pip install -e ".[dev]"
python3 -m pytest -q tests
```

Database-backed tests run only when `TRELLION_TEST_DSN` is set (scheme
`postgresql+asyncpg://`, a throwaway database migrated with the command above;
the tests write rows). Otherwise they skip, so read the summary line for
`skipped` instead of assuming coverage.

## Add-ons

A commercial add-on can extend retrieval via entry points in the
`trellion.addons` group, using the interface in
[trellion-protocols](https://github.com/personifAI/trellion-protocols)
(Apache-2.0); see [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md). Trellion runs
without any add-on: with none installed, retrieval is vector-only.

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md) (contributions require signing the
[CLA](CLA.md)) and [SECURITY.md](SECURITY.md).

## Licence

AGPL-3.0-only (see [LICENSE](LICENSE)). A commercial licence is available for
use without AGPL obligations; see [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md)
or write to contact@noomesh.ai. Bundled sample data (`data/sample/`) is CC BY 4.0; see [DATA-LICENSE.md](DATA-LICENSE.md).
