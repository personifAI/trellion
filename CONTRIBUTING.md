# Contributing to trellion

Thanks for your interest. Trellion is licensed AGPL-3.0-only, with a
commercial licence available (see [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md)).

## Contributor licence agreement

Every contribution needs a signed [CLA](CLA.md); it is what lets PersonifAI
LLC keep offering the commercial licence. The CLA bot comments on your first
pull request. Reply with the sentence it asks for:

> I have read the CLA Document and I hereby sign the CLA

The signature is recorded once and covers later pull requests. A pull request
cannot be merged until the check is green. If you contribute on behalf of an
employer, make sure you are allowed to; questions go to contact@noomesh.ai.

## Development setup

Python 3.11 or newer.

```bash
git clone https://github.com/personifAI/trellion.git
cd trellion
pip install -e ".[dev,clustering]"
```

The live subsystem needs PostgreSQL with the `pgvector` and `pg_trgm`
extensions; see the README. Most tests do not.

## Tests and checks

```bash
python3 -m pytest -q tests                        # DB-backed tests skip without TRELLION_TEST_DSN
pip install ruff==0.15.21                         # the version CI pins
ruff check --select E4,E7,E9,F src/               # CI's lint gate
python3 scripts/spdx.py . AGPL-3.0-only --check   # licence headers
```

To run the database-backed tests, point `TRELLION_TEST_DSN`
(`postgresql+asyncpg://...`) at a throwaway database migrated with
`alembic -c src/trellion/db/alembic.ini upgrade heads`. The tests write rows,
so never use a database you care about.

`tests/test_no_private_identifiers.py` fails if the tree carries private network addresses or credentials (and, when a maintainer supplies a private name list, private
names). Keep fixtures neutral (`alice`, `node-1`,
`192.0.2.x`, `example.ts.net`).

## Licence headers

Every source file carries two comment lines at the top (after any shebang):

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC
```

`python3 scripts/spdx.py . AGPL-3.0-only` adds them where missing; `--check`
is what CI runs. Markdown, JSON and SQL files are not stamped. Never edit an
existing migration's SQL: add a new revision instead.

## Pull requests

- Open pull requests against the repository's default branch.
- Keep a change focused, add or update a test with it, and run the checks
  above first.
- Describe what changed and why in the pull request body.

Report security problems privately, not as issues; see [SECURITY.md](SECURITY.md).
