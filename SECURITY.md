# Security policy

## Reporting a vulnerability

Email **contact@noomesh.ai** with the subject line "trellion security". Please
do not open a public issue, pull request or discussion for a vulnerability.

Include what you found, the version or commit, steps to reproduce, and the
impact you expect. We will acknowledge your report, keep you informed while we
investigate, and credit you in the fix if you wish.

## Scope

In scope: the code in this repository, including the Alembic migrations, the
`trellion` CLI, and the way the package handles credentials, SQL, access
scoping by audience and user, and erasure of a user's data.

Out of scope: the noomesh stack and its gateway (report those to the same
address, naming the component), third-party dependencies (report them
upstream; tell us if trellion's use of one makes it exploitable), and findings
that need an already-compromised host or database superuser.

## Supported versions

Fixes land on the default branch and ship in the next release. Only the latest
release is supported.
