# Python interpreter resolution for repository scripts

Issue: #1606

## Problem

The helper scripts require Python 3.11+ — `enum.StrEnum` in
`scripts/ci/change_policy.py`, `tomllib` in `check-api-consistency.py`,
`check-guide-dependency-snippets.py` and `check-docs-site.py` — but nothing
declared or enforced it, and every shell entry point invoked a bare `python3`.

On macOS the system `python3` is 3.9, so `scripts/check-pr-fast.sh`, the local
gate contributors are told to run, died inside `change_policy.py` with
`AttributeError: module 'enum' has no attribute 'StrEnum'` before reaching a
single check. CI never saw it: no workflow uses `actions/setup-python`, so the
scripts run on the runner default (3.12 on `ubuntu-latest`).

Two of the affected scripts already raise `RuntimeError("Python 3.11+ is
required for tomllib support")`, i.e. the requirement was known — but only
reported after the wrong interpreter had already been chosen.

## Change

`scripts/lib/python.sh` resolves an interpreter once and exposes `py`. The
seven shell scripts that invoke Python source it and call `py` instead of
`python3`: `check-pr-fast.sh`, `build_docs_site.sh`, `serve_docs_site.sh`,
`check-repo-settings.sh`, `configure-repo-settings.sh`, `create-pr.sh`,
`monitor-pr-checks.sh`.

Resolution order, first hit wins:

1. `$PYTHON` — an explicit override (rejected, loudly, if it is too old)
2. `python3.13` / `python3.12` / `python3.11` on `PATH`
3. `python3`, when it is already 3.11+
4. `uv run --no-project --python 3.12 python`, when `uv` is installed

Otherwise it fails with a message naming the requirement and all three
remedies.

`TENFERRO_PYTHON` is an array because the uv fallback is a multi-word command;
`py` forwards every argument, so the existing `py - <<'PY'` heredocs and the
`policy_args=(py …)` array in `check-pr-fast.sh` keep working unchanged.

## Why uv is a fallback rather than the runner

Making `uv run` the only entry point would be more reproducible but would turn
uv into a hard prerequisite for every contributor and for CI. As a fallback it
costs nothing: CI keeps using its default interpreter, contributors with a
modern `python3` are unaffected, and a macOS user with uv installed gets a
working gate with no manual step (uv fetches and caches a managed 3.12).

`--no-project` matters — the repository has no `pyproject.toml`, and without it
uv would try to resolve one. `python-dotenv`, the only third-party import in
these scripts, is imported lazily and only when a `.env` exists, so running
under a managed interpreter without the dev extras still behaves correctly and
still prints its own install hint.

## Verification

- `bash scripts/test-python-resolver.sh` — six cases covering the override,
  the versioned-interpreter path, the uv fallback (asserting it yields 3.11+),
  and the failure message. Each builds a `PATH` exposing only the interpreters
  under test rather than trusting what this machine has installed.
- `bash scripts/check-pr-fast.sh --coverage-reviewed --test 'bash
  scripts/test-python-resolver.sh'` passes.
- The same gate run with `PATH` reduced to uv plus the 3.9 system `python3`
  reaches and passes the Python steps (`change_policy.py`, `doc-snippets-ok`),
  confirming the fallback carries the real workload and not just a version
  probe.

## Residual

The resolver pins the uv fallback to 3.12. If the scripts ever need a newer
feature, that literal and the `3.11` floor in `tenferro_python_is_new_enough`
are the two places to update; `TENFERRO_MIN_PYTHON` documents the floor for
readers but the check itself uses a tuple comparison.
