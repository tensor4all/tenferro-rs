# uv-pinned Python interpreter for repository scripts

Issue: #1606

## Problem

The helper scripts require Python 3.11+ — `enum.StrEnum` in
`scripts/ci/change_policy.py`, `tomllib` in `check-api-consistency.py`,
`check-guide-dependency-snippets.py` and `check-docs-site.py` — but nothing
declared or enforced it and every shell entry point invoked a bare `python3`.

On macOS the system `python3` is 3.9, so `scripts/check-pr-fast.sh`, the local
gate contributors are told to run, died inside `change_policy.py` with
`AttributeError: module 'enum' has no attribute 'StrEnum'` before reaching a
single check. CI never saw it: no workflow uses `actions/setup-python`, so the
scripts run on the runner default (3.12 on `ubuntu-latest`).

Two of the affected scripts already raise `RuntimeError("Python 3.11+ is
required for tomllib support")` — the requirement was known, but only reported
after the wrong interpreter had been chosen.

## Change

`uv` is now the source of the interpreter, pinned by `.python-version` (3.12),
so a contributor and CI execute the same version rather than whatever each
machine ships. `scripts/lib/python.sh` installs a `python3`/`python` shim that
forwards to `uv run --no-project --python <pin> python` and prepends it to
`PATH`; the seven shell entry points that invoke Python source it and are
otherwise unchanged.

`PYTHON` remains an escape hatch for someone who cannot install uv, accepted
only if it is 3.11+.

## Why a PATH shim rather than rewriting call sites

The first attempt renamed every `python3` invocation to a `py` wrapper
function. CI rejected it, and the failure was the informative kind:

- `scripts/ci/run_profile.py` runs its profile commands with `shell=True`, and
  those command strings spell `python3` themselves. Rewriting the shell call
  sites fixed the parent process and left every child on the 3.9 interpreter —
  `scripts/ci/run_profile.py docs` still died in `tomllib`. Locally this was
  invisible because `check-pr-fast.sh` does not run the `docs` profile.
- `scripts/ci/tests/test_run_profile.py`, `test_workflow_contracts.py` and
  `scripts/test-doc-consistency.py` deliberately pin those command strings so
  local and hosted CI stay in step. The rename broke `docs-site` and
  `CI configuration checks` on exactly those assertions.

Fixing the NAME instead of the call sites resolves both: the interpreter is
inherited by children through `PATH`, and every pinned command string stays
literally correct.

`--no-project` matters — the repository has no `pyproject.toml`, and without it
uv would try to resolve one. `python-dotenv`, the only third-party import in
these scripts, is imported lazily and only when a `.env` exists, so a managed
interpreter without the dev extras still behaves correctly and still prints its
own install hint.

## Verification

- `bash scripts/test-python-resolver.sh` — five cases: the pinned interpreter,
  **propagation into a child process** (the property the shim exists for), the
  `PYTHON` override, a too-old override rejected, and the missing-uv message.
  Each builds a `PATH` exposing only the tools under test.
- `bash scripts/check-pr-fast.sh --coverage-reviewed --test 'bash
  scripts/test-python-resolver.sh'` passes.
- `python3 scripts/ci/run_profile.py docs` — the profile that caught the first
  attempt — passes under the shim.

## Follow-up

CI still runs `python3 …` directly in workflow steps rather than through these
shell entry points, so hosted runs use the runner's interpreter, not the pinned
one. `scripts/ci/setup-python-shim.sh` exists for that: it appends the shim
directory to `$GITHUB_PATH` so every later step in a job uses the pinned
interpreter. Wiring it (plus `astral-sh/setup-uv`) into the ~23 checkout points
across six workflow files is a separate change: workflow edits cannot be
validated locally, and it should land where a red CI run is expected and
reviewable rather than riding along with this one.

## CI adoption (follow-up PR)

`.github/actions/setup-uv-python` installs uv (pinned to 0.11.18) and runs
`scripts/ci/setup-python-shim.sh`, which appends the shim directory to
`$GITHUB_PATH` so every later step in the job uses the interpreter pinned by
`.python-version`. One step is added after the checkout of each job that runs
the repository's Python tooling, carrying that checkout's own `if:` condition so
a skipped job stays skipped:

- `ci.yml` — policy, fmt, clippy, test-inject-blas, coverage, docs-site,
  ci-config
- `ci-pr-workspace-tests.yml` — changes, ci-fork, ci-maintainer, ext-samples
- `review_bot.yml` — review-bot, review-bot-no-llm, review-bot-waived
- `runpod-gpu-test.yml` — authorize, runpod-contract

**Not** added to the GPU pod-side steps in `CI_gpu.yml` and the rest of
`runpod-gpu-test.yml`. Those `python3` calls are not this repository's tooling —
they parse CUDA manifests, run the pod smoke test, and `pip download` inside a
container — and pinning them to a uv-managed interpreter would change unrelated
behaviour on the pod for no benefit.

Security note: the two `pull_request_target` / `workflow_run` workflows check
out the TRUSTED base or default-branch revision (both say so in a comment and
treat PR contents as data only), so `uses: ./.github/actions/setup-uv-python`
resolves to trusted content rather than PR-controlled code. That is the reason
the composite action is safe to reference by path in those jobs.

Verified with `actionlint` (clean), plus `run_profile.py ci-config` — which runs
`test_workflow_contracts.py`, the tests that pin workflow text — and
`run_profile.py docs`. Workflow behaviour itself can only be confirmed by the
hosted run on this PR.
