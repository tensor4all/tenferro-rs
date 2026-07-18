# Trusted CI Cache Publication And Immutable GPU Artifact Reuse

## Session summary

Implemented issue #1403: made every RunPod-workflow job a cache reader only,
moved shared cache publication to a new trusted default-branch workflow, and
let GPU allocation retries reuse the per-key archive artifact from a prior
trusted run instead of rebuilding it.

## Context read

- Issue #1403 and umbrella #1401 (acceptance criteria and constraints)
- `.github/workflows/runpod-gpu-test.yml` (authorize gating, cuda-archive,
  run-gpu-tests cache/artifact flow, security header)
- `.github/workflows/ci.yml`, `ci-pr-workspace-tests.yml` (existing CPU-lane
  cache posture and push-to-main coverage)
- `scripts/ci/runpod_client.py` transport-injection style,
  `scripts/ci/recover_runpod_pr.py` retry dispatch path
- `scripts/ci/tests/test_workflow_contracts.py` existing invariants

## Chosen design

- New `ci-cache-publish.yml` (push to main / weekly schedule / dispatch)
  is the only writer of the shared CUDA/PJRT archive cache, the rust-cache
  prefix family, the cuTENSOR cache, and the CUDA runtime trees.
- `cuda-archive` and `run-gpu-tests` use `actions/cache/restore` and
  rust-cache `save-if: false`; `cuda-archive` gains `actions: read` for
  artifact lookup only.
- Archive artifact names encode the content key;
  `scripts/ci/find_archive_artifact.py` locates a reusable artifact and
  verifies the producing run is trusted (same repo, trusted workflow path,
  default-branch-defined event) before download; failures fall back to a
  fresh build.
- Archive cache key drops `runpod_config.json` and the workflow YAML hash,
  adds the rustc version, keeps a manual `vN` for build-command changes.
- Pod-shared caches move to the fixed root `/opt/tenferro-ci` so trusted
  hosted-runner saves restore to identical absolute paths on the pod; shared
  install logic extracted to `scripts/ci/install_cuda_toolkit_hosted.sh`,
  `install_cuda_runtime_tree.sh`, and `install_cutensor.sh`.
- Trust and data-flow model recorded in `docs/design/ci-cache-trust.md`.

## Rejected alternatives

- Giving `cuda-archive` cache-write permission on main-push runs only:
  job permissions are static per job, so a PR-triggered run would hold the
  same write scope while compiling PR-controlled build scripts.
- Publishing caches from the pod (status quo): exposes a main-scope
  cache-write credential to a root self-hosted runner executing test code.
- Hashing the whole workflow file into the key (status quo): every unrelated
  workflow edit invalidated the archive; replaced by material inputs plus a
  manual `vN` for build-command changes.
- In-run artifact download only (status quo): each retry dispatch rebuilt an
  identical archive for ~13 minutes before any GPU work.

## Verification

- `python3 -m unittest` over `scripts/ci/tests` (new finder tests plus new
  workflow contract tests; 107 tests, pre-existing local Python 3.9
  `LocalGateTests` environment failures reproduce identically on main)
- `actionlint` on both workflows
- Live end-to-end cache publication and retry reuse require hosted CI runs
  and are acceptance items on the PR

## Residual risks

- Manual `vN` bump discipline for build-command changes
- First post-merge main push must populate the new caches before consumers
  see hits; until then behavior equals today's build-per-run
- Paid-run validation of pod-side fixed-root cache restores is outstanding
