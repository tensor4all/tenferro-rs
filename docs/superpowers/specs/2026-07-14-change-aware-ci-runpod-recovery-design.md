# Change-aware CI and trusted RunPod recovery

**Date:** 2026-07-14
**Status:** Approved design, pending implementation plan
**Issue:** [#1379](https://github.com/tensor4all/tenferro-rs/issues/1379)

## Problem

The current pull-request workflows apply most Rust-heavy validation to every
change. A documentation-only change still runs workspace tests, coverage,
provider tests, extension samples, and the GPU gate. A CI-only follow-up also
reruns those lanes even when no Rust source, manifest, or generated artifact
has changed.

The NUMA implementation in PR #1376 exposed related feedback and recovery
problems:

- The CPU-BLAS-only workspace command caught a provider-scope test assumption
  only after push because local preflight and GitHub Actions do not share one
  command definition.
- A one-line RunPod workflow correction restarted the complete non-GPU matrix.
- The RunPod pod-creation loop retried a deterministic HTTP 400 schema error
  five times, then treated temporary HTTP 500 capacity failures the same way.
- Automatic and manual GPU runs rebuilt equivalent CUDA archives because the
  cache key included ref identity in addition to content.
- Recovering the required GPU check needed raw ref and SHA inputs and a manual
  wrapper rerun. That path is correctable, but too easy to invoke incorrectly.

The workflows must reduce latency without weakening branch protection,
comprehensive main-branch validation, or the trusted RunPod boundary.

## Goals

- Give local contributors exact entry points for the command profiles used by
  GitHub Actions.
- Skip irrelevant expensive work for docs-only and CI-only pull requests while
  preserving successful required check names.
- Validate RunPod request configuration before building a CUDA archive or
  requesting paid hardware.
- Distinguish permanent request failures from retryable service and capacity
  failures.
- Reuse equivalent CUDA archives across automatic and manual recovery runs.
- Provide a PR-number-oriented recovery path that always executes the workflow
  and secret-bearing helpers from trusted `main`.

## Non-goals

- Do not change tenferro's public Rust API, crate graph, backends, feature
  flags, numerical behavior, or AD semantics.
- Do not weaken required checks for changes that can affect compiled code.
- Do not skip comprehensive validation on pushes to `main`.
- Do not use RunPod Community Cloud or source a secret-bearing workflow/helper
  from a pull-request branch.
- Do not turn an infrastructure failure into a successful GPU result without
  an actual successful GPU validation.
- Do not introduce a third-party CI service or a repository dependency for
  workflow classification.

## Chosen approach

Use repository-owned, standard-library-only Python helpers as the testable
policy layer and keep GitHub workflow YAML as thin orchestration adapters.

Two alternatives were rejected:

- Keeping all logic inline in YAML is initially smaller, but it would preserve
  the command drift and make classification, schema parsing, and retry policy
  difficult to test locally.
- Moving the policy into an external reusable action or service would create a
  new trust and versioning boundary for a repository-local problem.

## Change classification

### Classifier contract

Add a single classifier under `scripts/ci/` that consumes changed paths and
returns:

- a primary class: `code`, `docs-only`, or `ci-only`;
- independent booleans for documentation, CI configuration, CPU/Rust-heavy,
  extension, and GPU validation needs;
- a human-readable explanation listing the paths that selected each flag.

Classification precedence is conservative:

1. Any code, manifest, build input, generated artifact source, or unknown path
   selects `code`.
2. If every path belongs to the explicit documentation or CI allowlists and at
   least one CI path changed, select `ci-only`.
3. If every path belongs to the documentation allowlist, select `docs-only`.

A mixed docs-and-CI change is therefore `ci-only`, with both the docs and CI
validation flags enabled. An empty or unrecognized diff is `code`; the
classifier never guesses that an unknown path is cheap.

Documentation paths include the repository documentation trees and selected
top-level prose files. CI paths are an explicit list covering workflow files,
the shared CI helper directory, and the existing PR-check entry points. The
implementation must not classify all of `scripts/` as CI-only because many
scripts validate source or generated project content.

### Event policy

Pull requests consume the classifier result. Pushes to `main` override it and
run the comprehensive non-GPU matrix. This preserves the current post-merge
backend coverage even when a pull request legitimately took a cheap path. Live
GPU validation remains a pull-request gate for changes that can affect GPU
behavior or its control plane; this design does not add a second paid GPU run
after merge.

The classifier also emits more specific safety decisions:

- A docs-only pull request runs docs/site/snippet/link/consistency validation.
- A CI-only pull request runs actionlint and CI helper/contract tests.
- A change to the RunPod control-plane workflow or its request helpers still
  requires a live GPU gate; unrelated CI-only changes do not.
- A mixed docs-and-CI pull request runs both relevant lightweight suites.

### Required check behavior

Required job names remain stable. Jobs that have no relevant expensive work
still start, report the classifier decision, and finish successfully. They are
not omitted in a way that leaves branch protection waiting for a missing or
skipped check.

The workspace aggregate gate accepts a successful explicit no-op only when the
classifier proves the pull request contains no code-affecting path. Code
changes retain the existing faer lane, conditional BLAS lane, extension
samples, and their aggregate requirements.

## Shared command profiles

Add a profile runner under `scripts/ci/` and make both local preflight and
GitHub Actions call it. The initial profiles are:

- `workspace-faer`
- `workspace-blas`
- `blas-inject`
- `extensions`
- `docs`
- `coverage`
- `ci-config`
- `full`

Profiles own exact Cargo feature flags, release mode, nextest arguments, and
the repository script sequence. Runner provisioning such as apt installation,
Rust component installation, and GitHub cache setup remains in YAML.

The runner supports a dry-run/list mode so unit tests can verify command
routing without compiling the workspace. `scripts/check-pr-fast.sh` may select
profiles, but it must delegate rather than duplicate the commands.

`full` composes the named profiles rather than maintaining a second command
list. Profile composition must detect duplicate execution where one profile
already subsumes another.

## RunPod request validation

### Trusted configuration

Store the cost-bounded GPU allowlist and request policy in one repository-owned
configuration file. The automatic and manual paths read the same file.

Before CUDA archive construction, a trusted GitHub-hosted job:

1. checks out the helper at the same trusted revision as the workflow;
2. fetches the authenticated RunPod OpenAPI document from
   `https://rest.runpod.io/v1/openapi.json`;
3. resolves the request schema used by `POST /pods`, including local `$ref`
   references;
4. compares the configured `gpuTypeIds` with the live enum;
5. fails with the invalid IDs before archive construction or pod creation.

The validator uses fixture-based tests for direct schemas, referenced schemas,
missing paths, missing enums, and invalid configured IDs. A missing or
structurally unexpected live schema is a hard failure, not permission to skip
validation.

### Retry policy

Move request construction and retry classification into a testable helper.
The policy is:

- 2xx: success;
- HTTP 408, 429, and 5xx: retryable;
- other 4xx: permanent and reported immediately;
- transient transport failures: retryable;
- malformed success responses or missing pod IDs: permanent protocol errors.

Retryable failures use bounded exponential backoff with jitter and a total
deadline. Logs state the attempt, status class, next delay, and redacted
provider error. Deterministic failures do not sleep or consume all attempts.

The request remains `cloudType: SECURE`, non-interruptible, and limited to the
reviewed allowlist. Cleanup remains unconditional and idempotent whenever a pod
ID was created.

## CUDA archive reuse

The archive cache key contains:

- an explicit cache-format version;
- hosted runner OS;
- CUDA/PTX/runtime configuration;
- relevant manifests, lockfile, source, and tests content hashes.

It does not contain the PR head SHA, merge SHA, branch, or dispatch identity.
Equivalent source trees therefore share a key. The trusted automatic and
manual workflows both run from `main`, keeping cache scope compatible while
checking out the requested tenferro ref only for archive/test content.

The workflow continues to upload a per-run artifact because the self-hosted
RunPod runner cannot directly restore the hosted-runner Actions cache. A cache
hit skips archive compilation but still uploads the archive for that run.

## Trusted manual recovery

Extend the existing `workflow_dispatch` path with a PR-number input. The
trusted `main` workflow resolves the pull request through GitHub API and checks:

- the actor has the required repository permission;
- the pull request belongs to this repository rather than a fork;
- the pull request is open;
- the resolved head SHA has not changed during authorization;
- the published required check targets that exact head SHA.

Raw ref/SHA inputs may remain for non-PR maintainer diagnostics, but PR-number
mode is the documented recovery path and rejects conflicting inputs.

Provide a small maintainer command that dispatches the workflow explicitly at
`--ref main`, passes the PR number, prints the run URL, waits for completion
when requested, and never reads or forwards repository secrets locally.

After a successful external GPU check, the command documents how the
pull-request wrapper gate observes the newest result. If GitHub requires a
wrapper rerun, the helper performs or clearly reports that repository-scoped
action; it never manufactures a success check.

## Workflow data flow

For pull requests:

1. A cheap classification job computes policy flags from the base/head diff.
2. Required jobs start and either run their shared profiles or report an
   intentional classifier-backed no-op.
3. The non-GPU aggregate gate verifies executed or explicitly unnecessary
   lanes.
4. The GPU wrapper either records that GPU validation is unnecessary for this
   diff or waits for the trusted RunPod result.
5. For a required live GPU run, the trusted workflow validates the RunPod
   schema before creating the CUDA archive, restores/builds the
   content-addressed archive, creates a SECURE pod with status-aware retries,
   runs GPU tests, cleans up, and publishes the result to the PR head.

For pushes to `main`, every non-GPU validation flag is forced on.

## Error handling and observability

- Classifier output includes the class, flags, and matching reason.
- Every no-op job says which class made its expensive steps unnecessary.
- Profile failures include the profile and exact command.
- OpenAPI failures distinguish fetch, parse, schema resolution, and invalid ID
  errors.
- RunPod failures distinguish permanent request errors, rate limiting,
  provider/service failures, transport failures, malformed responses, runner
  registration timeout, test failure, and cleanup failure.
- The final GPU check summary identifies the failing phase without treating
  skipped downstream work as an independent root cause.

## Testing strategy

### Unit and contract tests

Use Python standard-library tests for:

- documentation, CI, mixed, code, empty, and unknown-path classification;
- per-lane flags, including GPU-control-plane changes;
- profile listing, composition, and dry-run command parity;
- OpenAPI enum extraction and invalid-ID reporting;
- HTTP/transport retry classification and bounded delay calculation;
- PR-number recovery validation and conflicting input rejection;
- required workflow names and their classifier-backed no-op contracts.

Use checked-in redacted fixtures derived from public API shapes, not responses
containing tokens, account IDs, pod IDs, or runner configuration.

### Local validation

At minimum:

```bash
python3 -m unittest discover -s scripts/ci/tests
actionlint
python3 scripts/ci/run_profile.py ci-config
python3 scripts/ci/run_profile.py workspace-faer
python3 scripts/ci/run_profile.py workspace-blas
```

Run profile dry runs for every named profile and source-contract tests against
all modified workflow files. Repository-wide formatting, clippy, tests,
coverage, docs, and rule review remain required before PR creation because the
implementation changes the policy that decides when future PRs may skip them.

### End-to-end validation

Exercise classifier fixtures representing:

- docs-only;
- CI-only without GPU-control-plane changes;
- CI-only with RunPod control-plane changes;
- mixed docs and CI;
- ordinary Rust code;
- an unknown new top-level path;
- push-to-main override.

Perform one trusted manual dry-run of PR resolution without starting a pod. A
live RunPod run is required before merge because the request and recovery
control plane changes. It must use SECURE Cloud and complete cleanup.

## Documentation and records

- Document the local profile commands and which PR classes run each lane.
- Document the PR-number RunPod recovery command and its trust boundary.
- Add a curated work log covering issue #1379, PR #1376 evidence, decisions,
  rejected alternatives, validation, and residual risks.
- Keep detailed implementation mechanics in the helpers and workflow comments;
  do not duplicate them across contributor documentation.

## Risks and mitigations

- **Misclassification could skip necessary tests.** Explicit allowlists,
  unknown-path fallback, main override, and table-driven tests mitigate this.
- **A required job could disappear instead of succeeding as a no-op.** Keep
  stable job names and test aggregate-gate behavior for every class.
- **Live OpenAPI changes could block GPU CI.** This is intentional when the
  configured request is invalid; the error must be early and actionable.
- **Broader retries could increase queue time or cost.** Retries end before pod
  creation when capacity is absent, use a bounded deadline, and retain the
  reviewed allowlist and SECURE Cloud.
- **Cache reuse could restore an incompatible archive.** Include every CUDA,
  runner, archive-format, manifest, and source input in the content key and
  bump the format version when packaging changes.
- **Manual dispatch could test the wrong commit.** Resolve PR number in the
  trusted workflow, recheck head stability, and publish only to the resolved
  SHA.

## Success criteria

- Docs-only pull requests run documentation validation without Rust-heavy,
  coverage, provider, extension, or live GPU work.
- CI-only pull requests run actionlint and CI helper tests; RunPod control-plane
  changes additionally run the live GPU gate.
- Code pull requests retain required GPU validation, and all pushes to `main`
  retain the comprehensive non-GPU matrix.
- Local and hosted CI use the same named profiles and command definitions.
- Invalid RunPod GPU IDs fail before CUDA archive construction.
- Permanent 4xx errors are not retried; retryable capacity/service failures use
  bounded jittered backoff.
- Equivalent automatic/manual source trees reuse the CUDA archive cache.
- Maintainers can recover a PR GPU gate safely from a PR number through the
  trusted `main` workflow.
- Required check names and branch-protection behavior remain stable.
