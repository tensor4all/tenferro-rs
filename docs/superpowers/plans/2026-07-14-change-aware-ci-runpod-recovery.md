# Change-aware CI and RunPod Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement issue #1379 so docs-only and CI-only pull requests avoid irrelevant expensive work, local and hosted CI share exact command profiles, and trusted RunPod validation/recovery handles schema drift, retries, caching, and PR-number dispatch safely.

**Architecture:** Put policy in standard-library-only Python modules under `scripts/ci/` and keep workflow YAML as orchestration. A conservative classifier emits lane flags, a profile runner owns exact commands, and trusted RunPod helpers own live-schema validation plus retry behavior. Required job names remain stable and report explicit successful no-ops; pushes to `main` force the comprehensive non-GPU matrix.

**Tech Stack:** Python 3 standard library and `unittest`, Bash entry points, GitHub Actions YAML, `gh`, RunPod REST/OpenAPI, Rust/Cargo/nextest/llvm-cov, actionlint.

---

## File map

Create:

- `scripts/ci/__init__.py`: marks the CI policy package.
- `scripts/ci/change_policy.py`: classifies changed paths and emits GitHub outputs.
- `scripts/ci/run_profile.py`: owns exact local/hosted CI command profiles.
- `scripts/ci/runpod_config.json`: single RunPod GPU allowlist and retry configuration.
- `scripts/ci/runpod_contract.py`: resolves the live OpenAPI pod schema and validates configured GPU IDs.
- `scripts/ci/runpod_client.py`: builds the redacted pod request and performs status-aware creation retries.
- `scripts/ci/recover_runpod_pr.py`: PR-number-oriented trusted manual-dispatch CLI.
- `scripts/ci/tests/__init__.py`: test package marker.
- `scripts/ci/tests/test_change_policy.py`: table-driven classification and lane tests.
- `scripts/ci/tests/test_run_profile.py`: profile composition and dry-run tests.
- `scripts/ci/tests/test_runpod_contract.py`: OpenAPI fixture and invalid-ID tests.
- `scripts/ci/tests/test_runpod_client.py`: retry/status/payload tests.
- `scripts/ci/tests/test_recover_runpod_pr.py`: trusted dispatch command tests.
- `scripts/ci/tests/test_workflow_contracts.py`: required-name, no-op, trust, and cache-key source contracts.
- `docs/worklogs/2026-07-14-issue-1379-change-aware-ci.md`: curated implementation record.

Modify:

- `scripts/check-pr-fast.sh`: delegate optional exact CI profiles.
- `.github/workflows/ci.yml`: consume shared policy/profile helpers and add CI configuration checks.
- `.github/workflows/ci-pr-workspace-tests.yml`: replace path regexes and keep aggregate checks successful for explicit no-ops.
- `.github/workflows/runpod-gpu-test.yml`: trusted classification, OpenAPI preflight, helper-based pod creation, content cache key, and PR-number recovery.
- `.github/workflows/CI_gpu.yml`: document and preserve wrapper behavior for classifier-backed external success.
- `CONTRIBUTING.md`: document local profiles and docs/CI-only policy at contributor level.

## Task 1: Conservative change classifier

**Files:**

- Create: `scripts/ci/__init__.py`
- Create: `scripts/ci/change_policy.py`
- Create: `scripts/ci/tests/__init__.py`
- Create: `scripts/ci/tests/test_change_policy.py`

- [ ] **Step 1: Write failing table-driven classifier tests**

Define expected behavior explicitly:

```python
from scripts.ci.change_policy import ChangeClass, classify_paths


def test_docs_only_runs_docs_without_rust_or_gpu() -> None:
    policy = classify_paths(["docs/guides/cpu.md", "README.md"])
    assert policy.change_class is ChangeClass.DOCS_ONLY
    assert policy.run_docs
    assert not policy.run_rust
    assert not policy.run_extensions
    assert not policy.run_gpu


def test_ci_and_docs_runs_both_lightweight_suites() -> None:
    policy = classify_paths([
        ".github/workflows/ci.yml",
        "docs/worklogs/ci.md",
    ])
    assert policy.change_class is ChangeClass.CI_ONLY
    assert policy.run_ci_config
    assert policy.run_docs
    assert not policy.run_rust


def test_runpod_control_plane_requires_gpu() -> None:
    policy = classify_paths(["scripts/ci/runpod_client.py"])
    assert policy.change_class is ChangeClass.CI_ONLY
    assert policy.run_ci_config
    assert policy.run_gpu


def test_unknown_and_empty_diffs_fall_back_to_code() -> None:
    for paths in ([], ["new-top-level-policy.toml"]):
        policy = classify_paths(paths)
        assert policy.change_class is ChangeClass.CODE
        assert policy.run_rust
        assert policy.run_extensions
        assert policy.run_gpu


def test_push_to_main_forces_comprehensive_non_gpu_lanes() -> None:
    policy = classify_paths(["README.md"], event="push")
    assert policy.change_class is ChangeClass.CODE
    assert policy.run_rust
    assert policy.run_extensions
    assert policy.run_docs
    assert policy.run_ci_config
```

- [ ] **Step 2: Run the tests and confirm the red state**

Run:

```bash
python3 -m unittest scripts.ci.tests.test_change_policy -v
```

Expected: import failure because `scripts.ci.change_policy` does not exist.

- [ ] **Step 3: Implement the classifier and GitHub-output CLI**

Use immutable output with explicit path allowlists:

```python
class ChangeClass(enum.StrEnum):
    CODE = "code"
    DOCS_ONLY = "docs-only"
    CI_ONLY = "ci-only"


@dataclasses.dataclass(frozen=True)
class ChangePolicy:
    change_class: ChangeClass
    run_rust: bool
    run_blas: bool
    run_extensions: bool
    run_docs: bool
    run_ci_config: bool
    run_gpu: bool
    reasons: tuple[str, ...]


def classify_paths(paths: Sequence[str], event: str = "pull_request") -> ChangePolicy:
    normalized = tuple(sorted({path.strip().removeprefix("./") for path in paths if path.strip()}))
    if event == "push" or not normalized:
        return full_policy("push-to-main override" if event == "push" else "empty diff fallback")

    docs = tuple(path for path in normalized if is_docs_path(path))
    ci = tuple(path for path in normalized if is_ci_path(path))
    unknown = tuple(path for path in normalized if path not in docs and path not in ci)
    if unknown:
        return full_policy("code or unknown paths: " + ", ".join(unknown))

    has_ci = bool(ci)
    return ChangePolicy(
        change_class=ChangeClass.CI_ONLY if has_ci else ChangeClass.DOCS_ONLY,
        run_rust=False,
        run_blas=False,
        run_extensions=False,
        run_docs=bool(docs),
        run_ci_config=has_ci,
        run_gpu=any(is_gpu_control_plane_path(path) for path in ci),
        reasons=("docs: " + ", ".join(docs), "ci: " + ", ".join(ci)),
    )
```

The CLI must accept `--event`, either repeated `--path` or `--base/--head`, print JSON to stdout, and append stable lowercase booleans plus `classification` and `reason` to `$GITHUB_OUTPUT` when present. `git diff` failures return nonzero rather than falling back cheap.

- [ ] **Step 4: Run classifier tests and CLI smoke checks**

Run:

```bash
python3 -m unittest scripts.ci.tests.test_change_policy -v
python3 scripts/ci/change_policy.py --path README.md --path docs/index.md
python3 scripts/ci/change_policy.py --path crates/tenferro-cpu/src/lib.rs
```

Expected: tests pass; first JSON is `docs-only`, second is `code`.

- [ ] **Step 5: Commit the classifier**

```bash
git add scripts/ci
git commit -m "ci: add conservative change classification"
```

## Task 2: Shared local and hosted command profiles

**Files:**

- Create: `scripts/ci/run_profile.py`
- Create: `scripts/ci/tests/test_run_profile.py`
- Modify: `scripts/check-pr-fast.sh`

- [ ] **Step 1: Write failing profile routing tests**

```python
from scripts.ci.run_profile import commands_for, expand_profiles


def test_workspace_blas_matches_ci_feature_contract() -> None:
    commands = commands_for("workspace-blas")
    assert commands == (
        "cargo nextest run --workspace --release --no-default-features --features cpu-blas --no-fail-fast",
        "cargo test --doc --workspace --release --no-default-features --features cpu-blas",
    )


def test_full_profile_expands_named_profiles_once() -> None:
    expanded = expand_profiles(["full"])
    assert expanded == (
        "workspace-faer",
        "workspace-blas",
        "blas-inject",
        "extensions",
        "docs",
        "coverage",
        "ci-config",
    )
    assert len(expanded) == len(set(expanded))


def test_duplicate_profile_composition_is_deduplicated() -> None:
    assert expand_profiles(["workspace-faer", "full"])[0] == "workspace-faer"
    assert expand_profiles(["workspace-faer", "full"]).count("workspace-faer") == 1
```

- [ ] **Step 2: Confirm tests fail before implementation**

Run:

```bash
python3 -m unittest scripts.ci.tests.test_run_profile -v
```

Expected: import failure for `scripts.ci.run_profile`.

- [ ] **Step 3: Implement exact profiles, composition, and dry run**

Represent commands as immutable static tuples. Execute with `subprocess.run(command, shell=True, check=True, env=...)`; commands are repository constants, never user-provided shell text. Set `RUSTFLAGS=-l dylib=openblas -l dylib=lapack` only for `workspace-blas`.

The profile map must contain the exact existing commands, including:

```python
PROFILE_COMMANDS = {
    "workspace-faer": (
        "cargo nextest run --workspace --release --no-fail-fast",
        "cargo test --doc --workspace --release",
    ),
    "workspace-blas": (
        "cargo nextest run --workspace --release --no-default-features --features cpu-blas --no-fail-fast",
        "cargo test --doc --workspace --release --no-default-features --features cpu-blas",
    ),
    "blas-inject": (
        'cargo test -p tenferro-cpu --test inject_tests --release --no-default-features --features "cpu-blas,provider-inject"',
    ),
    "extensions": (
        "cargo test --manifest-path ext/tropical/Cargo.toml --release --features autodiff",
        "cargo test --manifest-path ext/sparse/Cargo.toml --release --features autodiff",
        "cargo check --manifest-path samples/kdv-pinn/Cargo.toml --release --all-targets",
    ),
    "docs": (
        "python3 scripts/test-check-docs-site.py",
        "python3 scripts/test-doc-consistency.py",
        "python3 scripts/test-repository-rules-review.py",
        "python3 scripts/check-guide-dependency-snippets.py",
        "python3 scripts/check-operation-categories.py --fail-on-findings",
        "bash scripts/build_docs_site.sh",
    ),
    "coverage": (
        "cargo llvm-cov --workspace --release --json --output-path coverage.json",
        "python3 scripts/check-coverage.py coverage.json",
    ),
    "ci-config": (
        "python3 -m unittest discover -s scripts/ci/tests -v",
        "actionlint",
    ),
}
```

Support `--list`, `--dry-run`, and one or more profile names. Print `+ <command>` before each execution and include the profile in failures.

- [ ] **Step 4: Extend fast preflight without duplicating commands**

Add repeatable `--ci-profile NAME` and `--ci-profile-dry-run` options to `scripts/check-pr-fast.sh`. The execution must be exactly:

```bash
python3 scripts/ci/run_profile.py "${profile_args[@]}"
```

The existing `--test` behavior remains available, but documentation tells contributors to prefer profiles for commands owned by CI.

- [ ] **Step 5: Verify profile tests and dry-run parity**

Run:

```bash
python3 -m unittest scripts.ci.tests.test_run_profile -v
python3 scripts/ci/run_profile.py --list
python3 scripts/ci/run_profile.py --dry-run full
bash scripts/check-pr-fast.sh --no-fetch --coverage-reviewed --skip-doc-snippets --ci-profile workspace-blas --ci-profile-dry-run
```

Expected: all tests pass; dry runs print each concrete command once and execute no Cargo command.

- [ ] **Step 6: Commit the profile runner**

```bash
git add scripts/ci/run_profile.py scripts/ci/tests/test_run_profile.py scripts/check-pr-fast.sh
git commit -m "ci: share local and hosted command profiles"
```

## Task 3: Integrate classifier-backed no-ops into non-GPU workflows

**Files:**

- Create: `scripts/ci/tests/test_workflow_contracts.py`
- Modify: `.github/workflows/ci.yml`
- Modify: `.github/workflows/ci-pr-workspace-tests.yml`

- [ ] **Step 1: Write failing workflow source-contract tests**

Read workflows as text and assert stable required names and shared helpers:

```python
def test_fast_ci_uses_shared_policy_and_profiles() -> None:
    text = read(".github/workflows/ci.yml")
    assert "python3 scripts/ci/change_policy.py" in text
    assert "python3 scripts/ci/run_profile.py blas-inject" in text
    assert "python3 scripts/ci/run_profile.py coverage" in text
    assert "python3 scripts/ci/run_profile.py docs" in text
    assert "name: CI configuration checks" in text


def test_required_names_remain_stable() -> None:
    fast = read(".github/workflows/ci.yml")
    heavy = read(".github/workflows/ci-pr-workspace-tests.yml")
    for name in ("rustfmt", "clippy", "coverage", "docs-site", "cargo test (blas inject)"):
        assert f"name: {name}" in fast
    assert "name: CI gate (PR workspace tests)" in heavy


def test_heavy_workflow_has_explicit_noop_matrix_and_gate_contract() -> None:
    text = read(".github/workflows/ci-pr-workspace-tests.yml")
    assert '"backend":"not-required"' in text
    assert "RUN_WORKSPACE" in text
    assert "RUN_EXTENSIONS" in text
    assert "Workspace tests not required" in text
```

- [ ] **Step 2: Run the contracts and verify failure**

Run:

```bash
python3 -m unittest scripts.ci.tests.test_workflow_contracts -v
```

Expected: failures because workflows still contain inline regexes and commands.

- [ ] **Step 3: Add a policy job and conditional expensive steps to `ci.yml`**

Add a first job that checks out full history and runs:

```yaml
- name: Classify change
  id: policy
  env:
    EVENT_NAME: ${{ github.event_name }}
    BASE_SHA: ${{ github.event.pull_request.base.sha }}
    HEAD_SHA: ${{ github.event.pull_request.head.sha }}
  run: >-
    python3 scripts/ci/change_policy.py
    --event "${EVENT_NAME}"
    --base "${BASE_SHA}"
    --head "${HEAD_SHA}"
```

Expose every policy flag as an output. Make each existing required job depend on the policy job, condition only expensive setup/execution steps, and add an unconditional summary step. The job itself must succeed when work is intentionally unnecessary.

Add `CI configuration checks`; it runs `python3 scripts/ci/run_profile.py ci-config` only when `run_ci_config == 'true'`, otherwise reports a no-op. Use a pinned actionlint installer/version in workflow setup.

- [ ] **Step 4: Replace backend regexes with policy output in the heavy workflow**

The selector uses the shared classifier. For `run_workspace=false`, emit one matrix entry:

```json
{"backend":"not-required","profile":"","rustflags":""}
```

For code changes emit faer and, when `run_blas=true`, BLAS entries with profile names. The matrix job invokes:

```yaml
python3 scripts/ci/run_profile.py "${{ matrix.cfg.profile }}"
```

only when the profile is nonempty; otherwise it prints `Workspace tests not required: <reason>`.

Make extension steps call `run_profile.py extensions` only when `run_extensions=true`. The aggregate gate receives both policy booleans and accepts successful no-op jobs only when the corresponding boolean is false.

- [ ] **Step 5: Run workflow contracts, policy tests, and profile dry runs**

Run:

```bash
python3 -m unittest discover -s scripts/ci/tests -v
python3 scripts/ci/run_profile.py --dry-run workspace-faer workspace-blas extensions coverage docs
git diff --check
```

Expected: all tests pass; no inline backend path regex remains.

- [ ] **Step 6: Run actionlint and focused real profiles**

Install the pinned actionlint release used in CI if absent, then run:

```bash
actionlint .github/workflows/ci.yml .github/workflows/ci-pr-workspace-tests.yml
python3 scripts/ci/run_profile.py blas-inject
```

Expected: actionlint passes and provider injection tests pass.

- [ ] **Step 7: Commit non-GPU workflow integration**

```bash
git add .github/workflows/ci.yml .github/workflows/ci-pr-workspace-tests.yml scripts/ci/tests/test_workflow_contracts.py
git commit -m "ci: skip irrelevant PR lanes conservatively"
```

## Task 4: Validate the live RunPod schema before archive work

**Files:**

- Create: `scripts/ci/runpod_config.json`
- Create: `scripts/ci/runpod_contract.py`
- Create: `scripts/ci/tests/test_runpod_contract.py`
- Modify: `.github/workflows/runpod-gpu-test.yml`

- [ ] **Step 1: Write failing OpenAPI contract tests**

Use in-memory fixtures with a `$ref` from the POST request body to a component schema:

```python
SCHEMA = {
    "paths": {
        "/pods": {
            "post": {
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/CreatePod"}
                        }
                    }
                }
            }
        }
    },
    "components": {
        "schemas": {
            "CreatePod": {
                "type": "object",
                "properties": {
                    "gpuTypeIds": {
                        "type": "array",
                        "items": {"enum": ["NVIDIA A40", "NVIDIA GeForce RTX 4090"]},
                    }
                },
            }
        }
    },
}


def test_extract_gpu_enum_follows_local_ref() -> None:
    assert extract_gpu_type_ids(SCHEMA) == frozenset({"NVIDIA A40", "NVIDIA GeForce RTX 4090"})


def test_validate_reports_every_invalid_configured_id() -> None:
    with assertRaisesRegex(ContractError, "Tesla T4.*Unknown GPU"):
        validate_gpu_type_ids(SCHEMA, ["Tesla T4", "Unknown GPU"])


def test_missing_post_schema_is_a_hard_error() -> None:
    with assertRaisesRegex(ContractError, "POST /pods"):
        extract_gpu_type_ids({"paths": {}})
```

- [ ] **Step 2: Confirm contract tests fail**

Run:

```bash
python3 -m unittest scripts.ci.tests.test_runpod_contract -v
```

Expected: import failure for `runpod_contract`.

- [ ] **Step 3: Implement configuration and local `$ref` resolution**

The JSON configuration contains `cloud_type`, `gpu_type_ids`, retry limits, and disk settings. Keep the current reviewed valid GPU list and no secret values.

Implement:

```python
def resolve_local_ref(document: Mapping[str, object], ref: str) -> Mapping[str, object]:
    if not ref.startswith("#/"):
        raise ContractError(f"unsupported non-local OpenAPI reference: {ref}")
    value: object = document
    for token in ref[2:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if not isinstance(value, Mapping) or token not in value:
            raise ContractError(f"unresolved OpenAPI reference: {ref}")
        value = value[token]
    if not isinstance(value, Mapping):
        raise ContractError(f"OpenAPI reference is not an object schema: {ref}")
    return value
```

Fetch with `urllib.request` and bearer authentication, parse JSON, validate all configured IDs, and print a concise success message. Never log the API key.

- [ ] **Step 4: Insert trusted change classification and contract validation before CUDA archive creation**

Extend the hosted `authorize` job with a `gpu_required` output. For PR runs, it
fetches the changed-file list through the GitHub API and passes that list to the
classifier checked out from trusted `main`; for push-to-main and explicit
revision validation it sets `gpu_required=true`. Only RunPod control-plane CI
paths, code paths, or unknown paths require GPU. Docs-only and unrelated
CI-only changes publish an explicit successful no-op.

Add a `runpod-contract` job after authorization, guarded by
`gpu_required == 'true'`. It checks out the same trusted workflow revision,
loads `runpod_config.json`, fetches `openapi.json`, and runs the validator. Make
`cuda-archive`, pod creation, and GPU tests depend on the same trusted output so
an invalid request cannot compile/upload an archive and an untrusted PR cannot
choose its own classification.

Add source-contract assertions that `RUNPOD_API_KEY` appears only in trusted
hosted jobs and never in `run-gpu-tests`, and that the final required gate
reports success for a skip only when the trusted `gpu_required` output is
`false`.

- [ ] **Step 5: Run fixtures, config validation against a saved live schema, and actionlint**

Run:

```bash
python3 -m unittest scripts.ci.tests.test_runpod_contract scripts.ci.tests.test_workflow_contracts -v
RUNPOD_OPENAPI_FILE=/tmp/runpod-openapi.json python3 scripts/ci/runpod_contract.py --schema-file "$RUNPOD_OPENAPI_FILE"
actionlint .github/workflows/runpod-gpu-test.yml
```

Expected: fixtures pass; the live-schema fixture accepts every configured ID; actionlint passes. The authenticated live fetch remains CI-only if no local key is available.

- [ ] **Step 6: Commit schema preflight**

```bash
git add scripts/ci/runpod_config.json scripts/ci/runpod_contract.py scripts/ci/tests/test_runpod_contract.py .github/workflows/runpod-gpu-test.yml scripts/ci/tests/test_workflow_contracts.py
git commit -m "ci: validate RunPod requests before GPU setup"
```

## Task 5: Status-aware RunPod pod creation

**Files:**

- Create: `scripts/ci/runpod_client.py`
- Create: `scripts/ci/tests/test_runpod_client.py`
- Modify: `.github/workflows/runpod-gpu-test.yml`

- [ ] **Step 1: Write failing status, delay, payload, and protocol tests**

```python
def test_retry_classification() -> None:
    for status in (408, 429, 500, 502, 503):
        assert classify_http_status(status) is RetryClass.RETRYABLE
    for status in (400, 401, 403, 404, 422):
        assert classify_http_status(status) is RetryClass.PERMANENT


def test_backoff_is_bounded_and_jittered() -> None:
    delays = [backoff_seconds(i, base=5, cap=60, jitter=lambda: 0.5) for i in range(1, 8)]
    assert delays == [2.5, 5.0, 10.0, 20.0, 30.0, 30.0, 30.0]


def test_payload_preserves_secure_trust_boundary(config: Mapping[str, object]) -> None:
    payload = build_pod_payload(config, "image", "startup", "jit")
    assert payload["cloudType"] == "SECURE"
    assert payload["interruptible"] is False
    assert payload["gpuTypeIds"] == config["gpu_type_ids"]


def test_success_without_pod_id_is_protocol_error() -> None:
    with assertRaisesRegex(PermanentRunPodError, "missing pod id"):
        parse_create_response(201, b"{}")
```

Use a fake transport and fake sleeper to assert permanent errors make one call, while two 500 responses followed by 201 make three calls and two bounded sleeps.

- [ ] **Step 2: Confirm the client tests fail**

Run:

```bash
python3 -m unittest scripts.ci.tests.test_runpod_client -v
```

Expected: import failure for `runpod_client`.

- [ ] **Step 3: Implement request construction and retry loop**

Use `urllib.request`, inject transport/sleep/random functions for tests, and define typed internal exceptions. The loop must stop at both `max_attempts` and `deadline_seconds`:

```python
for attempt in range(1, max_attempts + 1):
    try:
        status, headers, body = transport(payload)
    except OSError as error:
        failure = RetryableRunPodError(f"transport failure: {error}")
    else:
        retry_class = classify_http_status(status)
        if 200 <= status < 300:
            return parse_create_response(status, body)
        message = redacted_error_message(body)
        if retry_class is RetryClass.PERMANENT:
            raise PermanentRunPodError(f"RunPod HTTP {status}: {message}")
        failure = RetryableRunPodError(f"RunPod HTTP {status}: {message}")

    if attempt == max_attempts or monotonic() >= deadline:
        raise failure
    delay = retry_after_or_backoff(headers, attempt, config, random_fn)
    sleep(min(delay, max(0.0, deadline - monotonic())))
```

Redact JIT configuration and configured secret field names from all payload and response logging.

- [ ] **Step 4: Replace the inline five-attempt curl loop**

Keep creation of the startup script in the trusted workflow. Check out the trusted helper revision and call:

```bash
python3 scripts/ci/runpod_client.py create \
  --config scripts/ci/runpod_config.json \
  --startup-script /tmp/runpod-startup.sh \
  --image-name "${RUNPOD_IMAGE}" \
  --response-file /tmp/runpod-response.json
```

The helper appends `pod_id` and `gpu_type_id` to `$GITHUB_OUTPUT`. The subsequent runner-registration wait and unconditional cleanup retain their existing trust and permission boundaries.

- [ ] **Step 5: Run client tests and workflow contracts**

Run:

```bash
python3 -m unittest scripts.ci.tests.test_runpod_client scripts.ci.tests.test_workflow_contracts -v
actionlint .github/workflows/runpod-gpu-test.yml
```

Expected: tests and actionlint pass; workflow source contains no `for attempt in $(seq 1 5)` pod-creation loop.

- [ ] **Step 6: Commit status-aware creation**

```bash
git add scripts/ci/runpod_client.py scripts/ci/tests/test_runpod_client.py .github/workflows/runpod-gpu-test.yml scripts/ci/tests/test_workflow_contracts.py
git commit -m "ci: classify and retry RunPod failures"
```

## Task 6: Content-addressed cache and trusted PR-number recovery

**Files:**

- Create: `scripts/ci/recover_runpod_pr.py`
- Create: `scripts/ci/tests/test_recover_runpod_pr.py`
- Modify: `.github/workflows/runpod-gpu-test.yml`
- Modify: `.github/workflows/CI_gpu.yml`

- [ ] **Step 1: Write failing cache and recovery tests**

```python
def test_cache_key_does_not_include_ref_identity() -> None:
    text = read(".github/workflows/runpod-gpu-test.yml")
    key_line = next(line for line in text.splitlines() if 'key="cuda-archive-' in line)
    assert "TENFERRO_REF" not in key_line
    assert "hashFiles(" in key_line


def test_dispatch_always_uses_trusted_main() -> None:
    command = build_dispatch_command(1379, wait=False)
    assert command == [
        "gh", "workflow", "run", "runpod-gpu-test.yml",
        "--ref", "main", "-f", "pr_number=1379",
    ]


def test_invalid_pr_numbers_are_rejected() -> None:
    for value in (0, -1):
        with assertRaisesRegex(ValueError, "positive"):
            build_dispatch_command(value, wait=False)
```

Workflow contracts must assert the manual `pr_number` input, same-repository check, open-state check, head-stability recheck, trusted changed-file classification, and final publication through the authorized target output.

- [ ] **Step 2: Confirm tests fail before changes**

Run:

```bash
python3 -m unittest scripts.ci.tests.test_recover_runpod_pr scripts.ci.tests.test_workflow_contracts -v
```

Expected: missing helper and missing workflow input failures.

- [ ] **Step 3: Make the archive key content/configuration based**

Remove `TENFERRO_REF` from the key. Keep a bumped format version, OS, CUDA binding/runtime versions, and `hashFiles` over lockfile, manifests, sources, and tests. Preserve per-run artifact upload on both cache hit and miss.

- [ ] **Step 4: Add trusted `pr_number` authorization**

Add optional numeric `pr_number` input. In manual PR mode, the hosted authorize job uses GitHub API to fetch the PR, verifies:

```bash
test "$(jq -r .state <<<"${pr_json}")" = open
test "$(jq -r .head.repo.full_name <<<"${pr_json}")" = "${GITHUB_REPOSITORY}"
```

It records `tenferro_ref` and `target_head_sha` from `.head.sha`, fetches the PR a second time, and rejects head movement. Conflicting `pr_number`, `tenferro_ref`, or `ci_gpu_gate_head_sha` combinations fail explicitly.

For automatic mode, expose the already validated PR head as the same
`target_head_sha` output. In either PR mode, fetch the complete changed-file
list with pagination and classify it using the helper from trusted `main`;
publish `gpu_required` from that result. Final gate publication consumes only
`needs.authorize.outputs.target_head_sha` and
`needs.authorize.outputs.gpu_required`.

- [ ] **Step 5: Implement the maintainer CLI**

`recover_runpod_pr.py PR_NUMBER [--wait]` validates `gh auth status`, dispatches exactly at `--ref main`, prints the URL returned by `gh workflow run`, and optionally watches that run. It never accepts a workflow ref override and never reads a secret.

Use `subprocess.run(..., check=True, text=True, capture_output=True)` with an injectable runner for unit tests.

- [ ] **Step 6: Verify recovery contracts and dry-run command**

Run:

```bash
python3 -m unittest scripts.ci.tests.test_recover_runpod_pr scripts.ci.tests.test_workflow_contracts -v
python3 scripts/ci/recover_runpod_pr.py --dry-run 1379
actionlint .github/workflows/runpod-gpu-test.yml .github/workflows/CI_gpu.yml
```

Expected: command shows `--ref main -f pr_number=1379`; tests and actionlint pass.

- [ ] **Step 7: Commit cache and recovery changes**

```bash
git add scripts/ci/recover_runpod_pr.py scripts/ci/tests/test_recover_runpod_pr.py .github/workflows/runpod-gpu-test.yml .github/workflows/CI_gpu.yml scripts/ci/tests/test_workflow_contracts.py
git commit -m "ci: add trusted PR GPU recovery"
```

## Task 7: Documentation, work log, and full verification

**Files:**

- Modify: `CONTRIBUTING.md`
- Create: `docs/worklogs/2026-07-14-issue-1379-change-aware-ci.md`
- Modify: `docs/superpowers/specs/2026-07-14-change-aware-ci-runpod-recovery-design.md` (record the trusted post-merge live-validation boundary)

- [ ] **Step 1: Add contributor-facing command and classification docs**

Document:

```bash
python3 scripts/ci/run_profile.py --list
python3 scripts/ci/run_profile.py workspace-faer
python3 scripts/ci/run_profile.py workspace-blas
python3 scripts/ci/recover_runpod_pr.py 1379 --wait
```

Explain docs-only, CI-only, unknown-path fallback, comprehensive `main` behavior, and that PR-number recovery executes the trusted workflow from `main`.

- [ ] **Step 2: Write the curated work log**

Record issue #1379, PR #1376 evidence, files/rules read, central-helper design, docs/CI classification, required-check no-op decision, SECURE trust boundary, OpenAPI validation, retry semantics, content cache key, rejected inline/external-service alternatives, verification, live GPU result, and residual risks.

- [ ] **Step 3: Run all CI helper and source-contract tests**

```bash
python3 -m unittest discover -s scripts/ci/tests -v
python3 scripts/ci/run_profile.py --dry-run full
actionlint
```

Expected: all helper tests and all workflows pass.

- [ ] **Step 4: Run exact non-GPU profiles**

```bash
python3 scripts/ci/run_profile.py workspace-faer
python3 scripts/ci/run_profile.py workspace-blas
python3 scripts/ci/run_profile.py blas-inject
python3 scripts/ci/run_profile.py extensions
python3 scripts/ci/run_profile.py docs
python3 scripts/ci/run_profile.py coverage
```

Expected: every profile passes; coverage thresholds pass for all tracked files.

- [ ] **Step 5: Run repository-required quality gates**

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy --manifest-path ext/tropical/Cargo.toml --all-targets -- -D warnings
cargo test --workspace --release
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
python3 scripts/repository-rules-review.py \
  --base origin/main \
  --head HEAD \
  --output-json /tmp/repository-rules-review-1379.json
```

Expected: every command passes and repository review reports no findings.

- [ ] **Step 6: Validate change classes end to end**

For checked-in fixture path lists, invoke the classifier CLI and assert:

- docs-only selects docs and skips Rust/extensions/GPU;
- unrelated CI-only selects CI config and skips Rust/docs/GPU;
- RunPod CI-only selects CI config plus GPU;
- mixed docs/CI selects docs plus CI config;
- Rust and unknown paths select full validation;
- push mode selects the comprehensive non-GPU matrix.

- [ ] **Step 7: Perform the trusted recovery dry run before merge**

After pushing the branch and opening the PR, run:

```bash
python3 scripts/ci/recover_runpod_pr.py --dry-run <PR_NUMBER>
```

Expected: the command targets `runpod-gpu-test.yml --ref main` and passes only
the PR number. Do not dispatch the PR branch workflow with repository secrets.

- [ ] **Step 8: Re-read repository rules and update the work log with pre-merge evidence**

Read `REPOSITORY_RULES.md` again. Add final local command outcomes, PR CI
evidence, the pending trusted post-merge live-validation requirement, and
residual risks to the work log.

- [ ] **Step 9: Commit documentation and pre-merge verification record**

```bash
git add CONTRIBUTING.md docs/worklogs/2026-07-14-issue-1379-change-aware-ci.md docs/superpowers/specs/2026-07-14-change-aware-ci-runpod-recovery-design.md
git commit -m "docs: record change-aware CI operations"
```

- [ ] **Step 10: Prepare and land the implementation PR**

Create a PR that links (but does not yet close) #1379, links the work log and design spec,
lists exact local verification, and calls out that required check names and the
SECURE RunPod trust boundary are unchanged. Enable auto-merge after all
pre-merge required checks succeed and land the PR; keep issue #1379 open until
the trusted post-merge live GPU validation succeeds.

- [ ] **Step 11: Run trusted post-merge live validation and close the issue**

After the implementation lands on `main`, dispatch the now-trusted workflow
against the landed revision:

```bash
gh workflow run runpod-gpu-test.yml --ref main -f tenferro_ref=main
```

Expected: schema validation passes before archive work, CUDA archive cache
behavior is reported, a SECURE pod is created, CUDA and OpenXLA PJRT GPU tests
pass, cleanup succeeds, and the workflow concludes successfully. Comment on
#1379 with the run URL, cache hit/miss, assigned GPU, test result, and cleanup
result, then close the issue. If it fails, keep the issue open and prepare a
corrective follow-up.
