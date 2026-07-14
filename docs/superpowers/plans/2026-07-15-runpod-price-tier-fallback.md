# RunPod price-tier fallback implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move trusted RunPod CI immediately through reviewed GPU price tiers when capacity is unavailable, bound creation to 60 seconds, and record the assigned GPU.

**Architecture:** `runpod_config.json` owns three ordered SECURE Cloud tiers. `runpod_client.py` distinguishes capacity exhaustion from generic transient failures: capacity changes tier without sleeping, while network/rate-limit/service failures get one short same-tier retry. Workflow YAML only forwards and displays the selected tier and GPU.

**Tech Stack:** Python 3 standard library, `unittest`, JSON, GitHub Actions YAML, actionlint.

---

### Task 1: Capacity-aware tier failover

**Files:**
- Modify: `scripts/ci/tests/test_runpod_client.py`
- Modify: `scripts/ci/runpod_client.py`

- [ ] **Step 1: Write the failing capacity tests**

Add imports for `CreateRequest` and `is_capacity_failure`, then add:

```python
def test_capacity_failure_requires_server_error_and_known_message(self) -> None:
    body = b'{"error":"This machine does not have the resources to deploy your pod"}'
    self.assertTrue(is_capacity_failure(500, body))
    self.assertFalse(is_capacity_failure(400, body))
    self.assertFalse(is_capacity_failure(500, b'{"error":"internal failure"}'))

def test_capacity_failure_moves_tier_without_sleep(self) -> None:
    requests = [
        CreateRequest("cost-preferred", b"cheap"),
        CreateRequest("premium", b"premium"),
    ]
    responses = iter([
        (500, {}, b'{"error":"This machine does not have the resources to deploy your pod"}'),
        (201, {}, b'{"id":"pod-1","machine":{"gpuTypeId":"NVIDIA L40S"}}'),
    ])
    seen: list[bytes] = []
    result = create_pod(
        CONFIG,
        requests,
        transport=lambda payload: (seen.append(payload), next(responses))[1],
        sleep=lambda _delay: self.fail("capacity failover must not sleep"),
    )
    self.assertEqual(seen, [b"cheap", b"premium"])
    self.assertEqual(result.gpu_tier, "premium")
    self.assertEqual(result.gpu_type_id, "NVIDIA L40S")
```

- [ ] **Step 2: Run the named tests and verify RED**

```bash
python3 -m unittest \
  scripts.ci.tests.test_runpod_client.RunPodClientTests.test_capacity_failure_requires_server_error_and_known_message \
  scripts.ci.tests.test_runpod_client.RunPodClientTests.test_capacity_failure_moves_tier_without_sleep
```

Expected: missing-symbol import errors.

- [ ] **Step 3: Implement the minimal tier request/result types and capacity recognizer**

```python
@dataclasses.dataclass(frozen=True)
class CreateRequest:
    tier_name: str
    payload: bytes

@dataclasses.dataclass(frozen=True)
class CreateResult:
    pod_id: str
    gpu_type_id: str
    gpu_tier: str
    body: bytes

_CAPACITY_MESSAGES = (
    "does not have the resources to deploy your pod",
    "no available machine",
)

def is_capacity_failure(status: int, body: bytes) -> bool:
    if not 500 <= status < 600:
        return False
    message = body.decode("utf-8", errors="replace").lower()
    return any(marker in message for marker in _CAPACITY_MESSAGES)
```

Change `parse_create_response` to accept `gpu_tier`. Change `create_pod` to accept `Sequence[CreateRequest]`; on a recognized capacity response, print the current and next tier and break immediately to the next request. On the last tier, raise `RetryableRunPodError`.

- [ ] **Step 4: Update old tests to wrap byte payloads and verify GREEN**

Wrap old payloads as `[CreateRequest("cost-preferred", b"{}")]`.

```bash
python3 -m unittest scripts.ci.tests.test_runpod_client -v
```

Expected: all tests pass.

- [ ] **Step 5: Add RED tests for one same-tier retry and the global deadline**

```python
def test_generic_service_failure_retries_same_tier_once(self) -> None:
    responses = iter([
        (503, {"Retry-After": "2"}, b'{"error":"busy"}'),
        (201, {}, b'{"id":"pod-1","machine":{"gpuTypeId":"NVIDIA L40S"}}'),
    ])
    seen: list[bytes] = []
    sleeps: list[float] = []
    result = create_pod(
        CONFIG | {"same_tier_retries": 1},
        [CreateRequest("premium", b"premium")],
        transport=lambda payload: (seen.append(payload), next(responses))[1],
        sleep=sleeps.append,
    )
    self.assertEqual(seen, [b"premium", b"premium"])
    self.assertEqual(sleeps, [2.0])
    self.assertEqual(result.gpu_tier, "premium")

def test_creation_deadline_caps_retry_sleep(self) -> None:
    ticks = iter([0.0, 59.0])
    sleeps: list[float] = []
    with self.assertRaises(RetryableRunPodError):
        create_pod(
            CONFIG | {"same_tier_retries": 1, "create_deadline_seconds": 60},
            [CreateRequest("cost-preferred", b"cheap")],
            transport=lambda _payload: (503, {"Retry-After": "30"}, b"{}"),
            sleep=sleeps.append,
            monotonic=lambda: next(ticks),
        )
    self.assertEqual(sleeps, [1.0])
```

- [ ] **Step 6: Verify RED, replace the old attempt budget, and verify GREEN**

Remove `max_create_attempts` use. Read `same_tier_retries`, retain one global monotonic deadline, and cap every sleep by remaining time.

```bash
python3 -m unittest scripts.ci.tests.test_runpod_client -v
git add scripts/ci/runpod_client.py scripts/ci/tests/test_runpod_client.py
git commit -m "ci: fail over RunPod capacity by GPU tier"
```

### Task 2: Reviewed price tiers and OpenAPI validation

**Files:**
- Modify: `scripts/ci/runpod_config.json`
- Modify: `scripts/ci/runpod_contract.py`
- Modify: `scripts/ci/tests/test_runpod_contract.py`
- Modify: `scripts/ci/tests/test_runpod_client.py`
- Modify: `scripts/ci/runpod_client.py`

- [ ] **Step 1: Write RED tests for ordered tier parsing**

Add `configured_gpu_tiers` tests:

```python
def test_configured_gpu_tiers_preserve_order(self) -> None:
    tiers = configured_gpu_tiers({"gpu_tiers": [
        {"name": "cheap", "gpu_type_ids": ["NVIDIA A40"]},
        {"name": "premium", "gpu_type_ids": ["NVIDIA L40S"]},
    ]})
    self.assertEqual(tiers, [
        ("cheap", ("NVIDIA A40",)),
        ("premium", ("NVIDIA L40S",)),
    ])

def test_configured_gpu_tiers_reject_duplicates(self) -> None:
    with self.assertRaisesRegex(ContractError, "duplicate GPU ID"):
        configured_gpu_tiers({"gpu_tiers": [
            {"name": "cheap", "gpu_type_ids": ["NVIDIA A40"]},
            {"name": "premium", "gpu_type_ids": ["NVIDIA A40"]},
        ]})
```

Run `python3 -m unittest scripts.ci.tests.test_runpod_contract -v`.
Expected: missing-symbol import failure.

- [ ] **Step 2: Implement strict tier parsing**

Implement:

```python
def configured_gpu_tiers(
    config: Mapping[str, object],
) -> list[tuple[str, tuple[str, ...]]]:
    value = config.get("gpu_tiers")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ContractError("runpod_config.json gpu_tiers must be a nonempty array")
    tiers: list[tuple[str, tuple[str, ...]]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping):
            raise ContractError("each GPU tier must be an object")
        name, ids = item.get("name"), item.get("gpu_type_ids")
        if not isinstance(name, str) or not name:
            raise ContractError("each GPU tier requires a nonempty name")
        if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes)) or not ids:
            raise ContractError(f"GPU tier {name} requires nonempty gpu_type_ids")
        if any(not isinstance(gpu_id, str) or not gpu_id for gpu_id in ids):
            raise ContractError(f"GPU tier {name} IDs must be nonempty strings")
        duplicate = seen.intersection(ids)
        if duplicate:
            raise ContractError(f"duplicate GPU ID across tiers: {sorted(duplicate)}")
        seen.update(ids)
        tiers.append((name, tuple(ids)))
    return tiers
```

Flatten all tiers before `validate_gpu_type_ids`.

- [ ] **Step 3: Replace the flat configuration with three reviewed tiers**

Use:
- `cost-preferred`: all current IDs plus `NVIDIA L4` and `NVIDIA RTX A6000`.
- `premium`: `NVIDIA RTX 6000 Ada Generation`, `NVIDIA L40`, `NVIDIA L40S`, `NVIDIA GeForce RTX 5090`.
- `a100`: `NVIDIA A100 80GB PCIe`, `NVIDIA A100-SXM4-80GB`.

Set `"same_tier_retries": 1` and `"create_deadline_seconds": 60`; remove `max_create_attempts`. Keep SECURE Cloud, one non-interruptible GPU, and existing storage settings.

- [ ] **Step 4: Build one payload per tier**

Change `build_pod_payload` to accept `gpu_type_ids: Sequence[str]`. In `main`, build:

```python
requests = [
    CreateRequest(
        tier_name,
        json.dumps(build_pod_payload(
            config,
            args.image_name,
            args.startup_script.read_text(encoding="utf-8"),
            jit_config,
            gpu_type_ids,
        )).encode(),
    )
    for tier_name, gpu_type_ids in configured_gpu_tiers(config)
]
```

Pass `requests` to `create_pod`.

- [ ] **Step 5: Run unit and live-contract tests, then commit**

```bash
python3 -m unittest \
  scripts.ci.tests.test_runpod_contract \
  scripts.ci.tests.test_runpod_client -v
python3 scripts/ci/runpod_contract.py
git add scripts/ci/runpod_config.json scripts/ci/runpod_contract.py \
  scripts/ci/runpod_client.py scripts/ci/tests/test_runpod_contract.py \
  scripts/ci/tests/test_runpod_client.py
git commit -m "ci: add cost-bounded RunPod GPU tiers"
```

Expected: the live OpenAPI accepts every configured ID. Never weaken validation to make an invalid ID pass.

### Task 3: Selected GPU observability

**Files:**
- Modify: `scripts/ci/tests/test_runpod_client.py`
- Modify: `scripts/ci/tests/test_workflow_contracts.py`
- Modify: `scripts/ci/runpod_client.py`
- Modify: `.github/workflows/runpod-gpu-test.yml`

- [ ] **Step 1: Write RED output and workflow contract tests**

Add:

```python
def test_publish_github_result_records_tier_and_gpu(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "output"
        summary = Path(directory) / "summary"
        result = CreateResult(
            pod_id="pod-1",
            gpu_type_id="NVIDIA L40S",
            gpu_tier="premium",
            body=b"{}",
        )
        publish_github_result(result, output_path=output, summary_path=summary)
        self.assertEqual(
            output.read_text(),
            "pod_id=pod-1\ngpu_type_id=NVIDIA L40S\ngpu_tier=premium\n",
        )
        self.assertIn("Selected GPU: `NVIDIA L40S`", summary.read_text())
        self.assertIn("Price tier: `premium`", summary.read_text())
```

Add to workflow contracts:

```python
def test_runpod_selected_gpu_is_forwarded_and_logged(self) -> None:
    text = read(".github/workflows/runpod-gpu-test.yml")
    self.assertIn("gpu_tier: ${{ steps.create_pod.outputs.gpu_tier }}", text)
    self.assertIn("needs.start-runpod.outputs.gpu_type_id", text)
    self.assertIn("needs.start-runpod.outputs.gpu_tier", text)
    self.assertIn("nvidia-smi --query-gpu=index,name", text)
```

Run the two named tests. Expected: missing helper and missing workflow output.

- [ ] **Step 2: Implement GitHub output and summary publication**

```python
def publish_github_result(
    result: CreateResult,
    *,
    output_path: Path | None,
    summary_path: Path | None,
) -> None:
    if output_path is not None:
        with output_path.open("a", encoding="utf-8") as output:
            output.write(f"pod_id={result.pod_id}\n")
            output.write(f"gpu_type_id={result.gpu_type_id}\n")
            output.write(f"gpu_tier={result.gpu_tier}\n")
    if summary_path is not None:
        with summary_path.open("a", encoding="utf-8") as summary:
            summary.write("### RunPod GPU selection\n\n")
            summary.write(f"- Price tier: `{result.gpu_tier}`\n")
            summary.write(f"- Selected GPU: `{result.gpu_type_id or 'unknown'}`\n")
```

Call it from `main` with `GITHUB_OUTPUT` and `GITHUB_STEP_SUMMARY` paths when present. Keep ordinary log lines for tier and GPU.

- [ ] **Step 3: Forward the tier in workflow YAML**

Add:

```yaml
gpu_tier: ${{ steps.create_pod.outputs.gpu_tier }}
```

to `start-runpod.outputs`. In `Check machine`, print the tier and provider GPU ID immediately before the existing `nvidia-smi --query-gpu=index,name,...` command.

- [ ] **Step 4: Verify and commit**

```bash
python3 -m unittest \
  scripts.ci.tests.test_runpod_client \
  scripts.ci.tests.test_workflow_contracts -v
actionlint .github/workflows/runpod-gpu-test.yml
git add scripts/ci/runpod_client.py scripts/ci/tests/test_runpod_client.py \
  scripts/ci/tests/test_workflow_contracts.py \
  .github/workflows/runpod-gpu-test.yml
git commit -m "ci: log the selected RunPod GPU"
```

### Task 4: Docs, full verification, and PR completion

**Files:**
- Modify: `docs/design/change-aware-ci.md`
- Modify: `docs/worklogs/2026-07-14-issue-1379-change-aware-ci.md`

- [ ] **Step 1: Update durable docs and the work log**

Document: immediate capacity tier changes, one same-tier transient retry, 60-second global deadline, H100 exclusion, A100 price cap, and selected GPU logs/job summary. Record failed run `29334073902`, its five capacity errors, successful cleanup, operational retry, and accepted follow-up design.

- [ ] **Step 2: Run focused verification**

```bash
python3 scripts/ci/run_profile.py ci-config
python3 scripts/check-docs-site.py
git diff --check
```

Expected: helper tests, actionlint, docs checks, and whitespace checks pass.

- [ ] **Step 3: Run the repository-required checks**

```bash
cargo fmt --all --check
cargo test --workspace --release
cargo llvm-cov --workspace --release --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
python3 scripts/repository-rules-review.py \
  --base origin/main --head HEAD \
  --output-json /tmp/repository-rules-review-1380.json
```

Also run clippy exactly as defined by the CI `clippy` job. Expected: every command passes.

- [ ] **Step 4: Commit docs**

```bash
git add docs/design/change-aware-ci.md \
  docs/worklogs/2026-07-14-issue-1379-change-aware-ci.md
git commit -m "docs: explain RunPod GPU tier fallback"
```

- [ ] **Step 5: Push and restore auto-merge**

```bash
git push origin codex/issue-1379-ci-resilience
gh pr merge 1380 --repo tensor4all/tenferro-rs \
  --auto --squash --delete-branch
gh pr checks 1380 --repo tensor4all/tenferro-rs --watch
```

Expected: the live RunPod log and summary name the selected tier/GPU, cleanup passes, and PR #1380 merges. After merge, dispatch the trusted `main` workflow once and record its URL on issue #1379 before closing it.

