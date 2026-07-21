# Phase 2E Atomic Non-Inferiority Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run protocol-v2 evidence tooling that proves the Phase 2 CPU execution changes are non-inferior to exact current main and to a common-lock normalized baseline.

**Architecture:** A shared Python protocol module owns immutable case inventories, sealed environments, hashes, atomic manifests, and the append-only attempt ledger. Separate build, timing, allocation, dispatch, and characterization tools emit self-validating child manifests; one outer orchestrator creates the initially empty evidence root and one aggregate validator is the only promotion gate. Rust test-only instrumentation and an external allocation-probe crate measure the hot path without adding production overhead.

**Tech Stack:** Python 3 standard library and `unittest`, Rust 2021, Cargo bench profile, Criterion 0.5, tenferro CPU/faer providers, Linux affinity/process monitoring, Git/GitHub CLI.

---

## File map

- `scripts/phase2e_protocol.py`: protocol constants, canonical cases, sealed environments, hashes, atomic JSON, manifest validation, and ledger transitions.
- `scripts/test_phase2e_protocol.py`: unit tests for roots, manifests, environments, and ledger closure rules.
- `scripts/phase2e_build.py`: exact/direct and normalized baseline construction, clean-worktree verification, lock ownership, binary builds, and build manifests.
- `scripts/test_phase2e_build.py`: baseline delta, role invariant, dirty/ignored-file, lock, environment, and binary provenance tests.
- `scripts/classify_criterion_noninferiority.py`: protocol-v2 complete-campaign classifier and retained result renderer.
- `scripts/test_classify_criterion_noninferiority.py`: protocol-v2 interval, inventory, state, and output-artifact tests.
- `scripts/run_phase1_eager_campaign.py`: atomic protocol-v2 timing runner; the historical filename remains the stable entry point.
- `scripts/test_run_phase1_eager_campaign.py`: fake-process tests for the 84-pair state machine, monitoring, timeouts, and exit codes.
- `scripts/phase2e/allocation-probe/Cargo.toml.in`: external probe manifest with one canonical tenferro-root substitution.
- `scripts/phase2e/allocation-probe/src/main.rs`: counting allocator and the frozen 28-case eager probe.
- `scripts/phase2e/allocation-probe/src/tests.rs`: module-local allocator and case tests; `main.rs` contains only the test-module declaration.
- `scripts/run_phase2e_allocation_campaign.py`: atomic 168-process allocation comparison.
- `scripts/test_run_phase2e_allocation_campaign.py`: allocation campaign ordering, invalidity, no-reuse, and manifest tests.
- `crates/tenferro-cpu/src/tests/phase2e.rs`: test-only 10-vector and CPU-owned 29-row executor/provider evidence output.
- `crates/tenferro-cpu/src/tests.rs`: registers the module under the repository's single production `mod tests` boundary.
- `crates/tenferro-ad/src/eager/tests/phase2e.rs`: ordinary eager session-entry proof plus the AD-owned 18 `E-N`/`E-D` rows.
- `crates/tenferro-ad/src/eager/tests.rs`: registers the phase2e eager test under the existing eager `mod tests` boundary.
- `crates/tenferro-ad/src/eager_backend.rs`: removes the substantive inline test fixture and keeps only the module declaration/minimal test-variant references.
- `crates/tenferro-ad/src/eager_backend/tests.rs`: owns the extracted `RecordingBackend` fixture, trait implementations, constructor, and backend-session counter.
- `crates/tenferro-cpu/benches/numa_execution.rs`: public-surface managed/external absolute-latency rows for the characterization manifest.
- `crates/tenferro-ad/benches/phase2e_characterization.rs`: AD-owned placement-bound eager latency rows.
- `crates/tenferro-ad/Cargo.toml`: no-harness characterization bench target.
- `scripts/run_phase2e_gates.py`: source-contract check plus 29-CPU/18-AD evidence composition into the canonical 47 rows.
- `scripts/test_run_phase2e_gates.py`: canonical inventory, source-scope, typed skip, and child-manifest tests.
- `scripts/run_phase2e.py`: outer orchestrator and aggregate root validator.
- `scripts/test_run_phase2e.py`: end-to-end fake-stage tests for root ownership and promotion rejection.
- `.gitignore`: ignores only the stable non-normative Phase 2E index-lock file.
- `docs/worklogs/2026-07-21-phase-2e-noninferiority.md`: commands, hashes, host facts, every attempt, results, skips, and review.
- `docs/worklogs/2026-07-21-phase-2e-index.json`: append-only candidate/root history and the exact current PASS root used after evidence commits change HEAD.

## Prerequisite: synchronize the accepted protocol

Before Task 1, commit and push this plan, the Phase 2E design, and the updated
Phase 2 parent spec/plan. Comment on #1436 with permanent branch links and
state that protocol v2 supersedes only the old Performance Gate and Task 11
performance-campaign procedure (now Step 6). Evidence-tool implementation
must not begin while the issue still points only to the old pair-retry
procedure.

The plans directory is ignored for new files, so stage this plan explicitly:

```bash
git add \
  docs/superpowers/specs/2026-07-21-phase-2e-atomic-noninferiority-campaign-design.md \
  docs/superpowers/specs/2026-07-21-phase-2-cpu-domain-executor-design.md \
  docs/superpowers/plans/2026-07-21-phase-2-cpu-domain-executor.md
git add -f docs/superpowers/plans/2026-07-21-phase-2e-atomic-noninferiority-campaign.md
git commit -m "docs: define atomic phase 2e evidence gate"
git fetch origin main
git merge --no-edit origin/main
bash scripts/check-pr-fast.sh \
  --coverage-reviewed \
  --test 'cargo test -p tenferro-cpu' \
  --test 'cargo test -p tenferro-ad'
python3 scripts/repository-rules-review.py \
  --base origin/main \
  --head HEAD \
  --output-json /tmp/phase2e-design-rules-review.json
git push origin codex/execution-engine-through-phase9
```

Then post the protocol-v2 links and supersession scope to #1436 and record the
comment URL before starting Task 1.

### Task 1: Add the protocol-v2 foundation

**Files:**

- Create: `scripts/phase2e_protocol.py`
- Create: `scripts/test_phase2e_protocol.py`

- [ ] **Step 1: Write failing protocol tests**

Add tests that assert the exact 28-case inventory, pair order, protocol
version, empty-root rejection, environment allowlist, collision-free atomic
JSON replacement that preserves any preexisting interrupted temporary, and
ledger lane rules:

```python
class ProtocolTests(unittest.TestCase):
    def test_inventory_and_orders_are_frozen(self):
        self.assertEqual(protocol.PROTOCOL_VERSION, 2)
        self.assertEqual(len(protocol.CANONICAL_CASES), 28)
        self.assertEqual(protocol.PAIR_ORDERS, ("A/B", "B/A", "A/B"))

    def test_runtime_environment_drops_ambient_injection(self):
        env = protocol.runtime_environment(
            path="/usr/bin", home="/tmp/empty-home", criterion_home="/tmp/c"
        )
        self.assertNotIn("LD_PRELOAD", env)
        self.assertNotIn("GLIBC_TUNABLES", env)
        self.assertNotIn("RAYON_RS_NUM_CPUS", env)
        self.assertEqual(env["RAYON_NUM_THREADS"], "1")
        self.assertEqual(env["OMP_DYNAMIC"], "FALSE")

    def test_complete_lane_cannot_be_reopened(self):
        ledger = protocol.new_ledger("a" * 40)
        ledger = protocol.open_attempt(ledger, "timing", "direct-current-main", 1)
        ledger = protocol.close_attempt(ledger, "timing", "direct-current-main", 1, "PASS")
        with self.assertRaises(protocol.ProtocolError):
            protocol.open_attempt(ledger, "timing", "direct-current-main", 2)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
python3 -m unittest scripts/test_phase2e_protocol.py -v
```

Expected: import failure because `scripts/phase2e_protocol.py` does not exist.

- [ ] **Step 3: Implement the protocol module**

Define immutable constants and typed helpers. Create each sibling temporary
with `O_EXCL` through `tempfile.mkstemp`; never reuse `<target>.tmp` or unlink a
partial temporary on failure. This makes every interrupted write a distinct
regular file that abandonment sealing can preserve. Use `os.replace` only
after writing, flushing, and `fsync`ing it, then `fsync` the parent directory:

```python
PROTOCOL_VERSION = 2
PAIR_ORDERS = ("A/B", "B/A", "A/B")
RUN_ROLES = ("sentinel_before", "first_target", "second_target", "sentinel_after")
THREAD_ENV = {
    "RAYON_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "MKL_DYNAMIC": "FALSE",
}

def atomic_write_json(path: pathlib.Path, payload: dict) -> None:
    encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.write-", suffix=".tmp", dir=path.parent
    )
    temporary = pathlib.Path(temporary_name)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)

def runtime_environment(*, path: str, home: str, criterion_home: str | None = None) -> dict[str, str]:
    result = {"PATH": path, "HOME": home, "LC_ALL": "C", "TZ": "UTC", **THREAD_ENV}
    if criterion_home is not None:
        result["CRITERION_HOME"] = criterion_home
    return result
```

Implement `prepare_empty_root`, `sha256_file`, `sha256_json`, manifest field
validation, `new_ledger`, `open_attempt`, and `close_attempt`. The ledger has
`allocation` and `timing` stages, each with ordered direct and normalized
lanes. Only validity `INCONCLUSIVE` permits another whole attempt.

- [ ] **Step 4: Run the tests and verify GREEN**

Run:

```bash
python3 -m unittest scripts/test_phase2e_protocol.py -v
python3 -m py_compile scripts/phase2e_protocol.py
```

Expected: all protocol tests pass and compilation is silent.

- [ ] **Step 5: Commit**

```bash
git add scripts/phase2e_protocol.py scripts/test_phase2e_protocol.py
git commit -m "test(perf): define phase 2e evidence protocol"
```

### Task 2: Build exact and normalized binaries with bound provenance

**Files:**

- Create: `scripts/phase2e_build.py`
- Create: `scripts/test_phase2e_build.py`
- Modify: `scripts/phase2e_protocol.py`

- [ ] **Step 1: Write failing build-contract tests**

Cover the two permitted baseline deltas, four root-owned lock copies, fresh
worktrees, ignored-file allowlist, invariant/role field separation, controlled
Cargo home, requested feature tuple, and executable re-hash:

```python
def test_role_comparison_allows_only_predeclared_differences(self):
    baseline = build.fake_manifest("direct-current-main")
    candidate = build.fake_manifest("candidate")
    build.validate_pair("direct-current-main", baseline, candidate)
    candidate["requested_features"] = ["cpu-blas"]
    with self.assertRaises(protocol.ProtocolError):
        build.validate_pair("direct-current-main", baseline, candidate)

def test_ignored_inventory_allows_only_root_owned_lock(self):
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        (root / "Cargo.lock").write_bytes(b"lock")
        build.validate_filesystem_inventory(root, {pathlib.Path("Cargo.lock")})
        (root / "target").mkdir()
        with self.assertRaises(protocol.ProtocolError):
            build.validate_filesystem_inventory(root, {pathlib.Path("Cargo.lock")})
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
python3 -m unittest scripts/test_phase2e_build.py -v
```

Expected: import failure because the build orchestrator does not exist.

- [ ] **Step 3: Implement baseline and build validation**

Define these immutable identities and commands:

```python
IMPLEMENTATION_BASELINE = "85855e272b1495611deb601a9ee06f3546772c3c"
HARNESS_COMMIT = "4471d6145c4d8793de3a96f8d99400c24ca8c6d1"
OLD_STRIDED = "10fc972d3c0f8cdfd4ecb45d21d815aebfd7d1f2"
COMMON_STRIDED = "6b0b4a46b7dd9a9ea1677a0d596c0b4adab1acbc"
BENCH_COMMAND = (
    "cargo", "bench", "--locked", "--no-run", "-p", "tenferro-ad",
    "--bench", "eager_dispatch_baseline", "--no-default-features",
    "--features", "cpu-faer",
)
INVARIANT_FIELDS = {
    "protocol_version", "toolchain", "target", "profile", "requested_features",
    "provider", "benchmark_sha256", "benchmark_stanza_sha256",
    "command_template", "config_chain_sha256",
}
ROLE_FIELDS = {
    "role", "head", "tracked_tree_sha256", "resolved_features_sha256",
    "lock_sha256", "worktree", "target_dir", "executable", "executable_sha256",
}
```

Create dedicated worktrees. Materialize the direct measurement commit by
applying only the frozen bench file/stanza, and the normalized commit by also
changing exactly five strided pins. Verify patches by path and hunk. Generate
the direct and common locks inside the sealed Cargo environment, copy their
bytes into `builds/locks`, install the correct copy in each worktree, and run
`cargo metadata --locked`, the build-matching feature query, and
`BENCH_COMMAND`. The timing feature query is exactly:

```bash
cargo tree --locked --target "$HOST_TARGET" -p tenferro-ad \
  --no-default-features --features cpu-faer -e features
```

Probe and later test/bench feature queries likewise use their actual
`--manifest-path` or package, `--target`, default-feature setting, feature
tuple, and root-owned lock. Never substitute a workspace-default
`cargo tree -e features` result.
Record actual argv/environment and parse the emitted bench executable path.
Use a 300-second deadline for lock/metadata/tree commands and 1,800 seconds for
each build; terminate the process group, wait five seconds, kill survivors,
and finalize build validity as inconclusive on timeout.

- [ ] **Step 4: Verify build-contract tests GREEN**

Run:

```bash
python3 -m unittest scripts/test_phase2e_build.py -v
python3 -m py_compile scripts/phase2e_build.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/phase2e_build.py scripts/test_phase2e_build.py scripts/phase2e_protocol.py
git commit -m "feat(perf): bind phase 2e build provenance"
```

### Task 3: Upgrade the classifier to protocol version 2

**Files:**

- Modify: `scripts/classify_criterion_noninferiority.py`
- Modify: `scripts/test_classify_criterion_noninferiority.py`

- [ ] **Step 1: Convert tests to terminal protocol-v2 artifacts**

Update synthetic manifests to use `validity_state`, `statistical_result`,
complete monitor samples, protocol/build/classifier digests, and normative
artifact inventories. Add assertions that classification writes both files and
returns the exact campaign result:

```python
result = classifier.classify_campaign(campaign_path, output_dir)
self.assertEqual(result["statistical_result"], "PASS")
self.assertEqual(len(result["cases"]), 28)
self.assertTrue((output_dir / "classification.json").is_file())
self.assertTrue((output_dir / "summary.md").is_file())
```

Also assert rejection of protocol 1, `RUNNING`, validity `INCONCLUSIVE`, a
partial case inventory, an unhashed estimate, and a mismatched classifier
digest.

- [ ] **Step 2: Run the classifier tests and verify RED**

```bash
python3 -m unittest scripts/test_classify_criterion_noninferiority.py -v
```

Expected: failures because the classifier still accepts version 1 and renders
only the legacy output.

- [ ] **Step 3: Implement protocol-v2 classification**

Keep the frozen interval rules and orient the middle pair before classifying:

```python
def classify_case(intervals):
    if all(interval["upper"] <= 0.05 for interval in intervals):
        return "PASS"
    if sum(interval["lower"] > 0.05 for interval in intervals) >= 2:
        return "FAIL"
    return "INCONCLUSIVE"

def campaign_result(results):
    if any(value == "FAIL" for value in results.values()):
        return "FAIL"
    if any(value == "INCONCLUSIVE" for value in results.values()):
        return "INCONCLUSIVE"
    return "PASS"
```

Write `classification.json` atomically, render `summary.md`, hash both, and
return their paths/digests to the caller for terminal `campaign.json`.

- [ ] **Step 4: Run the tests and verify GREEN**

```bash
python3 -m unittest scripts/test_classify_criterion_noninferiority.py -v
```

Expected: all classifier tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/classify_criterion_noninferiority.py scripts/test_classify_criterion_noninferiority.py
git commit -m "feat(perf): classify atomic phase 2e campaigns"
```

### Task 4: Rewrite the timing runner as one atomic campaign

**Files:**

- Modify: `scripts/run_phase1_eager_campaign.py`
- Modify: `scripts/test_run_phase1_eager_campaign.py`
- Modify: `scripts/classify_criterion_noninferiority.py`
- Modify: `scripts/test_classify_criterion_noninferiority.py`
- Modify: `scripts/phase2e_protocol.py`
- Modify: `scripts/test_phase2e_protocol.py`

The classifier and protocol files are in scope because an atomic runner must
classify an in-memory terminal view while the durable manifest remains
`RUNNING`, and must commit artifacts relative to caller-retained directory
descriptors.  These are prerequisites for preventing false `COMPLETE` records
and pathname-swap redirection; they do not create a second campaign state
machine.

- [ ] **Step 1: Add failing whole-campaign tests**

Replace the five shallow tests with fake benchmark/monitor fixtures that
exercise all 336 process roles. Assert a valid 84-pair PASS, first-invalid
termination, no retry namespace, fresh roots, timeout termination, complete
environment recording, atomic exception finalization, and exit codes 0/2/3/4.

```python
def test_first_invalid_process_closes_the_whole_campaign(self):
    fake = FakeProcessFactory(invalid_at=17)
    code = runner.run_campaign(self.arguments(), process_factory=fake)
    self.assertEqual(code, 2)
    self.assertEqual(fake.launch_count, 17)
    manifest = json.loads(self.campaign_path().read_text())
    self.assertEqual(manifest["validity_state"], "INCONCLUSIVE")
    self.assertNotIn("_rejected", json.dumps(manifest))
```

- [ ] **Step 2: Run the runner tests and verify RED**

```bash
python3 -m unittest scripts/test_run_phase1_eager_campaign.py -v
```

Expected: failures because the current runner retries individual pairs.

- [ ] **Step 3: Implement the atomic state machine**

Delete `--max-attempts` and every pair-local retry path. Require comparison
kind, baseline/candidate build manifests, ledger, artifact root, and Criterion
root. Register the attempt before launch, write `RUNNING`, execute the fixed
inventory once, and stop on first invalid observation. Bound quiet wait to 300
seconds and each process to 30 seconds plus a five-second kill grace period.

Use one terminal mapping:

```python
EXIT_BY_RESULT = {
    ("COMPLETE", "PASS"): 0,
    ("INCONCLUSIVE", None): 2,
    ("COMPLETE", "FAIL"): 3,
    ("COMPLETE", "INCONCLUSIVE"): 4,
}
```

Copy every Criterion estimate into its pair directory before hashing. Treat
`CRITERION_HOME` as non-normative scratch and never resume a `RUNNING`
manifest.

Retain descriptors for campaign roots and benchmark executables, validate
their identities after every child process, and perform owned I/O relative to
those descriptors.  Revalidate builds through `BuildConfig` at the public
entry point.  At successful completion, classify the in-memory terminal view,
prepare and fsync a fixed final-campaign stage and hash marker, close the ledger,
then atomically replace the sole durable `RUNNING` manifest with `COMPLETE`.
The marker permits finalization-only recovery without benchmark remeasurement
after interruption before or after ledger closure, rename, directory fsync, or
marker cleanup; unknown or mismatched partials are rejected.  This is not
measurement resume.

Recovery must fully revalidate a published `COMPLETE` campaign through the
retained artifact descriptor: strict campaign/run/build/Criterion schemas,
every artifact digest and directory, both classifier outputs, and the exact
closed ledger attempt/result.  A committed marker write whose parent fsync
fails retains the RUNNING campaign, active ledger, stage, and marker for that
same recovery.  A `COMPLETE` campaign with a matching closed ledger remains
idempotently recoverable even after marker removal; it is never treated as a
fresh measurement root.  Keep the exact final stage until marker removal and
its directory fsync have completed, and reject contradictory active/closed
ledger combinations or any mismatched partial.

The recovery recognizer is an exact allow-list.  With columns
`campaign / marker / stage / publish / ledger-active`, the only reachable
states are:

```text
RUNNING / no  / no  / no  / yes  -> ordinary incomplete campaign; do not resume
RUNNING / no  / yes / no  / yes  -> pre-marker crash; preserve and do not resume
RUNNING / yes / yes / no  / yes  -> finish finalization
RUNNING / yes / yes / yes / yes  -> finish finalization
RUNNING / yes / yes / no  / no   -> finish finalization
RUNNING / yes / yes / yes / no   -> finish finalization
COMPLETE / yes / yes / no / no   -> validate, then clean marker and stage
COMPLETE / no  / yes / no / no   -> validate, then clean stage
COMPLETE / no  / no  / no / no   -> validate idempotently
```

Reject every other combination before launching a benchmark.  A surviving
publish partial must be the same regular-file inode as the stage hard link,
not merely byte-identical.  A retained stage beside `COMPLETE campaign.json`
must likewise be the same regular-file inode left by publish-and-rename; a
markerless `COMPLETE` with no stage represents completed cleanup and remains
valid.  If an arbitrary `BaseException` interrupts marker publication after
rename, probe the canonical marker through the held root descriptor as
`EXACT`, `ABSENT`, `MISMATCH`, or `UNKNOWN_IO`.  `EXACT` resumes finalization;
`MISMATCH` is preserved and rejected; `UNKNOWN_IO` conservatively preserves
the transaction and original exception; only `ABSENT` permits pre-commit
stage cleanup.  Only a `FileNotFoundError` from the dirfd-relative leaf open is
`ABSENT`; after that open succeeds, every read, stability-check, or close
exception is `UNKNOWN_IO`, including another `FileNotFoundError`.
Missing-file cleanup still fsyncs the root directory so that recovery has one
durability rule.

Create and pin fresh roots through their parent directory descriptors before
checking emptiness.  Keep classifier reads, inventory, and output publication
relative to the retained artifact descriptor without resolving
`/proc/self/fd/N` back to a pathname.  Snapshot each validated executable into
a write/grow/shrink/seal-protected Linux memfd and launch only that immutable
snapshot.  Record and strictly validate each run's exact argv, sealed
environment and digest, source/snapshot executable identity, Criterion logical
root and actual descriptor binding, and process-group cleanup outcome.  Observe
normal child exit with non-reaping `waitid(..., WNOWAIT)` so the final host
sample is captured before reap; after reap, detect and kill any surviving
process-group members.  Cleanup must preserve `KeyboardInterrupt` and
`SystemExit` while still terminating the whole process group.

Open every absolute parent component from `/` with retained directory
descriptors and `O_DIRECTORY | O_NOFOLLOW`; do not resolve-check a parent and
then reopen it by pathname before creating the fresh child root.
Consume each traversed descriptor in caller state before attempting its close.
If close reports an exception, never retry that descriptor number: it may
already have been closed and reassigned to an unrelated file descriptor.

- [ ] **Step 4: Run runner and classifier tests GREEN**

```bash
python3 -m unittest \
  scripts/test_run_phase1_eager_campaign.py \
  scripts/test_classify_criterion_noninferiority.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_phase1_eager_campaign.py scripts/test_run_phase1_eager_campaign.py
git add scripts/classify_criterion_noninferiority.py scripts/test_classify_criterion_noninferiority.py
git add scripts/phase2e_protocol.py scripts/test_phase2e_protocol.py
git commit -m "feat(perf): make eager campaign atomic"
```

### Task 5: Implement the external allocation probe

**Files:**

- Create: `scripts/phase2e/allocation-probe/Cargo.toml.in`
- Create: `scripts/phase2e/allocation-probe/src/main.rs`
- Create: `scripts/phase2e/allocation-probe/src/tests.rs`
- Modify: `scripts/phase2e_build.py`
- Modify: `scripts/test_phase2e_build.py`

- [ ] **Step 1: Add failing allocator reference tests**

Add only `#[cfg(test)] mod tests;` to `main.rs`. Put substantive tests for
reset, `alloc`, `alloc_zeroed`, `realloc`, `dealloc`, injected failure, and
checked overflow in module-local `src/tests.rs`, following the repository's
unit-test organization rule. Tests use the wrapper's test-only failure/seed
controls and assert monotonic positive counts/bytes; they do not assert whether
the system reallocates in place.

Define `TestAllocator` inside `src/tests.rs` as the same counter state machine
with an injectable delegate; production `CountingSystem` uses `System`
directly.

```rust
#[test]
fn realloc_counts_one_event_and_the_new_size() {
    let mut probe = TestAllocator::new();
    let ptr = probe.alloc(Layout::from_size_align(8, 8).unwrap());
    probe.reset();
    let ptr = probe.realloc(ptr, Layout::from_size_align(8, 8).unwrap(), 32);
    assert!(!ptr.is_null());
    assert_eq!(probe.snapshot(), Snapshot { allocations: 1, bytes: 32, failures: 0 });
}
```

- [ ] **Step 2: Generate a temporary manifest and verify RED**

Substitute the current repository path into `Cargo.toml.in`, then run:

```bash
PHASE2E_PROBE_DEV_ROOT="$(mktemp -d -p /tmp phase2e-allocation-probe.XXXXXX)"
# Generate $PHASE2E_PROBE_DEV_ROOT/Cargo.toml from the tracked template.
cargo test --manifest-path "$PHASE2E_PROBE_DEV_ROOT/Cargo.toml"
```

Expected: compile failure because the probe implementation is absent.

- [ ] **Step 3: Implement the counting allocator and 28 cases**

The manifest uses path dependencies with
`default-features = false, features = ["cpu-faer"]`. The global allocator
delegates once to `System` and records successful allocation events with
checked atomics:

```rust
unsafe impl GlobalAlloc for CountingSystem {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: Forward the caller's GlobalAlloc layout contract unchanged
        // to System; this wrapper never dereferences the returned pointer.
        let pointer = unsafe { System.alloc(layout) };
        self.record(pointer, layout.size());
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: Forward the caller's GlobalAlloc layout contract unchanged
        // to System; this wrapper never dereferences the returned pointer.
        let pointer = unsafe { System.alloc_zeroed(layout) };
        self.record(pointer, layout.size());
        pointer
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: Forward the caller's pointer/layout/new-size contract
        // unchanged to the same System allocator; no pointer is dereferenced.
        let result = unsafe { System.realloc(pointer, layout, new_size) };
        self.record(result, new_size);
        result
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: The caller guarantees pointer/layout came from this global
        // allocator, whose sole delegate is System; no pointer is dereferenced.
        unsafe { System.dealloc(pointer, layout) };
    }
}
```

Keep every allocator call in an explicit unsafe block with its adjacent
boundary-specific `// SAFETY:` explanation; do not rely on the implicit unsafe
body of `unsafe fn`. The module-local tests must exercise the valid pointer and
layout lifecycle plus null/failure behavior across all four delegate methods.

Parse exactly the canonical case names. Construct tensors/runtime before 256
warm-ups, reset, execute 4,096 iterations, consume lazy results with
`shape()` plus `tensor_read()`, consume materialized results with
`materialized()` plus the first `f64`, and print one sorted JSON record with
case, repetitions, checksum, allocation count/bytes/failures, and overflow.
Extend `phase2e_build.py` with `verify-allocation-probe --repository <path>`.
It creates a unique temporary generated-manifest root and runs the exact fmt,
test, all-target clippy, and bench-build commands below with bounded subprocess
deadlines. Add fake-process tests for all four commands and nonzero/timeout
propagation.

- [ ] **Step 4: Run probe tests GREEN**

```bash
cargo fmt --manifest-path "$PHASE2E_PROBE_DEV_ROOT/Cargo.toml" -- --check
cargo test --manifest-path "$PHASE2E_PROBE_DEV_ROOT/Cargo.toml"
cargo clippy --manifest-path "$PHASE2E_PROBE_DEV_ROOT/Cargo.toml" \
  --all-targets -- -D warnings
cargo build --locked --profile bench \
  --manifest-path "$PHASE2E_PROBE_DEV_ROOT/Cargo.toml"
python3 scripts/phase2e_build.py verify-allocation-probe \
  --repository "$PWD"
```

Expected: formatting is clean, all tests pass, clippy reports no warnings, the
bench-profile binary builds, and the helper independently reproduces all four
checks from a fresh generated root.

- [ ] **Step 5: Commit**

```bash
git add scripts/phase2e/allocation-probe/Cargo.toml.in \
  scripts/phase2e/allocation-probe/src/main.rs \
  scripts/phase2e/allocation-probe/src/tests.rs \
  scripts/phase2e_build.py scripts/test_phase2e_build.py
git commit -m "feat(perf): add frozen eager allocation probe"
```

### Task 6: Add the atomic allocation campaign

**Files:**

- Create: `scripts/run_phase2e_allocation_campaign.py`
- Create: `scripts/test_run_phase2e_allocation_campaign.py`
- Modify: `scripts/phase2e_build.py`
- Modify: `scripts/test_phase2e_build.py`

- [ ] **Step 1: Write failing 168-process and probe-build tests**

Assert three probe identities, root-owned direct/common probe locks, sealed
bench-profile builds, six processes per case in `A/B`, `B/A`, `A/B` order,
first-invalid stop, no observation reuse, and PASS only when every candidate
count/byte observation is no greater.

```python
def test_complete_comparison_launches_exactly_168_processes(self):
    fake = FakeProbeFactory()
    code = allocation.run_comparison(self.arguments(), process_factory=fake)
    self.assertEqual(code, 0)
    self.assertEqual(fake.launch_count, 2 * 3 * 28)
```

- [ ] **Step 2: Run tests and verify RED**

```bash
python3 -m unittest \
  scripts/test_phase2e_build.py \
  scripts/test_run_phase2e_allocation_campaign.py -v
```

Expected: failures because probe builds and allocation campaigns are not
implemented.

- [ ] **Step 3: Build three probes and implement the campaign**

Extend the build tool to create candidate, direct-baseline, and
normalized-baseline probe manifests using the probe lock/profile contract.
Implement allocation manifest states and exit codes 0 PASS, 2 validity
inconclusive, and 3 deterministic FAIL. Require a fresh attempt directory and
register it in the allocation ledger before the first process. Compare
corresponding oriented observations without averaging or selecting minima.
Give every probe process a 30-second deadline and apply the common five-second
process-group termination/kill protocol before writing validity inconclusive.

- [ ] **Step 4: Run tests GREEN**

```bash
python3 -m unittest \
  scripts/test_phase2e_build.py \
  scripts/test_run_phase2e_allocation_campaign.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add \
  scripts/phase2e_build.py scripts/test_phase2e_build.py \
  scripts/run_phase2e_allocation_campaign.py \
  scripts/test_run_phase2e_allocation_campaign.py
git commit -m "feat(perf): run atomic allocation comparisons"
```

### Task 7: Emit dispatch and 47-row characterization evidence

**Files:**

- Create: `crates/tenferro-cpu/src/tests/phase2e.rs`
- Modify: `crates/tenferro-cpu/src/tests.rs`
- Create: `crates/tenferro-ad/src/eager/tests/phase2e.rs`
- Modify: `crates/tenferro-ad/src/eager/tests.rs`
- Modify: `crates/tenferro-ad/src/eager_backend.rs`
- Create: `crates/tenferro-ad/src/eager_backend/tests.rs`
- Modify: `crates/tenferro-cpu/benches/numa_execution.rs`
- Create: `crates/tenferro-ad/benches/phase2e_characterization.rs`
- Modify: `crates/tenferro-ad/Cargo.toml`
- Modify: `scripts/phase2e_build.py`
- Modify: `scripts/test_phase2e_build.py`
- Create: `scripts/run_phase2e_gates.py`
- Create: `scripts/test_run_phase2e_gates.py`

- [ ] **Step 1: Add failing inventory and count-vector tests**

Register the CPU file from the existing `tests.rs`; do not add a second test
module to production `lib.rs`. Its serialized test emits the five direct and
five borrowed CPU-session downstream vectors plus the CPU-owned 29-row
inventory: 27 `D-N`/`D-D`/`G-O` rows and `U-O`/`U-I`. Assert those exact
counts:

```rust
const DIRECT_NATIVE: Counts = Counts::new(0, 1, 1, 1, 0, 0);
const EAGER_NATIVE: Counts = Counts::new(1, 1, 1, 1, 0, 0);
const DIRECT_DOT: Counts = Counts::new(0, 1, 1, 1, 0, 1);
const EAGER_DOT: Counts = Counts::new(1, 1, 1, 1, 0, 1);

#[test]
fn phase2e_characterization_evidence() {
    let evidence = run_cpu_owned_rows();
    assert_eq!(evidence.canonical_vectors.len(), 10);
    assert_eq!(evidence.characterization.len(), 29);
    assert!(evidence.characterization.iter().all(|row| row.gating_passed()));
    write_evidence(&evidence).unwrap();
}
```

Extract the existing `cfg(test)` `RecordingBackend`, its delegate macro, all
trait implementations, and its constructor from production
`eager_backend.rs` into module-local `eager_backend/tests.rs` before extending
it. The production module retains only `#[cfg(test)] mod tests;`, the minimal
private type reference/enum dispatch arms needed for its test-only variant, and
no substantive fixture implementation. The AD test adds the session counter
to that extracted fixture, emits a separate eager-session JSON file, and proves
`neg`, `add`, `reduce_sum`, `slice`, and `dot_general` materialization each call
`with_backend_session` once. The same AD artifact owns and executes all 18
public placement-bound `E-N`/`E-D` rows using AD-local instrumented public CPU
executor/provider fixtures:

```rust
#[test]
fn phase2e_eager_characterization_evidence() {
    let evidence = run_ad_owned_rows();
    assert_eq!(evidence.session_entries.len(), 5);
    assert!(evidence.session_entries.iter().all(|count| *count == 1));
    assert_eq!(evidence.characterization.len(), 18);
    assert!(evidence.characterization.iter().all(|row| row.gating_passed()));
    write_evidence(&evidence).unwrap();
}
```

The gate wrapper composes each AD session entry with the matching CPU
borrowed-session downstream vector and combines the 29 CPU plus 18 AD rows. It
rejects duplicate, absent, or wrong-owner row keys before accepting exactly 47
rows. CPU tests never depend on `tenferro-ad`, so no dependency cycle is
introduced. Add a source-organization assertion that rejects `struct
RecordingBackend`, its delegate macro, or its trait impls in
`eager_backend.rs`.

Across the two crate-owned artifacts, the 45 ownership/budget/surface rows use
the exact D-N/E-N/D-D/E-D/G-O requests and budget-dependent modes from the
spec. Add explicit low-level CPU-owned U-O and U-I rows. Every success row
checks numerical results, typed error and unwind recovery, counts, and
post-recovery success. Hardware rows record every observed CPU and typed skips.

Extend `numa_execution.rs` with the CPU-owned direct/grouped/executor
success-row benchmark ids and add `phase2e_characterization.rs` for the
AD-owned placement-bound `E-N`/`E-D` rows,
both with fixed 2s/5s/100/95% Criterion settings. Build both from the immutable
candidate with its root-owned common lock and sealed environment, then record
their source/binary digests in the characterization manifest. The private
single-entry/count proof remains in the unit test; the benchmarks contribute
only absolute non-gating latency and observed-affinity artifacts.

- [ ] **Step 2: Add failing Python gate-wrapper tests**

Test the exact `cargo test --locked --no-run` CPU and AD build commands,
package-specific locked feature queries, sealed build environment, emitted
test-executable selection and digest, direct executable run argv/environment,
JSON digest, source hot-path function inventory, banned identifiers,
hardware-skip schema, 29/18 owner partitions, duplicate/missing/wrong-owner row
rejection, and manifest candidate/protocol binding. Evidence mode
must reject direct `cargo test`, a default-feature feature graph, a binary not
owned by its build manifest, or a test build from a dirty/non-candidate tree.

- [ ] **Step 3: Run both test layers and verify RED**

```bash
cargo test -p tenferro-cpu phase2e_characterization_evidence --lib -- --nocapture
cargo test -p tenferro-ad phase2e_eager_characterization_evidence --lib -- --nocapture
python3 -m unittest scripts/test_run_phase2e_gates.py -v
```

Expected: failures because the test module and wrapper do not exist.

- [ ] **Step 4: Implement test-only counters and gate wrapper**

Reuse instrumented `CpuDomainExecutor` and provider fixtures; do not add
production counters. The Python source check owns an explicit function/module
list from `EagerTensor` through `CpuExecSession` and rejects identifier tokens
`TypeId`, `Any`, `HashMap`, formatting dispatch, and string-key lookup in that
steady-state scope. Extend the build tool to build the CPU and AD library-test
executables from the immutable candidate worktree with the root-owned common
lock, `--locked --no-run --no-default-features --features cpu-faer
--message-format=json`, fresh external target directories, and the sealed
environment. Bind each executable to its exact source, lock, toolchain,
package-specific feature graph, argv/environment, and SHA-256 in
`dispatch-gates/{cpu,ad}-test-build.json`.

Run the two hashed executables directly with their exact evidence filters and
`--nocapture`; do not let Cargo rebuild them during evidence collection. Hash
their JSON, compose the CPU and AD artifacts, and write
`dispatch-gates/manifest.json` and `characterization/manifest.json`
atomically. Build both Criterion binaries under the same source/lock rules and
record matching locked feature graphs. Use Criterion's 2s/5s/100/95%
configuration for absolute non-gating latency rows. Each correctness test
binary has a 120-second deadline and every filtered Criterion row has a
30-second deadline; either timeout is typed validity inconclusive after the
five-second process-group kill grace period. The direct `cargo test` commands
in Steps 3 and 5 are development RED/GREEN checks only and are never accepted
as provenance-bound evidence.

- [ ] **Step 5: Run focused tests GREEN**

```bash
cargo test -p tenferro-cpu phase2e_characterization_evidence --lib -- --nocapture
cargo test -p tenferro-ad phase2e_eager_characterization_evidence --lib -- --nocapture
python3 -m unittest scripts/test_run_phase2e_gates.py -v
cargo test -p tenferro-cpu provider::tests --lib
cargo test -p tenferro-cpu dot_runtime::tests --lib
```

Expected: all tests pass and production builds contain no instrumentation.

- [ ] **Step 6: Commit**

```bash
git add \
  crates/tenferro-cpu/src/tests.rs crates/tenferro-cpu/src/tests/phase2e.rs \
  crates/tenferro-cpu/benches/numa_execution.rs \
  crates/tenferro-ad/src/eager/tests.rs crates/tenferro-ad/src/eager/tests/phase2e.rs \
  crates/tenferro-ad/src/eager_backend.rs crates/tenferro-ad/src/eager_backend/tests.rs \
  crates/tenferro-ad/Cargo.toml crates/tenferro-ad/benches/phase2e_characterization.rs \
  scripts/phase2e_build.py scripts/test_phase2e_build.py \
  scripts/run_phase2e_gates.py scripts/test_run_phase2e_gates.py
git commit -m "test(cpu): emit phase 2e parallelism evidence"
```

### Task 8: Add the outer orchestrator and aggregate validator

**Files:**

- Create: `scripts/run_phase2e.py`
- Create: `scripts/test_run_phase2e.py`
- Modify: `scripts/phase2e_protocol.py`
- Modify: `.gitignore`

- [ ] **Step 1: Write failing fake-stage orchestration tests**

Test initial outer-root emptiness, stage order, all child hashes, all attempt
retention, direct-before-normalized opening, stop on non-pass, foreign
candidate rejection, external scratch exclusion, existing-root digest
revalidation, retryable-validity transitions, indexed-root closure,
abandoned-root indexing, ignored normative lock/log staging, Git-index versus
manifest inventory equality, fresh-checkout validation, parallel `start`,
parallel `record-index`, record-versus-rerun races, initialization crash
recovery, and aggregate PASS-only exit. Assert global `ACTIVE` reservation,
the fixed index-then-root lock order, canonical experiment-identity binding,
exact candidate provenance, closure after every statistically complete result,
and the `ACTIVE` -> `PENDING_PRESERVATION` -> `PRESERVED` lifecycle. Prove that
no fresh root starts before its predecessor is present on the remote branch
and reported on #1436, that evidence/worklog-only commits cannot create a new
experiment identity, and that only a preserved validity-inconclusive or
abandoned event permits another unique root for the same exact candidate. For
abandonment, cover partial `.log` and `.tmp` files,
missing/tampered/extra paths, symlink and special-file rejection,
no-live-process confirmation, seal digest ownership, and fresh-checkout
reconstruction:

The read-only identity-comparison tests use synthetic Git trees to prove that
commit metadata and `docs/worklogs/**` changes compare equal, while any other
mode/path/blob, protocol, command, feature, revision, or threshold change
compares unequal.

Also assert that `.gitignore` ignores exactly
`docs/worklogs/.phase2e-index.lock`, and that an initialization failure after
the ACTIVE event creates a complete abandonment seal and
`PENDING_PRESERVATION` terminal event in the same locked operation without
requiring a later `record-index` call, returns exit 5, and prints exactly the
stable token `ABANDONED_INITIALIZATION`. Fake Git/GitHub adapters must prove that
`record-preserved` rejects an unpushed preservation commit, a commit missing
the exact staged root/index blobs, a non-#1436 comment, a comment missing the
commit/root/status, and an unpushed PRESERVED index before the next `start`.

```python
def test_root_pass_requires_every_gate(self):
    root = make_complete_fake_root()
    self.assertEqual(orchestrator.validate_root(root), "PASS")
    manifest = json.loads((root / "characterization" / "manifest.json").read_text())
    manifest["gating_result"] = "FAIL"
    protocol.atomic_write_json(root / "characterization" / "manifest.json", manifest)
    with self.assertRaises(protocol.ProtocolError):
        orchestrator.validate_root(root)
```

- [ ] **Step 2: Run tests and verify RED**

```bash
python3 -m unittest scripts/test_run_phase2e.py -v
```

Expected: import failure because the outer orchestrator does not exist.

- [ ] **Step 3: Implement stage orchestration and validation**

Use this fixed order: root/ledger initialization; three timing binaries and
three probe builds; direct and normalized allocation; two candidate dispatch
test-binary builds and dispatch gates; two characterization bench builds and
characterization; direct timing; normalized timing; aggregate validation.
Every subprocess receives the constructed sealed environment. Update
`phase2e-evidence.json` after each child, hash all four locks, every build/gate
manifest, every attempt, and the ledger. Create every Criterion root with
`mktemp` beneath a caller-supplied scratch parent outside both the repository
and evidence root; record only its diagnostic path and leave its contents
outside the normative hash tree.

The root CLI has `start`, `rerun-invalid-lane`, `continue`, `validate`,
`record-index`, `record-preserved`, and read-only
`compare-experiment-identity` subcommands. The comparison command computes the
same canonical digest for any two commits and exits 0 only when they are
identical; it never mutates index state. All index operations use an exclusive
`fcntl` lock
on `docs/worklogs/.phase2e-index.lock`, which is ignored by its exact path in
`.gitignore` and never staged. Every operation needing both locks acquires the
index lock before the root lock. `start` requires an empty root, rejects any
global `ACTIVE` or `PENDING_PRESERVATION` campaign, and, when an index already
exists, requires its bytes to match the fetched remote-branch blob. It verifies
the root was never reserved, creates the root and locks it while holding the
index lock, and appends an `ACTIVE` event
before initializing any normative child or launching any subprocess. The
event binds reservation id, exact candidate SHA, root, protocol, and two
separate identities. `candidate_provenance` binds the exact commit and full
tree. `experiment_identity_digest` is the SHA-256 of the sorted Git
`(mode,path,blob)` inventory after excluding only `docs/worklogs/**`, plus the
protocol/classifier/probe source hashes, immutable revisions, feature/command
matrix, and thresholds. Commit metadata and issue text are not inputs. Index
closure is keyed by `experiment_identity_digest`, never by raw candidate SHA;
for a permitted validity/abandonment retry, the exact original candidate SHA
must also remain unchanged.

If initialization fails after the ACTIVE event, `start` itself, while still
holding index then root locks and before any child process has been launched,
rejects symlinks/special files, hashes every existing regular file including
unique partial temporaries, atomically writes `abandoned-inventory.json`, and
appends the terminal `ABANDONED` event containing its digest. That event
transitions ACTIVE to `PENDING_PRESERVATION`; it does not permit a fresh root.
It is immediately eligible for the normal abandoned-root
stage/commit/push/report workflow; a later `record-index --abandoned` is
neither required nor permitted for that already terminal reservation. A
failure before the durable ACTIVE append creates no campaign or index event
and launches no subprocess.

After exit 2, `rerun-invalid-lane` takes both locks in that order and re-hashes
the aggregate manifest, ledger, every previously registered child, all
locks/builds, exact candidate provenance, canonical experiment identity,
protocol, active reservation, and
current lane before it mutates anything. It rejects a finalized reservation,
a digest mismatch, a complete lane, or any `RUNNING` child; otherwise it
allocates the next attempt id plus fresh child and external scratch roots and
executes that whole lane. `continue` performs the same complete revalidation
and proceeds only after the replacement lane passes. Aggregate status may
transition from retryable validity `INCONCLUSIVE` back to `RUNNING`, but valid
`FAIL`, statistical `INCONCLUSIVE`, `PASS`, and recorded-stop roots are
permanently closed.

No command resumes a `RUNNING` attempt, reopens a complete lane, deletes
evidence, or reuses a child or scratch root. Exit codes are 0 only for
aggregate PASS, 2 for validity inconclusive, 3 for a deterministic/statistical
failure, 4 for statistical inconclusive, and 5 only when `start` self-seals an
initialization-time `ABANDONED` root. Exit 5 also prints the stable status token
`ABANDONED_INITIALIZATION`; the matching PENDING_PRESERVATION index event and
seal are authoritative, so an operator never infers this branch from partial
files. `validate` without
`--require-pass` exits successfully for a structurally and cryptographically
valid terminal non-PASS root while printing its status; `--require-pass` is the
promotion gate. `record-index` atomically appends any validated terminal root,
including a validity `INCONCLUSIVE` on which retries are intentionally stopped,
and transitions its ACTIVE reservation to `PENDING_PRESERVATION`. It never
clears the global reservation or updates `current_evidence_root`.
`record-index --abandoned --confirm-no-live-processes` is reserved for an
uncatchable interruption of an otherwise active root. It also verifies all
recorded process groups are gone, rejects symlinks/special files, hashes every
preexisting regular file including partial `.log`/`.tmp` files without
deleting or renaming them, and atomically writes
`abandoned-inventory.json`. The terminal event hashes that seal.

`record-preserved` requires the pending root, durable index, and curated
worklog to exist byte-for-byte in a named preservation commit reachable from
`origin/codex/execution-engine-through-phase9`. It reads those Git blobs,
reruns structural reconstruction from that commit, and fetches the provided
permanent GitHub comment URL. The comment must be on issue #1436 and name the
preservation commit, exact root, exact candidate, and terminal status. Only
then may it append a `PRESERVED` event. A preserved PASS sets
`current_evidence_root`; a statistically complete PASS/FAIL/INCONCLUSIVE closes
the canonical experiment identity permanently, while a preserved validity
INCONCLUSIVE or abandonment permits a new unique root only for the same exact
candidate and experiment identity. Before any later `start`, the full local
index including its PRESERVED event must itself be committed and pushed: the
command fetches the branch and requires the remote index blob to equal the
local index byte-for-byte. Thus neither a local event nor a root commit without
its issue report unlocks measurement.

The append-only events retain exact candidate provenance,
`experiment_identity_digest`, root, status, root/ledger or abandonment-seal
digests, preservation commit, and issue URL; only a preserved PASS changes
`current_evidence_root`. Repeating exact events is idempotent; changing history
is rejected.
`validate --git-index` compares the exact root-owned manifest inventory with
the staged Git blobs, including ignored `Cargo.lock` and `*.log` paths, rejects
missing or extra staged root files, and can validate the reconstructed tree in
a fresh checkout. For an abandoned root it uses the complete abandonment seal
instead of the incomplete aggregate manifest, and requires the seal itself to
match the terminal index-event digest. It never treats the working-tree copy
alone as proof that evidence will survive a push.

- [ ] **Step 4: Run every Python unit test**

```bash
python3 -m unittest \
  scripts/test_phase2e_protocol.py \
  scripts/test_phase2e_build.py \
  scripts/test_classify_criterion_noninferiority.py \
  scripts/test_run_phase1_eager_campaign.py \
  scripts/test_run_phase2e_allocation_campaign.py \
  scripts/test_run_phase2e_gates.py \
  scripts/test_run_phase2e.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add .gitignore scripts/run_phase2e.py scripts/test_run_phase2e.py scripts/phase2e_protocol.py
git commit -m "feat(perf): orchestrate phase 2e evidence"
```

### Task 9: Verify tooling and freeze the evidence candidate

**Files:**

- Modify: `docs/superpowers/plans/2026-07-21-phase-2-cpu-domain-executor.md`
- Modify: `docs/superpowers/specs/2026-07-21-phase-2-cpu-domain-executor-design.md`
- Modify: `docs/superpowers/specs/2026-07-21-phase-2e-atomic-noninferiority-campaign-design.md`

- [ ] **Step 1: Run focused and repository verification**

```bash
test -z "$(git status --porcelain=v1)"
git fetch origin main
git merge --no-edit origin/main
test -z "$(git status --porcelain=v1)"
python3 -m unittest discover -s scripts -p 'test_*phase2e*.py' -v
python3 -m unittest \
  scripts/test_run_phase1_eager_campaign.py \
  scripts/test_classify_criterion_noninferiority.py -v
python3 scripts/phase2e_build.py verify-allocation-probe \
  --repository "$PWD"
git check-ignore -q docs/worklogs/.phase2e-index.lock
cargo fmt --all --check
cargo test -p tenferro-cpu
cargo test -p tenferro-ad
cargo clippy -p tenferro-cpu -p tenferro-ad --all-targets -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc -p tenferro-cpu -p tenferro-ad --no-deps
python3 scripts/check-doc-snippets.py --check
python3 scripts/test-doc-consistency.py
python3 scripts/test-check-docs-site.py
bash scripts/check-pr-fast.sh \
  --coverage-reviewed \
  --test 'python3 -m unittest discover -s scripts -p "test_*phase2e*.py" -v' \
  --test 'cargo test -p tenferro-cpu' \
  --test 'cargo test -p tenferro-ad'
git diff --check
```

Expected: every command succeeds.

- [ ] **Step 2: Commit any verification-only corrections**

```bash
# Inspect every correction and stage only its exact reviewed path; do not use
# directory-wide `git add` here.
git status --short
git diff --check
git diff --cached --check
git commit -m "chore(perf): finalize phase 2e tooling"
test -z "$(git status --porcelain=v1)"
```

Run the commit only after at least one exact path has been staged. If the tree
was already clean, omit the empty commit and record that no correction commit
was needed. Parent Task 11 commits all parent-owned docs and benchmark changes
before this child plan starts, so no earlier-phase file may remain dirty here.
The resulting clean HEAD is the candidate precursor. Step 3 checks that main
has not advanced since the full Step 1 verification and only then captures the
immutable candidate used by all evidence.

- [ ] **Step 3: Push the immutable candidate and update issue #1436**

```bash
test -z "$(git status --porcelain=v1)"
git fetch origin main
git merge-base --is-ancestor origin/main HEAD
test -z "$(git status --porcelain=v1)"
bash scripts/check-pr-fast.sh \
  --coverage-reviewed \
  --test 'python3 -m unittest discover -s scripts -p "test_*phase2e*.py" -v' \
  --test 'cargo test -p tenferro-cpu' \
  --test 'cargo test -p tenferro-ad'
python3 scripts/repository-rules-review.py \
  --base origin/main \
  --head HEAD \
  --output-json /tmp/phase2e-candidate-rules-review.json
PHASE2E_CANDIDATE="$(git rev-parse HEAD)"
test "${#PHASE2E_CANDIDATE}" -eq 40
git push origin codex/execution-engine-through-phase9
```

If the ancestry check fails because `origin/main` advanced after Step 1, do
not run only the fast gate. Return to Step 1, merge main, and rerun the complete
verification matrix before freezing a candidate.

Update the existing protocol-v2 comment on #1436 with the immutable candidate
SHA stored in `PHASE2E_CANDIDATE` and the verification results. Copy that exact
40-hex value into Task 10; after this point no evidence/worklog commit may
redefine the measurement candidate from `HEAD`. Do not start evidence
collection until the pushed candidate and issue agree.

### Task 10: Run the complete Phase 2E evidence campaign

**Files:**

- Create: `docs/worklogs/2026-07-21-phase-2e-noninferiority.md`
- Create: `docs/worklogs/2026-07-21-phase-2e-index.json`
- Create: `docs/worklogs/artifacts/2026-07-21-phase-2e-<candidate-short-sha>-run-<root-id>/phase2e-evidence.json`
- Create: `docs/worklogs/artifacts/2026-07-21-phase-2e-<candidate-short-sha>-run-<root-id>/evidence-ledger.json`
- Create: child evidence named by the aggregate manifest under the same root.

- [ ] **Step 1: Confirm the evidence root is absent and host is eligible**

```bash
PHASE2E_CANDIDATE="<exact 40-hex candidate recorded and pushed in Task 9>"
test "${#PHASE2E_CANDIDATE}" -eq 40
git cat-file -e "$PHASE2E_CANDIDATE^{commit}"
PHASE2E_CANDIDATE_SHORT="${PHASE2E_CANDIDATE:0:12}"
PHASE2E_ROOT_ID="0001"
PHASE2E_EVIDENCE_ROOT="docs/worklogs/artifacts/2026-07-21-phase-2e-$PHASE2E_CANDIDATE_SHORT-run-$PHASE2E_ROOT_ID"
PHASE2E_INDEX="docs/worklogs/2026-07-21-phase-2e-index.json"
PHASE2E_SCRATCH_PARENT="$(mktemp -d -p /tmp phase2e-scratch.XXXXXX)"
test ! -e "$PHASE2E_EVIDENCE_ROOT"
test -d "$PHASE2E_SCRATCH_PARENT"
df -h . /tmp
taskset -pc $$
python3 - <<'PY'
import os
print(os.getloadavg())
print(sorted(os.sched_getaffinity(0)))
PY
pgrep -af 'cargo|rustc' || true
```

Expected: the evidence root does not exist, at least one CPU is allowed, and
no overlapping build process remains when measurement starts. The scratch
parent is outside the repository and is never staged. If an unrecoverable
interruption later requires another outer root for the same candidate, first
take that abandoned root through the complete PENDING_PRESERVATION/PRESERVED
workflow in Steps 4-6, then retain the same literal `PHASE2E_CANDIDATE` and
`experiment_identity_digest` while incrementing `PHASE2E_ROOT_ID`; never
derive candidate identity from the new evidence-commit `HEAD` or reuse the old
path.

- [ ] **Step 2: Run the outer orchestrator once**

```bash
python3 scripts/run_phase2e.py start \
  --repository "$PWD" \
  --candidate "$PHASE2E_CANDIDATE" \
  --evidence-root "$PHASE2E_EVIDENCE_ROOT" \
  --index "$PHASE2E_INDEX" \
  --scratch-parent "$PHASE2E_SCRATCH_PARENT"
```

Expected: exit 0 only after both 168-process allocation comparisons, all
dispatch/47-row gates, and both 336-process timing campaigns pass. If validity
is inconclusive in a registered allocation or timing lane and the aggregate
manifest names that lane as retryable, retain the attempt and run:

```bash
python3 scripts/run_phase2e.py rerun-invalid-lane \
  --evidence-root "$PHASE2E_EVIDENCE_ROOT"
python3 scripts/run_phase2e.py continue \
  --evidence-root "$PHASE2E_EVIDENCE_ROOT"
```

The rerun command first revalidates every existing digest and creates a fresh
whole-lane attempt plus fresh external scratch; `continue` then proceeds from
the completed stage. Neither command resumes a `RUNNING` attempt. On FAIL or
statistical inconclusive, stop and preserve the result; do not rerun the
complete lane. If validity inconclusive will not be retried, treat it as the
terminal non-PASS branch below and seal it through `record-index`. A validity
failure during build, dispatch, or characterization is never lane-retryable:
record and preserve that complete outer root, then use a different empty root
id only after its PRESERVED event has been committed and pushed. If `start`
itself reports an initialization-time `ABANDONED`, it has already sealed the
root and appended `PENDING_PRESERVATION`; do not call `record-index` again.
It returns exit 5 plus `ABANDONED_INITIALIZATION`; verify the indexed seal and
proceed directly to Steps 4-6 for that root.

- [ ] **Step 3: Validate the aggregate root independently**

```bash
python3 scripts/run_phase2e.py validate \
  --evidence-root "$PHASE2E_EVIDENCE_ROOT"
PHASE2E_STATUS="$(python3 scripts/run_phase2e.py validate \
  --evidence-root "$PHASE2E_EVIDENCE_ROOT" --print-status)"
test "$PHASE2E_STATUS" = PASS
python3 scripts/run_phase2e.py validate \
  --evidence-root "$PHASE2E_EVIDENCE_ROOT" --require-pass
python3 scripts/run_phase2e.py record-index \
  --evidence-root "$PHASE2E_EVIDENCE_ROOT" \
  --index "$PHASE2E_INDEX"
```

Expected: exit 0 and aggregate status `PASS`; the index reservation is now
`PENDING_PRESERVATION`, not yet reusable and not yet
`current_evidence_root`.

For `FAIL` or terminal `INCONCLUSIVE`, omit `--require-pass`, run
`record-index` with the same root/index, write the negative worklog in Step 4,
commit and push it in Step 5, comment #1436 with the exact status and root, and
stop Phase 2E. For an uncatchable interruption that leaves `RUNNING`, first
verify no campaign process remains, then use:

```bash
python3 scripts/run_phase2e.py record-index \
  --evidence-root "$PHASE2E_EVIDENCE_ROOT" \
  --index "$PHASE2E_INDEX" \
  --abandoned \
  --confirm-no-live-processes
```

Take that abandoned root through Steps 4-6 before choosing a fresh
`PHASE2E_ROOT_ID`. The command must leave every partial file untouched, create
and index `abandoned-inventory.json`, and the later `--git-index` validation
must use that seal as the complete ownership inventory. Negative and abandoned
entries never set `current_evidence_root`.

- [ ] **Step 4: Write the worklog**

For an initialized campaign, record the candidate/direct/normalized commits,
all four lock digests, all ten build manifests, exact host/toolchain/config
facts, allocation observations, 47 characterization rows, both timing case
tables, every attempt including negative evidence, hardware skips, total wall
time, commands, and aggregate validator result. Link normative manifests
rather than copying classifier claims by hand.

For an initialization-time ABANDONED root, do not claim nonexistent later
stages. Record the exact failure point and error, candidate/experiment
identities, exit-5 status token, abandonment-seal digest and complete sealed
inventory, every artifact that actually exists, and an explicit `NOT_STARTED`
entry for locks, builds, allocation, dispatch, characterization, and timing.

- [ ] **Step 5: Commit the curated evidence**

Exclude non-normative Criterion scratch and external target directories. Add
the root/child manifests, copied estimates/logs required by their hash trees,
and worklog:

```bash
git add docs/worklogs/2026-07-21-phase-2e-noninferiority.md \
  "$PHASE2E_INDEX"
git add -f -- "$PHASE2E_EVIDENCE_ROOT"
python3 scripts/run_phase2e.py validate \
  --evidence-root "$PHASE2E_EVIDENCE_ROOT" \
  --git-index
git commit -m "docs: record phase 2e non-inferiority evidence"
bash scripts/check-pr-fast.sh \
  --base "$PHASE2E_CANDIDATE" \
  --no-fetch \
  --coverage-reviewed \
  --test 'python3 -m unittest discover -s scripts -p "test_*phase2e*.py" -v' \
  --test 'cargo test -p tenferro-cpu' \
  --test 'cargo test -p tenferro-ad'
python3 scripts/repository-rules-review.py \
  --base "$PHASE2E_CANDIDATE" \
  --head HEAD \
  --output-json /tmp/phase2e-evidence-commit-rules-review.json
git push origin codex/execution-engine-through-phase9
```

Because every Criterion root and build target is outside the repository, the
complete evidence root contains only curated normative evidence. Force-add is
restricted to that exact validated root because its normative `Cargo.lock`
and `*.log` paths match repository ignore rules; the Git-index validator must
prove no owned path is missing and no extra path is staged. For a
non-PASS or abandoned root use commit message `docs: record non-passing phase
2e evidence`; the same pre-push gate and committed-HEAD rules review are
mandatory because that branch does not proceed to Task 11.

- [ ] **Step 6: Report and durably confirm preservation**

After Step 5 has pushed, post one comment on #1436 naming the exact evidence
commit, root, literal measurement candidate, canonical experiment identity,
and terminal status. Capture its permanent comment URL, then prove both remote
preservation channels before unlocking the reservation:

```bash
PHASE2E_PRESERVATION_COMMIT="$(git rev-parse HEAD)"
PHASE2E_COMMENT_URL="<permanent https://github.com/tensor4all/tenferro-rs/issues/1436#issuecomment-... URL>"
python3 scripts/run_phase2e.py record-preserved \
  --evidence-root "$PHASE2E_EVIDENCE_ROOT" \
  --index "$PHASE2E_INDEX" \
  --preservation-commit "$PHASE2E_PRESERVATION_COMMIT" \
  --issue-comment-url "$PHASE2E_COMMENT_URL"
```

The command must fetch the branch, verify the named commit is reachable,
reconstruct the exact root/index/worklog blobs from that commit, and verify the
live #1436 comment body. Record the returned comment URL in the worklog, then
commit and push the PRESERVED index event:

```bash
git add docs/worklogs/2026-07-21-phase-2e-noninferiority.md "$PHASE2E_INDEX"
git diff --cached --check
git commit -m "docs: confirm phase 2e evidence preservation"
bash scripts/check-pr-fast.sh \
  --base "$PHASE2E_CANDIDATE" \
  --no-fetch \
  --coverage-reviewed \
  --test 'python3 -m unittest discover -s scripts -p "test_*phase2e*.py" -v' \
  --test 'cargo test -p tenferro-cpu' \
  --test 'cargo test -p tenferro-ad'
python3 scripts/repository-rules-review.py \
  --base "$PHASE2E_CANDIDATE" \
  --head HEAD \
  --output-json /tmp/phase2e-preserved-rules-review.json
git push origin codex/execution-engine-through-phase9
test -z "$(git status --porcelain=v1)"
```

Before any later `start`, the orchestrator verifies that the fetched remote
index blob contains this exact PRESERVED event. Only a preserved PASS proceeds
to Task 11. A preserved FAIL/statistical INCONCLUSIVE stops Phase 2E; a
preserved validity INCONCLUSIVE or ABANDONED may use a fresh root with the same
literal candidate and experiment identity.

### Task 11: Independent review, final verification, and issue handoff

**Files:**

- Modify: `docs/worklogs/2026-07-21-phase-2e-noninferiority.md`

- [ ] **Step 1: Request independent performance and rules reviews**

Reviewers must verify source/binary provenance, exact current-main versus
normalized roles, ledger closure, every child digest, allocation accounting,
47-row conformance, interval orientation, thresholds, negative evidence, and
the aggregate validator. Resolve every Critical, Important, and untracked
Minor finding with a fresh review.

- [ ] **Step 2: Invalidate evidence when a review fix changes experiment input**

If a fix changes Rust, benchmark/probe source, protocol, runner, classifier,
validator, build inputs, or any other path/constant owned by the canonical
experiment identity, preserve the old evidence root, commit the fix as a new
identity, and restart Tasks 9-10 with a new initially empty candidate-scoped
root. Do not run only the validator against evidence produced by the old
identity. A prose-only worklog/link correction under excluded
`docs/worklogs/**` changes no experiment identity and may continue to Step 3;
it must not authorize another measurement. A defect in a preserved normative
evidence artifact cannot be patched in place or used by itself to reopen the
closed identity: fix the owning generator/protocol so the canonical identity
changes, retain the defective root, and then repeat the entire campaign.

- [ ] **Step 3: Re-run final verification after review fixes**

```bash
PHASE2E_EVIDENCE_ROOT="$(python3 -c 'import json; print(json.load(open("docs/worklogs/2026-07-21-phase-2e-index.json"))["current_evidence_root"])')"
python3 -m unittest discover -s scripts -p 'test_*.py' -v
python3 scripts/phase2e_build.py verify-allocation-probe \
  --repository "$PWD"
cargo fmt --all --check
cargo test -p tenferro-cpu
cargo test -p tenferro-ad
cargo clippy -p tenferro-cpu -p tenferro-ad --all-targets -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc -p tenferro-cpu -p tenferro-ad --no-deps
python3 scripts/run_phase2e.py validate \
  --evidence-root "$PHASE2E_EVIDENCE_ROOT" \
  --require-pass
python3 scripts/repository-rules-review.py \
  --base origin/main \
  --head HEAD \
  --output-json /tmp/phase2e-rules-review.json
git diff --check
```

Expected: every command succeeds, the repository-rules JSON reports zero
unwaived findings, and reviewers report zero unresolved findings. Record the
review command, result, and JSON digest in the worklog.

- [ ] **Step 4: Commit review evidence**

```bash
git add docs/worklogs/2026-07-21-phase-2e-noninferiority.md
git diff --cached --check
git commit -m "docs: close phase 2e evidence review"
```

At this point only the exact prose worklog path above may remain. Any review
fix to scripts, Rust, a benchmark/probe, a protocol, a design, or another
normative artifact returns to Step 2 and creates a new candidate and complete
evidence run; do not stage it into this review-only commit. If review required
no worklog change, omit the empty commit.

- [ ] **Step 5: Review the final committed HEAD and push**

The repository-rules review in Step 3 precedes the optional worklog commit.
Run it once more against the actual committed head that will be pushed:

```bash
test -z "$(git status --porcelain=v1)"
PHASE2E_EVIDENCE_ROOT="$(python3 -c 'import json; print(json.load(open("docs/worklogs/2026-07-21-phase-2e-index.json"))["current_evidence_root"])')"
PHASE2E_CANDIDATE="$(python3 -c 'import json, os, sys; print(json.load(open(os.path.join(sys.argv[1], "phase2e-evidence.json")))["candidate"])' "$PHASE2E_EVIDENCE_ROOT")"
git fetch origin main
git merge-base --is-ancestor origin/main HEAD
python3 scripts/phase2e_build.py verify-allocation-probe \
  --repository "$PWD"
bash scripts/check-pr-fast.sh \
  --coverage-reviewed \
  --test 'python3 -m unittest discover -s scripts -p "test_*phase2e*.py" -v' \
  --test 'cargo test -p tenferro-cpu' \
  --test 'cargo test -p tenferro-ad'
python3 scripts/repository-rules-review.py \
  --base origin/main \
  --head HEAD \
  --output-json /tmp/phase2e-final-rules-review.json
git diff --check
test -z "$(git status --porcelain=v1)"
git push origin codex/execution-engine-through-phase9
```

If the ancestry check fails, stop before the remaining commands, merge
`origin/main`, and run `compare-experiment-identity` between the literal
measured candidate and the new HEAD. If the identities are equal (for example,
main changed only excluded worklogs), the preserved measurements still apply,
but rerun Task 11 Steps 3-5 against the merged committed HEAD. If they differ,
return to Task 9 Step 1, run its full matrix, freeze the new candidate, and
repeat all of Task 10 in a fresh root while retaining the already preserved
root:

```bash
git merge --no-edit origin/main
python3 scripts/run_phase2e.py compare-experiment-identity \
  --repository "$PWD" \
  --first "$PHASE2E_CANDIDATE" \
  --second HEAD
```

Do not use an excluded-path-only main advance to reopen a closed experiment,
and do not reuse old evidence after a changed canonical identity.

Expected: zero unwaived findings on the exact committed `HEAD`, a clean tree,
and a successful push. Report this final review result in the issue handoff;
do not amend the reviewed commit afterward.

- [ ] **Step 6: Update #1436 and #1433**

Comment on #1436 with commit, tests, aggregate PASS, worklog, root manifest,
and reviewer results. Update #1433's Phase 2 row only after all Phase 2 exit
criteria pass. Keep production BLAS build-mode discovery/scoped controllers
listed as a post-Phase-9 unresolved decision; do not implement or silently
close it in Phase 2E.
