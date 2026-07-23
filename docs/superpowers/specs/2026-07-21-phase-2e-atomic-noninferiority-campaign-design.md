# Phase 2E Atomic Non-Inferiority Campaign Design

**Parents:** [#1433](https://github.com/tensor4all/tenferro-rs/issues/1433),
[#1436](https://github.com/tensor4all/tenferro-rs/issues/1436)

**Implementation candidate branch:** `codex/execution-engine-through-phase9`

**Immutable implementation baseline:**
`85855e272b1495611deb601a9ee06f3546772c3c`

**Frozen benchmark harness:**
`4471d6145c4d8793de3a96f8d99400c24ca8c6d1`

**Original lower-layer revision:**
`strided-rs@10fc972d3c0f8cdfd4ecb45d21d815aebfd7d1f2`

**Common lower-layer revision:**
`strided-rs@6b0b4a46b7dd9a9ea1677a0d596c0b4adab1acbc`

## Status and scope

This child design defines the Phase 2E evidence gate for the CPU execution
domain and provider migration. It replaces the current pair-local retry loop
with atomic whole-campaign runs and separates a direct current-main gate from
a common-lower-layer attribution guard.

The phase measures non-inferiority. It does not require a positive eager
speedup and does not authorize post-hoc threshold changes. Production BLAS
build-mode discovery and scoped thread controllers are intentionally outside
this phase and remain unresolved until after Phase 9.

Phase 2E contains four deliverables:

1. an atomic 28-case timing campaign runner and validator used for both
   required comparisons;
2. reproducible direct-current-main and common-lock normalized baselines;
3. allocation, source-contract, and entry-count gates over the same eager
   surface; and
4. placement-bound and multi-thread characterization with gating conformance
   checks and non-gating latency values.

## Authoritative synchronization

This protocol-v2 child design supersedes only the Phase 2 design's
`Performance Gate` section and the Phase 2 implementation plan's Task 11
performance-campaign procedure (now Step 6). It does not change the accepted
CPU domain/executor semantics. The Phase
2 design, plan, and issue #1436 must link this document and state the same dual
gate, atomic retry, provenance, and acceptance rules before evidence-tool
implementation begins. Issue #1433 remains the umbrella; #1436 owns Phase 2
acceptance and receives the implementation and evidence updates.

## Chosen approach

The runner is rewritten around a whole-campaign state machine. One invocation
owns all 28 cases and their three ordered pairs. It never retries a pair and
never carries a valid pair into another invocation. The orchestrator invokes
it once for the end-to-end current-main gate and, only after that gate passes,
once for the common-lock attribution guard. Each invocation is independently
indivisible; Phase 2E requires both to pass.

This was selected over two alternatives:

1. Wrapping the current runner with `--max-attempts=1` would stop pair-local
   retries but would still leave partial manifests without a terminal campaign
   classification and would retain retry-oriented artifact semantics.
2. A manual protocol would rely on operator discipline rather than making
   selective reuse structurally impossible.

The atomic runner directly enforces the repository's performance-gated
experiment protocol and produces a terminal artifact even when a validity
gate fails.

## Campaign unit and state machine

The indivisible unit for each of the two comparison kinds is:

- 28 canonical cases;
- three pair orders per case: `A/B`, `B/A`, `A/B`;
- four benchmark processes per pair: candidate sentinel before, two target
  runs in the declared order, and candidate sentinel after; and
- 84 pairs / 336 benchmark processes for a complete valid campaign.

`campaign.json` has an explicit state:

```text
RUNNING -> COMPLETE
        -> INCONCLUSIVE
```

This state-machine and build-provenance contract is protocol version 2. The
runner and classifier reject version-1 manifests rather than interpreting
their pair-local retry semantics as atomic evidence.

The manifest stores these as separate fields. `validity_state` is `RUNNING`,
`COMPLETE`, or `INCONCLUSIVE`; `statistical_result` is null until validity is
`COMPLETE` and is then `PASS`, `FAIL`, or `INCONCLUSIVE`. A validity
`INCONCLUSIVE` records the first invalid case/pair/role plus its typed reason
and is never passed to the statistical classifier. A valid finalization
atomically stores the campaign result, every case result, the rendered
`classification.json` and `summary.md` digests, and the exact normative input
inventory in `campaign.json`.

The runner returns exit code 0 only for validity `COMPLETE` and statistical
`PASS`. It returns 2 for validity `INCONCLUSIVE`, 3 for a valid statistical
`FAIL`, and 4 for a valid statistical `INCONCLUSIVE`. Exit code 1 is reserved
for failure before a manifest can be created or for failure to atomically
record a terminal result. Recoverable benchmark, artifact, monitoring, and
validation failures must become terminal validity `INCONCLUSIVE` first. Thus
shell and CI callers cannot promote a non-passing campaign by checking only
the process status.

The manifest is atomically rewritten after every completed process and pair.
A catchable exception after manifest creation is converted to terminal
validity `INCONCLUSIVE`. An uncatchable interruption such as `SIGKILL` may
leave `RUNNING`; that manifest is evidence of an incomplete campaign and
cannot be resumed. A new run requires a different empty artifact root and a
different empty Criterion root.

## Fresh-root contract

Both roots are caller-supplied and exclusive to one invocation:

- `--artifact-root` must not exist or must be empty;
- `--criterion-root` must not exist or must be empty;
- the two resolved paths must be distinct and neither may contain the other;
- the runner creates both roots and sets `CRITERION_HOME` to the Criterion
  root for every benchmark process; and
- an existing `campaign.json`, partial case directory, Criterion baseline, or
  unrelated file rejects startup before a benchmark is launched.

The runner never deletes either root. A rerun uses newly created paths. This
makes artifact reuse and stale Criterion baselines impossible without an
explicit violation outside the runner.

The Criterion root is always created beneath a caller-supplied scratch parent
outside both the repository and the outer evidence root. Its resolved path is
recorded for diagnostics, but neither the directory nor its contents are
normative evidence. Every estimate used by the classifier is copied into the
attempt artifact root before it is hashed. This separation makes staging the
complete evidence root safe and prevents a later broad documentation add from
accidentally committing Criterion scratch.

## Outer evidence-root lifecycle

The fresh-root rule above applies to each timing or allocation *attempt*. The
outer candidate evidence root has a different, explicit lifecycle. `start` is
the only command that accepts a nonexistent or empty outer root; it creates
`phase2e-evidence.json`, `evidence-ledger.json`, and the root lock before any
child process. A validity-inconclusive child may then be rerun inside that
same outer root so the ledger can retain the failed attempt and the successful
replacement together.

Before `rerun-invalid-lane` or `continue` changes an existing root, the
orchestrator takes the global index lock and then the root lock, in that fixed
order, and revalidates all of the following from
disk: candidate and protocol identities, the aggregate manifest, the complete
ledger, every registered child manifest and digest, every root-owned lock and
build manifest, the matching `ACTIVE` reservation, the current stage and lane,
and the absence of an open `RUNNING` attempt. Every command that needs both
locks uses index-then-root; no reverse acquisition is permitted. Any mismatch
rejects the command without changing a file.
`rerun-invalid-lane` is legal only when the current lane's last attempt is
validity `INCONCLUSIVE`, the lane has no statistically complete attempt, and
the root has not been recorded as closed in the durable index. It allocates
the next attempt id, a new empty child artifact directory, and a new external
Criterion scratch directory before returning the aggregate root from
retryable `INCONCLUSIVE` to `RUNNING`. `continue` is legal only after that
whole replacement attempt passes; it never reuses a partial result.

A valid `FAIL` or statistical `INCONCLUSIVE` closes the lane and the outer
root permanently. A `PASS` closes the lane and advances the fixed stage order.
Only allocation and timing lanes have the whole-lane retry transition. A
validity failure during lock/build/probe construction, dispatch, or
characterization closes that outer root as non-retryable `INCONCLUSIVE`; after
it is indexed and preserved, reconsideration starts from a different empty
outer root and repeats every stage. This prevents successful setup or
conformance artifacts from being selectively carried across an incomplete
root.

Recording a retryable validity `INCONCLUSIVE` in the durable index means the
operator has chosen to stop; it seals the root against later reopening. No
command resumes a child or outer `RUNNING` state. If an uncatchable
interruption leaves such a state, the root is retained read-only and indexed
as `ABANDONED`; after verifying that no campaign process remains, any new run
uses a different initially empty outer-root path even when the candidate is
unchanged. Thus recovery never edits, deletes, or silently replaces incomplete
evidence.

## Validity failure semantics

Before a pair, the runner may wait for the host to become quiet. Waiting before
measurement is not a retry and does not consume or select data.

Normalized load is exactly `getloadavg()[0] / len(sched_getaffinity(0))`, where
the denominator is the process-allowed CPU count before pinning. Load and
overlapping `cargo`/`rustc` processes are sampled immediately before and after
every benchmark process and once per second while it runs. Affinity is sampled
at the same endpoints and cadence.

The quiet-host wait is bounded to 300 seconds per pair and polls once per
second. Each benchmark process has a 30-second wall-clock deadline. A process
timeout terminates its process group, waits up to five seconds, then kills any
survivors. Either deadline produces terminal validity `INCONCLUSIVE` with the
observations and process outcome recorded; no wait or benchmark can block the
campaign indefinitely.

After measurement starts, any of the following terminates the whole campaign
as validity `INCONCLUSIVE`:

- benchmark process non-zero exit or incomplete termination;
- observed CPU affinity different from the selected singleton CPU;
- normalized one-minute load above 0.25 at any monitor sample;
- overlapping `cargo` or `rustc` process at any monitor sample;
- missing or malformed Criterion estimate;
- binary, lock, validity, or estimate digest mismatch;
- candidate/candidate sentinel confidence interval wholly outside
  `[-5%, +5%]`;
- unexpected case, pair, role, order, or process count; or
- a quiet-wait or benchmark-process timeout; or
- an exception after manifest creation that prevents a complete pair.

The runner records already completed cases but does not classify or promote
them. It does not continue to collect the remaining cases after the campaign
is invalid, because no later result can repair the indivisible experiment.

No CLI retry count exists. Reconsideration requires a complete rerun with the
same protocol and fresh roots. A valid statistical `FAIL` or `INCONCLUSIVE`
also remains final evidence for that invocation and is not automatically
rerun.

Every allocation or timing invocation is registered before its first measured
process in one candidate-scoped `evidence-ledger.json`. The ledger has ordered
allocation and timing stages, each with `direct-current-main` and
`common-lock-normalized` lanes, and retains every attempt id, fresh roots,
candidate/protocol identity, terminal result, and attempt-manifest digest. A
validity-inconclusive attempt permits only a whole-campaign rerun in the same
lane with unchanged protocol and candidate, using the next id and fresh roots.
A gate `FAIL` or statistical `INCONCLUSIVE` closes that lane and the overall
evidence as non-passing. A `PASS` closes its lane; only direct-current-main
`PASS` opens the corresponding normalized lane, and both passes complete that
stage. No complete lane may be rerun to search for a favorable sample. A code
change requires a new candidate commit and evidence ledger; all previous
ledgers and negative results remain linked from the worklog.

## Immutable comparison contract

The direct comparison measures the complete user-visible change. The
normalized comparison then isolates execution-engine changes with the
required lower-layer infrastructure held identical.

### Candidate

The candidate is a clean commit on
`codex/execution-engine-through-phase9` after all Phase 2E evidence code is
integrated. The candidate commit, benchmark binary SHA-256, rustc/cargo
versions, target, profile, and complete `Cargo.lock` SHA-256 are recorded
before measurement.

### Direct current-main baseline

The primary end-to-end gate preserves tenferro implementation source and all
five original strided pins from
`85855e272b1495611deb601a9ee06f3546772c3c`. Its dedicated measurement commit
may add only the benchmark target stanza and benchmark source from
`4471d6145c4d8793de3a96f8d99400c24ca8c6d1`. Before any timing or allocation
measurement, the orchestrator generates and freezes this baseline's own lock
under the declared toolchain. The baseline and candidate lock digests differ
by design in this comparison: this is the user-visible end-to-end test that
includes the required strided update and prevents a lower-layer regression
from being normalized away.

### Common-lock normalized baseline

The baseline preserves tenferro implementation source from
`85855e272b1495611deb601a9ee06f3546772c3c`. A dedicated normalization commit
may contain only:

- the five root `Cargo.toml` strided revisions changed from
  `10fc972d3c0f8cdfd4ecb45d21d815aebfd7d1f2` to
  `6b0b4a46b7dd9a9ea1677a0d596c0b4adab1acbc`;
- the frozen benchmark target stanza from
  `4471d6145c4d8793de3a96f8d99400c24ca8c6d1`; and
- the frozen
  `crates/tenferro-ad/benches/eager_dispatch_baseline.rs` from that same
  harness commit.

No other baseline source may change. This is a required attribution guard, not
a replacement for the direct current-main gate: it prevents a lower-layer
speedup from masking execution-engine overhead. A verifier compares both
measurement commits against the immutable implementation baseline and rejects
extra paths or extra hunks.

### Lock contracts

The orchestrator generates the common lockfile from the candidate and the
original-pin lock from the direct baseline before any probe or benchmark. It
copies their normative bytes to
`builds/locks/common.Cargo.lock` and
`builds/locks/direct-current-main.Cargo.lock` under the evidence root. The
root manifest owns both files and their SHA-256 digests. Byte-identical common
lock bytes are installed into the candidate and normalized-baseline
worktrees; the direct baseline receives the root-owned original-pin lock. All
three worktrees must pass with their declared copy:

```bash
cargo metadata --locked --format-version 1
```

One build orchestrator creates all three binaries and their manifests. It uses
the requested command `cargo bench --locked --no-run -p tenferro-ad --bench
eager_dispatch_baseline --no-default-features --features cpu-faer`, the Cargo
`bench` profile, separate fresh external `CARGO_TARGET_DIR` values, and the
same host target.

The build process starts from an empty environment rather than inheriting the
operator's shell. Its allowlist contains only controlled absolute tool paths,
a minimal `PATH`, an empty temporary `HOME`, `LC_ALL=C`, `TZ=UTC`, the
role-specific `CARGO_TARGET_DIR`, `CARGO_INCREMENTAL=0`,
`CARGO_NET_OFFLINE=true`, and the five thread-count variables fixed to one.
It uses a controlled `CARGO_HOME` with pre-seeded registry/git cache data but
no config or credentials. The orchestrator rejects untracked ancestor or
repository `.cargo/config*`; a tracked repository config is permitted only
when byte-identical for all three roles and named by digest. Consequently no
ambient `CARGO_PROFILE_*`, `CARGO_TARGET_*`, `CARGO_BUILD_*`, compiler/linker,
pkg-config, CMake, make, wrapper, or flags variable reaches a build.

The orchestrator creates dedicated fresh build worktrees for these checks.
Before and after each build, tracked candidate source must be byte-identical
to its declared commit. Its ignored `Cargo.lock` is a separate controlled
input and must equal the root-owned common lock. Each baseline worktree may
differ from its declared measurement commit only at its ignored `Cargo.lock`,
which must equal the appropriate root-owned lock. No worktree may have another
tracked, untracked, or ignored file. The verifier combines
`git status --porcelain=v1 --untracked-files=all` with an ignored-file
inventory and a filesystem allowlist containing only `.git`, tracked paths,
and the one root-owned `Cargo.lock`; ignored files are not assumed clean.
Build products live outside all worktrees. A tracked-file content digest, the
allowed lock input, HEAD, both worktree inventories, benchmark source digest,
benchmark target stanza digest, actual argv and full constructed environment,
Cargo-config-chain proof, rustc/cargo verbose versions, host target, profile,
requested root feature tuple, selected `Faer` provider identity, role-specific
resolved feature graph digest, lock digest,
worktree/target-dir identity, executable path, and executable SHA-256 are
written directly from the build process to each read-only build manifest.

The timing graph is generated with the same lock, target, package, defaults,
and feature tuple as the build:

```text
cargo tree --locked --target <host-target> -p tenferro-ad \
  --no-default-features --features cpu-faer -e features
```

Using a workspace-wide or default-feature `cargo tree` result is a provenance
failure even when it happens to list the same leaf dependencies.

The validator separates invariants from permitted role differences:

- toolchain, host target, bench profile, requested `cpu-faer` feature tuple,
  selected provider, benchmark source/stanza, command template, controlled
  environment except paths, and Cargo-config proof must be identical;
- HEAD, tracked source digest, resolved feature graph, worktree,
  `CARGO_TARGET_DIR`, executable path/digest, and direct-versus-common lock are
  role-specific and must match that role's predeclared source and root-owned
  lock; and
- the normalized baseline and candidate must share the common lock, while the
  direct baseline must use the original-pin lock. No other difference is
  accepted.

The runner records its comparison kind, declared lock paths and SHA-256,
candidate commit, immutable implementation baseline, direct measurement
commit, normalization commit, harness commit, and both strided revisions. All
binaries are rejected unless their build manifests match every field above
and still hash to their executable digests. The normalized comparison also
requires byte-identical candidate/baseline common locks; the direct comparison
requires the exact root-owned original-pin baseline lock and candidate common
lock. The invariant fields above must be identical in either comparison.

The timing runner requires `--comparison-kind` (`direct-current-main` or
`common-lock-normalized`), `--baseline-build-manifest`, and
`--candidate-build-manifest`. Both manifests must already exist outside the fresh
timing and Criterion roots, must validate against protocol version 2, and are
read-only inputs. Their resolved paths and SHA-256 digests are frozen into
`campaign.json` before the first benchmark process. The runner also verifies
the tracked-file and worktree-state proofs again immediately before the
campaign and verifies that each declared executable hashes to the digest in
its owning manifest. Callers cannot override provenance fields independently
on the command line.

The same sealed-environment constructor is used for lock generation, `cargo
metadata`, `cargo tree`, every build, all benchmark and allocation-probe
processes, and characterization. Runtime processes start from an empty
environment with the controlled `PATH`, empty `HOME`, `LC_ALL=C`, `TZ=UTC`,
the five count variables, `OMP_DYNAMIC=FALSE`, and `MKL_DYNAMIC=FALSE`;
Criterion processes additionally receive only their fresh `CRITERION_HOME`.
Engine-owned multi-thread characterization still leaves every ambient/vendor
thread variable at one because its explicit executor owns `B`. Process-local
artifact identifiers are explicit allowlist entries. Variables such as
`LD_PRELOAD`, `GLIBC_TUNABLES`, `RAYON_RS_NUM_CPUS`, target-specific flags, and
all unlisted shell state are absent. Every process record stores the complete
sorted constructed environment and its digest, not only selected variables.

All subprocess classes have predeclared wall-clock deadlines: 1,800 seconds
for a Cargo build, 300 seconds for lock generation/metadata/feature queries,
30 seconds for one allocation or Criterion case, and 120 seconds for the
dispatch/characterization correctness test. A timeout terminates its process
group, waits five seconds, kills survivors, records stdout/stderr and the typed
deadline, and makes the owning gate validity `INCONCLUSIVE`. No build, probe,
or characterization process may hang the autonomous campaign indefinitely.

## Timing matrix and classification

The frozen matrix covers lazy and materialized execution for:

- `neg` and `add` at sizes 1, 8, and 64;
- `reduce_sum` at sizes 1, 8, and 64;
- `slice` at sizes 1, 8, and 64; and
- `dot_general` at square sizes 1 and 2.

Criterion settings are fixed at 2 seconds warm-up, 5 seconds measurement,
100 samples, and 95% confidence. Thread environment is fixed to one thread
for Rayon, OpenMP, OpenBLAS, MKL, and Accelerate. `RUSTC_WRAPPER` is unset and
`CARGO_INCREMENTAL=0` during all builds and runs.

For each case, after orienting the middle `B/A` interval back to `A/B`:

- `PASS`: all three mean relative-change confidence-interval upper bounds are
  at most +5%;
- `FAIL`: at least two lower bounds exceed +5%; and
- otherwise `INCONCLUSIVE`.

The campaign is `PASS` only when all 28 cases pass. Any case `FAIL` makes the
campaign `FAIL`; otherwise at least one statistical `INCONCLUSIVE` makes the
campaign `INCONCLUSIVE`.

## Allocation and dispatch gates

Timing alone is insufficient. An external, frozen allocation-probe crate lives
under `scripts/phase2e/allocation-probe/`; it is not copied into any
tenferro measurement worktree. Its source and manifest-template SHA-256 values
are fixed in the root evidence manifest. The orchestrator generates three
probe manifests whose manifest-template substitution is limited to the
canonical tenferro path, links one probe executable to each tenferro build with
the frozen dependency/feature contract below, and records the probe
binary, resolved feature graph, tenferro source, lock, toolchain, profile, and
sealed build-environment digests.

The orchestrator separately generates and stores
`builds/locks/direct-current-main-probe.Cargo.lock` and
`builds/locks/common-probe.Cargo.lock`. The latter is byte-identical input for
the candidate and normalized probe builds. Each probe is built with
`cargo build --locked --profile bench --manifest-path <generated>/Cargo.toml`
in a fresh role-specific target directory using the same empty environment,
controlled Cargo home, config-chain proof, absolute toolchain, host target,
and invariant-versus-role validator as the timing binaries. Probe requested
dependencies/features, source/template digests, allocator implementation, and
profile are invariant; generated manifest path, tenferro source identity,
resolved graph, root-owned probe lock, target directory, and binary digest are
role-specific. Probe build-manifest validation has the same dirty-source,
actual-argv/environment, lock-byte, and executable re-hash requirements as the
timing build validation.

Each probe's resolved graph is produced with its actual generated manifest,
root-owned lock, and host target:

```text
cargo tree --locked --manifest-path <generated>/Cargo.toml \
  --target <host-target> -e features
```

The generated manifest has exactly three direct dependencies at canonical
repository-root-relative paths: `crates/tenferro-ad`,
`crates/tenferro-cpu`, and `crates/tenferro-tensor`. All three set
`default-features = false`; AD and CPU enable exactly
`features = ["cpu-faer"]`, while tensor enables no features. The tracked
template contains one kind of repository-root token, substituted only as
TOML-safe string content. The verifier parses the generated TOML and rejects
wrong dependency names, paths, default-feature settings, feature lists,
foreign roots, paths outside the repository, or noncanonical spellings.

Every generated probe is a unique external temporary crate. The verifier
byte-copies the tracked `src/main.rs` and `src/tests.rs` into its own `src/`,
rejecting symlinks, special files, extra inventory, and digest drift before or
after verification. Only `Cargo.toml` is generated; a `[[bin]]` entry may not
point back into tracked source. Build evidence binds the template plus tracked
and generated source digests. `cargo test` creates the generated
`Cargo.lock`; subsequent locked clippy and build commands must leave its bytes
unchanged. Generated source, target, HOME, and controlled Cargo-cache state are
cleaned on success, nonzero exit, timeout, `KeyboardInterrupt`, and
`SystemExit`, without replacing the primary failure.

The probe also has a nonmeasurement `--list-cases` mode. It prints the compact
JSON array of all 28 canonical case strings in protocol order followed by one
LF, emits no other stdout, and exits 0. Rust tests bind every name to its exact
operation, mode, and size. The Python verifier executes the bench-built binary
and byte-compares its inventory and order with
`tuple(protocol.CANONICAL_CASES)`. Unknown arguments exit 1, emit no allocation
record, and diagnose only on stderr.

The probe uses one fixed `System`-allocator counting wrapper and launches a
fresh single-thread process for each case. Inputs and runtime are constructed,
then 256 warm-up operations run; every warm-up output is consumed and dropped
and all temporaries settle before the allocation counters reset. Exactly 4,096
operations then run inside the measured region. Each measured input and output
is black-boxed, every output is dropped within its iteration, and only a
primitive checksum accumulator survives. The final snapshot is taken
immediately after the loop and output drops; recording is disabled before any
serialization or printing, so output allocations are excluded. Each
comparison runs six processes per case in the fixed
orders `A/B`, `B/A`, `A/B`, producing three observations per binary across one
indivisible 28-case matrix. A crash, missing case, unequal repetition count,
or inconsistent within-binary count is a whole-probe validity `INCONCLUSIVE`,
never a case-local retry.

Each comparison therefore owns exactly 168 probe processes: two roles times
three ordered sets times 28 cases. Its fresh manifest transitions from
`RUNNING` to validity `COMPLETE` plus gate `PASS`/`FAIL`, or to validity
`INCONCLUSIVE`; exit codes are 0, 3, and 2 respectively. An invalid process
stops the complete comparison, and a rerun requires the same ledger rules and
a new whole-comparison root. The candidate probe build manifest may be shared
as immutable build provenance, but no candidate observation, case result, or
partial process set is reused between the direct and normalized comparisons
or between attempts.

For one allocator call, a non-null return from `alloc` or `alloc_zeroed`
increments allocation count by one and allocated bytes by the requested
`Layout::size()`. A non-null `realloc` increments count by one and bytes by
`new_size`, including an in-place reallocation. `dealloc` never subtracts from
either monotonic counter. A null result for a nonzero request increments a
separate failure counter but not count/bytes; any failure or checked `u64`
counter overflow makes the process invalid. The wrapper delegates each event
to `System` exactly once and the reference tests cover allocation,
zero-initialized allocation, in-place-or-moving reallocation, deallocation,
failure injection, reset, and overflow without relying on optimizer-elidable
allocations.

Allocation count, allocated bytes, and failures are independent `AtomicU64`
counters. Updates are checked and never wrap: an overflowing counter remains
unchanged, `counter_overflow` becomes true, and any successful updates to the
other counters remain visible as invalid evidence. Tests independently seed
each overflow and cover zero-delta state transitions only through the safe
state-machine API. Reset and snapshot are protocol-defined quiescent points
with no concurrent allocator calls. Relaxed atomic ordering is sufficient for
the counters but does not make concurrent reset or snapshot valid.

Every unsafe allocator delegation is an explicit `unsafe` block with an
adjacent boundary-specific `// SAFETY:` note. The note states that the
`GlobalAlloc` caller's pointer/layout contract is forwarded unchanged to the
same `System` allocator, and that the wrapper never dereferences or changes the
returned pointer; it classifies the returned null/success result before
updating counters. The implementation does not rely on the implicit unsafe
context of an `unsafe fn`. Boundary tests use only nonzero valid layouts,
non-null live pointers, and positive realloc sizes. They never allocate a
zero-sized layout, reallocate or deallocate null, or deallocate an invalid
pointer. One valid lifecycle covers `alloc`, `alloc_zeroed`, `realloc`, and
`dealloc`. Injected null is tested only for `alloc`, `alloc_zeroed`, and
`realloc`; after failed realloc the original pointer remains live and is
deallocated exactly once with its original layout. `dealloc` has no failure
result or injected-failure test.

Measured consumption is allocation-free. `shape_token` starts from rank and
folds every dimension using wrapping base 257, with exact-value tests. Lazy
cases black-box `shape()`, compute the shape token, invoke and match
`tensor_read()`, black-box its tensor, and add storage tag `Tensor = 1` or
`View = 2`. Materialized cases call `materialized()`, black-box its shape and
first `f64`, and combine that value with the shape token. Both tokens must be
finite. Iteration `i` performs
`checksum += token * ((i + 1) as f64)` and black-boxes the accumulator. Exact
checksums for all 28 cases are frozen using triangular factor
`4096 * 4097 / 2` and known outputs/tags.

A measured run emits exactly one compact JSON object, with lexicographically
sorted keys, followed by one LF and no other stdout:
`allocation_count: u64`, `allocated_bytes: u64`,
`allocation_failures: u64`, canonical-string `case`, finite-f64 `checksum`,
`counter_overflow: bool`, and `repetitions: 4096`. Unknown keys are forbidden.
Exit 0 requires zero failures, false overflow, and all run checks valid.
Failure or overflow emits the same record and exits 2. Unknown case or argv,
runtime/tensor failure, or serialization failure emits no stdout, writes a
diagnostic to stderr, and exits 1. Task 6 revalidates exact JSON types (a bool
is not an integer), checksum finiteness, repetition count, canonical case,
one-line framing, and record/exit-code consistency.

Because the generated probe crate lives outside the Cargo workspace, its
dedicated verifier creates a fresh generated crate and runs, in order,
`cargo fmt --check` with a 300-second deadline, complete tests with 1,800
seconds, locked all-target clippy with `-D warnings` with 1,800 seconds, locked
bench-profile build with 1,800 seconds, and the built binary's `--list-cases`
with 30 seconds. Every subprocess starts a new process group and uses the
existing bounded runner: timeout or control exception sends TERM, waits five
seconds, then KILLs and reaps the group. Verification stops at the first
failure. Fake-process tests cover all five commands, exact environment and
provenance, nonzero exit, timeout, cleanup, control exceptions, and
first-failure stop; real commands do not replace these orchestration tests.
The requested Cargo profile is `bench`; Cargo's built-in bench profile places
the actual probe executable at
`$CARGO_TARGET_DIR/release/phase2e-allocation-probe`, whose regular-file
identity and digest are bound before and after `--list-cases`.
These checks run when the probe is implemented, when the evidence candidate is
frozen, and again on the final committed head; workspace-wide fmt/clippy is
not treated as coverage for this external crate.

The direct-current-main and common-lock comparisons have separate atomic
allocation manifests. For every observation of every case, candidate
allocation count and allocated bytes must be no greater than that comparison's
baseline. Both comparisons must pass; the normalized result cannot replace a
direct-current-main failure.

A source-contract test follows the complete path from `EagerTensor` through
`EagerRuntime`, backend session, `CpuExecSession`, and provider dispatch. The
steady-state path must not contain string-key lookup, `TypeId`, `Any`
downcasting, `HashMap` lookup, or formatting-based dispatch. Registration,
construction, diagnostics, and error-only formatting remain outside the
measured boundary.

Independent test-only counters record the canonical family inventory at size
8 (`neg`, `add`, `reduce_sum`, and `slice`) and size 2 (`dot_general`) on both
the direct `CpuBackend` and ordinary materialized eager surfaces in one-thread
sequential mode. They record:

- backend session entries;
- CPU operation-scope entries;
- resource-arbiter permits;
- executor `install` calls;
- executor indexed `submit` calls; and
- provider calls.

The required vector `(backend session, operation scope, permit, install,
submit, provider)` is `(0, 1, 1, 1, 0, P)` for direct and
`(1, 1, 1, 1, 0, P)` for ordinary eager, where `P = 1` for `dot_general` and
`P = 0` for the four native families. Lazy graph construction is outside this
counter window; materialization must produce exactly that vector. A
placement-bound eager callback similarly enters one backend session and one
permit for the callback, and each single core operation uses the existing CPU
operation scope with exactly one executor entry and no nested session,
additional permit, second install, or second fan-out layer. Test-only
instrumentation is compiled out of production builds.

The evidence is split without creating a crate cycle. CPU module-local tests
measure the five direct vectors and the five downstream vectors inside one
borrowed `CpuBackend::with_backend_session`. AD module-local tests extend the
existing test-only recording backend and prove that ordinary `EagerTensor`
materialization enters exactly one such backend session for each family. As
part of that touch, the complete `RecordingBackend` fixture, delegate macro,
trait implementations, constructor, and counters move from production
`eager_backend.rs` to module-local `eager_backend/tests.rs`; the production
module retains only the minimal `#[cfg(test)] mod tests;` declaration, private
test-module type reference, and enum dispatch arms required to compile the
test-only variant. No substantive fixture or test implementation remains
inline. The
dispatch manifest composes those two hashed test artifacts into the five eager
vectors; `tenferro-cpu` never depends on `tenferro-ad`.

Evidence execution never invokes `cargo test` as an unbound build-and-run
operation. The build orchestrator uses the immutable candidate worktree, the
root-owned common lock, a fresh external target directory, and the sealed build
environment to run these two build commands with JSON messages:

```text
cargo test --locked --no-run -p tenferro-cpu --lib \
  --no-default-features --features cpu-faer --message-format=json
cargo test --locked --no-run -p tenferro-ad --lib \
  --no-default-features --features cpu-faer --message-format=json
```

It identifies each emitted test executable from Cargo's messages, hashes it,
and writes `dispatch-gates/{cpu,ad}-test-build.json`. The manifests bind the
candidate source/tree inventory, common lock, exact build argv, full sealed
environment, host target, toolchain, profile, executable, and the matching
package-specific resolved feature graph generated with `--locked`,
`--target <host-target>`, `--no-default-features`, and `--features cpu-faer`.
The gate wrapper accepts only those manifests, re-hashes the binaries, and
runs the test executables directly with the exact filter and `--nocapture`
arguments under the sealed runtime environment. Each binary has the
predeclared 120-second deadline and five-second process-group kill grace.
Development-time RED/GREEN commands may use `cargo test`; they are not
measurement evidence.

## Placement and parallelism characterization

The canonical matrix is generated for every budget `B` in `{1, 2, 4}`
and each ownership fixture: managed exact, external exact-declared, and
external advisory-declared. Managed uses the first usable NUMA node and its
process-allowed CPU subset. Both external fixtures use an instrumented
caller-owned Rayon executor with `worker_count = B`, outer support, Rayon
inner support, rejected re-entry, and caller-owned shutdown. Real-hardware
rows declare the first `B` allowed CPUs. When fewer than `B` CPUs are
available, topology-independent correctness/count/recovery fixtures instead
declare all `min(B, available)` nonempty allowed CPUs and oversubscribe their
`B` workers to that set; only real affinity/latency rows skip. The exact
fixture pins and audits every worker; the advisory fixture makes no placement
claim. This fixture audit does not upgrade general `ExternalManaged`
live-affinity semantics.

For each ownership/budget combination, the following five rows are mandatory:

| Row | Surface and request | Effective mode/result | Expected `(session, scope, permit, install, submit, provider)` |
|---|---|---|---|
| `D-N` | direct `CpuBackend::add`, `f64[65536]` | `Sequential` if `B=1`, otherwise `Inner` | `(0, 1, 1, 1, 0, 0)` |
| `E-N` | placement-bound `with_eager_session` add, `f64[65536]` | `Sequential` if `B=1`, otherwise `Inner` | `(1, 1, 1, 1, 0, 0)` |
| `D-D` | direct faer matmul-shaped `dot_general`, `f64[128,128]` | `Sequential` if `B=1`, otherwise `Inner` | `(0, 1, 1, 1, 0, 1)` |
| `E-D` | placement-bound session with the same dot request | `Sequential` if `B=1`, otherwise `Inner` | `(1, 1, 1, 1, 0, 1)` |
| `G-O` | public engine grouped dot composite, `J = 2B + 1` independent `f64[64,64]` requests | `Sequential` fallback if `B=1`, otherwise `Outer` | if `B=1`: `(0, 1, 1, 1, 0, 1)`; otherwise `(0, 1, 1, 0, 1, J)` |

At `B = 1`, native work uses sequential execution and faer receives
`Par::Seq`. At `B = 2` and `4`, native work receives a Rayon execution policy
bounded by `B`, and faer receives exactly `Par::rayon(B)`. At `B = 1`, `G-O`
must use the public grouped sequential fallback, enter the executor once, and
make one `grouped_gemm` provider call containing all `J` jobs. At `B > 1`, it
submits once with
`min(J, B) = B` indexed lanes, executes all `J` provider requests exactly
once, and gives every child a sequential context.

Two low-level external capability rows are also mandatory at budget 2. `U-O`
calls `CpuOperationEntry::submit_outer` explicitly with an executor whose
`outer_parallelism = false`; this explicit request, unlike the public grouped
fallback row, must return the typed scheduling error before calling submit or
provider, with vector `(0, 1, 1, 0, 0, 0)`. `U-I` supplies
`inner_parallelism = None`; the direct dot request remains supported with a
sequential faer child and vector `(0, 1, 1, 1, 0, 1)`. Neither row may fall
back to ambient Rayon or a different domain.

The inventory is therefore exactly 47 rows: 45 successful
ownership-by-budget-by-surface rows plus the typed-error `U-O` and successful
fallback `U-I`.

The inventory follows crate ownership. The CPU module-local artifact owns the
27 `D-N`/`D-D`/`G-O` rows plus `U-O` and `U-I`, for exactly 29 rows. The AD
module-local artifact owns the 18 public placement-bound `E-N`/`E-D` rows and
executes them through `CpuPlacementBoundEager::with_eager_session`; it builds
its own instrumented public CPU executor/provider fixtures because
`tenferro-ad` depends on `tenferro-cpu`, never the reverse. The gate wrapper
hashes both artifacts and rejects any duplicate, missing, or wrong-owner key
before composing the canonical 47-row manifest. CPU-only evidence may not
stand in for an AD-owned eager surface.

Inputs use deterministic nonuniform values. Native results must match exactly;
dot results must have relative Frobenius error at most `1e-12`. Every success
row also injects one typed operation/executor error and one unwind, proves the
permit and resources are released, then reruns successfully without count or
affinity contamination. Every participating worker records its observed CPU.
Managed and external-exact fixture observations must be subsets of their
declared sets; advisory observations are retained but not compared to an exact
set.

Latency is the only non-gating field. Each success row uses a two-second
warm-up, five-second measurement, 100 samples, and a 95% absolute-time
confidence interval, with setup and input construction outside the measured
loop. The public CPU/grouped rows come from the `numa_execution` bench and the
direct/placement-bound eager rows from the candidate-only
`phase2e_characterization` bench. Both are built with `--locked`, the
root-owned common lock, bench profile, `cpu-faer`, and the sealed environment;
their source, argv/environment, resolved graph, binary, and lock digests are
owned by `characterization/{cpu,eager}-bench-build.json`. There is no relative
threshold and no retry based on latency. If fewer
than `B` process-allowed CPUs exist, only that real-hardware budget's affinity
and latency rows receive typed `InsufficientAllowedCpus { required: B,
available }` skips. The topology-independent managed and external unit
fixtures use the nonempty allowed set with an instrumented `B`-worker executor,
so deterministic count, correctness, capability, and recovery rows still run
under oversubscription. Real cross-socket locality is a separate typed
hardware skip when fewer than two usable NUMA nodes exist. Neither skip can
hide a fixture failure or turn a failed timing campaign into a pass.

## Deferred post-Phase-9 decision

Production discovery of an external BLAS library's threading build mode is
not part of this implementation campaign. In particular, Phase 2E does not
attempt to distinguish pthread, OpenMP, or other vendor worker runtimes, and
does not install a production scoped controller based on that distinction.
After Phase 9, the project will decide how build/runtime identity is proven,
what an unknown identity means for strict placement, and which per-vendor
count or placement controls can be supported without process-global races.
Until then this remains an explicit unresolved item, not an implicit
acceptance dependency.

## Artifact schema

One orchestrator initially receives a nonexistent or empty outer evidence root
and rejects any pre-populated file before work begins. Only the lifecycle
commands defined above may subsequently reopen that root, and only after
complete digest and ledger revalidation. The orchestrator creates the
candidate-scoped root manifest and ledger first, then owns every child. No
standalone gate may import or overwrite a sibling from another root or
candidate:

```text
.orchestrator.lock
phase2e-evidence.json
evidence-ledger.json
abandoned-inventory.json  # present only when an interrupted root is sealed
builds/
  locks/
    direct-current-main.Cargo.lock
    common.Cargo.lock
    direct-current-main-probe.Cargo.lock
    common-probe.Cargo.lock
  direct-current-main-baseline.json
  common-lock-normalized-baseline.json
  candidate.json
allocation/
  probe-builds/
    candidate.json
    direct-current-main-baseline.json
    common-lock-normalized-baseline.json
  direct-current-main/
    attempt-0001/manifest.json
  common-lock-normalized/
    attempt-0001/manifest.json
dispatch-gates/
  manifest.json
  cpu-test-build.json
  ad-test-build.json
  cpu-evidence.json
  ad-evidence.json
characterization/
  manifest.json
  cpu-bench-build.json
  eager-bench-build.json
timing-attempts/
  direct-current-main/attempt-0001/
    timing/campaign.json
    timing/classification.json
    timing/summary.md
    timing/<case>/pair{1,2,3}/
  common-lock-normalized/attempt-0001/
    timing/...
```

`phase2e-evidence.json` has aggregate status `RUNNING`, `PASS`, `FAIL`, or
`INCONCLUSIVE` and contains protocol version and digest, classifier
source digest, canonical inventories, immutable source revisions, candidate
SHA/full-tree provenance, canonical experiment identity, host/toolchain/build
contract, status of every required gate, and
SHA-256 of all four normative lock copies, `evidence-ledger.json`, and every
required child manifest. The root
validator recomputes the complete ownership tree and exits 0 only when both
allocation comparisons, dispatch gates, characterization conformance, and
both timing comparisons are `PASS`.

Each allocation manifest owns exactly three fixed-order observations for all
28 cases and binds them to its two probe-build manifests, tenferro build
manifests, source-template digest, repetitions, and locks. The dispatch
manifest owns the source-contract module/function inventory and all ten
canonical direct/eager count vectors plus the hashes and owner partitions of
the 29-row CPU and 18-row AD artifacts. The characterization manifest owns its
composed complete mode/budget/ownership matrix, typed hardware skips, gating
conformance fields, non-gating latency fields, and both characterization bench
build-manifest digests. Every child manifest repeats
the candidate, protocol, classifier or probe, lock, toolchain, and build
manifest digests relevant to that gate; a mismatch invalidates the root rather
than being silently repaired.

Each timing pair directory contains the four process stdout/stderr logs,
copied target and sentinel estimate JSON, monitor observations, and
`validity.json`. `campaign.json` owns the SHA-256 inventory of every normative
pair file plus `classification.json` and `summary.md`, and records the expected
28 cases rather than discovering directories from disk. The fresh
Criterion tree lives outside the repository and evidence root: it is never
reused and is not normative evidence. Every estimate needed for classification
is copied byte-for-byte into its pair directory before hashing, so later
Criterion scratch mutation cannot alter a result.

An invalid campaign may have a strict prefix of case/pair directories. Its
manifest records `validity_state: INCONCLUSIVE`, `completed_at`, the invalid
location, reason, and prefix inventory. A valid `COMPLETE` campaign must have
exactly the canonical directory inventory and no pair-local rejected-attempt
namespace. Whole-campaign attempt directories exist only because the ledger
retains every permitted validity-inconclusive rerun; ids are allocated before
measurement and never reused.

The root manifest hashes every registered allocation and timing attempt, not
only the terminal one. A lane may name a gate result only from the sole
statistically or deterministically complete attempt after a strict prefix of
validity-inconclusive attempts. Since a complete attempt closes the lane,
there is no choice among multiple complete results and no favorable-result
selection rule.

The append-only `docs/worklogs/2026-07-21-phase-2e-index.json` is outside an
individual evidence root. All index operations take an exclusive `fcntl` lock
on the stable sibling `docs/worklogs/.phase2e-index.lock`; that non-normative
lock file is ignored by that exact path in `.gitignore` and is never staged.
While holding it, `start` rejects any global `ACTIVE` or
`PENDING_PRESERVATION` campaign and, when a durable index already exists,
requires its bytes to match the fetched remote-branch blob. It verifies that
the exact root has never been reserved, then
takes the new root lock in index-then-root order and appends an `ACTIVE`
reservation event before initializing any normative child or launching any
subprocess. The event
binds the immutable candidate SHA, root, protocol, reservation id, and two
separate identities. `candidate_provenance` binds the exact commit and its full
Git tree. `experiment_identity_digest` hashes the sorted Git
`(mode,path,blob)` inventory after excluding only `docs/worklogs/**`, together
with the protocol/classifier/probe hashes, frozen baseline and harness
revisions, feature/command matrix, and thresholds. Commit metadata and issue
text are not identity inputs. Durable closure is keyed by that canonical
experiment identity rather than raw candidate SHA, so an evidence/worklog-only
commit cannot reopen an experiment. A permitted validity/abandonment retry
must additionally retain the exact original candidate SHA. Root mutation
commands require the exact active reservation and both identities. This
prevents concurrent host campaigns, lost index updates, and post-result
cherry-picking.

If initialization fails after the durable ACTIVE append, `start` handles it
inside the same index-then-root locked operation and before any child process
exists. It rejects symlinks and special files, hashes every regular file then
present (including unique partial write temporaries), atomically creates
`abandoned-inventory.json`, and appends a terminal `ABANDONED` event that owns
the seal digest. The event moves ACTIVE to `PENDING_PRESERVATION`; it never
erases the reservation and does not yet permit a fresh root. That root
must be staged, committed, pushed, and reported like every other abandoned
root; the already terminal reservation cannot be finalized again through
`record-index --abandoned`. A failure before ACTIVE is durably appended creates
no campaign/index event and launches no subprocess.

The outer `start` command returns 0 only for aggregate PASS, 2 for validity
inconclusive, 3 for deterministic/statistical FAIL, 4 for statistical
INCONCLUSIVE, and 5 only for the self-sealed initialization-time abandonment
above. Exit 5 prints the stable status token `ABANDONED_INITIALIZATION`; the
matching seal and PENDING_PRESERVATION event are authoritative. Its worklog
records the failure, seal, and every existing file while marking every
unstarted build/evidence stage explicitly instead of requiring nonexistent
locks, manifests, rows, or timing tables.

`record-index` acquires the index lock and then root lock and moves the active
reservation to `PENDING_PRESERVATION` by appending a terminal event;
it never rewrites a history event. Each terminal event binds candidate,
resolved root path, aggregate/root-manifest status, root-manifest and ledger
digests, whether the root was abandoned, and the recording commit when
available. It retains `PASS`, `FAIL`, statistical or validity `INCONCLUSIVE`,
and explicitly abandoned `RUNNING` roots, but neither clears the global
reservation nor changes `current_evidence_root`.

`record-preserved` is the only `PENDING_PRESERVATION` transition. It requires
the exact root, durable index, and curated worklog blobs to exist in a named
preservation commit reachable from
`origin/codex/execution-engine-through-phase9`, reconstructs and validates the
root from that commit, and fetches a permanent issue-comment URL. The comment
must belong to #1436 and name the preservation commit, root, candidate, and
terminal status. It then appends a `PRESERVED` event binding that commit and
URL. A preserved statistically complete `PASS`, `FAIL`, or `INCONCLUSIVE`
closes the canonical experiment identity against every later candidate SHA. A
preserved validity-inconclusive or abandoned result permits a fresh uniquely
named root only for the same experiment identity and exact candidate SHA,
because it contains no selectable complete result. Only a preserved PASS may
set `current_evidence_root`; negative or abandoned events never replace the
last PASS pointer. Before a later `start`, the index including the PRESERVED
event must itself be committed and pushed; byte equality with the fetched
remote index is mandatory. Repeating exact transitions is idempotent, while
changing an event or reserving two roots concurrently is rejected. Every root
is therefore linked from the worklog, committed, pushed, and reported to #1436
before Phase 2E either continues with a permitted fresh root or stops.

For an uncatchable interruption of an active initialized root,
`record-index --abandoned` first acquires the
index lock and then the root lock, verifies every recorded process group is
gone, and requires an explicit operator confirmation. It rejects symlinks and
special files, hashes every regular file currently present under the root—including
partial logs and temporary files—without deleting or renaming any, then writes
`abandoned-inventory.json` atomically. Atomic JSON writes always use a fresh
`O_EXCL` sibling temporary with a collision-free name, never a shared
`<target>.tmp`; a failed write remains a distinct regular file for a later
seal and is not overwritten or unlinked. That inventory owns all preexisting
paths and is itself hashed by the terminal index event. Abandoned structural
and Git-index validation use this immutable inventory rather than the
incomplete aggregate manifest and require the exact same files and bytes.

The evidence root deliberately contains normative files whose names match
repository ignore rules, including root-owned `Cargo.lock` copies and process
`*.log` files. After terminal validation, the exact owned root is therefore
staged with `git add -f -- <root>`; no directory above that root is force-added.
The orchestrator's `validate --git-index` mode reads staged blob identities,
requires every manifest-owned normative path to be present with byte-identical
content, rejects every extra staged path under the root, and separately
requires the worklog and durable index to be staged normally. A fresh-checkout
test reconstructs the root from the Git index and reruns structural validation.
This staged-inventory gate applies to PASS and ordinary negative roots from
their manifest ownership trees, and to abandoned roots from
`abandoned-inventory.json`; it prevents ignored or interrupted evidence from
disappearing after push.

## Tests

Runner unit tests use fake benchmark processes and synthetic Criterion JSON;
they do not run release benchmarks. They prove:

- the canonical inventory remains exactly 28 cases and three pair orders;
- valid 84-pair completion atomically stores validity, per-case statistical
  results, campaign result, and rendered artifact hashes;
- one invalid run finalizes the entire campaign as `INCONCLUSIVE` and stops;
- no pair is retried and no pair-local `_rejected` or attempt namespace exists;
- a non-empty artifact root is rejected before process launch;
- a non-empty Criterion root is rejected before process launch;
- nested or identical roots are rejected;
- a partial/RUNNING manifest cannot be resumed;
- `CRITERION_HOME` is the declared fresh root;
- quiet-wait and benchmark-process deadlines terminate as validity
  `INCONCLUSIVE`;
- terminal exception handling preserves a readable atomic manifest;
- the classifier rejects validity-inconclusive and partial campaigns;
- direct and normalized baseline verifiers reject every extra file or hunk;
- build manifests reject dirty trees, unbound binaries, wrong commands,
  environments, features/providers, locks, toolchains, profiles, harnesses,
  or revisions;
- all three probe build manifests reject a wrong source/template, dependency
  feature tuple, profile, sealed environment, role lock, resolved graph, or
  executable digest;
- allocation manifests reject missing/retried cases and non-identical probe
  source/templates;
- the ledger permits only whole reruns after validity failure and closes after
  any statistically complete result;
- outer-root reopening revalidates every existing digest, cannot resume
  `RUNNING`, cannot reopen an indexed root, and allocates fresh attempt and
  external scratch paths;
- the durable index retains terminal negative and abandoned roots while only
  a remotely preserved `PASS` updates `current_evidence_root`;
- canonical experiment identity ignores evidence/worklog-only commits, closes
  every statistically complete identity, and permits validity/abandonment
  retries only for the same exact candidate SHA;
- the index lock and `ACTIVE`/`PENDING_PRESERVATION` reservations serialize
  parallel starts/records, enforce index-then-root lock order, reject
  record-versus-rerun races and candidate/input-identity drift, preserve
  initialization crash history, and require remote commit plus #1436 report
  before a `PRESERVED` event can unlock another root;
- Git-index validation includes ignored normative locks/logs, rejects missing
  or extra staged root files, and reproduces validation from a fresh checkout;
- the root validator rejects a missing, stale, foreign-candidate, unhashed, or
  non-passing sibling manifest.

Existing classifier tests continue to cover interval inversion, PASS/FAIL/
INCONCLUSIVE thresholds, sentinel drift, artifact hashes, complete inventory,
and rendered summaries.

## Acceptance

Phase 2E is complete only when:

1. the runner and validator tests pass without `ignore` or `no_run`;
2. the direct and normalized baseline verifiers accept exactly their declared
   deltas and all three builds pass provenance validation;
3. both atomic allocation comparisons, source-contract, and entry-count gates
   pass;
4. the direct-current-main timing campaign reports `PASS`, followed by a
   common-lock-normalized timing campaign that also reports `PASS`;
5. placement/parallelism correctness, recovery, counts, and affinity
   conformance pass, with non-gating latency and hardware skips explicit;
6. all commands, commits, hashes, host configuration, intervals, and limits
   plus every ledger attempt including negative evidence are saved in a
   worklog;
7. the aggregate root validator exits 0 and its root/index/worklog reach the
   remote branch plus #1436 before a PRESERVED event is committed and pushed;
8. an independent performance reviewer reports no unresolved Critical,
   Important, or untracked Minor finding; and
9. the final candidate and evidence commit pass the repository-rules review
   with no unwaived finding.

If either timing or allocation comparison is validity `INCONCLUSIVE`,
statistically `INCONCLUSIVE`, or `FAIL`, Phase 2E does not pass. Every result
is retained as evidence and the architecture branch is not promoted on the
strength of a different or secondary metric.

An audit fix that changes Rust source, benchmark/probe source, protocol,
runner, classifier, validator, build inputs, or another canonical experiment
input creates a new experiment identity. The previous root remains preserved
evidence, and Tasks 9-10 restart with a new initially empty candidate-scoped
root; re-running only the validator is forbidden. A prose-only worklog or
issue-link correction under excluded `docs/worklogs/**` changes no identity,
does not authorize another measurement, and may be reviewed without repeating
measurement. A defect in a preserved normative artifact cannot be patched in
place or used alone to reopen the closed identity: the owning generator or
protocol must be fixed so the canonical identity changes before a complete new
campaign. When `origin/main` advances, a read-only comparison computes the
prospective HEAD identity: an excluded-path-only equal identity reuses the
preserved evidence after full final-HEAD verification, while a changed identity
returns to Tasks 9-10.
