# Phase 2E dispatch and characterization evidence

Date: 2026-07-22

## Scope

Task 7 adds crate-owned dispatch evidence without introducing a CPU-to-AD
dependency or production counters. The CPU artifact owns 29 characterization
rows and ten direct/borrowed-session vectors. The AD artifact owns 18 public
placement-bound rows and five ordinary eager single-session-entry proofs. The
gate composes exactly 47 unique canonical keys and rejects duplicates,
omissions, wrong ownership, count/mode mismatches, failed recovery fields, and
untyped hardware skips.

The row fixtures are executable rather than declarative. Scoped `#[cfg(test)]`
recorders measure permit, operation-scope, executor install/submit, provider,
session-entry, selected-mode, and operation-local worker-CPU events at their
real runtime boundaries. They compile out of production builds. AD owns only
its raw eager session and backend observations; the gate composes each AD row
with the matching CPU borrowed-session vector, so neither crate duplicates the
other crate's facts or gains a dependency on it.
External exact fixtures pin their caller-owned Rayon workers when enough CPUs
are visible, while advisory fixtures retain observations without upgrading the
placement claim. D-N uses nonuniform `f64[65536]` inputs with exact comparison,
D-D/E-D use nonuniform `f64[128,128]` inputs with a `1e-12` relative Frobenius
bound, and G-O executes `J = 2B + 1` real grouped requests. U-O executes the
typed pre-submit rejection and U-I executes the no-inner sequential fallback.
Every success row exercises a typed operation error and an unwind on the same
public workload surface before a successful post-recovery rerun. U-O is the
only validated path without a worker observation and binds the exact typed
pre-submit scheduling source.

The complete `RecordingBackend` fixture now lives in the module-local
`eager_backend/tests.rs`. Production `eager_backend.rs` retains only the test
module declaration, private test-only type reference, and dispatch arms.

## TDD record

The plan-prescribed initial commands produced the following baseline:

- CPU and AD filtered Cargo tests reported zero matching tests and exited zero,
  which is Cargo's normal missing-filter behavior.
- `python3 -m unittest scripts/test_run_phase2e_gates.py -v` failed to import the
  nonexistent module.

After registering the Rust tests, the compile RED exposed obsolete
`SliceConfig` field names (`start_indices` and `limit_indices`); these were
replaced by `starts` and `limits`. The first CPU Criterion compile then exposed
the missing `TensorElementwise` trait import. Both defects were fixed before
the corresponding GREEN runs.

## Provenance and deadlines

`run_phase2e_gates.py` is the single Task 7 owning CLI. It validates a clean
immutable candidate, installs the root-owned common lock, invokes
`phase2e_build.py` for four fresh external targets, executes the hashed test
binaries directly with the evidence filters and `--nocapture`, composes the 47
rows, and directly executes 45 filtered Criterion rows. Each Criterion process
uses a fresh `CRITERION_HOME` and `--bench --noplot`; its exact stdout, stderr,
and `new/estimates.json` bytes are copied to normative storage before scratch
cleanup. The terminal manifest records their SHA-256 digests plus the absolute
mean point estimate and 95% confidence interval.

Build and terminal manifests bind the protocol, candidate commit/tree, exact
Task 7 source inventory, common lock, feature graph, toolchain, sealed
environment, argv, executable, evidence, and row artifacts. A final recursive
validator recomputes the exact 148-file inventory and every digest/key. Timeout
or nonzero execution writes owning `INCONCLUSIVE` terminals with captured
stdout/stderr and TERM, five-second grace, KILL, and reap metadata.

The Cycle 3 provenance hardening separates runtime execution from build
environments. Direct test processes receive the exact
`protocol.runtime_environment` allowlist plus only their evidence-directory
variable, and Criterion rows receive only the allowlist plus
`CRITERION_HOME`; build-only Cargo variables are rejected. The terminal source
contract is a canonical list keyed by every `(path, signature)` pair and binds
both whole-file and extracted-item SHA-256 values, so multiple hot functions in
one source file cannot overwrite each other. The caller-supplied scratch root
is resolved before evidence mutation and must be disjoint in both directions
from both the repository and evidence trees.

Cycle 3 also makes low-CPU exact fixtures auditable instead of silently
unpinned. `CpuContext::with_pinned_cpus` now permits more workers than CPUs and
cycles worker assignments over the nonempty exact set. CPU and AD fixtures use
a barrier-backed Rayon broadcast to record every worker index and current CPU;
exact audits must cover `0..B` and remain inside the declared set. When a
managed correctness fixture has fewer allowed CPUs than `B`, it uses the same
topology-independent exact `B`-worker fixture rather than reducing the worker
count. Correctness and recovery still run and never hardware-skip.

Real-hardware validity is represented separately. Affinity and Criterion
latency records receive typed `InsufficientAllowedCpus` skips when distinct
resources are unavailable, skipped rows launch no benchmark and contain no
synthetic estimates or artifacts, and cross-socket locality records typed
`InsufficientNumaNodes`. AD records actual placement-bound session entry at a
crate-local `#[cfg(test)]` boundary and independently owns its executor/provider
CPU observations and all-worker placement audit. Session-entry CPU observations
are named separately and are not presented as downstream worker placement.
The gate composes only the matching CPU borrowed-session downstream count/mode
contract; it no longer substitutes CPU observations into AD rows.

Correctness executables have a 120-second deadline. Each Criterion row has a
30-second deadline. Process groups receive a five-second termination grace
before forced kill. The two Criterion harnesses use fixed two-second warmup,
five-second measurement, 100 samples, and 95% confidence.

## Cycle 4 evidence hardening

Managed exact construction now retains the requested nonzero worker budget
even when the placement contains fewer logical CPUs. The owned pinned Rayon
pool assigns workers cyclically over the exact nonempty CPU set; it is never
replaced by an `ExternalManaged` fixture. A one-CPU, three-worker engine test
binds Managed ownership, exact placement, thread budget, and worker count.

AD E-N rows use the internal, non-default `phase2e-observe` build feature. It
exports no Rust API. The actual closures passed to the strided elementwise
kernel record their Rayon lane and `sched_getcpu` value to a row-specific
temporary file, deduplicated by path/lane/CPU. The AD test reads and removes
that file immediately. The gate requires nonempty operation-participating
`[lane, cpu]` pairs and exact-subset membership independently of the separate
all-worker placement audit and eager session-entry observation.

Every correctness row now serializes a fresh-reset recovery subrecord. CPU
recovery independently remeasures the complete six-counter vector, selected
mode, operation CPUs, numerical result, and exact-subset result. AD recovery
independently remeasures eager entry, optional external install/submit,
provider count, operation lane/CPU pairs, operation CPUs, numerical result,
and subset result; the gate composes its downstream recovery vector and mode
from the matching CPU row.

Every unskipped Criterion process receives a row-specific affinity artifact
path. Before `b.iter`, the actual fixture performs a barrier-backed
`rayon::broadcast` through its backend executor and writes ownership,
guarantee, budget, worker count, declared CPUs, and all worker lane/CPU pairs.
The measured loop contains no audit. The runner validates exact placement,
copies `affinity.json` beside the Criterion estimates and logs, and binds its
SHA-256 in the terminal inventory. G-O operands are deterministic nonuniform
matrices rather than constant vectors.

Cross-socket validity is now CPU-executable evidence instead of Python-side
hardware inference. When two usable topology nodes exist, per-node managed
backends first-touch independent inputs inside their executors, synchronize
two scoped threads, execute elementwise work on both nodes, and record node,
declared CPU set, observed operation CPUs, numerical success, and subset
success. Only a host with fewer than two usable nodes may emit the typed
`InsufficientNumaNodes` skip.

## Verification

- CPU and AD focused evidence tests passed and their emitted JSON composed to
  exactly 47 rows.
- All 97 combined Python build/gate contract tests passed.
- The full CPU suite passed 513 tests and the full AD suite passed 71 tests.
- The focused provider filter passed 57 tests and the dot-runtime filter passed
  43 tests.
- Both characterization Criterion binaries compiled with `cpu-faer` and no
  default features.
- Clippy passed for both libraries with warnings denied; rustfmt and
  `git diff --check` passed.
- The repository-rules review passed with no findings.
- A clean temporary candidate completed the owning CLI end to end: all four
  builds, two direct evidence runs, 47 composed rows, and 45/45 Criterion
  estimates passed. Independent terminal validation found 148 exact files.
  Dispatch terminal SHA-256 was
  `df8d9c3a64d7e401ea60bb3508deae3a8d004b9c7e062c8605a9aabd7e3c45e9`;
  characterization terminal SHA-256 was
  `1e694a83fe96c740f5980240b12e990de5d70670feab618f75229dfbd5239f66`.
- Cycle 3 Python hardening passed 57 protocol/gate unit tests and Python bytecode
  compilation; `git diff --check` also passed.
- Cycle 3 Rust hardening passed 515 CPU tests, 71 AD tests, 142 combined
  protocol/gate/build Python tests, both exact `cpu-faer` characterization bench
  builds, and CPU/AD Clippy with warnings denied.
- A clean temporary Cycle 3 candidate at `324771ad` completed the Task 7 owning
  CLI end to end: four fresh builds, two direct evidence tests, 47 composed
  correctness rows, and 45/45 real Criterion latency rows passed. This host had
  no allowed-CPU skip and one usable NUMA node, so cross-socket locality recorded
  typed `InsufficientNumaNodes { required: 2, available: 1 }`. Recursive terminal
  validation found exactly 148 files. Dispatch terminal SHA-256 was
  `fffe2c2efbfa0d8ee8adf4bccbffd285f362b9587edd1bc96291f8d1f99fd746`;
  characterization terminal SHA-256 was
  `9dd04de48479bb594841388bfdf5ce0aac5fe461d4a9179e174aa28bec96c6a1`.
  The complete 19/19 gate-test suite, including the focused synthetic
  insufficient-CPU cases, also passed and proves that skipped latency rows
  launch no benchmark and contain neither estimates nor artifacts.
- Cycle 4 pre-E2E verification passed 515 CPU tests, 71 AD tests, 143 combined
  protocol/build/gate Python tests, both exact characterization release builds,
  CPU and AD Clippy with warnings denied (including the observe-feature build),
  rustfmt, and `git diff --check`. A directly executed managed-exact B=2 D-N
  Criterion row emitted worker observations `[[0, 0], [1, 63]]` inside the
  declared CPU set. This host exposes one usable NUMA node, so the executable
  cross-socket probe emitted only the typed
  `InsufficientNumaNodes { required: 2, available: 1 }` skip.
