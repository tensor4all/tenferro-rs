# NUMA-aware CPU execution

Issue #1345 adds process-affinity-aware NUMA topology, pinned fixed CPU engines,
shared arbitration, and a documented provider boundary.

The implementation treats OS node IDs as sparse and intersects node cpusets
with the process mask. `CpuBackend` clones share engines and caches. faer owns
managed placement; external BLAS accepts only `Auto` and runs exclusively
because its worker affinity is not controlled by tenferro. Graph execution
keeps a permit across supported Host/native/FFI instructions, entering Rayon
for native BLAS-backend segments and leaving it for provider calls.

Rejected alternatives were thread-count-only placement, exposing concrete BLAS
provider names as stable identities, and allowing explicit BLAS node placement
without a verifiable worker-affinity contract.

Verification covers sparse topology parsing, affinity pinning, overlapping and
disjoint permits, clone sharing, graph session count, Host work inside the
session, and the BLAS Rayon/provider transition. The online guide is
`docs/guides/cpu-execution.md`.

The opt-in `numa_execution` benchmark reports allowed CPUs, topology,
requested/resolved placement, stable backend kind, diagnostic provider text,
worker count, and problem shape. It covers 64, 256, and 512 square matrices at
one worker and the process-default worker budget. It compares concurrent disjoint-node managed
sessions, an all-allowed session, and (when compiled and linked)
provider-default exclusive execution. On single-node hosts it still measures
the portable cases and skips only the disjoint-node comparison. The CPU crate benchmark uses the same
multi-op `BackendSession` boundary as compiled graph execution without adding a
reverse dev-dependency from `tenferro-cpu` to `tenferro-runtime`.

Local benchmark verification on 2026-07-14:

- `cargo bench -p tenferro-cpu --bench numa_execution --no-run` passed.
- `cargo bench -p tenferro-cpu --bench numa_execution -- --sample-size 10`
  passed and reported a skip because the process-visible topology contained one
  usable node (64 allowed logical CPUs). Live two-node timing evidence is
  therefore unavailable in this environment; fixture/unit tests cover sparse
  multi-node resolution and disjoint arbitration.

Repository verification evidence:

- `cargo test --workspace --release` passed, including workspace doctests.
- `cargo llvm-cov --workspace --release --json --output-path coverage.json --
  --test-threads=1` passed; `scripts/check-coverage.py` reported 159/159 files
  meeting thresholds (3 excluded).
- The first parallel coverage run exposed a pre-existing graph-metadata registry
  race in one linalg test. The same instrumented test passed alone; serializing
  the coverage harness made the full run deterministic. Normal parallel release
  tests passed.
- `cargo doc --workspace --no-deps` and `scripts/check-docs-site.py` passed,
  including guide dependency snippets and 13 workspace API crates.
- CI-equivalent workspace and tropical-extension clippy commands passed with
  `-D warnings`.

Pre-PR review found that the fused segmented executor still opened one backend
session per fused segment. The revised executor keeps elementwise fusion and
terminal lazy values while running every session-capable fused, Host, and FFI
segment inside one cached session. Regression tests first failed with two
session entries, then passed with one for both owned and value outputs; the
fixture includes native/FFI/Host/native transitions.

The same review tightened the remaining safety contracts:

- arbitration is process-wide, so independently constructed backends cannot
  overlap an unmanaged external-provider call with any tenferro CPU domain;
- process-wide arbitration propagates a logical execution owner across Rayon
  pools for direct synchronous nesting, while backend calls from parallel
  Rayon child tasks or unrelated work on the active context are rejected so
  siblings cannot bypass engine or provider exclusion;
- reentrant operations use transient engine resources so a nested tensor or
  provider session does not re-lock the outer session's scratch/cache mutex;
- fallible constructors expose `CpuBackendError` and preserve typed placement
  and `CpuTopologyError` detail, while only infallible construction may use the
  documented compatibility fallback;
- unsupported-affinity platforms resolve faer `Auto` to compatibility mode and
  reject explicit managed placement;
- affinity-mask allocation rejects extreme public CPU IDs using checked,
  fallible allocation;
- reduced worker budgets are spread over the logical CPU domain; and
- typed diagnostics now report topology, worker count, and managed,
  provider-exclusive, or compatibility execution mode.

Focused post-review verification passed for 287 CPU unit tests, fused runtime
owned/value session tests, the provider-inject integration suite, and benchmark
compilation. A `wasm32-unknown-unknown` portability check stopped in the
third-party `atomic-wait` crate before compiling tenferro; the unavailable
affinity branch is covered by an injected capability contract test.
