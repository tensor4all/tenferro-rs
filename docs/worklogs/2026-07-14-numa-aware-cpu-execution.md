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
worker count, and problem shape. It compares concurrent disjoint-node managed
sessions, an all-allowed session, and (when compiled and linked)
provider-default exclusive execution; it exits successfully with an
explanation on single-node hosts. The CPU crate benchmark uses the same
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
