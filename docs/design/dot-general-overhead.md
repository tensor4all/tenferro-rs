# Dot-General Fixed Overhead

This note records the current interpretation of the
`tenferro-tensor/benches/dot_general_overhead.rs` benchmark. It is a
developer-facing performance note for small, repeated contractions such as MPS
inner products.

## Benchmark Shape

The benchmark contracts a length-32 complex MPS inner product using two
`dot_general_with_conj` calls per site. It compares three execution styles:

| Variant | What it measures |
|---------|------------------|
| `fresh_backend_cache` | Calls `backend.dot_general_with_conj` for every local contraction. |
| `persistent_backend_cache` | Reuses the backend runtime cache but still calls the backend method for every contraction. |
| `single_exec_session` | Opens one `backend.with_backend_session_cached` around the whole MPS contraction and calls `BackendSession` methods inside it. |

Representative one-thread timings from the May 2026 investigation:

| `L = 32`, `d = 2`, `Complex64` | `chi = 4` | `chi = 8` | `chi = 16` | `chi = 32` | `chi = 64` |
|--------------------------------|----------:|----------:|-----------:|-----------:|-----------:|
| `fresh_backend_cache` | ~445 us | ~446 us | ~445 us | ~1.53 ms | ~8.69 ms |
| `persistent_backend_cache` | ~421 us | ~434 us | ~444 us | ~1.52 ms | ~8.67 ms |
| `single_exec_session` | ~45.8 us | ~61.2 us | ~190 us | ~1.12 ms | ~8.12 ms |

Exact numbers vary by machine and allocator state. The important signal is the
scaling difference between per-op backend calls and a single execution session.

## Interpretation

For small contractions, the dominant overhead is not GEMM analysis cache misses,
hash-map-heavy label handling, or contraction path search. The main fixed cost is
opening the full backend wrapper path for each local contraction:

```text
backend.dot_general_with_conj(...)
  -> backend.with_backend_session(...)
       -> dtype dispatch
            -> Tensor wrapper / enum dispatch
                 -> CpuExecSession::dot_general_with_conj(...)
                      -> GEMM planning / execution
```

When this sequence is repeated for every tiny local contraction, the
`backend.with_backend_session` / dtype dispatch / wrapper path cost dominates.
Earlier measurements also included `rayon::ThreadPool::install`; current CPU
execution no longer enters a tenferro-owned Rayon pool. Reusing a persistent
runtime cache helps only slightly in this benchmark, which indicates that GEMM
analysis caching is not the principal bottleneck for low bond dimensions.

For larger bond dimensions the actual GEMM work becomes dominant, so the gap
between `fresh_backend_cache` and `single_exec_session` naturally shrinks.

## Design Consequences

- Prefer executing many small contractions inside one `BackendSession` when
  the caller naturally owns a loop or compiled program.
- Do not add ad hoc MPS-specific fast paths to bypass the public backend API.
  The general abstraction to improve is an execution-session surface that can be
  reused by tensor-network code and compiled graph execution alike.
- Further SmallVec/hash-map tuning in contraction metadata is unlikely to move
  this benchmark unless profiling shows that the workload has shifted away from
  backend/session fixed costs.
- A future public or crate-private batched eager execution API should make
  session lifetime explicit, so callers can avoid opening one backend session
  per small contraction without reaching into backend internals.

## Reproduction

Run with one CPU worker and one BLAS worker to avoid hiding fixed costs behind
parallel scheduling noise:

```bash
RAYON_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
cargo bench -p tenferro-tensor --bench dot_general_overhead
```
