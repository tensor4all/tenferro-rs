# Structural-copy performance for borrowed CPU reads (#2029, #2037)

Covers the [terasaki performance batch][batch] rows that share one root cause:
CPU structural materialization used the generic element-wise strided map even
where a blocked full-overwrite copy was available, and an owned `reshape_read`
copy was pinned to a serial host copy.

[batch]: https://github.com/tensor4all/tenferro-rs/issues/2010

## Decisions

- **Full-overwrite view materialization uses the strided full-overwrite entry.**
  `typed_materialize_view_with_pool`, `typed_transpose_view_with_pool`, and
  `typed_reshape_view_with_pool` now call `strided_kernel::copy_into_uninit`
  instead of `map_into(.., MaybeUninit::new)`. The destination layouts are
  compact column-major, so the entry's injectivity precondition holds. At the
  pinned strided-rs revision the entry reinterprets storage only for exact f32
  and f64 and only when the element count is at or below `MINTHREADLENGTH` or
  the scheduler is sequential; integers, bool, and complex scalars, and
  native-float copies above the threshold in a parallel context, keep the same
  element-wise map.
- **Owned `reshape_read` copies on the session pool above the strided
  parallel threshold.** `CpuExecSession::reshape_read` keeps the existing serial
  `Tensor::to_vec` fast path when the context cannot parallelize the copy or the
  input is at or below `SERIAL_RESHAPE_MAX_ELEMS`
  (`strided_kernel::execution::MINTHREADLENGTH`). A larger owned input in a
  parallel-capable context routes through `reshape_read_with_pool`, which now
  borrows the owned tensor as a view and runs the same strided copy on the
  configured pool. That branch uses `run_native`, not `run_native_fresh`, so the
  output keeps the input placement; the metadata-only reshape contract requires
  that an owned reshape is not retagged with the session domain.
  - Rejected: routing *every* owned reshape through the pool. It paid the engine
    entry for copies too small to parallelize and changed pool-buffer reuse for
    tiny reshapes (the `tensor_stack_reuses_reclaimed_cpu_buffer` case).
  - Rejected: `run_native_fresh` for the pooled owned branch. It overwrites the
    output affinity with the session domain and makes placement depend on size.
- **A caller-owned payload keeps its typed refusal.** `reshape_read_with_pool`
  returns the existing unsupported-dtype error for a `TensorRead::Tensor` with
  an externally defined dtype instead of borrowing `Tensor::tensor_view`, which
  has no external variant and would otherwise panic.
- **`CpuExecutionContext::uses_inner_parallelism` is now `pub(crate)`** so the
  session can ask the same predicate the native dispatch uses instead of
  re-deriving thread policy at the operation.
- **The serial threshold reuses the lower-layer constant**
  (`strided_kernel::execution::MINTHREADLENGTH`) rather than a new literal, so
  it cannot drift from the scheduler it mirrors.
- **Not taken here.**
  - `extract_diagonal` (#2038) measured no gain from `copy_into_uninit` (its
    diagonal view is a large-stride gather, not a permutation); the gap is real
    and larger on this host (76.7 ms at 4T against the JAX reference's 14.5 ms),
    so it needs a different kernel rather than this copy entry.
  - Metadata-only view construction (#2040) is already allocation-free for
    ranks <= 8 and its comparison is against a compile-time Julia operation;
    the absolute cost is a few hundred ns and the fix direction is the storage
    redesign, not a local change.
  - Complex Frobenius norm (#2035), reductions (#2034), elementwise and
    activation kernels (#2030-#2033), and FFT (#2039) need strided-rs kernel
    work (complex sum-of-squares, SIMD reduce, vectorized transcendental math)
    or are intrinsic engine differences (RustFFT vs oneMKL DFTI); none is a
    tenferro-side quick fix.

## Verification conclusions and constraints

- `cargo test -p tenferro-cpu` passes in full (417 lib tests plus integration
  and doctests). The source-contract test that pins the pool-aware
  materialization path now requires `copy_into_uninit` and was updated.
- Focused regression tests cover: an owned reshape above the threshold and one
  below it through a two-thread session; affinity preservation for an owned
  reshape above the threshold in a two-thread session; and the external-payload
  refusal in the same parallel route. The three parallel tests assert
  `num_threads() == 2`, so the pooled branch is exercised explicitly rather than
  assumed from the requested degree.
- Measurements come from the frozen issue harness
  (`tenferro-benchmark` `mwe/cpu_followup` at `59ed19bd`) and two local probes,
  built in release with OpenBLAS (no MKL on the host). Host: AMD EPYC 7713P, 64
  cores, Linux, rustc 1.97.1; strided-rs pin `12ff2de9`. The fixed cases use the
  harness's own Rust `strided-rs` arm; the Python/Julia reference arms exist
  locally (`tenferro-benchmark` `.venv` with torch 2.12.0+cu130 and jax 0.10.1,
  Julia 1.12.5) but were not run for this record except for the `#2038` JAX
  number above. oneMKL is not installed, so the `#2039` DFTI reference is not
  reproducible here.
- **These are exploratory before/after comparisons, not a performance-gated
  paired experiment.** The issue rows predeclare the workload and the >=1.2x
  need, but no complete AB/BA run with predeclared thresholds, A/A noise floor,
  pinned-core idle observations, or confidence intervals was captured: the host
  is shared (1-minute load average around 2-4 during these runs). The tables
  are per-operation medians, and the direction and rough size are the claim; the
  exact ratios are not. Effective degrees were verified in-process: the probes
  report `requested=1 effective=1` and `requested=4 effective=4` via
  `CpuBackend::num_threads`, and the harness arms construct one backend per
  process with an explicit degree. The Python/Julia reference arms are not part
  of these fixed-case measurements because both fixed cases are compared against
  a Rust arm (strided-rs) or a plain memcpy.

  Permutation/transpose materialization, 1 thread, median ms (before → after,
  with the strided-rs reference for scale):

  | case | before | after | strided-rs |
  | --- | ---: | ---: | ---: |
  | `perm.reverse_15d_3` | 168.8 | 118.1 | 113.7 |
  | `perm.reverse_23d_2` | 59.1 | 32.0 | 64.6 |
  | `perm.cyclic_15d_3` | 67.5 | 66.3 | 60.0 |
  | `perm.transpose_3d_256_201` | 131.8 | 85.9 | 83.3 |
  | `perm.transpose_3d_256_102` | 93.3 | 90.4 | 88.0 |
  | `perm.tn_light_415_24d_scattered_to_colmajor` | 80.6 | 77.2 | 75.7 |
  | `perm.tn_light_415_24d_contiguous_same_perm` | 73.1 | 69.3 | 71.2 |
  | `structural.transpose` | 112.8 | 86.9 | 113.8 |

  Every listed issue row is one thread, and each moves to the reference
  envelope. The same harness at four threads is unchanged within host noise
  (`perm.reverse_15d_3` 65.8 → 63.4 ms, `perm.transpose_3d_256_201` 36.2 →
  39.1 ms, `structural.transpose` 35.7 → 32.1 ms); those layouts exceed the
  threshold, so the entry resolves to the element-wise map exactly as before.
  Tenferro's parallel permutation copy remains slower than the strided-rs
  parallel copy for some layouts and needs an upstream uninitialized parallel
  permutation entry.

  `transpose_read` materialization through a borrowed session, median ms
  (before → after), shows the below-threshold shift and no parallel regression:

  | shape | 1T before | 1T after | 4T before | 4T after |
  | --- | ---: | ---: | ---: | ---: |
  | 256x256 | 0.1462 | 0.0304 | 0.1137 | 0.1136 |
  | 32x32x16 | 0.0304 | 0.0063 | 0.0453 | 0.0229 |
  | 4096x4096 | 115.73 | 90.51 | 34.43 | 34.81 |

  Owned reshape copy, f64 4096x4096 -> 8192x2048 in one borrowed session,
  median of 40: before 74.4 ms at 1T and 74.5 ms at 4T (no scaling); after
  74.5 ms at 1T and 21.9 ms at 4T (3.4x from threads). A 256-element reshape
  is 0.0002 ms in both states, so the small-input fast path is unaffected.
- The full `tenferro-benchmark` reference arms (JAX/PyTorch/Julia, MKL) were not
  reproduced here; the issue's absolute ratios are host-specific. Regression
  rows for these cases belong to `tenferro-benchmark`, not this repository.
