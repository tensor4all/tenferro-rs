# CUDA bug remediation ledger

## Session summary

This work log is the classification ledger for issues #1353 and #1356 through
#1366. It records the scope approved for the single CUDA remediation batch; it
does not itself claim that any close condition has been met. The baseline
`cargo test --workspace --release` passed before remediation edits began.

Remote state was refreshed on 2026-07-11: every issue in this ledger is open,
and no open or closed pull request matched these issue numbers. Historical
issue reports and CUDA-devcontainer repro comments remain evidence, while the
working-tree source is the authority for implementation decisions.

## Scope decisions

- #1353 is maintainer-approved in the interactive session: floating-point
  operations preserve IEEE/NumPy/JAX-style exceptional values rather than
  converting them to typed domain errors. Integer domain errors remain
  unchanged.
- #1362, #1363, and #1364 are explicitly approved capability-parity work, but
  only for behavior already implemented by the CPU backend. They must not add
  CPU-new behavior. In particular, Bool additive scatter is excluded because
  CPU rejects Bool data tensors.
- #1360 was narrowed from `Verify First` and implemented for the three confirmed
  CUDA paths: `dynamic_slice`, `gather`, and `scatter`. Its contract is exact
  CPU parity, including the float-index bounds `F32: 2^24` and `F64: 2^53` and
  the exact `InvalidConfig` variant, operation label, and message parity.

## Classification ledger

| Issue | Classification | Current evidence | Close condition and verification target | Residual risk |
| --- | --- | --- | --- | --- |
| #1353 | Implemented locally; focused verification passed | Active tensor/primitive specs now state the approved IEEE-style value-propagation policy. Focused CPU `F32`/`F64` tests cover division and remainder with signed-zero divisors, NaN, infinity, and signed-zero results; integer `I32`/`I64` zero divisors remain typed `DivisionByZero`. An ignored CUDA parity test covers the same floating cases by result class and sign bit. No production kernel change was required. | Focused CPU test passes and the CUDA test compiles with the `cuda` feature; the ignored CUDA test still requires execution on a CUDA 12.8+ GPU. Overall batch verification remains incomplete. | Other analytic operations can have independent IEEE edge discrepancies; scan them, but do not silently expand this issue. |
| #1356 | Implemented locally; focused verification passed | Generic CubeCL `F::new(0.0)` / `F::new(1.0)` literals in the float unary kernels were replaced with explicitly typed `f32` literals. | The focused ignored CUDA test passed on an NVIDIA A100, the CUDA feature test build emitted no fallback warning for these sites, and a source check found no remaining untyped `F::new(0.0)` / `F::new(1.0)` literals in `kernels/elementwise.rs`. Overall batch verification remains incomplete. | CubeCL or rustc upgrades may expose equivalent literals in other kernel families. |
| #1357 | Implemented locally; focused A100 verification passed | CUDA `div`/`rem` now use a private scalar-indexed launch only when exactly one same-dtype operand is rank-0. The kernel maps the scalar to index zero without `broadcast_typed` or a full-size temporary. Integer scalar paths retain the device error flag and exact `DivisionByZero` fields, plus wrapping `MIN / -1` and `MIN % -1` behavior. | The ignored regression passes on an NVIDIA A100 for scalar LHS/RHS with `F32`, `F64`, `I32`, and `I64`; the source contract and CUDA feature compilation pass. The A100 run used `CUDA_PATH=/usr/local/cuda-12.6`, `CUBECL_DEBUG_LOG=0`, and CUDA/cuTENSOR library paths in `LD_LIBRARY_PATH`. Overall batch verification remains incomplete. | The special path is intentionally limited to exactly one rank-0 operand; arbitrary implicit broadcasting remains rejected. |
| #1358 | Stale / Out of Scope (false positive guarded) | Current CPU `pow` requires equal shapes in `typed_pow_with_pool` and rejects both rank-0/vector operand orders with exact `ShapeMismatch { op: "pow", lhs, rhs }`. The historical claim that CPU accepts scalar `pow` is false at the working hash. CUDA now performs the same operand-order shape preflight before its raw launcher. | An ignored CPU/CUDA regression covers both operand orders and `F32`, `F64`, `I32`, and `I64`; a source contract prevents `pow` from entering the CUDA scalar launcher and preserves the exact preflight. | No scalar `pow` behavior was added. A future change requires an accepted CPU/public semantic contract first. |
| #1359 | Implemented locally; focused A100 verification passed | Before the fix, the focused A100 regression failed because a full-axis CUDA reduction exposed public shape `[1]` while CPU and the expected scalar contract returned `[]`. CUDA reduction public metadata now retains the exact rank-0 `[]` shape when all axes are reduced. The existing CubeCL binding boundary continues to translate rank-0 tensors into private one-element shape/stride metadata, so no invalid zero-length CUDA metadata or buffer reinterpretation is introduced. Reduction input binding validation now precedes the zero-output shortcut. | After separating public and launch metadata, the ignored A100 regression passes with CPU/CUDA shape and value parity for full-axis sum/prod over `F32`, `F64`, `I32`, `I64`, `C32`, and `C64`, and min/max over their supported real/integer dtypes. A focused metadata test guards the public/private shape separation. The A100 RED/GREEN runs used `CUDA_PATH=/usr/local/cuda-12.6`, `CUBECL_DEBUG_LOG=0`, and CUDA/cuTENSOR library paths in `LD_LIBRARY_PATH`. Overall batch verification remains incomplete. | CubeCL still receives `[1]` as private launch metadata for rank-0 bindings; that workaround must remain confined to the dispatch boundary. |
| #1360 | Implemented locally; narrowed current paths; focused A100 verification passed | The A100 RED matrix confirmed that CUDA `dynamic_slice`, `gather`, and `scatter` all accepted fractional, NaN, positive/negative infinity, and values just outside the exact float-index bounds for both `F32` and `F64`; the earlier report that gather already rejected a fraction was stale. Integral values and both signs of the exact boundaries (`F32`: 2^24, `F64`: 2^53) already matched CPU. CUDA now scans float index tensors on device, atomically selects the first invalid flat index, copies only that value into the same two-element validation flag, and reads that flag after synchronization; it never downloads the index tensor or falls back to CPU. | The unchanged A100 matrix now passes all three operations for `F32`/`F64`: integral valid, fractional, NaN, positive/negative infinity, both signed exact boundaries, and both signed just-outside values, with exact CPU `InvalidConfig { op: "index_tensor", message }` parity. RED and GREEN used `CUBECL_DEBUG_LOG=0`, `CUDA_PATH=/usr/local/cuda-12.6`, and CUDA/cuTENSOR paths in `LD_LIBRARY_PATH`. CUDA feature no-run compilation also passes. | Float starts for Bool `dynamic_slice` remain excluded because CPU does not provide that capability. The device check adds one small validation flag allocation and synchronization for non-empty float index tensors; empty validated inputs skip the flag entirely. Overall batch verification remains incomplete. |
| #1361 | Implemented locally; focused verification passed | CUDA `abs` now canonicalizes both zero signs to `+0.0`, and CUDA `sign` explicitly preserves NaN before its zero/finite branches. The ignored CUDA regression test covers `F32`/`F64` `abs(-0.0)` bit parity and `sign(NaN)` plus finite and zero sanity cases. | The regression test first failed on `F32` `abs(-0.0)` (`0x80000000` versus CPU `0x00000000`) and then passed for both dtypes on an NVIDIA A100 after the kernel fix. Overall batch verification remains incomplete. | CubeCL or CUDA fast-math changes could affect NaN/signed-zero semantics; retain the hardware regression test. |
| #1362 | Implemented locally; focused A100 verification passed | CUDA explicit `cast` now covers all 49 CPU-supported pairs across `F32`, `F64`, `I32`, `I64`, `Bool`, `C32`, and `C64` through shared numeric, Bool-truthiness, complex projection/injection, and complex-width kernel families. Real or complex-real to integer casts atomically select the first invalid value on device, download only a two-scalar validation flag, and reproduce CPU's exact typed `InvalidConfig`; empty inputs allocate no flag. Checked `convert` remains gated by the existing promotion lattice. | The source-derived 49-pair CPU/CUDA matrix passes on an NVIDIA A100, including positives, negatives, zero, NaN truthiness, complex projection/injection, integer narrowing, and exact NaN/infinity/out-of-range errors. CUDA feature no-run compilation passes. | The validation path adds one small flag allocation and synchronization only for non-empty fallible casts; no full tensor download or host fallback is used. Overall batch verification remains incomplete. |
| #1363 | Implemented locally; focused A100 verification passed | Before the fix, the A100 repro failed at `transpose(Bool)` with `UnsupportedOpDType { op: "transpose", dtype: Bool, backend: Cuda }`. CUDA now routes CPU-supported Bool shape/data-movement operations through one-byte copy/index kernels rather than numeric traits. CPU and CUDA both reject Bool additive scatter with exact `BackendFailure { op: "scatter", message: "Bool data tensors are not supported by additive scatter" }` parity. | After the fix, 2 focused Bool A100 tests pass, covering non-empty values/shapes, empty/zero-length outputs, and exact invalid-config error parity for transpose, broadcast, diagonal extraction/embedding, triangular masks, slice, dynamic slice with `I32`/`I64` starts, pad, concatenate, reverse, and gather. The complete CUDA structural module passes 15/15 and indexing module passes 7/7; launch contracts pass 25/25, kernel metadata contracts 3/3, public-surface contracts 12/12, and CUDA feature no-run compilation passes. Runs used `CUBECL_DEBUG_LOG=0`, `CUDA_PATH=/usr/local/cuda-12.6`, and `LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:/usr/local/cuda-12.6/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu/libcutensor/12:$LD_LIBRARY_PATH`. This closes only the CPU-supported Bool structural/indexing gap; float-start validation remains #1360. | Bool arithmetic, reductions, linalg, additive scatter, float-start validation, and CPU-new behavior remain excluded. |
| #1364 | Implemented locally; focused A100 verification passed | CUDA `abs` now maps `C32 -> F32` and `C64 -> F64` through CubeCL's native complex `abs` intrinsic, which lowers to stable CUDA `hypotf`/`hypot`; no CPU behavior or other complex analytic operation changed. The ignored parity test covers zero, `(3,4)`, `(5,12)`, scaled very large/small components, and CPU-relevant NaN/infinity cases. | The focused regression first failed with `UnsupportedOpDType { op: "abs", dtype: C32, backend: Cuda }`, then passed on an NVIDIA A100 for both complex dtypes with real output dtype and CPU value/class parity. Overall batch verification remains incomplete. | CubeCL/CUDA intrinsic edge semantics could change across toolchain upgrades; retain the hardware parity test. No broader complex analytic support is implied. |
| #1365 | Implemented locally; narrowed current contract; focused A100 verification passed | The current bug is confined to the optional CUDA `TensorFusion` hook: classification now returns `Ok(None)` for identity-view inputs with incompatible runtime shapes. The direct regression first failed with `ShapeMismatch { op: "fused_elementwise", lhs: [3], rhs: [] }`, then passed for `[3]` and `[]`; a companion regression confirms a plan/runtime dtype descriptor mismatch remains a typed `BackendFailure`. The reported user-visible scalar-broadcast fallback path is stale/unproven: current segmented scalar broadcasting uses explicit `BroadcastInDim` metadata, which exits fusion classification earlier as a nonidentity view, while ordinary unfused CUDA `add`/`mul` do not themselves accept `[3]` with `[]`. | Direct fusion shape-defusal and dtype-corruption regressions pass on an NVIDIA A100; the full CUDA fusion module, CUDA feature compilation, and relevant launch contracts pass. Overall batch verification remains incomplete. | `Ok(None)` only declines this optional backend optimization; it does not promise that every shape combination is executable by the caller's fallback path. Malformed plans and corrupted tensor descriptors remain hard errors. |
| #1366 | Implemented locally; focused A100 verification passed | CUDA now accepts the CPU-established matching-precision `F32` rank-0/`C32` and `F64` rank-0/`C64` cases for `add`, `sub`, `mul`, and `div` in both operand positions. A shared private kernel reads the real scalar once and promotes it in registers while operating directly on interleaved complex components; it performs no host transfer and allocates no full-size promoted scalar tensor. Every operation preserves CPU's explicit `Complex(real, +0)` promotion and generic `num_complex` component operation order, including zero cross terms and division norm squares, so overflow, underflow, NaN, infinity, and signed-zero behavior remains aligned. | The focused regression first failed with `DTypeMismatch { op: "add", lhs: F32, rhs: C32 }`, then passed on an NVIDIA A100 for both dtype pairs, all four operations, both operand orders, signed zero, NaN/infinity cross terms, ordinary finite and overflow-scale complex values, output dtype/shape, and exact rejection of mixed non-scalars, cross-precision scalars, and matching-precision mixed `pow`/`rem` in both orders. The CUDA no-run build and launch/source contracts pass. Overall batch verification remains incomplete. | Promotion is intentionally limited to matching-precision real rank-0 scalars. General mixed non-scalar broadcasting, cross-precision promotion, `pow`, and `rem` remain unsupported. |

## Final-review indexing preflight follow-up

Final review found that CUDA float-index value validation could allocate a
device flag and synchronize before operation metadata was validated, and that
scatter zero-domain returns could bypass operand/index/update binding and
atomic-capability checks. The source-order regression was RED before the fix:
`dynamic_slice_typed` reached the checked output-domain assertion only after
`I::validate`. After reordering, metadata and checked launch counts precede all
input residency/binding and capability checks, which precede nonempty float
index scans; allocation, copy, zero-domain returns, and launches follow.

The A100 evidence covers exact structural errors for mixed invalid
configuration plus NaN/fractional indices across `F32`/`F64` dynamic slice,
float/complex/Bool gather data, and float/complex scatter. Separate valid-config
cases assert exact CPU/CUDA invalid-index `Error` equality. Bool dynamic slice
uses its permitted integer start dtype and checks exact invalid-config parity.
CPU currently converts float index values before operation-specific config
validation, so when both inputs are independently invalid its error precedence
intentionally differs from the CUDA cheap-metadata-before-device-scan contract;
CPU behavior was not changed.

Zero-output and zero-update scatter tests cover float and complex data,
cuda:1 placement metadata over real cuda:0 buffers, malformed device-placement
host buffers, and a second same-device `CudaBackend`. Runtime residency is by
CUDA device ordinal, not wrapper identity, because same-device CubeCL clients
share the primary context. The focused tests and the full CUDA indexing module
passed on an NVIDIA A100 (`11/11`) with `CUBECL_DEBUG_LOG=0`,
`CUDA_PATH=/usr/local/cuda-12.6`, and CUDA/cuTENSOR paths in
`LD_LIBRARY_PATH`. Launch and kernel-metadata contracts plus CUDA no-run
compilation also passed. Final overall batch verification remains pending.

## Batch close conditions

An issue receives `Closes #...` only after its row's focused verification
passes. The final batch also requires the repository checklist, the ignored
CUDA suite in the supported CUDA environment, a same-pattern neighborhood
scan, updated active capability documentation, and reconciliation of this
ledger with the latest issue comments before the PR is opened. Partially
resolved or deferred items remain `Refs #...` with the residual risk stated
explicitly. A narrowed issue may still receive `Closes #...` when its narrowed,
current close condition is fully resolved.
