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
- #1360 remains `Verify First`. Its contract is exact CPU parity, including
  the float-index bounds `F32: 2^24` and `F64: 2^53` and the exact
  `InvalidConfig` variant, operation label, and message parity. The affected
  operation family must be narrowed from current source and repros before
  implementation.

## Classification ledger

| Issue | Classification | Current evidence | Close condition and verification target | Residual risk |
| --- | --- | --- | --- | --- |
| #1353 | Auto Fix (maintainer-approved policy) | CPU/CUDA floating remainder kernels use value propagation, while integer division/remainder retain zero-divisor checks. The branch already contains the approved IEEE policy design. | Active tensor/primitive specs state the policy; focused CPU and ignored CUDA tests cover `F32`/`F64` division and remainder by signed zero, result class, and signed-zero behavior. | Other analytic operations can have independent IEEE edge discrepancies; scan them, but do not silently expand this issue. |
| #1356 | Auto Fix | Generic CubeCL `F::new(0.0)` / `F::new(1.0)` literals produce rustc future-incompatible fallback warnings. | CUDA feature build/test emits no warning for these sites; source check confirms all same-pattern literals in the kernel family are typed. | CubeCL or rustc upgrades may expose equivalent literals in other kernel families. |
| #1357 | Auto Fix | CUDA `div`/`rem` use direct equal-shape launches; issue repros confirm rank-0 scalar RHS failures while CPU accepts scalar broadcast. | CPU/CUDA parity tests pass for scalar LHS and RHS, float and integer `div`/`rem`; integer zero-divisor errors remain typed. | Avoid broadcast materialization and preserve equal-shape launch contracts. |
| #1358 | Auto Fix | CUDA `pow` uses a direct checked-binary path; the CUDA repro rejects vector/rank-0 input accepted by CPU. | Scalar LHS/RHS parity tests pass for supported float/integer cases, including integer negative-exponent errors. | Scalar dispatch must not weaken existing integer domain validation. |
| #1359 | Auto Fix | CUDA reduction metadata substitutes `[1]` when all reduced axes leave an empty output shape; CPU returns rank-0 `[]`. | All-axis reductions return public shape `[]` with CPU/CUDA value parity across supported reductions/dtypes. | CubeCL may still require an internal one-element launch shape; it must not leak into public metadata. |
| #1360 | Verify First | CPU `try_index_tensor` rejects non-finite, fractional, and out-of-bound float indices using `InvalidConfig { op: "index_tensor", .. }`; exact bounds are `F32` 2^24 and `F64` 2^53. CUDA repros confirm acceptance in `dynamic_slice` and `scatter`; `gather` reportedly already errors for the tested fraction. | First narrow every affected current path. Then test CPU/CUDA exact `InvalidConfig` variant, `op`, and message parity for non-finite, fractional, boundary-accepted, and just-outside-bound values in each confirmed operation. | Device validation must avoid lossy cast-before-check. Gather may be stale or fail for a different reason; do not claim it fixed without exact error parity. |
| #1361 | Auto Fix | CUDA comparison-based `abs` preserves `-0.0`, and comparison-based `sign` maps `NaN` to a finite value; CPU uses IEEE-compatible behavior. | Bit/class parity tests cover `F32`/`F64` `abs(-0.0)` and `sign(NaN)` plus ordinary values. | CubeCL intrinsic behavior and fast-math settings may affect NaN/signed-zero semantics. |
| #1362 | Auto Fix (explicitly approved CPU capability parity) | CUDA `cast` has unsupported branches for CPU-supported explicit cast families; the issue repro confirms `F32 -> I32` and identifies further CPU-supported cases. | A source-derived CPU cast matrix is tested on CUDA for every newly enabled pair, including edge/projection behavior already defined by CPU; capability docs match. | No new cast semantics may be invented. NaN, infinity, overflow, complex projection, and Bool conversion must follow current CPU results exactly. |
| #1363 | Auto Fix (explicitly approved CPU capability parity) | CUDA dispatch rejects Bool for several shape/data-movement operations that CPU implements dtype-polymorphically; transpose has a confirmed repro. CPU explicitly rejects Bool additive scatter. | Focused parity tests cover only CPU-supported Bool structural/indexing paths; docs reflect the resulting matrix. Bool scatter remains rejected with CPU-parity error behavior. | The issue wording overstates scatter support; arithmetic or additive semantics must not be smuggled into shape-only parity work. |
| #1364 | Auto Fix (explicitly approved CPU capability parity) | CPU `abs` maps `C32 -> F32` and `C64 -> F64`; CUDA currently returns `UnsupportedOpDType` for complex input. | CUDA magnitude tests match CPU for both complex dtypes, including zero and representative finite values, without changing CPU behavior. | Stable hypot/overflow behavior and output dtype must match CPU; no broader complex analytic support is implied. |
| #1365 | Auto Fix | CUDA fusion classification returns `ShapeMismatch` for scalar-broadcast candidates instead of `Ok(None)`, preventing the existing unfused fallback. | Direct fusion test returns `Ok(None)` for unsupported shape combinations, and graph execution succeeds via unfused CPU/CUDA-parity behavior. | Do not accidentally classify invalid plans as merely unsupported; preserve hard errors for malformed plans. |
| #1366 | Auto Fix | CUDA rejects real rank-0 plus complex tensor with `DTypeMismatch`; CPU promotes the real scalar and broadcasts it. | CPU/CUDA parity tests cover both operand orders and the existing supported operation family for `F32`/`C32` and `F64`/`C64`. | Restrict promotion to CPU-established scalar cases; do not add general mixed-dtype broadcasting. |

## Batch close conditions

An issue receives `Closes #...` only after its row's focused verification
passes. The final batch also requires the repository checklist, the ignored
CUDA suite in the supported CUDA environment, a same-pattern neighborhood
scan, updated active capability documentation, and reconciliation of this
ledger with the latest issue comments before the PR is opened. Deferred or
narrowed items remain `Refs #...` with the residual risk stated explicitly.
