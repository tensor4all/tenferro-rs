# Incremental Householder QR

**Issue:** [#1735](https://github.com/tensor4all/tenferro-rs/issues/1735)

**Status:** approved for implementation. `reviewer-flash` Round 3 returned
**Correct-to-merge** after the Round-2 findings were fixed; see the
[design work log](../worklogs/2026-09-01-incremental-householder-qr-design.md).

## Purpose

Adaptive algorithms need to append column blocks without refactorizing the
accumulated matrix or materializing its full orthogonal factor. The existing
`qr` API returns explicit `(Q, R)` and discards provider reflector state, so it
cannot provide that operation efficiently.

This design adds an opaque, backend-neutral compact QR state to
`tenferro-linalg`. The state stays on its input device. CPU and CUDA providers
execute Householder factorization and reflector application natively; no path
may download to the host, reconstruct the accumulated matrix, or call one-shot
QR during append.

## V1 scope

V1 supports:

- rank-2 matrices;
- `F32`, `F64`, `C32`, and `C64`;
- initialization from a matrix;
- initialization from compatible factors `(Q, R)`;
- functional column-block append;
- extraction of the thin upper-trapezoidal `R`;
- materialization of a contiguous range of thin-`Q` columns;
- concrete `Tensor`, eager `EagerTensor`, and traced `TracedTensor` surfaces;
- CPU-faer, CPU-BLAS/LAPACK, and CUDA execution;
- JVP and VJP through every operation on the full-rank differentiable domain.

V1 does not add pivoting, rank revelation, batching, arbitrary Q-column index
lists, serialization of provider-private handles, or WebGPU/ROCm execution.

## Public state and operations

The public state is:

```rust
pub struct HouseholderQr<T> { /* private */ }
```

`T` is `Tensor`, `EagerTensor`, or `TracedTensor`. The payload is private and
has a bounded summary `Debug` implementation; users cannot treat packed
reflectors as ordinary tensor values.

The canonical operation vocabulary is:

```text
A.householder_qr(...)                  -> HouseholderQr
HouseholderQr::from_factors(Q, R, ...) -> HouseholderQr
state.append_columns(B, ...)           -> HouseholderQr
state.r(options, ...)                   -> R
state.q_columns(start..end, options, ...) -> Q[:, start..end]
```

Concrete operations take an explicit `LinalgBackend`. Eager and traced
operations use their owning runtime. `append_columns` consumes and returns the
state where the Rust surface permits it, allowing provider buffer reuse without
observable mutation.

`QrOptions` controls output gauge for `r` and `q_columns`. The compact state
itself always uses raw provider-neutral Householder gauge. Calling `r` and
`q_columns` with the same options yields mutually consistent factors. Gauge
post-processing is part of the selected provider execution; it must not call
the existing host-only gauge helper for backend-resident tensors.

No additional public raw-factor accessors or provider-specific state types are
added.

## Mathematical state

For an accumulated matrix `A` of shape `m x n`, let `k = min(m, n)`. The
private logical state contains:

- `packed`, shape `m x n`: `R` on and above the diagonal and Householder tails
  below it;
- `coeff`, shape `k`: one coefficient per implicit-unit reflector.

The provider-neutral convention is

```text
H_j = I - coeff[j] * v_j * v_j^H
A   = Q R
Q   = H_0 H_1 ... H_{k-1}
```

`v_j` is zero before row `j`, one at row `j`, and stores its remaining tail in
`packed`. Fixed-position zero reflectors remain present, including for
rank-deficient inputs, so every provider exposes the same shapes and append
semantics.

LAPACK and cuSOLVER `tau` values map directly to `coeff`. A provider with a
blocked or reciprocal coefficient representation converts at this state
boundary. Provider handles and blocked workspace are execution resources, not
part of `HouseholderQr`.

## Validation

Initialization from `A` requires rank 2 and a supported dtype.

`from_factors(Q, R)` requires:

- rank-2 inputs with the same supported dtype and placement;
- `Q.shape = [m, s]` and `R.shape = [s, n]`;
- `s <= min(m, n)`;
- `R` is exactly upper trapezoidal, as produced by QR factorization;
- the same eager runtime or concrete backend provider.

It does not assume that `Q` is already orthonormal. The provider factorizes
`Q = Q_h T` and folds the small triangular factor into `R_h = T R`, producing
a compact state for the mathematical product `Q R`. Requiring upper-trapezoidal
`R` keeps `R_h` upper trapezoidal and makes this bridge possible without
refactorizing the full `m x n` product. Rank-deficient factors remain valid
primal inputs.

For append, `B` must have shape `m x p` and match the state's dtype, placement,
runtime, and provider. A zero-column block returns the state unchanged.

`q_columns(start..end)` requires `start <= end <= k` and returns shape
`m x (end - start)`. `r()` returns shape `k x n`. Zero-sized matrix dimensions
and empty Q ranges preserve these shape rules without issuing invalid provider
calls.

Known invalid metadata is rejected at API or graph-build time. Symbolic
constraints are retained and reported through the existing compile/execution
error phases. Provider, numerical, and validation failures use existing typed
linalg/tensor error categories and preserve their source chain.

## Append algorithm

For old width `n`, `k = min(m, n)`, and a block `B` of width `p`:

1. Apply the existing reflector sequence to compute `Y = Q^H B` without
   materializing `Q`.
2. Copy `Y[0..k, :]` into the upper-right block of the enlarged `R`.
3. Householder-factor `Y[k..m, :]` in place.
4. Store the new reflector tails and coefficients after the old sequence.
5. Return state shapes `packed: m x (n + p)` and
   `coeff: min(m, n + p)`.

When `k == m`, append only applies `Q^H` and extends `R`; it creates no new
reflectors. Existing columns are never refactorized.

`r()` extracts the upper trapezoid. `q_columns(start..end)` initializes only the
requested identity columns and applies the reflector sequence, so full `Q` is
materialized only when the caller requests the full range.

## Gauge and rank deficiency

Raw state follows provider QR gauge. `QrGauge::PositiveDiagonal` applies the
same phase/sign convention as existing `qr_with_options` at output time. The
phase is computed from the matching diagonal of `R` and applied consistently
to requested Q columns and R rows.

Primal factorization and append are defined for rank-deficient inputs and do
not choose a numerical-rank tolerance. Small or zero diagonal entries are
returned as produced by the provider. AD is defined only on the ordinary
full-rank differentiable domain; a singular triangular solve in a derivative
reports the existing typed numerical failure.

## Eager, traced, and runtime representation

The linalg extension vocabulary gains pure operations and matching AD-manifest
entries for:

- factorization from one matrix;
- factorization from `(Q, R)`;
- append from `(packed, coeff, B)`;
- R extraction from `(packed, coeff)`;
- Q-column materialization from `(packed, coeff)`.

The phase-2 manifest marks each entry `Unsupported`; phase 3 changes an entry
only after its oracle and numerical tests land.

State-producing operations return ordered `(packed, coeff)` tensor outputs;
`HouseholderQr<T>` is the only public wrapper that can construct or consume
that pair. Payloads record all shape-affecting attributes, including the Q
column range and gauge, so extension fingerprints and transform-cache keys stay
structural and deterministic.

The runtime prepares these operations through the existing linalg extension
module. Missing provider support is explicit; extension execution never falls
back to eager execution, CPU, full QR, or host reconstruction.

## Backend contract

Hidden `LinalgBackend` hooks operate on the internal compact-state result type.
Their defaults return `Unsupported`. The hooks are factor, `from_factors`,
append, R extraction, and Q-column application; they do not expose raw LAPACK
routines publicly.

### CPU-BLAS/LAPACK

Use `*geqrf`, `*ormqr`/`*unmqr`, and trailing-block `*geqrf`. Dimension and
workspace sizes are checked before integer conversion and FFI. Scratch comes
from the existing session-owned linalg pool and threading remains provider
controlled.

### CPU-faer

Use faer's public `make_householder_in_place` elementary-reflector primitive
inside the configured `CpuExecutionContext`, storing its scalar `tau` in
`coeff`. Apply reflectors with faer matrix primitives in the required order.
The existing convenience QR path uses a blocked `block_size x k` coefficient
matrix, so it is not the state constructor and is not converted by merely
reading one row. A later blocked implementation may derive transient WY factors
from the scalar state, but those factors remain session scratch. Fixed-position
zero reflectors must be preserved.

### CUDA

cuSOLVER provides `*geqrf` and `*orgqr`/`*ungqr`, but not LAPACK's
`*ormqr`/`*unmqr`. V1 therefore uses cuSOLVER `*geqrf` for new reflectors and
applies the stored sequence on the active stream with cuBLAS:

1. `gemv` with conjugate transpose computes `w = v^H C`;
2. scale by `tau` (or `conj(tau)` for the adjoint reflector application);
3. real `ger` or complex unconjugated `geru` performs `C -= v w`.

Reflectors are enqueued one at a time in the mathematically required order:
`Q^H C` applies `j = 0, ..., k - 1` with `conj(tau[j])`, while `Q C`
applies `j = k - 1, ..., 0` with `tau[j]`. Each `gemv`/rank-1 update covers the
matrix block in parallel and no GPU thread loops over an unbounded tensor
domain. The same routine applies `Q^H` during append and applies `Q` to
requested identity columns. A later blocked-WY
optimization may replace this routine only if the performance gate requires it;
there is no full-Q, one-shot-QR, host, or CPU fallback.

`QrGauge::PositiveDiagonal` is also provider-owned on CUDA: a linalg-owned
same-device kernel computes diagonal phases and scales requested Q columns and
R rows consistently. Phase 4 shares that kernel with existing CUDA
`qr_with_options`, replacing its currently inapplicable host-only default gauge
path. The kernel launches over the output domains and does not download scalar
phases.

All CUDA work runs inside the scoped `CudaExecSession::with_raw` or typed
CubeCL session appropriate to the call. Workspace and stream-retention behavior
follow `gpu-backend-design.md`; synchronization failure must not release
allocations that an unfinished vendor call can still access. Missing cuSOLVER
or cuBLAS symbols are typed provider errors, not fallback triggers.

## AD semantics

AD differentiates the mathematical accumulated matrix, not provider packed
bytes. The opaque state has an abstract tangent:

- the tangent/cotangent of the `packed` output means `dA` / `A_bar` for the
  accumulated matrix;
- `coeff` is auxiliary and has no tangent or cotangent.

Because raw fields are private, this interpretation cannot leak into ordinary
tensor operations.

Rules are:

```text
factor(A):
    dstate = dA

from_factors(Q, R):
    dstate = dQ R + Q dR

append(state(A), B):
    dstate_new = concatenate(dA, dB, axis=1)

transpose(append):
    split A_new_bar by the old width into A_bar and B_bar
```

The append transpose obtains the old width from the packed-state input through
a runtime `DimExpr::InputDim` shape source. It must not require an exact static
extent or reinterpret an upper bound as the slice position.

For `r` and `q_columns`, the rule materializes the required primal thin Q/R in
the derivative program, reuses the existing QR gauge and derivative formulas,
and selects the requested output tangent. The raw-state factors supplied to the
existing QR derivative must use exactly the gauge represented by the public
output; gauge-consistency reconstruction and finite-difference tests enforce
this invariant. Reverse mode remains `SupportedViaLinearize`. AD execution may
materialize full thin factors as residual work; primal execution may not.

The `from_factors` transpose follows the real-inner-product convention:

```text
Q_bar = A_bar R^H
R_bar = Q^H A_bar
```

Complex JVP/VJP follows the repository's JAX-style complex convention. Before
these rules become supported, the linalg AD manifest records the operations as
`PendingOracle` or `Unsupported`, and the corresponding `tensor-ad-oracles`
family must land.

## Tests

Primal tests cover all four dtypes where the provider supports them:

- reconstruction after initialization and each of multiple appends;
- orthogonality of full and selected Q columns;
- agreement with one-shot QR under the selected gauge;
- `from_factors` with orthonormal, non-orthonormal, and rank-deficient factors;
- rank-deficient matrices, zero-column append, empty Q ranges, empty matrix
  dimensions, tall/square/wide inputs, and tall-to-wide transitions;
- invalid rank, shape, dtype, range, placement, runtime, and provider cases;
- CPU-faer/CPU-BLAS parity and CUDA hardware coverage;
- source-contract checks against hidden host transfer and full refactorization.

AD tests cover JVP and VJP with respect to the initial matrix, both factors,
and each of at least two appended blocks. They include Q-only, R-only, and
combined losses; all four dtypes; tall, square, and wide transitions; gauge
parity; finite differences; and the accepted oracle family. Rank-deficient AD
is outside the differentiable test domain.

## Performance gate

The primary comparison is compact Householder append against tensor4all's
current explicit-Q BCGS2 append, not only against one-shot QR. The paired
experiment also records one-shot QR as a diagnostic baseline.

Before measurements begin, the phase work log must freeze the exact baseline
and candidate commits, benchmark source, SRC-derived matrix shapes and block
widths (including rank increment 3 and maximum rank 32), provider/thread and
hardware settings, case list, statistic, repetitions, noise gate, and numeric
pass thresholds. The accepted tensor4all-rs #694 one-thread bonds 32, 64, and
128 are mandatory CPU cases. CUDA cases use the same logical matrices when
memory permits.

The gate reports every case as `PASS`, `FAIL`, or `INCONCLUSIVE`. Append must
show no full-refactorization scaling, hidden transfer, or correctness
regression. Thresholds may not be changed after candidate results are seen.

## Delivery and review gates

1. Land this durable design after an independent design-review verdict.
2. Land opaque API/IR, `from_factors`, CPU-BLAS and CPU-faer primal execution;
   mark AD unsupported in the manifest.
3. Land the oracle family, then JVP/VJP support.
4. Land CUDA bindings, execution, and hardware tests.
5. Run the predeclared performance gate and finish user documentation.

Each phase is independently mergeable and receives design-before-code and
post-diff review. The umbrella issue remains open until all phases and the
repository-scale final audit pass.

## References

- Camaño, Epperly, and Tropp, adaptive randomized tensor algorithms,
  [arXiv:2504.06475](https://arxiv.org/abs/2504.06475), especially Appendix C.3.
- LAPACK `GEQRF`, `ORMQR`/`UNMQR`, and `ORGQR`/`UNGQR` operation contracts.
- tenferro [`AD contract`](../spec/ad-contract.md) and
  [`GPU backend design`](gpu-backend-design.md).
