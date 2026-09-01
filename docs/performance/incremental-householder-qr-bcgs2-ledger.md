# Incremental QR BCGS2 benchmark correspondence ledger

The Phase-5 primary comparator reproduces the explicit-Q two-pass BCGS2 append
from tensor4all-rs#694 at commit
`da0775a208006352f6e5eab18bc6bb09ca39a1f6`, source
`crates/tensor4all-tensorbackend/src/incremental_qr.rs`.

| #694 operation | Phase-5 benchmark operation |
| --- | --- |
| `q_adjoint = matrix_adjoint(q)` | rank-2 transpose of real F64 Q |
| `first_projection = q_adjoint * columns` | `dot_general(Qᵀ, B)` |
| `first_residual = columns - q * first_projection` | `B - dot_general(Q, first_projection)` |
| `correction = q_adjoint * first_residual` | second `dot_general(Qᵀ, residual)` |
| `residual = first_residual - q * correction` | second reconstruction/subtraction |
| `projection = first_projection + correction` | elementwise add |
| `factorize_backend(residual)` | backend-native QR of only the residual block |
| `q.append_columns(appended_q)` | backend concatenate on Q columns |
| `assemble_r(old, projection, appended_r)` | block R assembly from top, bottom-left zeros, and residual R |

All bulk operations use the same tenferro backend session as the compact path.
Inputs, initial rank, block schedule, provider object lifetime, synchronization,
and timed boundaries are identical.

The benchmark intentionally omits #694's rank-deficiency fallback and
inverse-adjoint/error-estimate update because the frozen deterministic matrices
are full rank and the tenferro compact state does not provide that estimator.
Omitting those costs favors BCGS2, so it cannot create a false compact-path
speedup. Final positive-diagonal canonicalization and correctness
materialization are outside both timed regions.

The correspondence is guarded by source tests that require two projection
passes, residual-only QR, and block assembly in the benchmark source. Any
change to this ledger, the pinned commit, or the benchmark BCGS2 implementation
invalidates existing performance artifacts and requires design re-review before
a new full paired run.
