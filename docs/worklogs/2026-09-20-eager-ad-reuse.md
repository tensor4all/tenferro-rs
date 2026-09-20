# Eager backward residual reuse and large-matrix solve gaps

## Decisions

Eager reverse mode retains declared residual values, not merely their metadata.
Retention shares one DAG of residual records along the existing per-value eager
trace lifetime, and stores value records rather than `EagerTensor` owners so the
runtime and gradient-slot cycles cannot close. A dropped decomposition handle
therefore cannot discard a residual that a surviving loss still needs.

Differentiation stays on the original semantic program; saved primal values are
bound only for numerical derivative execution. Returned functional gradients keep
the original mathematical derivative program so higher-order AD still sees the
primal dependency, and saved values never become constants in the differentiable
source. Prepared-derivative caches hold programs and input-position maps, never
sample-specific residual tensors.

All active input gradients are consolidated into a single VJP that binds retained
residuals as explicit execution inputs; the existing residual declarations are the
retention authority. No decomposition-specific arithmetic or replacement kernel
was added.

Tracked eager `solve` reuses the existing traced composite (`LuFactor` plus
`LuSolvePrepared`) instead of re-deriving a factorization. This follows PyTorch
`linalg_solve_backward`, which reuses saved LU and pivots for the adjoint solve and
the saved solution for the matrix cotangent while keeping an explicit dependency on
the matrix for higher-order AD. The prepared op already carries the matrix
separately from its numerical factors, so its existing JVP/VJP semantics already
preserve that dependency; its semantic transpose now consumes the saved solution
instead of emitting a second primal solve. Untracked eager solve keeps the direct
path.

The LU kernels' destructive-`getrf` input copy was the located cost at large
sizes. That fix is not part of this change: `main` already took the LAPACK scratch
from the session buffer pool (`pooled_copy`/`pooled_zeroed`/`release_scratch`,
landed as the LAPACK scratch-pool change), so an earlier local variant of the same
optimization was dropped during rebase rather than duplicated. This change keeps
only the residual reuse and the eager solve reuse above.

## Verification conclusions and constraints

Eager residual binding is accepted: the complete paired AD-family check passed with
every predeclared variability and regression gate, and an independent LAPACK call
counter shows the eager SVD forward+backward decomposition count falling from two
to one per workflow. Eager `solve` factorization counts fall from four to one; the
prepared trace remains at one. Numerically it is covered by real and complex,
batched and non-batched, first- and higher-order finite-difference and
gradient-of-gradient checks, plus dropped-handle lifetime, active-mask isolation,
cache-rebinding, and Hessian-vector tests.

The first LU-reuse performance candidate is **not** promoted: its geometric-mean
gate of 2.0x was missed at 1.978x, and the earlier intermediate pair missed at
1.262x. Those numbers, and the failed first pair, are retained rather than rounded
or selectively rerun. The accepted change is the reuse itself plus the located
data-movement fix.

LU overhead above the `dgetrf` floor was measured with an interleaved A/B on the
solve row plus provider probes separating the LAPACK call from the kernel's own
overhead; it dropped from about 6.6 ms to about 1 ms at n=1024 once the destructive
copy came from the pooled scratch instead of a fresh dense `Vec`. That measurement
justified the pool change that `main` now carries; a locally identical variant was
dropped here during rebase. See the benchmark report and raw archive for the tables
and protocols.

Remaining gap, measured rather than assumed. The suite's eager rows are slower
than their prepared-trace counterparts, but a dedicated probe of the n=1024 solve
forward+backward row in a fresh, clean-context process shows the gap is small and
is not caused by per-operation session entry:

| variant at n=1024, 1T, release | ms |
|---|---:|
| forward, untracked public eager (direct `Solve`) | 20.7-21.9 |
| forward, tracked eager (`LuFactor` + `LuSolvePrepared`, two entries) | 23.0-24.7 |
| forward, both ops inside one entered session, warm pool | 22.0 |
| forward, prepared trace | 22.6-23.6 |
| row, tracked eager forward + sum + backward | 27.0-27.9 |
| row, prepared trace of the same graph | 24.6 |

One entered session for both linalg operations is not faster than the two
separate entries once the pooled workspace is warm, so session batching is not the
gap. The first, cold measurement of the one-session variant was 28-30 ms, which is
pool allocation rather than entry cost. The residual eager-vs-trace difference in
a clean context is about 2.5 ms (10%), consistent with AD-side bookkeeping, the
gradient extraction/store, and rebinding the derivative program's inputs. The much
larger eager solve numbers seen inside the full suite (up to about 45 ms) are a
run-context effect of memory/buffer-pool state, not an execution-strategy cost, so
they must not be read as an eager architecture penalty. Session reuse and the
prepared-program executor already exist and are used; the located per-call copies
were the fixable part. No GPU measurement, no sub-64 latency claim, and no claim
that value-only SVD or checkpoint scheduling improved.

Evidence: benchmark `result/amd-cpu/cpu/large_ad.md` and `large_ad_raw.tar.gz`
contain the paired tables, protocols, source snapshots, raw samples, instrumented
call counts, and reproduction commands.
