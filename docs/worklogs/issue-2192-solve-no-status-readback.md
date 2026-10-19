# CUDA solve without per-call status readbacks

## Decisions

- The CUDA forward solve path (`solve`/`lu_factor_solve`) no longer scans the
  packed LU diagonal for near-zero pivots. That scan ran five kernels, a full
  stream synchronize, and two scalar downloads on every call; `torch.linalg.solve`
  does not perform it.
- The same path skips the cuSOLVER `getrf` `info` download and the permutation
  parity kernel. A singular input therefore reaches the triangular solve and is
  left to the caller's residual check. The public `lu_factor`/`lu` operations
  keep both the status read and parity, so their `Singular` contract is
  unchanged.
- No public strictness knob was added. The contract is documented on the public
  solve surfaces and in `docs/design/linalg.md` and
  `docs/design/gpu-backend-design.md`.

## Verification conclusions and constraints

A100, `tenferro-benchmark` `harness-main`, eager `tenferro-cuda-eager`, median
of three independent invocations (five timed runs each), same harness for both
arms. `harness-main` did not contain the per-case clock ramp from
`c3c432f`; it was applied locally for the measurement and not committed. The
process was not pinned to one CPU because that confined the CUDA driver threads
and inflated every number (512 solve 3.26 ms pinned vs 2.08 ms unpinned).

| Case | origin/main e921b059a | this branch c9744cbf1 | Change |
|---|---:|---:|---:|
| `dense_solve_f64_512_rhs16` | 2.375 ms | 2.081 ms | -12.4% |
| `dense_solve_f64_1024_rhs16` | 4.701 ms | 4.413 ms | -6.1% |
| ad-hoc `dense_solve_f64_64_rhs16` | 0.742 ms | 0.430 ms | -42.0% |

The published PyTorch CUDA 512 number is 2.054 ms, so the after row is 1.01x
and the before row is 1.16x. The measured `origin/main` baseline is lower than
the issue's 3.087 ms, so these ratios are not a reproduction of the issue's
1.50x; they are the same-harness effect of this change. The 64 case changes the
sampler's operations-per-sample between runs (2, 4, or 8), so its median is
noisier than the 512/1024 rows.

CUDA `solve` on an exactly singular matrix now returns provider-defined values
instead of `Singular`; a new ignored CUDA test pins that return. The public
`lu_factor` still rejects the same matrix, and the CPU providers keep their
existing exact-zero singularity reporting.
