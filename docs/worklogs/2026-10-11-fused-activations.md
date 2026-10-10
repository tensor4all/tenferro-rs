# Fused CPU activations (#2030, #2032)

The eager and concrete activation surface (`sigmoid`, `silu`, `softplus`,
`gelu`, `gelu_tanh`) was evaluated as a composite of 5-10 separately
materialized elementwise ops. A single eager op already costs about 0.15 ms for
65536 f32 values, so the composite paid that cost per intermediate.

## Decisions

- **A backend opts into a fused activation through one optional trait method.**
  `BackendSession::fused_activation_read(ActivationOp, TensorRead)` returns
  `Ok(None)` by default, so every existing backend keeps the shared composite
  formulation unchanged. The CPU backend returns `Some` and evaluates the whole
  expression in one `map`. The five variants live in the new public
  `tenferro_tensor::ActivationOp` vocabulary; the maintainer approved the
  addition.
- **The fused path is taken only when no AD record would be lost.** The eager
  session checks the same eligibility the recorded primitive path uses
  (`untracked_fast_path_allowed`): an untracked operand with no active semantic
  capture. A tracked operand, or active capture, keeps the recorded composite, so
  first- and higher-order AD are unchanged.
- **The concrete and typed session surfaces use the same fast path** before
  falling back to `run_session_composite`.
- The fused formulas mirror `tenferro_runtime::composite` exactly (same
  overflow-free sigmoid branch, same softplus split, same GELU constants) so the
  result matches elementwise up to float association order.

## Verification conclusions and constraints

- `cargo test -p tenferro-cpu` passes in full, including new tests that check the
  fused value against the reference formula for all five ops on `F64` and that
  `C32`/`I32` inputs decline the fast path. `tenferro-tensor`,
  `tenferro-runtime`, `tenferro-ad` (including the `nn_composites` AD
  integration suite), and clippy on all four crates pass.
- Isolated in-process A/B on 65536 f32 (EagerRuntime, same process, median of
  30): `exp` 151 us (one op), `sigmoid` 721 -> 171 us, `silu` 772 -> 181 us,
  `gelu_tanh` 1152 -> 793 us. The composite is the sum of its per-op costs, so
  fusing removes the intermediate passes.
- Against the real PyTorch reference in the frozen harness (4 threads, 5 runs):
  `activation.sigmoid` 2.49x, `activation.silu` 3.14x, `activation.gelu_tanh`
  13.65x (was 21x before this change), `activation.softplus` 5.64x,
  `activation.erf` 17.97x. The host is shared, so these are direction and rough
  size only.
- **This does not close #2030/#2032 on its own.** The residual is dominated by
  (a) the per-backend-call overhead, which is already about 0.12-0.15 ms for this
  size (tracked by #1904/#1945), and (b) scalar `tanhf`/`erf` in `gelu_tanh` and
  `erf`. The second needs vectorized transcendental math (#2033); the first is
  the small-call-overhead design. `activation.erf` has no fused variant here
  because `erf` is already a single op and is limited by the same scalar math.
