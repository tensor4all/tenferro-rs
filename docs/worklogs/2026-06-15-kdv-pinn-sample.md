# Work Log: KdV PINN Sample

Date: 2026-06-15
Branch: `kdv-pinn` (worktree `.worktrees/kdv-pinn`)

## Goal

Implement an independent `kdv_pinn` sample crate that demonstrates how to build a Physics-Informed Neural Network (PINN) for the Korteweg–de Vries (KdV) equation using tenferro-rs `TracedTensor` graphs.

## Context Read

- Reviewed the implementation plan in `docs/superpowers/plans/2026-06-15-kdv-pinn-implementation.md` and the design doc in `docs/superpowers/specs/2026-06-15-kdv-pinn-design.md`.
- Confirmed that `tenferro-rs` uses column-major tensor layout by default and that `Tensor::from_vec_row_major` is required for interleaved `(x, t)` collocation data.
- Verified that `TracedTensorAdExt::grad` requires a scalar output, while `TracedTensorAdExt::jvp` can compute per-element derivatives for vector outputs.

## Chosen Design

- **Graph-based training**: Build the MLP once with `TracedTensor` placeholders for weights/biases, compile scalar loss + per-parameter gradient programs, and evaluate them each epoch with `GraphExecutor::run_with_inputs`. Update `Tensor` parameter buffers in place with a small SGD helper.
- **Per-element PDE derivatives**: Because `u` is a vector (`[N, 1]`), compute `u_t`, `u_x`, `u_xx`, `u_xxx` via repeated JVPs with tangent tensors of ones. This exploits the pointwise structure of the MLP to obtain elementwise derivatives.
- **Module organization**: Each source file focuses on one concern (`sampler`, `network`, `pde`, `loss`, `optimizer`). Unit tests live in module-local `tests.rs` files.
- **Binary-only crate**: `kdv_pinn` is a sample binary; public items use `pub(crate)` and doc examples are avoided because there is no library target for doctests.

## Rejected Alternatives

- **Eager AD (`EagerTensor::backward`)**: Rejected because it does not expose a `create_graph` equivalent and cannot compose to third-order derivatives. `TracedTensor` + `TracedTensorAdExt` is the supported path for higher-order AD.
- **Using `TracedTensorAdExt::grad` for the residual**: Rejected because `grad` requires a scalar output. The residual is computed over a batch, so JVP is the correct primitive.
- **Adding a `[lib]` target**: Considered to enable doctests, but rejected to keep the sample a simple binary and avoid restructuring the planned `main.rs`.

## Key Adjustments During Implementation

- Fixed a column-major layout bug in `Sampler::collocation` by switching to `Tensor::from_vec_row_major` for interleaved data.
- Changed `Sampler::collocation` to return `(Tensor, Tensor)` directly to avoid fragile buffer slicing in `main.rs`.
- Stratified `Sampler::boundary` so both `x = -5` and `x = 5` are always represented.
- Lowered the learning rate from `0.01` (used in the initial-condition-only loop) to `0.001` because the full PDE + boundary objective diverges to NaN at the higher rate.
- Moved the `grad` helper into `pde/tests.rs` because it is only used by the third-derivative PoC test.

## Residual Risks

- **Training accuracy**: After 500 epochs the L2 relative error at `t = 0.5` is around `1.0` (≈100%). The loss decreases from O(100) to O(1), gradients flow, and all unit tests pass, but the network has not converged to the analytical soliton. Improving accuracy would require hyperparameter/architecture tuning (larger MLP, adaptive LR, better collocation sampling, annealed loss weights) that is outside the scope of this sample.
- **Higher-order AD coverage**: The JVP-based residual relies on the MLP being pointwise. If the architecture were changed to introduce batch mixing (e.g., batch normalization), the per-element derivative interpretation would break.

## Verification

- `cargo fmt --all --check` ✅
- `cargo clippy -p kdv_pinn -- -D warnings` ✅
- `cargo test -p kdv_pinn` ✅ 12/12 tests pass
- `cargo run -p kdv_pinn --release` ✅ completes 500 epochs and prints final loss + L2 error
