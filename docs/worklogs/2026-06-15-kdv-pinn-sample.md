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
- **Critical bug fix**: The design doc specified the PDE as `u_t + u * u_x + u_xxx = 0`, but the reference soliton `u = 2 sech^2(x - 4t)` satisfies the *standard* KdV equation `u_t + 6 u u_x + u_xxx = 0`. Updated `kdv_residual` and the design doc to use the coefficient `6`. After this fix the residual of the exact solution is numerically zero and training converges to a meaningful solution.
- Switched from SGD to **Adam** for the full training loop; added first- and second-moment buffers to `optimizer.rs`.
- Tuned hyperparameters for the best accuracy within a reasonable runtime: MLP `[2, 64, 64, 1]`, `N_COL = 512`, `N_IC = N_BC = 64`, `LR = 0.001`, `EPOCHS = 1000`, balanced loss weights `λ_pde = λ_ic = λ_bc = 1.0`.
- Added an optional `--gif <path>` CLI flag that renders 30 frames comparing the predicted (red) and analytic (blue) solution curves over `t ∈ [0, 1]` and encodes them as an animated GIF using the `plotters` crate (high-quality axes, grid, legend, and caption).
- Added `.gitignore` rules to exclude generated `*.gif` files.

## Residual Risks

- **Training accuracy**: After 1000 epochs the L2 relative error at `t = 0.5` is around `0.15` (≈15%). This is a large improvement over the original ~100%, but further reduction would require a larger network, longer training, adaptive collocation sampling, or a learning-rate schedule.
- **Higher-order AD coverage**: The JVP-based residual relies on the MLP being pointwise. If the architecture were changed to introduce batch mixing (e.g., batch normalization), the per-element derivative interpretation would break.

## Verification

- `cargo fmt --all --check` ✅
- `cargo clippy -p kdv_pinn -- -D warnings` ✅
- `cargo test -p kdv_pinn` ✅ 21/21 tests pass
- `cargo test --workspace --release` ✅ passes
- `cargo run -p kdv_pinn --release` ✅ completes 1000 epochs and prints final loss + L2 error
