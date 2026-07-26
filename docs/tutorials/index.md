# Tutorials

These tutorials are ordered, runnable introductions to the main tenferro
workflows. They complement the guides: tutorials show one complete path, while
guides describe the broader APIs and tradeoffs.

Short tutorial programs in this section are sourced from `docs/tutorial-code`
and are run by the workspace test workflow. Extension tutorials live in
standalone nested crates under `ext/` and are tested through their manifest
paths. Longer application samples, such as the KdV PINN tutorial, point at
standalone sample packages and may use compile-only CI coverage when execution
would be too slow for every pull request.

## Suggested Order

| Tutorial | Use it when |
| --- | --- |
| [TypedTensor for numeric computation without autodiff](typed-tensor-non-ad.md) | You know the scalar type in Rust and want ndarray-like CPU tensor computation without AD. |
| [Eager autodiff, PyTorch style](eager-autodiff-pytorch-style.md) | You want immediate execution, scalar losses, `backward()`, accumulated gradients, or the broader functional eager AD entry point. |
| [Traced autodiff, JAX style](traced-autodiff-jax-style.md) | You want to build a graph, compile/run it, and use `grad` or `jvp` on the traced graph. |
| [Einsum: subscripts to gradients](einsum-subscripts-to-gradients.md) | You contract more than two tensors and want planned contraction order plus AD. |
| [XLA backend: einsum to StableHLO](xla-einsum-backend.md) | You want to lower a fixed-shape N-ary einsum path through the experimental XLA executor. |
| [Dynamic shapes: truncated SVD](dynamic-shape-truncated-svd.md) | Output ranks depend on runtime values such as singular-value thresholds. |
| [Tropical extension](tropical-extension.md) | You want a complete extension crate for non-standard arithmetic, runtime registration, and AD rules. |
| [Sparse tensor extension](sparse-extension.md) | You want a fixed-pattern sparse COO extension with sparse-sparse contraction and value AD. |
| [KdV PINN sample](kdv-pinn.md) | You want a full traced-graph PINN training loop with PDE residuals and scalar loss gradients. |

## Running The Tutorial Code

From the repository root:

```bash
cargo test -p tenferro-tutorial-code --release
```

The CI workflow runs this package through the existing workspace test command,
so tutorial execution does not add a second tenferro compilation step after
unit tests.

The tropical and sparse extension tutorials are tested as standalone crates:

```bash
cargo test --manifest-path ext/tropical/Cargo.toml --release --features autodiff
cargo test --manifest-path ext/sparse/Cargo.toml --release --features autodiff
cargo test --manifest-path ext/tenferro-cpu-tblis/Cargo.toml --release
```

The KdV PINN sample is compile-checked separately:

```bash
cargo check --manifest-path samples/kdv-pinn/Cargo.toml --release --all-targets
```
