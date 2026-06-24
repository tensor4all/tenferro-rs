# Tutorials

These tutorials are ordered, runnable introductions to the main tenferro
workflows. They complement the guides: tutorials show one complete path, while
guides describe the broader APIs and tradeoffs.

Short tutorial programs in this section are sourced from `docs/tutorial-code`
and are run by the workspace test workflow. Longer application samples, such
as the KdV PINN tutorial, point at their workspace package and provide a
separate package-level test command.

## Suggested Order

| Tutorial | Use it when |
| --- | --- |
| [TypedTensor for numeric computation without autodiff](typed-tensor-non-ad.md) | You know the scalar type in Rust and want ndarray-like CPU tensor computation without AD. |
| [Eager autodiff, PyTorch style](eager-autodiff-pytorch-style.md) | You want immediate execution, scalar losses, `backward()`, and accumulated gradients. |
| [Traced autodiff, JAX style](traced-autodiff-jax-style.md) | You want to build a graph, compile/run it, and use `grad` or `jvp` on the traced graph. |
| [Einsum: subscripts to gradients](einsum-subscripts-to-gradients.md) | You contract more than two tensors and want planned contraction order plus AD. |
| [XLA backend: einsum to StableHLO](xla-einsum-backend.md) | You want to lower a fixed-shape N-ary einsum path through the experimental XLA executor. |
| [Dynamic shapes: truncated SVD](dynamic-shape-truncated-svd.md) | Output ranks depend on runtime values such as singular-value thresholds. |
| [KdV PINN sample](kdv-pinn.md) | You want a full traced-graph PINN training loop with PDE residuals and scalar loss gradients. |

## Running The Tutorial Code

From the repository root:

```bash
cargo test -p tenferro-tutorial-code --release
```

The CI workflow runs this package through the existing workspace test command,
so tutorial execution does not add a second tenferro compilation step after
unit tests.

The KdV PINN sample is tested separately:

```bash
cargo test -p kdv_pinn
```
