# Tutorials

These tutorials are ordered, runnable introductions to the main tenferro
workflows. They complement the guides: tutorials show one complete path, while
guides describe the broader APIs and tradeoffs.

All non-trivial code in this section is sourced from `docs/tutorial-code` and
is run by the workspace test workflow.

## Suggested Order

| Tutorial | Use it when |
| --- | --- |
| [TypedTensor for numeric computation without autodiff](typed-tensor-non-ad.md) | You know the scalar type in Rust and want ndarray-like CPU tensor computation without AD. |
| [Eager autodiff, PyTorch style](eager-autodiff-pytorch-style.md) | You want immediate execution, scalar losses, `backward()`, and accumulated gradients. |
| [Traced autodiff, JAX style](traced-autodiff-jax-style.md) | You want to build a graph, compile/run it, and use `grad` or `jvp` on the traced graph. |
| [Einsum: subscripts to gradients](einsum-subscripts-to-gradients.md) | You contract more than two tensors and want planned contraction order plus AD. |
| [Dynamic shapes: truncated SVD](dynamic-shape-truncated-svd.md) | Output ranks depend on runtime values such as singular-value thresholds. |

## Running The Tutorial Code

From the repository root:

```bash
cargo test -p tenferro-tutorial-code --release
```

The CI workflow runs this package through the existing workspace test command,
so tutorial execution does not add a second tenferro compilation step after
unit tests.
