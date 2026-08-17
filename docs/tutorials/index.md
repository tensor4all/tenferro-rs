# Tutorials

These tutorials are ordered by user workflow. Standard scientific computing
comes first; extension authoring and execution-model internals are advanced
paths. The repository is extensible internally, but ordinary use stays
conventional: construct a value, enter a session, call an operation, and keep
the result.

## Start here

| Tutorial | Use it when |
| --- | --- |
| [Ordinary CPU scientific computing](../getting-started/index.md#quickstart-a-direct-tensor-and-linalg) | You want matmul, solve, and singular values in one bounded backend session. |
| [TypedTensor for numeric computation without autodiff](typed-tensor-non-ad.md) | You know the scalar type in Rust and want ndarray-like CPU tensor computation without AD. |
| [CUDA and explicit device movement](../guides/devices-and-gpu.md#cuda-quickstart) | You want to upload inputs, run supported operations on CUDA, and download values explicitly. Hardware-executed CUDA tutorial validation lives in the GPU CI lane; CPU CI only compile-checks that artifact. |
| [Using tenferro with ndarray/faer data](../getting-started/ndarray-nalgebra-mapping.md#zero-copy-interop-keep-your-faerndarray-buffers) | Your application already owns arrays and needs an explicit borrowed-view round trip. |
| [Calling faer or BLAS/LAPACK directly](../guides/external-linalg-interop.md) | One specialized routine is outside the standard operation families; borrow compact host storage, call the external library, and continue with tenferro. |
| [Eager autodiff, PyTorch style](eager-autodiff-pytorch-style.md) | You want immediate execution, scalar losses, `backward()`, accumulated gradients, or the functional eager AD entry point. |
| [Traced autodiff, JAX style](traced-autodiff-jax-style.md) | You want to build a graph, compile/run it, and use `grad` or `jvp` on the traced graph. |

## Advanced topics

| Tutorial | Use it when |
| --- | --- |
| [Einsum: subscripts to gradients](einsum-subscripts-to-gradients.md) | You contract more than two tensors and want planned contraction order plus AD. |
| [XLA backend: einsum to StableHLO](xla-einsum-backend.md) | You want to lower a fixed-shape N-ary einsum path through the experimental XLA executor. |
| [Dynamic shapes: truncated SVD](dynamic-shape-truncated-svd.md) | Output ranks depend on runtime values such as singular-value thresholds. |
| [Tropical extension](tropical-extension.md) | You want a complete extension crate for non-standard arithmetic, runtime registration, and AD rules. |
| [Sparse tensor extension](sparse-extension.md) | You want a fixed-pattern sparse COO extension with sparse-sparse contraction and value AD. |
| [KdV PINN sample](kdv-pinn.md) | You want a full traced-graph PINN training loop with PDE residuals and scalar loss gradients. |

The [custom operations guide](../guides/custom-operations.md) explains the
extension architecture only when you need to add a new operation family.

## Running the tutorial code

From the repository root:

```bash
cargo test -p tenferro-tutorial-code --release
```

The CI workflow runs this package through the existing workspace test workflow.
The CPU tutorial binaries remain hardware-independent. The CUDA tutorial is
compiled and archived on the non-GPU CUDA lane and executed with deterministic
value assertions on the trusted GPU lane; see [Devices and GPU](../guides/devices-and-gpu.md)
for the exact transfer contract.

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
