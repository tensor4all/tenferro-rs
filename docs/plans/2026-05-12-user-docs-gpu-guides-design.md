# User Documentation and GPU Guide Refresh Design

## Goal

Update the public tenferro documentation so the site root is a normal
user-facing documentation landing page, not an API/design selector. Remove stale
claims, especially around GPU support, and expand beginner guides with checked
CPU and CUDA examples.

## Current Problems

- The published root page at `/tenferro-rs/` looks like an API/design index
  rather than the main documentation entry point.
- Some user-facing pages still claim GPU support is planned, even though a
  partial CUDA backend exists.
- Beginner guides are sparse for installation choices, API selection, memory
  order, and GPU device transfers.
- Some examples live only in Markdown, which makes them easy to drift from the
  public API.
- The implementation detail name `cubecl` leaks into the user-facing GPU story.

## Public Information Architecture

The user-facing site should be organized around tasks:

- `/tenferro-rs/`: tenferro overview, quick start, and navigation into the main
  user workflows.
- `/getting-started/`: installation and first working examples.
- `/guides/`: task-oriented guides for everyday users.
- `/api/`: Rustdoc entry point only, clearly labeled as API reference.
- `/internals/`: contributor and architecture entry point.
- `/design/`: historical and developer design documents reachable through
  internals, not the primary beginner path.

## Guide Set

Add or refresh these guides:

- Choosing an API: when to use `Tensor`, `TypedTensor`, `EagerTensor`, and
  `TracedTensor`.
- Installation: default CPU/faer, CPU BLAS providers, and CUDA feature setup.
- Devices and GPU: explicit CPU/GPU transfer, CUDA quickstart, current coverage,
  and limitations.
- Memory Order: column-major default, row-major owned import/export, and
  conversion helpers.
- From PyTorch/JAX: migration mapping and common surprises.
- Troubleshooting: CUDA library loading, dtype mismatch, GPU host access, and
  layout mismatch.

## CUDA Naming

Use `GPU` as the guide category, but use `CUDA` for the concrete public API.
The current backend targets NVIDIA CUDA only, so naming the public type
`GpuBackend` would overpromise future AMD/ROCm support.

Expose a facade-level module:

```rust
#[cfg(feature = "cubecl")]
pub mod cuda {
    pub use tenferro_tensor::cubecl::{
        download_tensor,
        upload_tensor,
        CudaBackend,
    };
}
```

Add a user-facing feature alias:

```toml
cuda = ["cubecl"]
```

Keep `cubecl` available as the implementation feature and keep
`tenferro-cubecl` documented only in internals/design material.

## Example Synchronization

Non-trivial user-facing examples must have an executable source of truth.
Markdown snippets should include or be generated from examples, tests, or
doctests instead of being manually copied.

The CUDA quickstart should be an executable example, for example
`tenferro/examples/cuda_quickstart.rs`, and the guide should include the same
source. CI should compile-check it with the CUDA feature enabled. Machines with
CUDA configured can run the example to execute the assertions.

CPU guide examples should follow the same rule. A workflow example is not
complete if it only constructs values or prints output; it should assert shapes,
dtypes, or deterministic values so readers and CI both verify behavior.

## Validation Strategy

Local and CI validation should include:

- formatting checks,
- doctests,
- docs-site rendering checks,
- compile-checks for examples referenced by guides,
- GPU example compile-checks with the `cuda` feature alias,
- optional CUDA execution command documented in the guide for hardware-backed
  validation.

The GPU quickstart runtime assertion should download the result to CPU and
check the deterministic result of a supported operation. CPU examples should
make equivalent meaningful assertions.
