# tenferro

AD-aware tensor interface layer on top of `tenferro-rs`.

## Status

The current public surface is intentionally narrow:

- Dynamic tensor frontend:
  - `Tensor`
  - `ScalarType` (`F32`, `F64`, `C32`, `C64`)
- Reverse-mode helpers:
  - `Tensor::requires_grad_`
  - `Tensor::grad`
  - `Tensor::backward`
  - free `grad(...)` / `backward(...)`
- Direct tensor methods:
  - elementwise/reduction: `add`, `exp`, `sum`
  - tensor contraction: `einsum`
  - linalg: `solve`, `det`, `norm`, `qr`, `svd`
- Runtime control:
  - `RuntimeContext`
  - `set_default_runtime`
  - `with_default_runtime`
  - `runtime::with_runtime`

`Tensor` is a public façade over `tidu::Value<DynTensor>`. Reverse-mode graph
state lives in the `Value` carrier; `tenferro` does not keep a second legacy
carrier layer.

## Runtime-backed operations

`add`, `exp`, and `sum` work directly on the dynamic carrier.

Operations that dispatch into tenferro runtimes must run under an installed
runtime:

- `Tensor::einsum`
- `Tensor::solve`
- `Tensor::det`
- `Tensor::norm`
- `Tensor::qr`
- `Tensor::svd`

Install a default runtime with `set_default_runtime(...)` or use
`runtime::with_runtime(...)` for an explicit scoped call.

```rust
use tenferro::{set_default_runtime, RuntimeContext, Tensor};
use tenferro_prims::CpuContext;

let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

let a = Tensor::from_slice(&[1.0_f64, 0.0, 0.0, 1.0], &[2, 2])?;
let qr = a.qr()?;
assert_eq!(qr.q.dims(), &[2, 2]);
# Ok::<(), tenferro::Error>(())
```

## Custom downstream operations

Downstream crates that need custom differentiable operations should implement:

- `LinearizableOp<DynTensor>`
- `LinearizedOp<DynTensor>`

The intended seam is `primal + linearize + jvp/vjp`.

## Development

```bash
cargo fmt --all
cargo clippy --workspace
cargo test --release -p tenferro
```

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](./LICENSE-APACHE))
- MIT license ([LICENSE-MIT](./LICENSE-MIT))

at your option.
