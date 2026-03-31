# tenferro

AD-aware tensor interface layer on top of `tenferro-rs`.

## Status

The current public surface is intentionally narrow:

- Dynamic tensor frontend:
  - `Tensor`
  - `ScalarType` (`F32`, `F64`, `C32`, `C64`)
- Reverse-mode helpers:
  - `Tensor::with_requires_grad`
  - `Tensor::grad`
  - `Tensor::backward`
  - free `grad(...)` / `backward(...)`
- Public JVP transform:
  - `jvp(...)`
  - `JvpResult` with `outputs` and `output_tangents`
- Direct tensor methods:
  - elementwise/reduction: `add`, `exp`, `sum`
  - tensor contraction: `einsum`
  - linalg: `solve`, `solve_triangular`, `det`, `inv`, `slogdet`, `cholesky`,
    `lstsq`, `lu`, `norm`, `qr`, `svd`, `eig`, `eigen`, `pinv`, `matrix_exp`
- Runtime control:
  - `RuntimeContext`
  - `set_default_runtime`
  - `with_default_runtime`
  - `runtime::with_runtime`

`Tensor` is a public façade over `tidu::Value<DynTensor>`. Reverse-mode graph
state lives in the `Value` carrier; `tenferro` does not keep a second legacy
carrier layer.

The public `jvp(...)` transform is a small forward-mode seam for the currently
wired tensor methods. It returns both the primal outputs and optional output
tangents in `JvpResult`. It is not a public dual-builder API, and it does not
promise higher-order forward-mode or HVP support.

## Tensor AD coverage

### Public `Tensor` methods wired into `jvp(...)`

| Operation | Public `Tensor` entrypoint | JVP dtypes |
|-----------|----------------------------|------------|
| add | `Tensor::add` | real + complex |
| exp | `Tensor::exp` | real + complex |
| sum | `Tensor::sum` | real + complex |
| einsum | `Tensor::einsum` | real + complex |
| solve | `Tensor::solve` | real + complex |
| solve_triangular | `Tensor::solve_triangular` | real + complex |
| det | `Tensor::det` | real only |
| inv | `Tensor::inv` | real only |
| slogdet | `Tensor::slogdet` | real only |
| cholesky | `Tensor::cholesky` | real only |
| lstsq | `Tensor::lstsq` | real only |
| lu | `Tensor::lu` | real + complex |
| norm | `Tensor::norm` | real only |
| qr | `Tensor::qr` | real + complex |
| svd | `Tensor::svd` | real + complex |
| eig | `Tensor::eig` | real input, complex outputs |
| eigen | `Tensor::eigen` | real only |
| pinv | `Tensor::pinv` | real only |
| matrix_exp | `Tensor::matrix_exp` | real only |

At the current `Tensor` seam, all operations with internal first-order
`frule/rrule` coverage are exposed on the public dynamic AD surface.

## Runtime-backed operations

`add`, `exp`, and `sum` work directly on the dynamic carrier.

Operations that dispatch into tenferro runtimes must run under an installed
runtime:

- `Tensor::einsum`
- `Tensor::solve`
- `Tensor::solve_triangular`
- `Tensor::det`
- `Tensor::inv`
- `Tensor::slogdet`
- `Tensor::cholesky`
- `Tensor::lstsq`
- `Tensor::lu`
- `Tensor::norm`
- `Tensor::qr`
- `Tensor::svd`
- `Tensor::eig`
- `Tensor::eigen`
- `Tensor::pinv`
- `Tensor::matrix_exp`

Install a default runtime with `set_default_runtime(...)` or use
`runtime::with_runtime(...)` for an explicit scoped call.

```rust
use tenferro::{jvp, Tensor};

let x = Tensor::from_slice(&[1.0_f64, 2.0], &[2])?;
let result = jvp(
    |inputs| {
        let y = inputs[0].add(&inputs[0])?.exp()?.sum()?;
        Ok(vec![y])
    },
    &[x],
    &[Some(Tensor::from_slice(&[1.0_f64, 0.0], &[2])?)],
)?;

assert_eq!(result.outputs.len(), 1);
assert_eq!(result.output_tangents.len(), 1);
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
