# tenferro

AD-aware tensor interface layer on top of `tenferro-rs`.

## Status

This repository currently provides:

- Dynamic public tensor objects:
  - `Tensor`
  - `ScalarType` (`F32`, `F64`, `C32`, `C64`)
- Runtime context:
  - `RuntimeContext`
  - `set_default_runtime`
  - `with_default_runtime`
- PyTorch-like direct methods on `Tensor`:
  - scalar/analytic: `exp`, `sqrt`, `sin`, `cos`, `tanh`, `add`, `pow`, `mean`, `sum`, `var`, `std`, ...
  - tensor: `einsum`
  - linalg: `svd`, `qr`, `lu`, `eigen`, `eig`, `lstsq`, `solve`, `det`, `slogdet`, ...

The preferred public surface is `Tensor` plus its direct methods. Runtime
selection uses an explicit runtime holder, reverse-mode graphs stay
homogeneous over one runtime-typed tensor payload, and rank-0 tensors carry
scalar AD values. Mixed-dtype tensor ops apply implicit result-type promotion
internally (`complex` beats `real`, `64-bit` beats `32-bit`), and reverse-mode
pullbacks cast gradients back to each input dtype. Explicit numeric casts use
`Tensor::to_scalar_type`.

Placement and transfer stay on the tensor object:

- `Tensor::memory_space()`
- `Tensor::preferred_compute_device()`
- `Tensor::to_memory_space(...)`
- `Tensor::to_cpu()` / `Tensor::to_gpu()`

Explicit runtime choice remains separate via `tenferro::runtime::with_runtime(...)`.

```rust
use tenferro::{set_default_runtime, RuntimeContext, Tensor};
use tenferro_prims::CpuContext;
use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};

let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
let a = DenseTensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor)?;
let qr_result = Tensor::from_tensor(a).qr()?;
assert_eq!(qr_result.q.dims(), &[2, 2]);
# Ok::<(), tenferro::Error>(())
```

## Development

```bash
cargo fmt --all
cargo clippy --workspace
cargo test --release --workspace
```

## Documentation

Build local docs site:

```bash
./scripts/build_docs_site.sh
```

Output:

- `target/docs-site/index.html` (top page)
- `target/docs-site/api/` (`cargo doc --workspace --no-deps` output)
- `target/docs-site/design/` (rendered design docs)

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](./LICENSE-APACHE))
- MIT license ([LICENSE-MIT](./LICENSE-MIT))

at your option.
