# tenferro-dyadtensor

AD-aware tensor interface layer on top of `tenferro-rs`.

## Status

This repository currently provides:

- Generic AD value model:
  - `AdValue<T>`
  - `AdScalar<T>`
  - `AdTensor<T>`
- Runtime dtype wrappers:
  - `DynScalar`, `DynTensor`
  - `DynAdTensor`
  - `ScalarType` (`F32`, `F64`, `C32`, `C64`)
- AD boundary traits:
  - `Differentiable`, `TensorKernel`, `OpRule`
- Runtime context:
  - `RuntimeContext`
  - `set_default_runtime`
  - `with_default_runtime`
- Builder-style operation API (`run()` terminal):
  - Einsum: `einsum(...)`, `einsum_ad(...)`
  - Linalg primal: `svd/qr/lu/eigen/lstsq/cholesky/solve/inv/det/slogdet/eig/pinv/matrix_exp/solve_triangular/norm`
  - Linalg AD: corresponding `*_ad(...)` operations

All operation entry points are builder-based and execute via `.run()` using
the default runtime context. Runtime selection uses an explicit runtime holder,
and reverse-mode bookkeeping attaches pullback rules directly to
`chainrules::Tape<DynTensor>`, where rank-0 tensors carry scalar AD values.

```rust
use tenferro_dyadtensor::{qr, set_default_runtime, RuntimeContext};
use tenferro_prims::CpuContext;
use tenferro_tensor::{MemoryOrder, Tensor};

let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
let a = Tensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor)?;
let qr_result = qr(&a).run()?;
# Ok::<(), tenferro_dyadtensor::Error>(())
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
