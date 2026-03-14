# tenferro-dyadtensor

AD-aware tensor interface layer on top of `tenferro-rs`.

## Status

This repository currently provides:

- Dynamic public tensor objects:
  - `DynAdTensor`
  - `DynScalar`
  - `ScalarType` (`F32`, `F64`, `C32`, `C64`)
- Public helper traits:
  - `TensorKernel`
  - `IndexLike`
  - `AllowedPairs`
- Runtime context:
  - `RuntimeContext`
  - `set_default_runtime`
  - `with_default_runtime`
- PyTorch-like eager methods on `DynAdTensor`:
  - scalar/analytic: `exp`, `sqrt`, `sin`, `cos`, `tanh`, `add`, `pow`, `mean`, `sum`, `var`, `std`, ...
  - tensor: `einsum`
  - linalg: `svd`, `qr`, `lu`, `eigen`, `eig`, `lstsq`, `solve`, `det`, `slogdet`, ...

The preferred public surface is `DynAdTensor` plus its eager methods. Runtime
selection uses an explicit runtime holder, and reverse-mode bookkeeping
attaches pullback rules directly to `chainrules::Tape<DynTensor>`, where
rank-0 tensors carry scalar AD values. Mixed-dtype tensor ops apply implicit
algebraic promotion internally, and reverse-mode pullbacks cast gradients back
to each input dtype. Explicit numeric casts use `DynAdTensor::to_scalar_type`.

```rust
use tenferro_dyadtensor::{DynAdTensor, set_default_runtime, RuntimeContext};
use tenferro_prims::CpuContext;
use tenferro_tensor::{MemoryOrder, Tensor};

let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
let a = Tensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor)?;
let qr_result = DynAdTensor::new_primal(a).qr()?;
assert_eq!(qr_result.q.dims(), &[2, 2]);
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
