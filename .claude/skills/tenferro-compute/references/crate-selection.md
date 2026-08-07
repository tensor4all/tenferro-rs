# Crate selection

There is deliberately no facade crate. Add the direct crates that own the
value layer, execution backend, and operation family you use.

## Choose by tier

| Program | Minimum direct crates | Add when needed |
| --- | --- | --- |
| Concrete `Tensor`/`TypedTensor` | `tenferro-runtime`, `tenferro-cpu` | `tenferro-einsum`, `tenferro-linalg`, or `tenferro-fft` for extensions |
| Eager forward/AD | `tenferro-ad`, `tenferro-cpu` | `tenferro-einsum`/`tenferro-linalg` with `autodiff` for those families |
| Traced graph | `tenferro-runtime`, `tenferro-cpu` | `tenferro-ad` for graph transforms and the operation crate for extensions |
| CUDA | the matching value/operation crates plus `tenferro-gpu` with `cuda` | Add `cuda` to crates whose operation surface needs it |
| XLA/PJRT | `tenferro-runtime`, `tenferro-xla` with `pjrt` | A PJRT plugin and the operation crates used by the graph |

A minimal release dependency block for a CPU concrete program is:

```toml
[dependencies]
tenferro-runtime = "0.2"
tenferro-cpu = "0.2"
```

For eager AD and linear algebra, add the owning crates and opt into the
operation family's AD rules:

```toml
[dependencies]
tenferro-ad = "0.2"
tenferro-runtime = "0.2"
tenferro-cpu = "0.2"
tenferro-linalg = { version = "0.2", features = ["autodiff"] }
```

The exact current publishable package set is discoverable without compiling:

```text
cargo metadata --no-deps --format-version 1
```

At this revision it contains 14 publishable `tenferro-*` packages. The
`tenferro-tutorial-code` package is a private CI helper, not a downstream
runtime dependency.

## CPU provider features

At least one of `cpu-faer` or `cpu-blas` must be compiled. They are additive;
`cpu-blas` may be enabled alongside `cpu-faer`, and the default provider is BLAS
when it is compiled, otherwise faer. For a BLAS build choose one provider
feature such as `blas-openblas`, `blas-mkl`, or `blas-accelerate`; those provider
choices are mutually exclusive.

```toml
[dependencies]
tenferro-runtime = { version = "0.2", default-features = false, features = ["cpu-blas"] }
tenferro-cpu = { version = "0.2", default-features = false, features = ["cpu-blas", "blas-openblas"] }
tenferro-linalg = { version = "0.2", default-features = false, features = ["cpu-blas", "blas-openblas"] }
```

Do not enable two BLAS provider implementations in one dependency graph.

## Scratch-crate workspace boundary

When experimenting inside a checkout, put an empty workspace table in the
scratch crate so Cargo does not enroll it in tenferro's parent workspace:

```toml
[workspace]

[dependencies]
tenferro-runtime = { path = "../tenferro-rs/crates/tenferro-runtime" }
```

Use crates.io versions for a real downstream project. Do not copy the workspace
path layout into a published package.
