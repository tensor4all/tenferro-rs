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
tenferro-runtime = { version = "0.2", default-features = false, features = ["cpu-faer"] }
tenferro-cpu = { version = "0.2", default-features = false, features = ["cpu-faer"] }
```

For eager AD and linear algebra, add the owning crates and opt into the
operation family's AD rules:

```toml
[dependencies]
tenferro-ad = { version = "0.2", default-features = false, features = ["cpu-faer"] }
tenferro-runtime = { version = "0.2", default-features = false, features = ["cpu-faer"] }
tenferro-cpu = { version = "0.2", default-features = false, features = ["cpu-faer"] }
tenferro-linalg = { version = "0.2", default-features = false, features = ["autodiff", "cpu-faer"] }
```

The exact current publishable package set is discoverable without compiling:

```text
cargo metadata --no-deps --format-version 1
```

The `publish` field identifies the packages intended for downstream use; the
`tenferro-tutorial-code` package is a private CI helper, not a runtime
dependency. Do not hard-code a package count in tooling or skill text; query
metadata at the revision you are using.

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

When experimenting with a checkout, use a sibling layout so the scratch
crate is its own workspace:

```text
work/
├── tenferro-rs/
└── scratch/
    └── Cargo.toml
```

The scratch `Cargo.toml` then needs an empty workspace table and a path back to
the checkout:

```toml
[workspace]

[dependencies]
tenferro-runtime = { path = "../tenferro-rs/crates/tenferro-runtime" }
```

If the scratch crate is placed inside the checkout instead, keep the empty
`[workspace]` table so Cargo does not enroll it in the parent workspace. Use
crates.io versions for a real downstream project; do not copy checkout paths
into a published package.
