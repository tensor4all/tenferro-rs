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
tenferro-runtime = { version = "0.2", default-features = false }
tenferro-cpu = { version = "0.2", default-features = false }
```

For eager AD and linear algebra, add the owning crates and opt into the
operation family's AD rules:

```toml
[dependencies]
tenferro-ad = { version = "0.2", default-features = false }
tenferro-runtime = { version = "0.2", default-features = false }
tenferro-cpu = { version = "0.2", default-features = false }
tenferro-linalg = { version = "0.2", default-features = false, features = ["autodiff"] }
```

The exact current publishable package set is discoverable without compiling:

```text
cargo metadata --no-deps --format-version 1
```

The `publish` field identifies the packages intended for downstream use; the
`tenferro-tutorial-code` package is a private CI helper, not a runtime
dependency. Do not hard-code a package count in tooling or skill text; query
metadata at the revision you are using.

## No CPU backend feature

The CPU crates declare no backend feature. Every build links one faer-backed
`tlinalg`/`cpueinsum`/`tcontract` stack, so nothing selects a provider and no
system BLAS/LAPACK has to be installed. A dependency list from an older release
still carrying `native`, `blas`, or `blas-openblas` fails with "feature does not
exist": remove the feature and the `default-features = false` that went with it.

Features that do exist on the CPU crates are orthogonal: `autodiff` on the
operation crates, and `ledger` where an engine-internal ledger test needs it.

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
