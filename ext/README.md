# `ext/` — optional adapters, experiments and standalone extensions

`ext/` holds two kinds of crate:

1. experiments that are **workspace members** and are never published;
2. external extension examples that are **standalone workspaces** (their own
   `Cargo.lock`, excluded from the root workspace).

The MPI and HDF5 adapters used to live here; they are now optional workspace
members under `crates/` and are listed below because their build contract is why the
restructure happened.

## Optional adapters

These moved from `ext/` to `crates/` so the root workspace can build them without
paying for MPI or HDF5. Their default features are empty; the adapter is enabled by a
feature.

| Path | Package | Default | Opt-in feature | Publish |
| --- | --- | --- | --- | --- |
| `crates/tenferro-mpi` | `tenferro-mpi` | `[]` | `system` (rsmpi-rt `mpi-sys-backend` → `rsmpi-sys 0.2.4`, needs MPI headers and libclang) or `runtime` (rsmpi-rt `mpi-rt-sys-backend` → `mpi-rt-sys 0.1.0`, loads MPI at run time) | `true` |
| `crates/tenferro-hdf5` | `tenferro-hdf5` | `[]` | `hdf5` (git `hdf5-rt`, needs a runtime `libhdf5` for tests) | `false` |

Both MPI backends build the published `rsmpi-rt 0.8.2` package (library target
`mpi`). `system` and `runtime` are mutually exclusive; rsmpi-rt rejects enabling both
backend features at once. `engine` compiles the MPI protocol with no binding.

Root lanes therefore need neither MPI nor HDF5:

```bash
cargo metadata
cargo test --workspace
```

The dedicated lanes are explicit:

```bash
cargo test -p tenferro-mpi --features system        # host with MPI + libclang
cargo test -p tenferro-mpi --features runtime       # requires MPIwrapper (MPI_RT_LIB); doctests initialize MPI
cargo test -p tenferro-hdf5 --features hdf5         # host with runtime libhdf5
scripts/check-mpi-features.sh tenferro-mpi
scripts/run-mpi.sh upstream                         # mpiexec -n 2 scenarios
```

## Experiments

Workspace members, `publish = false`, built and tested by the root lanes:

| Path | Package | Why it exists |
| --- | --- | --- |
| `ext/experiments/bf16-proof` | `tenferro-bf16-proof` | carries standard `half::bf16` through the external-scalar boundary |
| `ext/experiments/df64-proof` | `tenferro-df64-proof` | the external `Df64` scalar, extension operations and AD rules |
| `ext/experiments/scalar-consumer-algorithm` | `tenferro-scalar-consumer-algorithm` | scalar-agnostic algorithm role |
| `ext/experiments/scalar-consumer-application` | `tenferro-scalar-consumer-application` | application role that binds one scalar set |

## Standalone extension examples

These are deliberately separate workspaces with their own `Cargo.lock`; the root
workspace excludes them and their tests run in the `extensions` profile.

| Path | Package | Publish |
| --- | --- | --- |
| `ext/sparse` | `tenferro-ext-sparse` | `false` |
| `ext/tropical` | `tenferro-ext-tropical` | `false` |

`ext/tenferro-cpu-tblis/` currently contains only a stale `Cargo.lock`; it is not a
manifest and not a workspace member.

## Follow-up: publication

`rsmpi-rt 0.8.2`, `rsmpi-sys 0.2.4` and `mpi-rt-sys 0.1.0` are on crates.io, so the
MPI adapter no longer has a git dependency and the temporary `tenferro-mpi-rt` crate
has been removed. Remaining work is publication, not restructuring:

1. review `cargo publish --dry-run -p tenferro-mpi` and publish `tenferro-mpi`
   (not done by this branch);
2. when `hdf5-rt` is on crates.io, switch that dependency to the registry, set
   `tenferro-hdf5` to `publish = true`, and re-run the `hdf5` lane;
3. keep both MPI feature lanes and the root workspace test in the release gate.
