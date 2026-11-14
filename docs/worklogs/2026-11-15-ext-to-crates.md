# Optional adapters moved under `crates/`

## Decisions

- `tenferro-mpi`, `tenferro-hdf5` and the temporary `tenferro-mpi-rt` left the
  standalone `ext/` workspaces and became root workspace members. The default build
  selects no system library, so the root lanes keep neither MPI nor HDF5; the
  adapters expose their backends as opt-in features instead of being excluded from
  the workspace.
- `tenferro-mpi` has `default = []` with mutually exclusive `system` and `runtime`
  features; rsmpi-rt itself rejects enabling both backend features at once. `engine`
  remains the protocol-only feature. `tenferro-hdf5` has `default = []` and a single `hdf5`
  feature, with `publish = false` until `hdf5-rt` is on crates.io.
- The proof and consumer crates moved to `ext/experiments/`; `ext/sparse` and
  `ext/tropical` remain standalone extension workspaces so their independent
  lockfiles and evolution stay outside the root build.
- Once `rsmpi-rt 0.8.2` and its sys crates were published, the MPI adapter switched
  to `rsmpi-rt` (aliased back to `mpi` because its library target kept that name),
  whose `mpi-sys-backend` resolves crates.io `rsmpi-sys 0.2.4` and whose
  `mpi-rt-sys-backend` resolves crates.io `mpi-rt-sys 0.1.0`. Both backends now name
  one `mpi` identity, so the temporary `tenferro-mpi-rt` crate was removed rather
  than kept as a duplicate API surface. The published `mpi 0.8.1`/`mpi-sys`
  dependency and the pinned git revisions were dropped.
- Dedicated adapter lanes stay explicit: the hosted CI MPI job enables `system`,
  the HDF5 job enables `hdf5`, and the runtime lane remains local because it needs
  MPIwrapper. `scripts/check-mpi-features.sh` pins the feature graphs rather than
  only the compile outcome.

## Verification conclusions and constraints

- `cargo metadata`, root `cargo test --workspace`, workspace clippy, and the root
  dependency/layout/doc-consistency checks build and pass without MPI or HDF5
  system libraries.
- `tenferro-mpi --features system` compiles and its single-process tests and
  doctests run on this Mac with Homebrew Open MPI; the two-rank scenarios run under
  `scripts/run-mpi.sh upstream`.
- `tenferro-mpi --features runtime` compiles and its unit/integration targets pass;
  the `mpi_p2p_runtime` binary skips when `MPI_RT_LIB` is unset. The runtime doctests
  call `mpi::initialize` and fail without MPIwrapper, so the `mpi-runtime` profile
  now requires `MPI_RT_LIB` before running; this host (no MPIwrapper) did not execute
  that lane. The `system` lane runs the same doctest bodies against the same API.
- `tenferro-hdf5 --features hdf5` builds without HDF5 headers; the test suite needs
  a runtime `libhdf5`, which Homebrew provides on this host.
- `cargo publish --dry-run -p tenferro-mpi` is the publication gate; the branch does
  not publish it. `tenferro-hdf5` stays unpublished until `hdf5-rt` is on
  crates.io, and its git dependency remains the only unpublished dependency.
