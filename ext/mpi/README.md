# tenferro-mpi

Blocking point-to-point MPI transport for tenferro host tensors: `send`, `recv` and
`recv_into` for `TypedTensor<T, R, Host>`, a host-resident `TypedTensor<T, R, Dynamic>`
and the dtype-erased `Tensor`.

```toml
[dependencies]
tenferro-mpi = "0.1"
```

```rust
use tenferro_mpi::mpi::traits::*;
use tenferro_mpi::{recv_into, send};

let universe = tenferro_mpi::mpi::initialize().unwrap();
let world = universe.world();
// ... one value per call, on a tag the caller owns ...
```

The adapter borrows the caller's communicator: it never initializes or finalizes MPI,
never duplicates a communicator and never moves data between host and device. A
`Gpu` tensor does not implement `Transport`, so an accidental device transfer is a
compile error. The wire format, the failure protocol and the limits are specified in
[`docs/design/mpi-transport.md`](../../docs/design/mpi-transport.md).

Needs system MPI headers and libclang at build time (the crates.io `mpi` binding). For
a build with no MPI headers at all, use `tenferro-mpi-rt`, which loads MPI at run time.

Run the two-rank scenarios with `scripts/run-mpi.sh upstream`; a plain `cargo test`
runs only the single-process checks.
