# tenferro-mpi

Blocking point-to-point MPI transport for tenferro host tensors: `send`, `recv` and
`recv_into` for `TypedTensor<T, R, Host>`, a host-resident `TypedTensor<T, R, Dynamic>`
and the dtype-erased `Tensor`.

The crate has `default = []`: pick exactly one MPI backend feature. Both build the
published [`rsmpi-rt`](https://crates.io/crates/rsmpi-rt) 0.8.2 package, whose library
target is still `mpi`.

```toml
[dependencies]
# rsmpi-rt's bindgen backend: needs system MPI headers and libclang at build time.
tenferro-mpi = { version = "0.1", features = ["system"] }
```

```rust
use tenferro_mpi::mpi::traits::*;
use tenferro_mpi::{recv_into, send};

let universe = tenferro_mpi::mpi::initialize().unwrap();
let world = universe.world();
// ... one value per call, on a tag the caller owns ...
```

For a build with no MPI headers, enable `runtime` instead:

```toml
tenferro-mpi = { version = "0.1", features = ["runtime"] }
```

`runtime` enables rsmpi-rt's MPIABI backend (`mpi-rt-sys 0.1.0`), which loads MPI at
run time and needs no headers or libclang. At run time it needs an
[MPIwrapper](https://github.com/eschnett/MPIwrapper) library:

```sh
git clone https://github.com/eschnett/MPIwrapper
cmake -S MPIwrapper -B MPIwrapper/build -DCMAKE_INSTALL_PREFIX="$HOME/.local"
cmake --build MPIwrapper/build && cmake --install MPIwrapper/build
export MPI_RT_LIB=$HOME/.local/lib/libmpiwrapper.so
```

`system` and `runtime` are mutually exclusive; `engine` compiles the protocol with no
binding for a caller that supplies its own backend implementation.

The adapter borrows the caller's communicator: it never initializes or finalizes MPI,
never duplicates a communicator and never moves data between host and device. A
`Gpu` tensor does not implement `Transport`, so an accidental device transfer is a
compile error. The wire format, the failure protocol and the limits are specified in
[`docs/design/mpi-transport.md`](../../docs/design/mpi-transport.md).

Run the two-rank scenarios with `scripts/run-mpi.sh upstream` (or `rt` for the
runtime backend); a plain `cargo test` runs only the single-process checks.
