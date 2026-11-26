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

`runtime` enables rsmpi-rt's `mpi-native-sys-backend`. It resolves MPI at run time in the
library the process already loaded (MPI.jl, mpi4py) or, for a Rust program that calls
`mpi::initialize()` itself, finds one by soname (`libmpi.so.12`, then `libmpi.so.40`, ...);
`MPI_NATIVE_LIB=/path/to/libmpi.so.N` overrides that search. No headers, no libclang, no wrapper
library to build. Calls go to that library with its own handles, so the MPICH ABI (MPICH, Intel MPI,
MVAPICH, Cray MPICH), the Open MPI ABI (Open MPI, Fujitsu MPI, IBM Spectrum MPI) and the MPI 5.0
standard ABI are supported. Any other MPI fails at start-up with an error that names the library
and its version string. There is no MPItrampoline fallback (removed in
[#2735](https://github.com/tensor4all/tenferro-rs/issues/2735)).

`system` and `runtime` are mutually exclusive: rsmpi-rt rejects enabling both backend
features at once. `engine` compiles the protocol with no binding for a caller that
supplies its own backend implementation.

For a combined graph with hataori-rs, use `runtime`: hataori still resolves
crates.io `mpi 0.8.1`/`mpi-sys`, whose `links = "mpi"` conflicts with `system`'s
`rsmpi-sys`. The two can share a graph only after hataori moves to `rsmpi-rt`;
`runtime` itself is co-installable because `mpi-native-sys` declares no `links` key.

The adapter borrows the caller's communicator: it never initializes or finalizes MPI,
never duplicates a communicator and never moves data between host and device. A
`Gpu` tensor does not implement `Transport`, so an accidental device transfer is a
compile error. The wire format, the failure protocol and the limits are specified in
[`docs/design/mpi-transport.md`](https://github.com/tensor4all/tenferro-rs/blob/main/docs/design/mpi-transport.md).

Run the two-rank scenarios with `scripts/run-mpi.sh upstream` (or `rt` for the
runtime backend, `capi` for the C entry points, see
[`tenferro-mpi-capi`](../tenferro-mpi-capi/README.md)); a plain `cargo test` runs only the single-process checks.
