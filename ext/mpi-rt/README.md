# tenferro-mpi-rt

The same MPI transport as `tenferro-mpi`, built against
[rsmpi-rt](https://github.com/tensor4all/rsmpi-rt): MPI is loaded at run time through
the MPIABI wrapper, so this crate needs no system MPI headers, no C compiler and no
libclang at build time, and the MPI implementation is chosen by `MPI_RT_LIB`.

Unpublished (a git dependency cannot be published on crates.io); `tenferro-mpi` is the
published default.

```sh
# once: build MPIwrapper against the system MPI
git clone https://github.com/eschnett/MPIwrapper && cmake -S MPIwrapper -B MPIwrapper/build
cmake --build MPIwrapper/build && cmake --install MPIwrapper/build --prefix "$HOME/.local"

export MPI_RT_LIB=$HOME/.local/lib/libmpiwrapper.so
scripts/run-mpi.sh rt          # launches mpiexec -n 2 itself
```

The protocol, wire format and error type are `tenferro-mpi`'s; this crate supplies
only the four MPI operations and re-exports its own `mpi` identity, which a caller must
use for the communicator and process handles.
