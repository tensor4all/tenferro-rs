# tenferro-mpi-capi

C entry points that let a host language use tenferro's blocking point-to-point MPI transport
on a communicator the host already owns.

The host (MPI.jl, mpi4py) passes its **raw C communicator handle** as a `uint64_t`.
tenferro reads that handle through the same MPI library object the host loaded, found by
walking the loaded objects, so the handle encoding cannot disagree. It never initializes or
finalizes MPI, never duplicates or frees a communicator, and never changes the communicator's
error handler. Give tenferro a communicator of its own (`MPI.Comm_dup`, `comm.Dup()`) so a
host message cannot collide with a tenferro tag.

The contract is `docs/design/mpi-native-abi.md` and `docs/design/mpi-transport.md`.

| Function | Purpose |
|---|---|
| `tfe_mpi_library_info` | family, library path, version string; callable before `MPI_Init` |
| `tfe_mpi_comm_attach` / `tfe_mpi_comm_detach` | borrow / release a communicator (process-local, no message) |
| `tfe_mpi_comm_info` | rank, size, thread level, whether the error handler is fatal |
| `tfe_mpi_send` | send a compact column-major host buffer with a dtype code and shape |
| `tfe_mpi_recv`, `tfe_mpi_recv_take` | receive into an object-held buffer, then copy it out (loss-free) |
| `tfe_last_error_message` | thread-local message of the last failure |

Every entry runs inside `catch_unwind`; the library must be built with `panic = "unwind"`
(a compile error otherwise). A panic returns `TFE_INTERNAL_ERROR` and poisons the
communicator object: later calls fail without touching MPI, and the peer rank may stay
blocked in its matching call, because a panic in mid-protocol is not recoverable on the wire.

Status codes beyond `capi-error-handling.md`: `-10` unsupported/ambiguous MPI library,
`-11` MPI not initialized, `-12` MPI finalized, `-13` an MPI call failed, `-14` transport
protocol error, `-15` wrong thread for the MPI thread level, `-16` no MPI library loaded yet
(retry after the host initializes MPI).
