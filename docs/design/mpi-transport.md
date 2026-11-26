# MPI transport adapter

`tenferro-mpi` moves one tensor value between two ranks over MPI point-to-point
messages. This note is the public contract: the wire format, the failure protocol, the
supported values and the boundaries the adapter does not cross. The API itself is
documented on the crate.

## Status

Version 1, blocking, two ranks per call. Deliberately small:

- a caller lends a communicator and owns the MPI lifecycle; the adapter never calls
  `MPI_Init`, `MPI_Finalize`, `MPI_Comm_dup` or any communicator constructor, and has
  no process-global state (a source scan enforces this);
- one message carries at most `i32::MAX` elements and its payload also has to fit one
  MPI byte count. A larger tensor is rejected, never chunked or truncated; chunking is
  v2 work;
- no collectives, no nonblocking traffic, no scheduling, no host/device transfer. A
  `Gpu`-representation tensor does not implement `Transport`, so an accidental device
  transfer is a compile error, and a device-only `Dynamic` owner is rejected at run
  time with `NotHostResident` because its representation is erased;
- payloads must be compact column-major host slices. The adapter does not materialize
  a view.

Distributed scheduling belongs to the caller (hataori and friends). This crate is
transport.

The planned native-handle runtime backend, which removes the MPItrampoline path and
adds a C entry for a host-owned communicator, is designed in
[mpi-native-abi.md](./mpi-native-abi.md) (#2735). It is not implemented; this note
still describes the code as it is.

## Crate layout

`tenferro-mpi` is a workspace member with `default = []`; exactly one of two
binding features is selected when a caller wants transport. Both features build the
published `rsmpi-rt 0.8.2` package, whose library target is still `mpi`; the
dependency is aliased back to `mpi` so the adapter and its callers share one type
identity.

| Feature | crates.io dependency | Build requirement |
| ------- | -------------------- | ----------------- |
| `system` | `rsmpi-rt 0.8.2` with `mpi-sys-backend` → `rsmpi-sys 0.2.4` (library `mpi_sys`) | system MPI headers, a C compiler and libclang |
| `runtime` | `rsmpi-rt 0.8.2` with `mpi-rt-sys-backend` → `mpi-rt-sys 0.1.0` | no headers or libclang; `MPI_RT_LIB` and MPIwrapper at run time |

`system` and `runtime` are mutually exclusive; rsmpi-rt itself rejects enabling both
backend features at once. `engine` compiles the protocol, error type and `Transport`
set with no MPI binding.

The crate is a **workspace member**. Its default features select no system library,
so the root `cargo clippy`/`cargo test` lanes build it without MPI headers, libclang
or an MPI launcher. The `mpi` job of the `CI PR workspace tests` workflow, which feeds
the required `CI gate (PR workspace tests)`, installs the MPI toolchain and runs the
scenarios as a real two-rank world in three arms: `system` against Open MPI, `system`
against MPICH (on ubuntu-22.04, whose MPICH package forms a two-rank world), and `runtime` against Open MPI through an MPIwrapper built from a
pinned release. `scripts/ci/run-mpi-ranks.sh` fails an arm unless both ranks report
success. The crate no longer carries its own `Cargo.lock`; the root lock and
`cargo metadata` cover it.

### One `mpi` identity per graph

Every `mpi` crate is a distinct type identity: two of them in one process produce
`Process`/`Communicator` types that cannot meet. Both backends therefore name the
same published `rsmpi-rt` package, and the adapter re-exports it as
`tenferro_mpi::mpi`; a caller must take its communicator and process handles from
that re-export.

`rsmpi-sys` declares `links = "mpi"`, while `mpi-rt-sys` (the MPIABI loader)
declares no link identity. Enabling both backend features would put both sys crates
in one graph, which rsmpi-rt rejects before the adapter is compiled.
`scripts/check-mpi-features.sh` asserts the outcome: one `rsmpi-rt 0.8.2` and one
`rsmpi-sys 0.2.4` in the `system` graph, one `rsmpi-rt` and one `mpi-rt-sys 0.1.0`
in the `runtime` graph, and no MPI package at all in the default and `engine` graphs.

### Co-installation with hataori-rs

`tenferro-mpi --features system` resolves crates.io `rsmpi-sys`, which declares
`links = "mpi"`. hataori-rs still resolves crates.io `mpi 0.8.1` and its
`mpi-sys`, which declares the same link identity, so Cargo cannot place both in one
graph. A process that needs both must move hataori to `rsmpi-rt`; `runtime` is the
co-installable backend because `mpi-rt-sys` declares no link identity and loads MPI
at run time. This is a dependency-identity constraint, not a feature of the
transport itself.

## Wire format v1

All frames are byte-oriented: the header, the ack and the payload go through the same
probe-then-receive path, so a frame is never received into a buffer shorter than its
probed length and MPI truncation is never reached.

Integers are little-endian on the wire. The payload is *not* converted: it is the
host's own representation of the elements, and the header carries the sender's byte
order so a big-endian peer is rejected with `ByteOrderMismatch` instead of being
mis-decoded.

### Header frame

| Offset | Size | Field |
| ------ | ---- | ----- |
| 0 | 4 | magic `b"TFMP"` |
| 4 | 2 | protocol version, `1` |
| 6 | 1 | frame kind: `1` header |
| 7 | 1 | dtype code (`docs/design/tensor-dtype-codes.md`) |
| 8 | 1 | status: `0` accept, otherwise an abort reason |
| 9 | 1 | byte order of the payload: `1` little, `2` big |
| 10 | 2 | reserved, `0` |
| 12 | 4 | rank |
| 16 | 8 | element count |
| 24 | 8·rank | shape extents |

Length is exactly `24 + 8·rank`. A rank above 64 is rejected, which bounds the header
at 536 bytes, so it lives on the stack.

An **abort** frame is a header with a non-zero status: dtype, byte order, rank and
element count are absent (`0`), the frame is exactly 24 bytes, and it expects no ack.

### Ack frame

| Offset | Size | Field |
| ------ | ---- | ----- |
| 0 | 4 | magic `b"TFMP"` |
| 4 | 2 | protocol version, `1` |
| 6 | 1 | frame kind: `2` ack |
| 7 | 1 | verdict: `0` accept, otherwise a reject reason |

Exactly 8 bytes.

### A call

1. The sender checks its own arguments and payload with no message: tag, self-send,
   host residence, compactness, dtype code, rank, element count, payload length.
2. The sender sends the header.
3. The receiver probes, checks the frame, validates it against its own destination,
   and answers with an ack.
4. On `0` the sender sends the payload and the call returns. On a non-zero verdict it
   returns `PeerRejected { reason }` **without** sending a payload, so a refused
   receive never leaves a payload in flight.

Two consequences are worth stating because they are contract, not implementation:

- A sender blocks until the receiver has answered, and its header send itself needs
  the receiver to have posted a matching receive. Calls must therefore be *complementary
  and ordered*: one rank sends while the other receives. Two ranks that call `send` at
  each other do not progress, and a tag per direction does not change that, because
  neither side ever posts a receive.
- The payload is only in flight after an accepted ack, so a failed `recv_into`
  never needs a drain. There is no drain path in v1.
- A frame that arrives where the *header* or the *ack* is expected is reported as
  `UnexpectedFrame` (an ack where a header belongs, a header where an ack belongs). A
  payload is unframed bytes, so a stray frame arriving where a payload is expected is
  indistinguishable from payload data: one call owns its tag until it returns.

### Reasons

Reject codes (ack): `1` dtype mismatch, `2` rank mismatch, `3` shape mismatch,
`4` unsupported version, `5` malformed header, `6` too many elements,
`7` destination not writable, `8` byte order mismatch. An unknown code becomes
`RejectReason::Unknown(code)` rather than a decode failure.

Abort codes (header status): `1` not host resident, `2` non-compact,
`3` unsupported dtype, `4` too many elements, `5` rank too large, `6` host access.

Locally refused before any message: negative tag (`InvalidTag`), blocking self-send
(`SelfMessage`).

## Scope decisions and rejected alternatives

- **Payload as bytes, dtype in the header.** The alternative — one MPI datatype per
  dtype — needs a dispatch arm per dtype on both sides, pulls `complex` into the
  binding's feature set for `Complex32`/`Complex64`, and reintroduces a second
  `num-complex` identity between the caller's complex type and MPI's. Bytes keep one
  path for all seven dtypes and let the error text name the dtype.
- **`bool` as `MPI_UINT8_T` with `0`/`1`.** A C `bool` byte is 1 byte in practice but
  the mapping is a datatype detail, and a `2` in a `bool` buffer is instant undefined
  behaviour. Received bytes are validated (`InvalidBoolPayload`) and the payload is
  built from validated values.
- **Probe before receive.** A receiver that receives into a fixed buffer would need
  truncation handling, and MPI truncation is an error class the protocol cannot report
  cleanly. Probing first turns a wrong frame into a typed error.
- **One tag per call, `UnexpectedFrame` on a foreign kind.** The decided API takes one
  caller tag, so two ranks that answer on the same tag are detected (`UnexpectedFrame`)
  instead of hanging on a payload that will never arrive. A second reserved tag per
  call was rejected: it would consume tags a caller cannot see.
- **No `MPI_Init`/`Finalize`, no communicator ownership.** The adapter must be callable
  from a library that was handed a communicator by a host language (MPI.jl, mpi4py) and
  must not decide when MPI starts or stops.
- **`MAX_WIRE_RANK = 64`.** A fixed bound keeps the header on the stack and the frame
  self-describing. A tensor of higher rank is a program error, not a wire case.

## Verification

- Unit tests pin the dtype table, the golden header/ack bytes, rank 0 and rank 64
  round trips, and every structural rejection.
- A `harness = false` scenario binary runs the same eight scenarios on both backends
  under `mpiexec -n 2`: all seven dtypes, rank 0 / empty extent / `Rank<2>`, typed ↔
  erased ↔ `Dynamic` crossings, allocation budgets, receiver rejections (including a
  device-backed destination, refused before the ack) with a recovery leg, sender aborts,
  tag behaviour, an invalid `bool` payload, and hand-crafted foreign frames.
- Allocation budget: a successful `send` and `recv_into` allocate nothing; `recv`
  allocates its result buffer, and a `bool` payload allocates a byte scratch on either
  side because a byte that is not 0 or 1 must never be written into a `bool`
  destination.
- The runtime backend needs MPIwrapper: build it once (`cmake` against the system
  MPI), set `MPI_RT_LIB`, and run `scripts/run-mpi.sh rt`. It has no hosted CI lane
  because that toolchain (CMake plus a Fortran compiler) is more than the adapter's
  own build; the local evidence is recorded in the work log.
