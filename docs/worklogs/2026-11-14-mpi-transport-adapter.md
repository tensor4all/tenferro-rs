# MPI transport adapter (#2612)

## Decisions

- **Two crates, not one.** `tenferro-mpi` is published against crates.io
  `mpi = "=0.8.1"`; `tenferro-mpi-rt` is unpublished and links the pinned
  `tensor4all/rsmpi-rt` rev, whose runtime-loading backend cannot be a published
  dependency. Both are excluded standalone workspaces, like `ext/sparse`: membership
  would force MPI headers, libclang and a launcher into every existing lane.
- **The protocol lives once.** `tenferro-mpi` owns the wire format, the error taxonomy
  and the `Transport` set; a backend crate implements four MPI operations
  (`MpiBackend`) and re-exports the protocol. To make that possible without publishing
  a second entry surface, `tenferro-mpi` gained an `engine` feature: protocol only, no
  MPI binding. Accepted surface beyond the decided API: `MpiBackend`, `Transport` and
  `tenferro_mpi::engine::{send, recv, recv_into}`.
- **Payload as bytes, dtype in the header.** Per-dtype MPI datatypes would need a
  dispatch arm per dtype, the binding's `complex` feature for `Complex32`/`Complex64`,
  and a second `num-complex` identity. Bytes keep one path for all seven dtypes; the
  header carries the dtype code and the sender's byte order, and a big-endian peer is
  rejected rather than mis-decoded.
- **Header → ack → payload, so a refusal costs no payload and no drain.** The payload
  is only in flight after an accepted ack, which is why `recv_into` needs no drain path
  and why a rejected receive can leave its destination untouched.
- **Probe before receive everywhere,** header, ack and payload. A frame received into
  a fixed buffer would need MPI truncation handling; probing makes a wrong frame a
  typed error instead.
- **Dtype code table shared with #2611** in `docs/design/tensor-dtype-codes.md`;
  `DType::External` has no code because a `TypeId` is process-local. A sender therefore
  aborts with `UnsupportedDType` and the receiver learns it instead of waiting.
- **Cross-tag overlap is not offered.** A send blocks until the receiver acks, so two
  tags cannot be driven from one thread; a caller that wants overlap needs a thread per
  tag. The scenario list asserts tags are independent, not that they interleave.

Review: `ollama-cloud/glm-5.3-flash:high` (read-only, plan review) — VERDICT: APPROVE,
0 BLOCKER, 4 IMPORTANT, 4 MINOR; every finding is folded into the design above
(`engine` feature explicit, publish metadata, the `links` claim restated, probe-before-
receive for the payload as well, `[profile.ci]` in both manifests, corrected citations,
one public engine path).

Post-implementation review: `openai/gpt-6.1-sol`, read-only, of the finished diff —
`CHANGES` (2 BLOCKER, 6 IMPORTANT, 2 MINOR), all resolved before the PR:

- BLOCKER: a `bool` payload was received into the destination, so a peer byte of `2`
  made an invalid `bool` - undefined behaviour the moment anything read it, and an
  unwind would have left it behind. Payloads for a type with invalid bit patterns now go
  through a byte scratch and are assigned only after validation; the destination keeps
  its values on rejection. Asserted by a hand-crafted `[2, 0]` payload scenario.
- BLOCKER: an erased `recv_into` checked dtype and shape before the ack but not
  writability, so a device-backed `Tensor` accepted and the sender then sent a payload
  nobody received. The writable-slice check moved before the ack and reports
  `DestinationNotWritable` to the sender; the scenario gained that leg.
- IMPORTANT: a TFMP header with a length its own rank field contradicts was consumed
  without a verdict, leaving that peer to wait forever. Such a frame now gets a reject
  ack (bounded scratch), an ack where a header belongs is `UnexpectedFrame`, and a frame
  above the scratch bound is still consumed silently because it cannot be ours.
- IMPORTANT: the runtime crate's doctests imported `tenferro_mpi::mpi`, which the
  `engine` feature leaves out, so they could not compile and its profile never ran them.
  Fixed to the crate's own `mpi` identity, and the profile now runs doctests there.
- IMPORTANT: several multi-rank examples sent on every rank, which self-sends on rank 1
  and pairs with nothing on rank 0. All examples now branch by rank.
- IMPORTANT: the `links = "mpi"` claim was wrong for `mpi-rt-sys` (it declares none), so
  the actual constraint is the single `mpi-sys` link identity plus the `mpi` type
  identity. Corrected in the design note, the manifest comment, the work log and the
  feature-check comment.
- IMPORTANT: the note claimed separate tags avoid a blocking-send deadlock and promised
  `UnexpectedFrame` for stray payload bytes. Neither is true of an unframed payload;
  the note now states that calls must be complementary and that one call owns its tag.
- MINOR: a 32-bit host and a `u8`-only overflow case, and a `Rank<2>`/inline-shape
  claim that the code did not match. The overflow fixture now overflows either width,
  the rank claim is corrected, and a receive collects its shape inline instead of
  through a `Vec`.
- Also deleted per review: the duplicate doctest invocation in the upstream profile and
  a reference to a build script that does not exist.

## Verification conclusions and constraints

- Both backends pass the same eight two-rank scenarios under `mpiexec -n 2`, on this
  host (OpenMPI 4.1.6, `-j 4`): all seven dtypes (bit-exact, including `NaN` and
  `-0.0`), rank 0, an empty extent, `Rank<2>`, typed ↔ erased ↔ `Dynamic` crossings,
  allocation budgets, receiver rejections (a dtype/rank/shape mismatch, and a
  device-backed destination refused before the ack) with a recovery leg on the same tag,
  sender aborts observed as `PeerAborted`, tag behaviour, an invalid `bool` payload, and
  hand-crafted foreign frames
  (bad magic, version 99, foreign byte order, `i32::MAX + 1` elements, an element count
  that disagrees with the shape) with the ack code each produces.
- Allocation budget: a successful `send` and `recv_into` allocate nothing; `recv`
  allocates its result buffer, and the shape of an ordinary rank stays inline. `bool` is
  the exception on both sides: its payload is received through a byte scratch, because a
  byte that is not 0 or 1 must never be written into a `bool` destination (that is
  undefined behaviour, not a wrong value).
- `tenferro-mpi-rt` was verified with MPIwrapper built from source against this host's
  OpenMPI into a scratch prefix (`MPI_RT_LIB`). The loader path is the one real risk: a
  wrong or missing `MPI_RT_LIB` fails at `initialize`, by construction, and no hosted
  lane covers it — building MPIwrapper needs CMake and a Fortran compiler. The CI lane
  therefore runs the `upstream` backend only.
- Not checked here: a genuinely big-endian peer (only the marker logic is tested), any
  MPI implementation other than OpenMPI 4.1.6, and behaviour above `i32::MAX` elements
  (rejected by design, so the rejection is what is tested).
- `scripts/check-mpi-features.sh` asserts the identity rules the two manifests rely on:
  a no-backend build fails with the crate's own message, `engine` alone pulls no `mpi`,
  the default graph carries exactly one `mpi`, and the runtime graph carries
  `mpi-rt-sys` and no `mpi-sys` (`mpi-sys` declares `links = "mpi"` in the crates.io
  and rsmpi-rt forms alike, so two of them cannot link; `mpi-rt-sys` declares none, and
  the `mpi` type identity is what keeps the two crates apart).
