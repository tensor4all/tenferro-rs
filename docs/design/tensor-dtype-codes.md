# Tensor dtype codes

One table, shared by every tenferro boundary format that has to name a tensor dtype
in bytes or in a header. #2611 (HDF5 adapter) and #2612 (MPI transport adapter) both
read it, and a future file, network or checkpoint format should extend this note
instead of inventing a second numbering.

## The codes

| Code | `DType` | Element width | Notes |
| ---- | ------------- | ------------- | ----- |
| 1 | `DType::Bool` | 1 byte | `0` is `false`, `1` is `true`; any other byte is invalid |
| 2 | `DType::I32` | 4 | |
| 3 | `DType::I64` | 8 | |
| 4 | `DType::F32` | 4 | |
| 5 | `DType::F64` | 8 | |
| 6 | `DType::C32` | 8 | `Complex32` (real, imaginary) |
| 7 | `DType::C64` | 16 | `Complex64` (real, imaginary) |

`0` is reserved and means "no dtype" (an abort frame carries no dtype at all).
Codes `8..=255` are unassigned; a reader must reject them rather than guess.

## Why `DType::External` has no code

`DType::External(TypeId)` names a caller-owned Rust type. A `TypeId` is a hash of one
compilation's view of a type: it is not stable across compiler versions, crates'
feature sets, or processes built at different times. Putting it on a wire or in a file
would produce a format that decodes only in the process that wrote it.

A caller that needs an externally defined scalar to cross a boundary supplies its own
codec and its own format. See `docs/guides/external-scalars.md` for the caller-side
story and `docs/design/mpi-transport.md` for how the MPI adapter refuses instead of
writing a process-local identifier.

## Why this note exists

The numbering is not derivable from `DType`'s declaration order, and re-deriving it
per format is how two adapters end up disagreeing about code 3. Both adapters assert
the table: a unit test pins every code to its `DType`, and a golden-byte test pins the
encoded header, so a reordering fails a test instead of a peer.
