# Sub-programs

Status: in progress under [#2691](https://github.com/tensor4all/tenferro-rs/issues/2691).
The approved step-1 design is the
[design comment on #2691](https://github.com/tensor4all/tenferro-rs/issues/2691#issuecomment-7072688632).
This document keeps the decisions that later work must follow and is updated by
each PR of the sequence.

A sub-program is a semantic program that an operation carries: a `scan` or
`cond` body, or the linear operator a Krylov solver applies. It is recorded
once and specialized for each concrete shape it is prepared at.

## What exists

| Piece | Where | Since |
|---|---|---|
| `Subprogram`: closed, shape-generic program | `tenferro_runtime::program::Subprogram` | step 1 (PR 1) |
| `TraceContext::record` and `RecordedProgram` | `tenferro_runtime` | step 1 (PR 1) |
| v2 semantic identity with sub-program calls | `program/identity.rs` | step 1 (PR 1) |
| Builder entry that attaches calls, preparation, execution | — | PR 2 |
| AD through sub-programs (`SubprogramAd`) | — | PR 3 |
| `scan`, `cond`/`switch`, `while_loop` | — | PRs 4–6 |

## Representation

- A `Subprogram` is an `Arc<SemanticProgram>` plus a **parameter count**. Its
  inputs are the declared parameters, in declaration order, followed by the
  **captures**. It has no `ProgramBindings`.
- **Captures are hoisted.** A tensor that the recorded closure binds from data
  (`from_vec_col_major`, `from_tensor_concrete_shape`) becomes a trailing input, and the tensor itself travels beside the program in
  `RecordedProgram::captures`. An operation that carries the sub-program
  receives captures as ordinary operands. Because of this, the identity never
  depends on tensor data: re-recording a closure over different weights of the
  same shape produces an equal `Subprogram`. Values from constant constructors
  (`full`, `zeros`, `ones`, `scalar`) are `Constant` ops: they stay in the
  program and are part of its identity, whatever their broadcast width, so
  recordings that differ only in such a value are different sub-programs. A
  test pins this contract.
- **Frozen at record time:** the operation list, value metadata with symbolic
  extents, shape guards and the cached fingerprint. **Not frozen:** concrete
  extents, engine placement, kernels and fusion, which are decided when the
  sub-program is prepared.
- `TraceContext::record(inputs, f)` traces `f` once on a fresh context whose
  parameters have symbolic extents (`concrete_shape()` fails inside `f`). Every
  declared parameter stays an input even if unused. A value of another trace
  context cannot enter the recording: combining it fails with the existing
  cross-context error, and returning it fails with `InvalidArgument`
  (`outputs`). An output that depends on a placeholder input `f` added itself
  fails with `InvalidArgument` (`f`), because that input would be neither a
  parameter nor a capture. A placeholder no output uses is pruned.
- `Subprogram::close(FrozenProgram)` from the step-1 design is not public yet.
  `record` is the only producer, and its snapshot already orders parameters
  before captures. Closing an arbitrary frozen program needs input reordering
  with `InputDim` remapping. It will be added when AD needs to turn a
  derivative program into a sub-program (PR 3).

## Calls and identity

- A `SemanticOperation` carries `subprograms: Box<[SubprogramCall]>`, which is
  empty for every operation today. A call is the callee plus `input_meta`: the
  callee's inputs stated in the carrying operation's **operation-local**
  symbols. In those symbols, `InputDim { input_idx }` names that operation's own
  input, the same convention that operation payloads use. Snapshots and imports
  renumber program inputs but never an operation's own inputs, so they copy
  calls unchanged.
- Sub-programs live on the operation, not in `ExtensionOp` payloads. The runtime
  owns their identity, validation and preparation, and extension authors never
  hash a program.
- The fingerprint tag is `tenferro.semantic-program.v2`. After an operation's
  placement, the encoder appends the call count and, for each call, the callee's
  cached 16-byte **sub-program** fingerprint and the encoded `input_meta`.
  `Subprogram::semantic_fingerprint` combines the program fingerprint with the
  parameter split, so it covers exactly what `Subprogram` equality compares.
  `SemanticProgram::semantic_fingerprint` of the same program ignores the split.
  Hashing therefore costs O(calls) per operation however deeply calls nest.
  Fingerprints are process-local cache keys (`DefaultHasher` is not stable
  across toolchains), and nothing persists them, so no v1 compatibility is
  needed.
- Exact equality compares calls pairwise. The callee comparison tries
  `Arc::ptr_eq` first and falls back to `semantic_eq`; the parameter counts and
  `input_meta` are compared as well. A snapshot copies calls only onto the one
  operation its replay appends, and it refuses a replay that appends anything
  else rather than guessing the owner. Recursion terminates because a callee is
  frozen before the operation that contains it.
- `Subprogram: Eq + Hash` uses the same structural definition, so two
  recordings of one closure are equal and caches keyed by a sub-program hit
  across recordings. The parameter split is part of the identity: the same
  program with one input treated as a capture instead of a parameter is a
  different sub-program.
- Retained-byte accounting charges a call's `input_meta` **and its callee**.
  A shared callee is counted at every call, which over-estimates sharing but
  never under-counts what a program keeps alive.
- Import copies calls without checking `input_meta` against the carrying
  operation, because no call can exist yet. The PR 2 builder entry validates
  calls, and import must apply the same check to calls arriving from a source
  program.
