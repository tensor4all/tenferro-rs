# API consolidation policy

What we do when the same fact is expressed more than once in a public surface, adopted after #2483 (linalg factorization surface),
#2484 (tensor core), #2485 (duplicate-systems audit) and #2486 (a public panic), each reviewed by an adversarial reader and a code
verifier.

This is a policy, not a proposal: it says which rules survived review, what was withdrawn and why, and in what order the work goes. The
proposals live in the issues named above.

## 1. Rules that survived

**R1. One fact, one expression — and collapse toward the one the code obeys.** When a fact is expressed N ways, the question is not which
is nicer; it is which one is *enforced at runtime*. Placement is the model case: a `Representation` type axis exists
(`types.rs:1051/1089/1118`), a `Placement` value rides on every tensor (`types.rs:357`), and a runtime check runs ~63 times — and the
runtime check is the one that holds. Collapse toward the enforced expression, then delete the others.

**R2. One entry shape per operation; `_into` is the primitive, the allocating form is sugar.** Signature:
`fn svd_into(exec, buffers, view, u, s, vh) -> Result<(), LinalgError>` with `fn svd(exec, buffers, view) -> Result<Svd<T>>` on top.
Implementation primacy and ergonomic primacy are separate decisions: the primitive gets the `_into` shape because it is the one that
makes allocation countable; the documented entry stays the allocating one because that is what a caller actually wants.

**R3. Never canonicalize implicitly.** The surface accepts every layout the selected provider handles directly, and *refuses* what it
cannot express. It does not silently copy. Concretely, on `d4cba68b`:
- `raw_view` refuses only negative strides (`cpu/linalg/mod.rs:92`); `faer_strided_ok` asks only for rank 2 and nonnegative strides
  (`faer_linalg.rs:2071`). So a transposed matrix already runs without staging, and a rule that refused non-contiguous input would
  remove capability while sounding like simplification. (This rule was written that way once; the adversarial review caught it.)
- The read path does the opposite of refusing: `cpu/backend.rs:362,403,525,561,585` fall back to `with_materialized_tensor_read`, and
  `provider_readable` (`mod.rs:116`) gathers negative strides for `solve`. That is the invisible cost this rule exists to remove.
- A caller who wants the staging writes `buffers.normalize(view)` themselves. One line, visible at the call site.

**R4. Erasure lives at the FFI edge and nowhere else.** One erased type for the bindings. `TensorRead` with 3788 mentions in this crate is
the *main* type, which is why the copy rule inside it is invisible — that is what to fix, not what to keep.

**R5. No foreign crate's type in a public API.** Not a style preference; a measured release cost. `faer::Par<'a>` carries
`threadteam::TeamExec`, so threadteam 0.4 → 0.5 forced `t4a-faer` 0.32.0 as a *major*, and then the whole stack had to move in lockstep:
`strided-basic 0.11` pinned threadteam ^0.4 while `t4a-faer 0.32` required ^0.5, giving 128 `E0308`s and two `Exec` types in one graph. The
public type should be the crate's own (`faer::Par`, `faer::Workspace`) with the foreign type behind it.

**R6. A public transition is not guarded by an invariant comment.** `Tensor::external(..)` is public (`types.rs:6055`), `tensor_view()` is
public (`:8200`), and the path ends in `unreachable!` ("views cover the preset scalars", `:7139`). The caller gets a process abort and the
comment asserts something the type does not carry. Public entry points return `Result`; `unreachable!` guards only states the type system
has actually excluded.

**R7. Push what the type can carry into the type, and consolidate the rest in one place.** `HostTensor`/`DeviceTensor` is worth doing, but
it *moves* the check rather than deleting it: the host check also rejects `DType::External` and tests for a backend buffer, and no type can
express managed memory. So: the axis where it helps (the linalg layer), one narrowing at the erased boundary for the rest. This is now how
the CPU entries work: `TensorRead::try_host(op)` / `TensorView::try_host(op)` is the one fallible `Dynamic -> Host` narrowing and returns a
`TensorView<'_, Host>`; each `LinalgBackend` method and each CPU reduction entry narrows its operands once, and everything below takes the
host-marked view (`TypedTensorView<'_, T, DynRank, Host>`) without re-checking placement. The managed-memory Cholesky route is dispatched
before the narrowing, and values the CPU engine itself produces (a materialized copy) are narrowed where they are produced. Write targets
cross the same boundary: `TensorWrite::try_mut_host(op)` / `TensorViewMut::try_mut_host(op)` returns a `TensorViewMut<'_, Host>` with the
read side's errors, so a CPU `_read_into` entry narrows its destination once and writes it without a refusal-only check. `TensorWrite` itself
stays unparameterized, like `TensorRead`: its `Tensor(&mut Tensor)` arm holds an owner whose placement no type proves.

## 2. Withdrawn — do not re-propose without new evidence

Recording these matters more than the rules, because each one is plausible and each one was wrong:

| withdrawn claim | why it is wrong |
|---|---|
| "non-contiguous input is `Unsupported`" | `raw_view` refuses only negative strides; a transposed view already runs without staging. The rule would have removed capability. |
| "two layout languages (tenferro vs strided-rs)" | `RawStridedRef` **is** `strided_view::RawStridedRef` (`faer_linalg.rs:11`). The disagreement is between two tenferro predicates, which is smaller and fixable. |
| "a buffer leaves the pool, so no steady state" | `finish_lease` → `assume_init_recycled` (`pooled_uninit_output.rs:241`) attaches a recycler; drop returns the buffer. The existing zero-allocation ceilings measure that steady state. |
| "the placement axis is ~95% dead" | 95% `Dynamic` means 95% go through the erased type. `erased_host.rs` uses `Host` 27 times. The real finding is narrower: **zero** `Host` in `tenferro-linalg`. |
| "breaking this is cheap, the rest are our own tests" | 88 of the 133 `LinalgBackend` references are non-test (`tensor_ext.rs` 39, `backend.rs` 34, `extension.rs` 6, gpu/cpu/lib/benches 11). `extension.rs` is an eager/traced execution boundary, not a binding adapter. |
| "a `Workspace` token solves dependency version identity" | A container that borrows the same team still needs a version-compatible boundary; a container that builds its own violates the "one scheduling owner" contract. The release train has to align regardless. |
| "`GlobalAlloc == 0` still needs proving" | `tests/cpu_linalg_allocation.rs:591-599` already asserts it for every borrowed entry on x86_64. The missing assertion is a **pool checkout counter** (one `svd_workspace`-sized checkout per call, same count for transposed and rank-3 input, unchanged output pointers). |

## 3. How a claim gets admitted

Three rules, all learned by being wrong:

1. **Numbers are re-derived on `origin/main` before they enter a design document.** A whole set of my counts came from a local branch
   (`adf9143c4`) that is not an ancestor of main: 1001 vs 1039 `TypedTensor<`, 39 vs 35 `Result<Vec<Tensor>>`, 1472 vs 2050 erased
   `Tensor`. Three readers then argued about three different populations (35 occurrences, 39 stale, 24 trait declarations) — all correct,
   none comparable.
2. **A reviewer's numbers are re-derived too.** One review supplied `24` where `grep -c` says `35` in the same file, and it was right about
   everything else. A review is not a verdict.
3. **A duplicated expression is a bug only when it is shown to disagree.** "Two types look alike" is not a finding: `Shape` and `Strides`
   are two facts. The admissible form is a case where the two expressions give different answers for one input — e.g. `set_placement`
   (`types.rs:2362`) let a caller relabel a host tensor as `Device`, after which the CPU path rejected it and the linalg path accepted it.
   That case is now closed the way this rule prescribes: the payload owns the fact (`is_host_memory`), every entry point asks it, and
   `set_placement` refuses a device label on host storage so the disagreement has no input left to disagree about (#2484).

## 4. Order of work

Not by importance; by what is blocked on what.

1. **Real spectrum in `tlinalg`** (`s: &mut [T::Real]`, and `eigh_into` at `faer_linalg/with_exec.rs:212` as well, which the proposal
   missed). Small, testable at the numerical and allocation level, agreed by both reviews, and it deletes the c32/c64 gather rather than
   moving it. Nothing below depends on it and it depends on nothing.
2. **#2486** (`Tensor::external` + `tensor_view()` panic). Independent of every design question; it is shipped behaviour.
3. **Linalg surface, rules 1-10.** Tenferro-local, on `tcontract-exec 0.10`, no faer change needed.
4. **Tensor core** (#2484). `tenferro-tensor` depends on neither strided-basic, threadteam nor faer, so this is free to move at any time.
   It should move *after* 1-3 only because the linalg work will show which erased shapes are actually load-bearing.
5. **`faer` addendum (A-F)** — the one item that is blocked: it is the release that carries threadteam 0.5, and it needs the chain
   `strided-* 0.12.0` (done) → `tcontract-exec`/`tcontract` 0.11 → `cpueinsum`/`tlinalg` 0.10 first.

## 5. Open, and how they get decided

- **Nine further duplicate rows** were reported by the audit but are a list, not evidence: two `TensorScalar` traits, two symbolic-dim
  expression trees (`DimExpr`/`RawSymDim`), two contiguity predicates that disagree at offset 1, four enums for operation identity, three
  backend-kind identifiers, `INLINE_RANK` declared in two crates with an assert to keep them equal. Each earns a row only with
  `file:line` plus a case where the two disagree.
- **Batch**: a separate function, not a `Batch<S>` shape parameter — batch semantics are runtime data in the eager/traced layers and a
  generic parameter does not move them into the type.
- **`&mut Exec` + `&mut BufferPool` vs one `Workspace`**: keep two for now. Their combination has not been shown to represent an invalid
  pairing, and the `Workspace` argument has been about argument count rather than ownership.
