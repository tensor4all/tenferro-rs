# Prototype findings

One file per prototype (p1.md … p7.md): what was built, what passed, what the
grammar could not express, and what was not verified.

Toolchain: `cargo +nightly` (rustc 1.93) with
`RUSTFLAGS="-Zcrate-attr=feature(atomic_try_update)"` and `--ignore-rust-version`,
because rustc 1.96 is not installed on the machine that ran them.

## Results (cb27284d + this branch)

All 79 prototype tests pass (`cargo +nightly test --ignore-rust-version -p
tenferro-api-grammar-proof`, flags above): P1 15, P2 13, P3 7, P4 10, P5 9, P6 8, P7 17 (15 + 2 in the
allocation binary) plus 2 `compile_fail` doctests. P7's hook also leaves the tenferro-cpu lib suite (347) and
the tenferro-linalg lib suite (149) passing.
P4's internal hooks also pass the tenferro-tensor error tests (28 + 7 doctests),
the tenferro-linalg lib suite (149) and the tenferro-einsum lib suite (210).

## Findings that change the design

| # | Finding | Where |
|---|---|---|
| P2 | N-ary einsum silently drops requested conjugation (`lhs_conj`/`rhs_conj` reset to false, call returns `Ok`); only the binary dot path honours it, and `DotGeneralAccumulation` cannot express conjugation of a third operand | `crates/tenferro-einsum/src/concrete.rs:1880`, `eager.rs:1235` |
| P1 | Full-matrices SVD cannot be combined with gauge options (`svd_with_options*` is thin only, `svd_full*` takes no options); `driver` is accepted but has no effect | p1.md |
| P1 | One implementation holds only by routing typed views through the erased path; the typed borrowed entry (`svd_with_exec`) refuses negative strides that the erased path gathers | p1.md |
| P4 | Native CPU einsum never materializes an input, so `Forbid` is vacuous there; only the SVD read gather (and BLAS packing) is real input materialization | p4.md |
| P4 | Eager fallback copies every operand (`to_contiguous_read`) even when compact; by the ruling's definition that is not materialization, so `Forbid` does not stop it | p4.md |
| P4 | Rank-3 strided SVD batches are refused under `Forbid` although a copy-free kernel exists (dispatch, not math) | p4.md |
| P5 | `tenferro_ad::Error` is `tenferro_runtime::Error`: two outer error types, not three; einsum returns its own error on every layer; transparent wrappers hide `tenferro_tensor::Error` from `source()` walks; traced `kind()` can disagree with the reachable cause | p5.md |
| P6 | `TraceContextEinsumExt` already owns `einsum`/`einsum_with` (with a vestigial `&mut self`), so the owner-receiver entry collides by name; `finish`+`compile_traced_graph` leaves `input_keys` empty while `compile_many` fills them; TracedTensor-path build costs 12-20% more allocations, time within noise at N=512 | p6.md |
| P7 | The six `*_with_exec` functions can be replaced by grammar on a borrowed engine: lifetimes, zero-allocation parity (0 vs 0 on every case, rank 8 included), nested and concurrent sub-team use, and no retention (pool recycler is `Weak`) all verified. Costs: a carrier engine must implement 48 required `Engine` methods (all `Unsupported`), a typed view cannot have both the gather and the refuse rule for negative strides, and the borrowed route has no gauge. A `with_lanes` split API works on top; per-lane serial engines (design C) ignore the share | p7.md |
| P3 | Session-receiver consuming FFT needs no hook (`EagerSession::engine()` is public); `?` drops the recoverable owner; in-place FFT has no length option, so its options diverge from out-of-place FFT | p3.md |

## Not verified

BLAS builds (compiled only), GPU engines, f32/c32 coverage, rustc 1.96.
