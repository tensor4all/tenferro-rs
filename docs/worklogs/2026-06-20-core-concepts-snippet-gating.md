# Work log: gate getting-started core-concepts snippets with compiled examples

- Date: 2026-06-20
- Type: AI-assisted bug fix (documentation correctness)
- Branch: `fix/core-concepts-direct-snippet-compile`

## Summary

`docs/getting-started/core-concepts.md` "Direct Tensor Execution" snippet did
not compile: `let a = Tensor::from_vec_col_major(...)` / `let b = ...` omitted
`?`/`.unwrap()`, so `a.matmul(&b, &mut backend)` was called on a
`Result<Tensor, _>` and failed with `E0599: no method named matmul ... for enum
Result`. A reader copy-pasting the snippet could not build it.

## How it was found

A **source-blind documentation audit** (the method added to
`tensor4all-agent-rules` `rules/common/docs-and-tests.md`): a doc-only agent was
given only README + `docs/getting-started/` (no source, no repo links) and asked
to write a minimal integration program from the docs alone. The main agent then
compile-checked the doc snippets against the real crates as an external
downstream crate, which surfaced the `E0599`.

## Root cause

`scripts/check-doc-snippets.py` only verifies code blocks wrapped in
`<!-- snippet-source: PATH -->` / `<!-- end-snippet-source -->` markers (it syncs
them from real example files, which CI compiles). The `index.md` First CPU
Program is marker-backed (`crates/tenferro-runtime/examples/cpu_quickstart.rs`)
and stayed correct; the core-concepts concept snippets were hand-written and
**unmarked**, so they were never compile-checked and the Direct snippet rotted.

## Change

Back all three concept snippets with real, CI-compiled examples and wire them
via snippet-source markers:

- `crates/tenferro-runtime/examples/direct_tensor_execution.rs` (fixed: `?`)
- `crates/tenferro-runtime/examples/traced_graph_execution.rs`
- `crates/tenferro-ad/examples/eager_backward.rs`

This both fixes the Direct snippet and prevents the whole section from rotting
again, consistent with the already-marker-backed Memory Model snippet
(`column_major_memory.rs`) in the same file.

## Rejected / deferred alternatives

- Minimal fix (add `.unwrap()` to the two lines only): rejected — leaves the
  snippet unmarked and able to rot again; does not address the root cause.
- New repository audit rule "flag unmarked code blocks in user-facing docs":
  considered; the mechanical snippet-source gate is stronger for these snippets.
  A general rule could still be proposed separately if unmarked blocks recur.

## Deferred (other audit findings, doc-clarity only — not compile bugs)

- `as_slice::<T>()` returns `Result<&[T]>` (not `Option`); undocumented.
- No copy-pasteable dependency (path placeholder / crates.io `"..."`; no git URL).
- Error type not named; no `Debug`/print example; MSRV/edition unstated.
- Eager example imports `Tensor` from `tenferro_ad` with no deps block shown.
- `pytorch-jax-mapping.md` column-major fragment is illustrative (not a program).

## Verification

- `cargo run -p tenferro-runtime --example direct_tensor_execution` — ok
- `cargo run -p tenferro-runtime --example traced_graph_execution` — ok
- `cargo run -p tenferro-ad --example eager_backward` — ok
- `python3 scripts/check-doc-snippets.py --check` — `doc-snippets-ok`
- `cargo fmt --all --check` — ok
- `cargo clippy -p tenferro-runtime -p tenferro-ad --examples` — ok
- Not run (unrelated to this change): full `cargo test --workspace --release`,
  coverage, `cargo doc`.
