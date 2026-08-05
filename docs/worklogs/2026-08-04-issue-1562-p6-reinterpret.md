# Issue #1562 — P6 representation reinterpretation

## Scope

This work implements only the explicitly selected P6 phase from #1555 and
#1562. The sealed host pairs are `Complex32 <-> f32` and `Complex64 <-> f64`.
Borrowed, exclusive mutable, and consuming typed forms share the same root
allocation and checked descriptor rules. Rank-changing complex-to-real views
prepend a component axis and publish a dynamic result rank.
Consuming reinterpretation publishes only through the existing allocation-group
descriptor; a non-group host buffer is returned unchanged with a typed
unsupported error, so no `Vec` is retagged with raw parts.

Backend-native reinterpretation, provider migration (P7/P8), compatibility,
recovery/quarantine, cryptographic identity, and repeated validation remain out
of scope.

## Evidence

- `cargo test -p tenferro-tensor --test storage_reinterpret`
- `cargo test -p tenferro-tensor --test storage_reinterpret_rank`

The tests cover compact, reverse, scalar, singleton/empty, mutable mapping,
sealed-pair rejection, dynamic dispatch, rank policy, and consuming failure
recovery. The P6 ledger rows are activated only after these commands and the
workspace checks pass on the final commit.
