# #1516A pooled uninitialized output guard

## Scope and sources

This work implements the accepted #1516A contract from issue #1516 and its
final contract comment, under umbrella #1535. The starting point is
`af9d566c`; strided uninitialized replay is consumed at pin `6885f52e`.
Reviewed: `AGENTS.md`, `REPOSITORY_RULES.md`, the existing buffer-pool and
indexing allocation code, issue #1516, issue #1535, and the merged strided
full-overwrite APIs.

## State-machine proof

`Fresh { actual_capacity }` and `Reused { actual_capacity }` are non-Copy checkout tokens. Acquisition
removes best-fit storage and subtracts its exact retained bytes; only Reused
increments the exact in-flight bucket. The owner holds `Vec<MaybeUninit<T>>`
and the token while incomplete. Drop discards the vector and decrements only
that token, so it cannot replenish, reinsert partial storage, clear unrelated
markers, or create typed references. `assume_init` consumes the owner and is
the only unsafe transition; one `Vec::from_raw_parts` preserves length and
capacity. Constructor failure or panic leaves the owner armed and Drop runs
exactly once; successful handoff disarms it.

## RED / GREEN

RED: added the lifecycle contract test
`pooled_uninit_guard_discards_partial_reused_storage_without_replacement`.
The focused test failed because the canonical owner, tracked token, and exact
discard inspection surface were absent.

GREEN: the focused lifecycle test passes. The guard now uses a non-Copy
`UninitCheckoutToken`, preserves the original vector allocation through
`ManuallyDrop`/`as_mut_ptr`, and keeps the token armed through typed-tensor
construction. Error and unwind cleanup pass an empty vector when ownership has
already moved, so the precise marker is still decremented exactly once.

## Upstream mapping

The owner is in `tenferro-internal-cpu-kernels`, adjacent to the pool because
the pool owns typed retention and in-flight accounting. Its pre-completion
surface is limited to `MaybeUninit` slices and byte/erased handoff helpers.

## Rejected alternatives

Unconditional zero-fill is rejected because it changes the full-overwrite
performance contract. Pool-wide `clear_in_flight_retained` and replacement
replenishment are rejected because they lose exact ownership. A second guard
in `tenferro-cpu` is rejected because the contract requires one cross-crate
owner. `ExecContext::ambient`, temporary pools, and scalar replacement loops
are out of scope and prohibited.

## Unsafe delta, performance, and final verification

Sol's high audit recorded the unsafe-count reduction as `671 -> 666` overall,
`71 -> 69` for internal kernels, and `103 -> 100` for CPU. The specialized
same-shape multiply paths retain pinned `mul_into_uninit`; scalar/broadcast
routing, SIMD/planner selection, wrapping integer dispatch, explicit contexts,
and one-time fused input-pointer conversion remain intact.

Final verification: internal unit tests **53 passed**, internal doctests **26
passed**, CPU library tests **506 passed**, nightly Miri guard-filter tests
**7 passed, 46 filtered**, and `git diff --check` passed. The source contract
checks every production source file in the internal crate, with the pool
definition's legacy public acquire path explicitly allowlisted. Internal
clippy with `-D warnings` passed. The pre-amend candidate `060527f`, based on
`origin/main` at `185d1256`, passed
`scripts/check-pr-fast.sh --coverage-reviewed --test 'cargo test
-p tenferro-internal-cpu-kernels
internal_full_overwrite_sources_use_the_guard_boundary'`. Its committed-head
repository-rules review also passed with no findings.

Fresh checkout tests assert the allocator-returned `actual_capacity` but do
not synthesize a C>len seam; deterministic C>len success, error, panic, and
invalid-byte cases are covered for Reused storage.

Earlier committed-head review findings on `4ab58773` and `f0bf6803` covered
scanner-sensitive receipt naming, concrete error documentation, and missing
adjacent SAFETY proofs; all were remediated. The reported `typed_map_with_pool`
hidden-public finding was verified false against `origin/main` and left
unchanged.

The valid committed-head review on `f0bf6803` then blocked every new internal
`out.assume_init` handoff lacking an adjacent proof. Each production handoff
now states the successful map, zip, compare, select, clamp, multiplication,
broadcast, outer-product, fused, fill, or materialization replay proof, and a
source-contract test rejects future omissions within the preceding two lines.


## #1516B deferrals

The remaining `tenferro-cpu` typed pooled callers, including strided dot/einsum,
structural and analytic helpers, are deliberately deferred. The legacy typed
`PoolScalar::pool_acquire` and typed-uninitialized helper remain until #1516B.
Only the duplicate indexing guard is canonicalized in this PR; unrelated
tenferro-cpu/linalg callers are not migrated.

## Ownership extraction follow-up

The guard owner and `checked_compact_strides` now live in the private
`pooled_uninit_output` module. The root crate keeps only the public re-export;
`UninitCheckoutToken` and all pool accounting remain owned by `buffer_pool`.
Guard-only coverage for `pooled_uninit_output.rs` is 94/96 lines (97.9%) after
the state simplification. `data` is always owned by the guard and the optional
checkout token is the sole completion/Drop state.
The root `lib.rs` remains a mixed-responsibility file and is not used as the
guard coverage target.

The public re-export is intentional: `tenferro-cpu` is a workspace sibling
and imports this canonical owner directly. Narrowing it to `pub(crate)` would
break the cross-crate ownership contract and invite a duplicate guard. The
pre-completion API exposes only `MaybeUninit`; typed ownership is transferred
only by the explicit unsafe completion methods.
