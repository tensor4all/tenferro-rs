# Held CPU session: public CPU fast path (#1945 U2-concrete)

**Status:** work package U2-concrete of
[tenferro-rs #1945](https://github.com/tensor4all/tenferro-rs/issues/1945), delivered with
the public held CPU session. Builds directly on
[`held-cpu-session-1945-u1.md`](./held-cpu-session-1945-u1.md), which owns the
ownership/drop/lock proof and the internal prototype.

Delivered here: the public entry and session surface, the doctested usage contracts, the
paired dispatch/allocation measurements, and the access-compatibility and
execution-routing checks. **Not** delivered here: the CPU phase/pool lease, and the
concrete CPU routes a *downstream* consumer builds on top (tensor4all-rs #859 B1).

## 1. Public surface

```text
impl CpuBackend {
    pub fn open_session(&self) -> Result<CpuHeldSession<'_>, SessionEntryError>;
}
impl CpuHeldSession<'_> {          // !Send + !Sync
    pub fn with_session<R>(&mut self, f: impl FnOnce(&mut dyn BackendSession) -> R) -> R;
    pub fn close(self) -> Result<(), CpuHeldSessionError>;
}
```

This is deliberately the *whole* API. #1945 states that concrete `Tensor`/`TensorRead`
plus `BackendSession` already support einsum and linalg and that "these are the non-AD
substrate, not a new tensor implementation", so the held session hands out the same
`BackendSession` the scoped entry does instead of growing a parallel operation surface:

- primitives, indexing, structural, reductions: the `BackendSession` methods;
- einsum, linalg and extension operations: the borrowed-session extension traits
  (`TensorSessionOpsExt`, `TypedTensorSessionOpsExt`) called with that session;
- borrowed reads and caller-provided output buffers: `TensorRead` / `TensorWrite`
  parameters of those same methods.

No method on the held session duplicates an operation, and the scoped and held entries
build their view through the same `with_operation_session` helper (U1 design §2), so the
two routes cannot drift.

Contract points, each covered by a runnable doctest in the rustdoc:

| Point | Behaviour |
| --- | --- |
| Lifetime | The session owns admission, the checked-out engine resources, the execution-owner marker and the caller's narrowed CPU mask. |
| Auto-traits | `!Send + !Sync`; the caller's CPU mask cannot be restored from another thread. |
| Reuse | One session entry serves any number of `with_session` calls; every operation gets a short-lived view, so no lock or cache borrow outlives the callback. |
| Reentrancy | Only one root session per engine. Nested `open_session`, any scoped entry inside a held session, a held root inside `with_execution_scope` and child handles are rejected with a typed `SessionEntryError::Reentered` before they can block. |
| Children | A child handle from `CpuExecSession::child_execution()` runs on a worker of an enclosing pool with inherited admission and its own private resources; it never opens a held root. |
| Management | While the resources are checked out, `buffer_pool_len`, `buffer_pool_stats` and `indexed_plan_cache_stats` report a typed `RuntimeState` error instead of waiting on a lock the session owns. |
| Cleanup | `close` reports the one fallible step (affinity restoration); dropping the session releases the same state best-effort. |

## 2. Paired dispatch and allocation measurements

Protocol: `crates/tenferro-cpu/benches/held_session_dispatch.rs`, Criterion, arms `scoped`
(`with_backend_session` per operation), `held` (one `open_session` outside the timed
region, one `with_session` per operation) and `held/chain16` (16 operations per timed
iteration, so divide by 16 for the marginal cost). Every arm validates its result outside
the timed region. `single` uses the allocation-returning `dot_general_read`; `into` uses
`dot_general_read_into` with a caller-owned output buffer that is reused across
iterations, so it measures dispatch without result allocation or zero-fill.

Environment: AMD EPYC 7713P (64 cores), Linux 6.8.0, rustc 1.97.1, tenferro-cpu 0.7.1 with
the default `native` (faer 0.24.4) provider, **one worker** (`CpuBackend::with_threads(1)`),
pinned with `taskset -c 0`, release/bench profile, 100 samples, 2 s warm-up and 5 s
measurement per case. Medians with Criterion's confidence interval (n = 100):

| case | median |
| --- | ---: |
| entry `scoped/empty` | 537.7 ns [530.8, 545.6] |
| entry `held/open_close` | 321.1 ns [315.6, 326.8] |
| dot 2 `scoped/single` | 5.88 µs |
| dot 2 `held/single` | 5.35 µs |
| dot 2 `scoped/into` | 5.44 µs |
| dot 2 `held/into` | 4.59 µs |
| dot 2 `held/chain16` | 87.96 µs (5.50 µs/op) |
| dot 95 `scoped/single` | 13.52 µs |
| dot 95 `held/single` | 12.70 µs |
| dot 95 `scoped/into` | 8.63 µs |
| dot 95 `held/into` | 6.74 µs |
| dot 95 `held/chain16` | 204.21 µs (12.76 µs/op) |
| dot 256 `scoped/single` | 54.34 µs |
| dot 256 `held/single` | 51.99 µs |
| dot 256 `scoped/into` | 17.43 µs |
| dot 256 `held/into` | 17.08 µs |
| dot 256 `held/chain16` | 832.32 µs (52.02 µs/op) |

Shapes are `size x size` times `size x 1` f64 contractions; `held/chain16` per-operation
figures track `held/single`, i.e. the session stays amortized and no per-operation entry
reappears.

What the numbers support:

- **Holding a session removes the per-operation entry.** One entry costs 537.7 ns scoped
  and 321.1 ns held; per operation the held path is 0.53 µs (size 2), 0.82 µs (95) and
  2.35 µs (256) faster than the scoped path for the allocating form, and 0.85 µs (2),
  1.89 µs (95) and 0.35 µs (256) faster for the output-buffer form. The scoped path itself
  is unchanged in structure and no scoped arm got slower than the recorded baseline of
  this harness (this is a new harness, so there is no earlier row to compare).
- **The output-buffer route is materially cheaper than the allocating one** (6.74 µs vs
  12.70 µs at size 95), which is the concrete meaning of #1945's "preserve/use
  caller-provided output" clause.

What the numbers do **not** support, reported as a miss:

- **The "#1945" small-GEMM fixed-cost target below 1 µs is not met on this machine.** The
  cheapest measured case, `held/into` at a 2x2 contraction, is 4.59 µs. Only 0.32 us of one
  entry and ~0.5-0.9 us of per-operation entry are attributable to session entry; the
  remaining ~4 µs is the in-session dispatch path (input canonicalization, plan lookup,
  extension dispatch, kernel setup) that exists identically on the scoped route. That
  cost is owned by the existing CPU per-operation overhead work (#1904 and the
  #2033-#2038 family), not by the held session, and reducing it is not part of this work.
  No claim of sub-microsecond fixed cost is made.
- Untracked eager and AD rows are not measured here: they live in other crates.

## 3. The "no eager records, no eager owner lock" invariant

The held route cannot create an `EagerTensor`, a trace node, a semantic node or a gradient
slot, and cannot take an eager owner lock, because `tenferro-cpu` has **no dependency on
`tenferro-ad`** (see `crates/tenferro-cpu/Cargo.toml`, dev-dependencies included) and the
dependency direction is enforced by the repository's crate-boundary checks. This is a
dependency-boundary argument rather than a counter, and it is recorded as such.

The part that *is* observable at this layer is measured instead:

- `an_unrelated_non_waiting_owner_is_rejected_while_the_session_is_held` — a Rayon worker
  opening a session while the held session owns the reservation gets a typed
  `Contended`, i.e. it neither waits on nor is confused with the held owner;
- `child_work_succeeds_while_the_parent_holds_the_root_reservation` — a child handle
  entered on an enclosing pool's worker is admitted under the parent's owner and runs
  concurrently with the parent's execution, then the parent's resources remain reusable;
- `a_queued_entry_succeeds_after_the_session_closes` — an ordinary scoped entry queued
  behind the session is admitted as soon as it closes, never observing a vacant slot;
- `host_values_cross_cpu_budget_contexts_without_a_conversion_copy` — a value produced
  under a one-worker context is read under a four-worker held session with its storage
  pointer unchanged, i.e. access compatibility is validated without an ownership
  conversion, and the producing backend need not run at all.

## 4. Remaining work

| Item | Owner |
| --- | --- |
| CPU phase/pool lease (park root execution, drive joined worker children under the owner's budget, cancellation and join) | U2-concrete, next step |
| Instrumented eager/AD counters on the concrete path | upstream of the downstream B1 gate; the crate boundary above is the current proof |
| Sub-microsecond small-GEMM fixed cost | #1904 / #2033-#2038 (in-session dispatch path) |
| Downstream explicit/concrete tensorbackend frontend and its bridges | tensor4all-rs #859 B1 |
| Held eager adapter (U2-AD), CUDA lifecycle (U3), asynchronous transfer (U4), adoption gates (U5) | later packages |
