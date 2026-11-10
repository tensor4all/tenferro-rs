# #2308 items 1 and 2: the duplicated CPU/linalg paths left after the threadteam migration

## Decisions

- **`Placement::Single(Pool)` is gone; a one-worker pool lands in `Placement::Serial`.**
  The issue asked to confirm first that a width-one `threadteam::Pool` holds no
  worker threads. It does hold one (`threadteam` 0.5.1 builds its pool through
  rayon's `ThreadPoolBuilder`, which spawns `n` workers, so `threads(1)` spawns
  one), and nothing that engine minted could dispatch: a budget-one token reports
  `is_serial`, `dispatch_cost` is `Duration::MAX` below width two, and
  `external`/`team`/`fork` all take their inline branch at budget one. The merge
  therefore releases the engine's ownership of an idle pool and changes no numerical
  behaviour: the worker itself may still be parked in threadteam's shared-worker
  cache. Dispatch costs are still read from the pool before the placement match, so
  the pre-entry gates keep the caller's pool numbers.
- **The owned/view twins collapsed view-into-owned, not owned-into-view.** The
  issue proposed `lu(x) = lu_view(x.as_view())`. That would give the owned route
  the view route's exact-rank-two check, and the owned route must accept a batched
  operand (rank 2 + trailing batch axes): `lu`/`cholesky`/`full_piv_lu`/`svd`/`qr`/
  `eigh`/`eig`/`rank_revealing_qr` all validate rank >= 2 in their shared impl, and
  the batched allocation test drives `[4, 4, 64]` through the owned entries. The
  thirteen crate-private `*_view` entries were reached only from the read hooks on
  `CpuEngine` (eleven through a `*_faer_view_entered` adapter, the two eig entries
  directly), and every one of those hooks is guarded by `faer_strided_host_ok`
  (rank two, non-negative strides) or `faer_rhs_host_ok` (rank two), so the
  exact-rank-two checks inside the deleted functions could never fire. Deleting
  them removes the duplicate function per family and leaves every reachable
  validation, error kind and error order as it was.
- **No `*_view` sibling is reintroduced for the read route.** The read adapters now
  hand the host view to the owned entry, which widens it with
  `into_dynamic`/`clone`. That copies view metadata (buffer ref, layout, placement),
  never the payload, and allocates nothing for a default placement.

## Verification conclusions and constraints

- The one behaviour that is not observationally identical is which engine holds a
  one-worker pool: it is released at construction instead of for the engine's
  lifetime, so such an engine now reports no ledger roots and the pool's waiting
  policy is no longer reachable through its token. `num_threads()` and
  `has_team()` are unchanged.
- The rank-two view checks that were deleted are dead in production but were the
  only guard against a future caller handing a `*_view` entry a higher-rank view.
  A new caller must now go through the owned entry like every other one.
- No recorded allocation ceiling covers the linalg read hooks, so the metadata copy
  added to that route is not machine-checked; the payload-copy contract the read
  hooks do have (`faer_eig_view_path_does_not_pool_an_input_copy`, and the borrowed
  ceilings of the public `*_with_exec` entries) is unaffected.
- Items 3 and later of the issue — the `FaerLinalg`/`LapackLinalg` host layer, the
  public batch-policy re-exports, the `cpu-faer` alias and CI labels, and
  `scripts/check-no-public-rayon.sh` — are untouched and stay open.
