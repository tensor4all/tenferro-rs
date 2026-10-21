# Dispatch and Engine Cost

There is no session entry to amortize any more. An engine is a value the caller
holds, and every operation borrows it through `&mut`: no admission permit, no
entry protocol, no nested-entry rejection. What remains per operation is
argument validation, a plan-cache lookup, the output buffer (pooled when the
engine has one to reuse) and the provider dispatch itself.

This page separates the costs that still exist and shows how to avoid paying
them more often than necessary. It replaces the older "CPU session entry and
Rayon dispatch cost" guide: the mechanisms described there (held sessions, phase
leases, child execution tickets) are gone with the caller-held engine (#2206).

Related records:

- [Execution engine ownership](../design/execution-engine-ownership.md): the
  engine/prepare/run model, pool ownership, and the removed mechanisms.
- [Ordinary and prepared execution](ordinary-and-prepared-execution.md):
  `Runtime::prepare` and `Runtime::run` with caller-supplied engines.
- [CPU backend execution](../design/cpu-backend-execution.md): provider routes
  and the engine's parallelism token.

## What one call costs

| Component | When it is paid | How to avoid repeating it |
| --- | --- | --- |
| Argument validation and shape/dtype checks | Every call | Nothing to remove; it is the error contract |
| Plan-cache lookup | Every call that has a plan | Share one `CpuPlanCache` across engines with `CpuEngine::with_plan_cache` |
| Output allocation | Every call that returns a tensor | Use the `_into` variants with a caller-owned, reused output buffer |
| Buffer-pool recycle | Every call that returns a tensor, when the pool has a spare buffer | Reuse one engine so the pool stays warm |
| Provider dispatch | Every call | Nothing to remove; a small operation stays on the calling thread |
| Rayon pool entry | Only when a provider chose more than one worker | A `CpuEngine::serial` engine never enters a pool |

The last row is the important one: a small operation on a pooled `CpuEngine`
runs on the calling thread and never touches the pool. The pool entry that used
to dominate microbenchmarks is now conditional on the provider actually
splitting work.

## What one prepared command costs

A prepared program replays the same table for each command it holds, plus one
fixed cost per run. The per-run part is the input-signature check, output
collection, extension-cache lending, and the event-domain drain; the
per-command part is submission to the event domain, the input and output slot
handling, and the operation itself.

Two shapes avoid most of the per-command handling:

- **One location.** When every part of the schedule lives at the same engine,
  storage class, and event domain and no transfer exists — the ordinary CPU
  program — each slot's value stays in the command array. Nothing is staged or
  republished per command.
- **Coalesced submission.** Such a schedule is also offered single-domain
  coalescing. A driver that takes the offer reports it, and the scheduler then
  submits each command with no dependency resolution and no completion token.
  The CPU immediate driver takes it; an asynchronous driver keeps the per-node
  completion contract unless it implements the same guarantee.

A run presenting the input signature the handle was prepared for skips the
specialization projection.

What is left per command is the operation itself plus about one microsecond of
executor work on this repository's measurement host (see
[#2186](https://github.com/tensor4all/tenferro-rs/issues/2186)). A caller that
needs less than that should not be replaying a general program: a direct engine
call, a batched driver, or a fused region is the cheaper shape.

## Amortizing a stage

1. **Hold one engine for the stage.** The engine owns the plan cache, the
   buffer pool and the extension caches, so a fit loop, a sweep or a batched
   driver should construct it once and pass `&mut engine` to every call. See
   the [typed non-AD tutorial](../tutorials/typed-tensor-non-ad.md).
2. **Reuse the output buffer.** The `_into` operation forms write into caller
   storage and allocate nothing for it; that is the caller-output route the
   allocation-returning form cannot offer.
3. **Prepare a traced program once.** `Runtime::prepare` resolves engines,
   plans and buffers for a program; `Runtime::run` replays it. Compile and
   prepare outside the loop, not per iteration
   ([ordinary and prepared execution](ordinary-and-prepared-execution.md)).
4. **Choose a serial engine for unavoidable tiny calls.** `CpuEngine::serial()`
   cannot enter a pool at all, so it cannot oversubscribe an outer pool either.
   It does not make validation, allocation or the numerical kernel free.
5. **A downstream non-AD kernel may use tensor/view storage directly.** Follow
   [external linalg interop](external-linalg-interop.md) for layout, borrowing
   and provider obligations. Direct faer calls can remove tenferro dispatch, but
   they do not remove a Rayon handoff if they still enter a pool.

## Reproduce the Rayon measurements

The standalone probes that measure a Rayon pool entry from outside and from
inside a pool are still useful when your own code, not tenferro, enters a pool.
They are available as
[install source](../performance/rayon-install-probe/install.rs.txt),
[scope source](../performance/rayon-install-probe/scope.rs.txt),
[manifest](../performance/rayon-install-probe/Cargo.toml.txt),
[lockfile](../performance/rayon-install-probe/Cargo.lock.txt), and raw
[install](../performance/rayon-install-probe/install-results.txt) /
[scope](../performance/rayon-install-probe/scope-results.txt) results. They are
standalone probes, not tenferro API examples or CI latency thresholds. From
the repository root:

```bash
probe_dir=$(mktemp -d)
mkdir -p "$probe_dir/src"
cp docs/performance/rayon-install-probe/Cargo.toml.txt "$probe_dir/Cargo.toml"
cp docs/performance/rayon-install-probe/Cargo.lock.txt "$probe_dir/Cargo.lock"
cp docs/performance/rayon-install-probe/install.rs.txt "$probe_dir/src/main.rs"
cargo run --release --locked --manifest-path "$probe_dir/Cargo.toml"
cp docs/performance/rayon-install-probe/scope.rs.txt "$probe_dir/src/main.rs"
cargo run --release --locked --manifest-path "$probe_dir/Cargo.toml"
```

Record compiler, OS/CPU, worker count, revisions, timing scope and load when
comparing results. Measure an empty pool entry, kernel work inside an already
entered pool, and end-to-end tenferro calls separately when locating a
regression; they are three different costs.

Do not infer tenferro's dispatch cost from a benchmark that also constructs a
pool, and do not generalize a Rayon `install` measurement into a fixed per-call
tenferro constant.

The historical measurements taken on the removed session-entry model are
preserved in [CPU session open cost](../design/cpu-session-open-cost.md); they
describe code that no longer exists and are not comparable with current
results.
