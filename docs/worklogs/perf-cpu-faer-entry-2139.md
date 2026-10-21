# The downstream faer entry is the caller's pool (#2139 code residual 1)

## Decisions

- Removed `FaerParallelismExt` (and the `CpuExecutionContext::with_faer_pool`
  helper its engine method called). The capability existed to hand a `faer::Par`
  to downstream code without exposing the Rayon pool (#1687, #1939). That
  premise no longer holds: since the caller-held engine landed (#2206),
  `CpuEngine::pool` already returns `&rayon::ThreadPool`, and `with_faer_pool`
  installed the pool whenever the engine had more than one worker, whatever the
  operation's size. That unconditional entry is the residual #2139 names.
- Downstream direct faer work now enters the pool the caller already owns:

  ```text
  let width = engine.num_threads();
  engine.pool().expect("pooled engine").install(|| {
      faer_operation(..., Par::rayon(width))
  });
  ```

  The caller chooses the width from its own work model, up to the engine's
  budget, exactly as an internal faer-backed provider chooses it before calling
  `CpuExecutionContext::with_faer_width`. A serial engine has no pool and takes
  `Par::Seq`; a width-one call on a pooled engine stays on the calling thread
  and enters no pool.
- The `tenferro_cpu::faer` re-export stays, but only as a build-alignment path.
  No faer value crosses a tenferro boundary now, so a separately pinned `faer`
  cannot surface as a type error at that boundary (#1939); using the re-export
  avoids a second faer build in one binary.
- The provider-side contract is unchanged: `CpuExecutionContext::with_faer_width`
  remains the width-gated entry for in-tree providers, and the `#[doc(hidden)]`
  `faer_parallelism` policy accessor remains for operation-family crates.

## Verification conclusions and constraints

- `crates/tenferro-cpu/tests/direct_faer_on_engine_pool.rs` runs a faer matmul
  on `engine.pool()` at two workers, asserts the pool width, and checks the
  result; a serial engine is asserted to have no pool. It replaces the former
  `faer_reexport_identity` test, whose callback contract no longer exists.
- `docs/tutorial-code/src/bin/faer_interop.rs` exercises the pool route and
  still passes (`faer_interop: all checks passed`).
- Removing the trait is a public API break. Nothing in this workspace used it
  outside the guides, the tutorial and its own tests.
- No speedup is claimed. The point is that a small sequential direct faer call
  is no longer forced into the engine's pool, not that a measured call got
  faster.
