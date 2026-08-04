# Phase 5 common scheduled-graph boundary

This worklog records the Phase 5 migration checkpoint for runtime-owned
compiled graph execution. It follows the Phase 4 preparation substrate recorded
in [`2026-07-24-phase-4-runtime-preparation.md`](2026-07-24-phase-4-runtime-preparation.md).

## Implemented boundary

- `CompiledGraph` is again a backend-neutral artifact: frozen semantic program
  plus the `CompilerOptions` used to compile it. It does not retain
  `ExecProgram`, compiler-owned staging, direct engine bindings, prepared
  operations, or runtime state.
- `Runtime::run_compiled` and `Runtime::run_compiled_values` are the current
  synchronous runtime-owned execution path. They derive input signatures,
  prepare through the runtime cache, validate specialization projection and
  shape guards, build a crate-private `ScheduledGraph`, and execute through a
  registered tensor backend bridge.
- `tenferro-cpu::runtime_engine_registration` is the public CPU helper that
  assembles direct core preparation capabilities, CPU cache-owner hooks, the
  executable witness, and the event-domain driver through the common runtime
  assembly path.
- `ScheduledGraph` is a crate-private executable boundary that can represent
  core operations, transfers, collectives, and barriers. Transfers use distinct
  source and destination event domains. Collectives are representable but remain
  unsupported by current runtime execution validation.
- The former executor-shaped compatibility path was retired at this
  checkpoint. `Runtime::prepare_compiled` and `Runtime::run_prepared` are the
  current steady-state execution path over `CompiledGraph`.

## Commit sequence

- `19739189` — `docs(phase5): plan common scheduled graph boundary`
- `845afcd6` — `docs(phase5): preserve compiler options across staging boundary`
- `45270516` — `refactor(runtime): move staging behind execution boundary`
- `0b67ad24` — `feat(runtime): add tensor backend execution bridge`
- `9df966c8` — `feat(runtime): execute compiled graphs through runtime`
- `e047f3c6` — `test(runtime): cover runtime execution bridge edge cases`
- `6df5c45d` — `feat(runtime): add scheduled graph boundary`
- closeout commit — expose the CPU runtime registration helper and update docs

## Focused verification evidence

The Phase 5 checkpoint has been verified incrementally with focused runtime,
CPU, and eager tests:

```text
cargo test -p tenferro-runtime graph::executor::tests::phase5_source_contracts --lib
cargo test -p tenferro-runtime runtime::tests::preparation::phase5 --lib
cargo test -p tenferro-runtime graph::compiler --lib
cargo test -p tenferro-runtime graph::executor --lib
cargo test -p tenferro-runtime runtime::tests::snapshot::engine_registration_records_tensor_backend_execution_bridge --lib
cargo test -p tenferro-runtime runtime::tests::snapshot --lib
cargo test -p tenferro-runtime --doc EngineRegistration
cargo test -p tenferro-runtime --test integration runtime_execution
cargo test -p tenferro-runtime runtime::tests::preparation --lib
cargo test -p tenferro-runtime runtime::tests::schedule --lib
cargo test -p tenferro-runtime
cargo test -p tenferro-cpu runtime_adapter --lib
cargo test -p tenferro-ad eager::tests::runtime_snapshot --lib
cargo test -p tenferro-ad --test integration runtime_snapshot_bridge
cargo fmt --all --check
```

The final closeout gate was also run with:

```text
python3 scripts/check-doc-snippets.py
python3 scripts/test-doc-consistency.py
python3 scripts/check-guide-dependency-snippets.py
python3 scripts/check-docs-site.py
python3 scripts/check-public-error-docs.py
cargo fmt --all --check
git diff --check
cargo test -p tenferro-runtime
cargo test -p tenferro-cpu
cargo test -p tenferro-ad
scripts/check-pr-fast.sh --no-fetch --coverage-reviewed --test 'cargo test -p tenferro-runtime' --test 'cargo test -p tenferro-cpu' --test 'cargo test -p tenferro-ad'
```

The deterministic repository-rules worktree review also passed, with the
external LLM call intentionally skipped until the PR gate:

- `/tmp/repository-rules-review-phase5-worktree-dry-run.json`

## Later-phase ownership

- Phase 6 owns operation-family derived caches and CPU native einsum execution.
- Phase 7 owns CUDA/WebGPU native engine work.
- Phase 8 owns XLA/subgraph integration through `SubgraphCompiler`.
- Umbrella Phase 9 remains the later multi-GPU scheduling phase.
