# Handoff: tenferro execution-path unification, Unification 6 WIP

Date: 2026-07-25 JST

Repository worktree:

`/home/shinaoka/tensor4all/tenferro-rs/.worktrees/execution-engine-phase9-restart`

Branch:

`codex/execution-engine-phase9-restart`

This is a checkpoint handoff, not a green completion handoff. The branch is intentionally committed with one newly-added RED regression test documenting the current Unification 6 gap.

## User goal and process constraints

Active goal:

> Unification 8まで順次進めて

Current user instructions:

- Implement locally through Phase/Unification 8 without opening a PR.
- After Unification 8 is implemented and verified, open PR and babysit merge including CI fixes.
- Do not preserve compatibility for old execution paths in this PR.
- Remove unnecessary old paths and compatibility surfaces instead of keeping shims.
- Do not spend time repeating benchmarks while the code is still structurally dirty; review and fix critical paths first.

Process constraints followed in this checkpoint:

- No subagents were spawned.
- CodeGraph was used first for code exploration where available.
- TDD was used for the new multi-input eager semantic VJP regression:
  - RED was observed.
  - The test is still RED at this handoff.
- This checkpoint is a WIP commit by explicit user request.

## Current stage

Current stage: Unification 6 / issue #1460.

Issue #1460 latest relevant scope checked on 2026-07-25:

1. S0: binding-tolerant `jvp_program` / `vjp_program` transform cache.
2. S1: symbolic-shape eager recording.
3. Eager recorded VJP moves toward `record -> TracedGraph -> SemanticProgram -> AD transform -> Runtime prepare`.
4. `tidu` remains intact and default through this issue; it is not the fallback design for later failure.

## What is already in the working tree

Large pre-existing dirty implementation from earlier phases is included in this checkpoint:

- Legacy `GraphExecutor` public path has been removed from the runtime surface.
- `ExtensionRuntime` / `ExtensionExecutor` compatibility surfaces are being removed.
- `HostReference` / `host_reference()` execution hook has been removed from core extension APIs.
- sparse and tropical reference execution is module-private rather than a public extension-op hook.
- `extension_runtime.rs` was renamed to `extension_execution_context.rs`.
- `Runtime::run_compiled` is the intended execution entry.
- Bound-program semantic AD transform cache already exists and is covered.

Two worklogs were present as untracked files before this handoff and should be kept with the checkpoint:

- `docs/worklogs/2026-07-25-unification-1-run-compiled-dead-weight.md`
- `docs/worklogs/2026-07-25-unification-4-extension-surface-cleanup.md`

## Fresh verification before this handoff

Passing checks run immediately before the new RED work:

```text
cargo test -p tenferro-ad semantic_program_transform_cache_reuses_bound_programs_without_stale_bindings --test integration -- --nocapture
result: pass, 1 passed

cargo test -p tenferro-ad eager_recording_retains_symbolic_semantic_trace_for_shape_churn --lib -- --nocapture
result: pass, 1 passed
```

Formatting before the checkpoint:

```text
cargo fmt --all
cargo fmt --manifest-path ext/sparse/Cargo.toml --all
cargo fmt --manifest-path ext/tropical/Cargo.toml --all
result: exit 0
```

Current RED regression:

```text
cargo test -p tenferro-ad eager_runtime_vjp_uses_semantic_trace_for_multi_input_graph_when_gate_enabled --lib -- --nocapture
result: fail
```

Observed failure:

```text
RuntimeStateSource {
  op: "semantic_eager_vjp",
  phase: GraphBuild,
  source: Build(Arity { expected: 2, actual: 1 })
}
```

The failure is expected for this checkpoint: the new test documents that gated semantic eager VJP does not yet handle a multi-input graph such as `x * y`.

## WIP code added in this checkpoint

New RED test:

- `crates/tenferro-ad/src/eager/tests.rs`
  - `eager_runtime_vjp_uses_semantic_trace_for_multi_input_graph_when_gate_enabled`
  - Builds `x * y`, requests `ctx.vjp(&output, &x, &seed)` and `ctx.vjp(&output, &y, &seed)`.
  - Expects correct values and exactly two semantic eager VJP executions.
  - Currently fails before semantic eager VJP execution count increments.

Runtime input-key plumbing started:

- `crates/tenferro-runtime/src/graph/program.rs`
  - `CompiledGraph` now stores ordered trace `input_keys`.
  - Added hidden accessors:
    - `input_keys(&self) -> &[TensorInputKey]`
    - `input_key_index(&self, key: &TensorInputKey) -> Option<usize>`

- `crates/tenferro-runtime/src/graph/compiler.rs`
  - `compile_many_with_descriptors` now carries ordered `TensorInputKey`s in parallel with descriptors.
  - Explicit input ordering reorders descriptors and input keys together.
  - `compile_frozen` currently constructs `CompiledGraph` with an empty input-key list because frozen semantic programs do not retain trace input keys.

Semantic eager VJP WIP:

- `crates/tenferro-ad/src/eager.rs`
  - `semantic_eager_vjp_optional` no longer hard-requires `source.input_count() == 1`.
  - It tries to map `wrt_trace.input_key()` to the compiled source input index.
  - It populates all bound primal inputs into the derivative program and adds the cotangent seed.

This WIP still fails. Do not treat it as a completed fix.

## What is bad / current root issue

The current failure shows that multi-input semantic eager VJP is not yet a valid execution path.

Confirmed facts:

- Single-input gated semantic eager VJP still passes:

  ```text
  cargo test -p tenferro-ad eager_runtime_vjp_can_use_semantic_trace_when_gate_enabled --lib -- --nocapture
  result: pass
  ```

- Multi-input gated semantic eager VJP fails during semantic VJP graph build before execution.
- The thrown source is `ProgramBuildError::Arity { expected: 2, actual: 1 }`.
- The failing path is `semantic_eager_vjp_optional -> AdContext::vjp_program -> semantic_vjp`.

Likely next diagnostic, in order:

1. Inspect the compiled `source` inside `semantic_eager_vjp_optional` for the RED test:
   - `source.input_count()`
   - `source.input_keys()`
   - `source.bindings().len()`
   - source semantic operations and each operation's input arity
2. If the source program for `x * y` has only one semantic input or a one-input `Mul`, fix semantic eager recording:
   - start at `record_semantic_eager_outputs`
   - confirm `tenferro_runtime::extension::apply_standard_op` receives both semantic input traces for binary eager ops
3. If the source program is correct, instrument or source-contract `SemanticProgramBuilder::add_op` / `vjp_core` enough to identify which VJP rule emits a one-input `CoreSemanticOp::Mul`.
4. Avoid papering over this by returning to tidu; tidu remains default/oracle for #1460, but the gated semantic path must become valid.

Design warning:

Do not conflate:

- which input cotangents are requested by the caller, and
- which primal inputs are needed as coefficients/residuals to compute those cotangents.

For `x * y`, computing `d/dx` needs `y` as a primal coefficient even if `dy` is not requested. If the fix changes semantic VJP activity semantics, keep this distinction explicit and add tests.

## Goal path from here to Unification 8

### Step 1: Finish #1460 S0/S1 evidence

Already mostly done:

- S0 bound-program transform cache is implemented.
- S0 behavior test passes:
  - `semantic_program_transform_cache_reuses_bound_programs_without_stale_bindings`
- S1 symbolic-shape eager recording has a passing test:
  - `eager_recording_retains_symbolic_semantic_trace_for_shape_churn`

Before declaring S0/S1 done, keep a short worklog note that these were verified on the checkpoint branch and name the exact commands.

### Step 2: Repair gated semantic eager VJP for multi-input core graphs

Required immediate next target:

- Make the RED test pass without removing the test's semantic execution-count assertion.

Relevant command:

```bash
cargo test -p tenferro-ad eager_runtime_vjp_uses_semantic_trace_for_multi_input_graph_when_gate_enabled --lib -- --nocapture
```

Add at least one runtime/compiler test for the new `CompiledGraph` input-key order behavior:

- default `compile_many` order for a binary graph
- explicit input order path if this API remains relevant

Then run:

```bash
cargo test -p tenferro-runtime graph_compile --test integration -- --nocapture
cargo test -p tenferro-ad eager_runtime_vjp_can_use_semantic_trace_when_gate_enabled --lib -- --nocapture
cargo test -p tenferro-ad eager_runtime_vjp_uses_semantic_trace_for_multi_input_graph_when_gate_enabled --lib -- --nocapture
```

### Step 3: Complete #1460 semantic eager frontend

After multi-input core graphs pass:

- Audit eager operations for `semantic_trace: None`.
- Keep extension ops out of semantic eager recording until extension module migration is ready, unless the issue explicitly requires a family.
- Confirm failure behavior: unsupported semantic trace should fall back to current tidu default in #1460, not silently claim semantic execution.
- Keep `TENFERRO_EAGER_SEMANTIC_VJP` / test override semantics until the issue says to flip default.

Focused checks:

```bash
cargo test -p tenferro-ad eager_runtime_vjp --lib -- --nocapture
cargo test -p tenferro-ad cache_management --test integration -- --nocapture
cargo test -p tenferro-ad semantic_transform --test integration -- --nocapture
```

### Step 4: Decide whether #1460 S2 is needed now

Do not benchmark repeatedly while the code is structurally unresolved.

Once the semantic eager frontend is valid:

- Review critical path for avoidable freeze/fingerprint/prepare work.
- If warm shape-churn still pays full freeze/semantic transform/prepare for identical structure, implement S2:
  - incremental structure digest at eager record time
  - lookup prepared derivative plan by structure digest
  - run full pipeline once per new structure
- If current cache hit behavior is enough to proceed to Unification 7, document the deferral.

### Step 5: Unification 7

Expected direction from #1460:

- Use tidu as oracle/default while comparing semantic eager path correctness.
- Do not keep tidu as the named fallback design.
- Move toward one semantic rule set with workload-appropriate execution.

Before performance work:

- Run code review over AD/eager/runtime critical paths:
  - hidden materialization
  - cache ownership
  - lock/mutex lifetime
  - old execution path references
  - extension-family registration path

### Step 6: Unification 8

Only after the code path is clean:

- Run the terminal performance gate harness.
- Include amendments already noted by the user:
  - requires_grad true/false split
  - warm/cold/shape-churn/graph-churn split
  - operation-size tiers
  - PyTorch numbers are non-gating references
  - gate remains non-regression against `main`
- For #1460 fallback policy:
  - fallback is not tidu
  - accepted fallback is semantic-rule-fragment direct executor if exact-dim specialization churn cannot be made non-regressive

### Step 7: PR and babysit

Only after Unification 8 implementation and local verification:

1. Re-read `REPOSITORY_RULES.md`.
2. Run focused PR gate proportional to touched surfaces.
3. Commit final cleaned changes in coherent commits if needed.
4. Push branch.
5. Open PR.
6. Babysit CI and fix failures until merge.

Do not open a PR before Unification 8 is implemented and verified.

## Commands worth rerunning immediately after resume

```bash
git status --short --branch

cargo test -p tenferro-ad eager_runtime_vjp_uses_semantic_trace_for_multi_input_graph_when_gate_enabled --lib -- --nocapture

cargo test -p tenferro-ad eager_runtime_vjp_can_use_semantic_trace_when_gate_enabled --lib -- --nocapture

cargo test -p tenferro-ad semantic_program_transform_cache_reuses_bound_programs_without_stale_bindings --test integration -- --nocapture
```

## Notes for the next agent

- This checkpoint intentionally contains failing test evidence. Do not claim the branch is green.
- Do not revert the RED test unless replacing it with a stricter test for the same behavior.
- Treat `CompiledGraph::input_keys` as WIP API. If a cleaner owner-scoped mapping exists, replace this instead of expanding public surface.
- `compile_frozen_program` has no trace input-key provenance; do not assume it can identify original eager leaves.
- The current implementation remains gate-controlled for semantic eager VJP. `tidu` is still default through #1460.
