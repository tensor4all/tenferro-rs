# Post-U8 stabilization worklog

Date: 2026-07-26

This worklog records the single-PR post-U8 stabilization sequence tracked by
#1477. The branch keeps cache cleanup, runtime dispatch cleanup, AD cleanup,
performance triage, and crate-split evidence in one coherent PR rather than
splitting the follow-up work into unrelated pull requests.

## Completed checkpoints so far

- #1476 docs/doc-pin cleanup: removed stale Phase 3/Phase 5 references that
  could send later agents toward the retired `GraphExecutor` or pre-semantic-AD
  contracts.
- #1478 cache lifetime and ownership cleanup: added explicit retained-byte
  bounds to runtime extension caches, preserved cache clear/hit/miss/eviction
  statistics, and documented cache owner and limit surfaces.
- #1456 production dispatch checkpoint: prepared roots now retain a
  crate-private `ScheduledGraph`; `Runtime::run_compiled*` walks that schedule
  synchronously; same-storage semantic operations execute through their
  selected engine bridge.
- #1471 U3 substrate checkpoint: `Runtime::submit` returns an
  `ExecutionHandle`, engine snapshots expose runtime-allocated event domains,
  transfer providers are registered by storage-class pair, and cross-storage
  linear execution calls the transfer provider from the production scheduled
  loop before dispatching the downstream operation.
- #1468 linalg semantic AD cleanup: semantic linalg `linearize` now emits JVP
  fragments directly into `SemanticProgramBuilder` instead of first recording
  and replaying a legacy fragment. The remaining linalg recorded-fragment usage
  is limited to reverse-mode construction paths that need a local linear
  fragment to transpose.
- #1464 CPU elementwise fusion classifier: replaced the hand-written
  add/multiply pattern arms with an explicit two-input, two-operation,
  one-output, identity-view binary-tail classifier. The classifier recognizes
  the benchmark-motivated `Add`/`Multiply` family only, including reversed
  commutative input order for the first op, and still rejects broadcast views,
  repeated inputs, multi-output plans, unsupported ops, and longer chains.
- #1473 tactical performance triage: rechecked the remaining #1426 findings
  against the post-U8 code, closed #1473 with one-line dispositions, and split
  the true positives into focused follow-ups: #1479 strided `dot_general`
  accumulation fallback, #1480 host identity copies, #1481 multi-axis CPU
  reductions, #1482 CPU indexing hot loops, #1483 eager AD accumulation
  allocation, #1484 FFT plan/scratch reuse, and #1485 cuTENSOR
  descriptor/plan/workspace caching.
- #1472 release build/crate split checkpoint: added
  `tenferro-internal-cpu-kernels` and moved CPU `elementwise` kernels plus
  `BufferPool` ownership there while preserving the public `tenferro-cpu`
  crate/API surface. `tenferro-cpu` now imports the internal kernel crate
  through crate-private re-exports, keeps `linalg_interop::BufferPool` as the
  owner-scoped public pool path, and retains source-contract coverage that the
  heavy elementwise/buffer-pool implementation files do not move back into the
  public CPU crate. The remaining fine-grained CPU release-codegen split is
  tracked separately in #1486.

## Current runtime boundary

- `CompiledGraph` remains backend-neutral.
- Runtime preparation owns the semantic-to-prepared binding, cache ownership,
  selected engine bindings, selected operation placements, and schedule
  construction.
- The synchronous scheduled executor tracks each slot's current storage class.
  Cross-storage handoff is supported only when a registered transfer provider
  can materialize the slot for the downstream operation's storage class.
- `ScheduledTransfer`, device-native event-domain bridging, pending-output
  composition, and full admission logic remain later #1471 follow-up scope.
- The extension execution hook is still the public
  hidden `PreparedOperationExecutor` bridge carried by
  `PreparedOperationPlan`. Public `PreparedOperation` is now metadata-only:
  binding, specialization, and retained-byte accounting. Core prepared
  operations use metadata-only plans; extension prepared operations attach an
  executor bridge for the scheduled loop.
- `run_compiled_values` preserves the metadata-only terminal lazy value path
  for layout/view-like outputs before falling back to normal scheduled
  execution.

## Verification evidence

```text
python3 scripts/test-doc-consistency.py
python3 scripts/check-doc-snippets.py --root-dir . --check
cargo test -p tenferro-runtime per_operation_placement_can_mix_same_storage_core_and_extension_engines -- --nocapture
cargo test -p tenferro-runtime --test integration runtime_run_compiled_dispatches_same_storage_extension_on_selected_engine -- --nocapture
cargo test -p tenferro-runtime --test integration runtime_run_compiled_transfers_between_storage_classes_on_scheduled_path -- --nocapture
cargo test -p tenferro-runtime --test integration runtime_run_compiled_reports_missing_transfer_provider_for_cross_storage -- --nocapture
cargo test -p tenferro-runtime --test integration runtime_submit_wait_uses_prepared_execution_path -- --nocapture
cargo test -p tenferro-runtime prepared_program_is_binding_free_and_shares_staged_root -- --nocapture
cargo test -p tenferro-runtime preparation -- --nocapture
cargo test -p tenferro-runtime --test integration runtime_execution -- --nocapture
cargo test -p tenferro-ad --test integration runtime_execution -- --nocapture
cargo test -p tenferro-linalg --test integration --features autodiff linalg_internal_path_contract::semantic_linalg_linearize_does_not_replay_recorded_legacy_fragments -- --nocapture
cargo test -p tenferro-linalg --test integration --features autodiff ad_support_manifest:: -- --nocapture
cargo test -p tenferro-linalg --test integration --features autodiff traced_ad_explicit:: -- --nocapture
cargo test -p tenferro-linalg --test integration --features autodiff oracle_replay:: -- --nocapture
RUN_ORACLE_REPLAY=1 ORACLE_REPLAY_JOBS=64 cargo test -p tenferro-linalg --test integration --features autodiff oracle_replay::oracle_replays_supported_db_cases_when_requested -- --nocapture
cargo test -p tenferro-cpu two_input_binary_tail_classifier --lib -- --nocapture
cargo test -p tenferro-cpu binary_tail_specialization --lib -- --nocapture
cargo test -p tenferro-cpu elementwise_fusion --lib -- --nocapture
cargo test -p tenferro-internal-cpu-kernels --lib
cargo test -p tenferro-internal-cpu-kernels --doc
cargo test -p tenferro-cpu --lib
cargo test -p tenferro-cpu --test integration backend_capability_contracts -- --nocapture
cargo test -p tenferro-cpu --test provider_boundary_allocation_tests warmed_public_session_request_provider_dispatch_does_not_allocate -- --nocapture
CARGO_INCREMENTAL=0 cargo llvm-cov --workspace --profile ci --no-report --test provider_boundary_allocation_tests -- --nocapture
CARGO_INCREMENTAL=0 cargo llvm-cov --workspace --profile ci --json --output-path /tmp/tenferro-post-u8-coverage.json
python3 scripts/check-coverage.py /tmp/tenferro-post-u8-coverage.json
cargo test -p tenferro-runtime --test integration runtime_public_api -- --nocapture
python3 scripts/gen_dep_graph.py --check-svg docs/assets/dependency-footprint.svg
python3 scripts/test-gen-dep-graph.py
python3 scripts/check-public-error-docs.py --root .
cargo metadata --no-deps --format-version 1
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py --root-dir . --site-index /tmp/tenferro-missing-api-index.html --docs-site-root /tmp/tenferro-missing-docs-site
cargo package -p tenferro-internal-cpu-kernels --allow-dirty --no-verify
CARGO_BUILD_JOBS=64 bash scripts/check-pr-fast.sh --coverage-reviewed --test 'cargo test -p tenferro-cpu --test provider_boundary_allocation_tests warmed_public_session_request_provider_dispatch_does_not_allocate -- --nocapture'
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1 CARGO_BUILD_JOBS=64 cargo bench -p tenferro-runtime --features __bench_unification_run_compiled_api --bench elementwise_fusion -- --sample-size 10 --warm-up-time 0.1 --measurement-time 0.3
CARGO_TARGET_DIR=/tmp/tenferro-1472-candidate-target.Y3WFns CARGO_BUILD_JOBS=64 CARGO_INCREMENTAL=0 RUSTC_WRAPPER= /usr/bin/time -v cargo build --workspace --release --timings
CARGO_TARGET_DIR=/tmp/tenferro-1472-candidate-target.Y3WFns CARGO_BUILD_JOBS=64 CARGO_INCREMENTAL=0 RUSTC_WRAPPER= /usr/bin/time -v cargo build -p tenferro-cpu --release --timings
CARGO_TARGET_DIR=/tmp/tenferro-1472-candidate-target.Y3WFns CARGO_BUILD_JOBS=64 CARGO_INCREMENTAL=0 RUSTC_WRAPPER= /usr/bin/time -v cargo test -p tenferro-cpu --release --lib --no-run --timings
gh issue close 1473 --reason completed
git diff --check
CARGO_BUILD_JOBS=64 cargo test -p tenferro-runtime capability::prepared_operation_source_contract_splits_metadata_from_execution_surface -- --nocapture
CARGO_BUILD_JOBS=64 cargo test -p tenferro-runtime --test integration runtime_execution -- --nocapture
CARGO_BUILD_JOBS=64 cargo test -p tenferro-runtime --test integration runtime_preparation_api -- --nocapture
CARGO_BUILD_JOBS=64 cargo test -p tenferro-runtime runtime::tests::preparation -- --nocapture
CARGO_BUILD_JOBS=64 cargo test --manifest-path ext/sparse/Cargo.toml --all-targets -- --nocapture
CARGO_BUILD_JOBS=64 cargo test --manifest-path ext/tropical/Cargo.toml --all-targets -- --nocapture
```

The full local oracle replay reported:

```text
ReplayRunSummary { total_records: 9585, supported_success_records: 2090, expected_error_records: 2, unsupported_records: 7493, skipped_by_filter_records: 0, replayed_success_records: 2090, replayed_expected_error_records: 2, parallel_jobs: 64 }
```

The #1464 Criterion run rebuilt the release bench target in `6m09s`, exposing
the same `tenferro-cpu` release-optimization long pole tracked by #1472. Median
times from the focused run were:

```text
runtime_elementwise_chain/f64/add_mul/segmented_graph/4096       43.776 us
runtime_elementwise_chain/f64/add_mul/segmented_graph/65536      125.64 us
runtime_elementwise_chain/f64/add_mul/segmented_graph/1048576    1.2320 ms
runtime_elementwise_chain/f64/broadcast_mul/segmented_graph/256x256       112.53 us
runtime_elementwise_chain/f64/broadcast_mul/segmented_graph/1024x1024     1.9468 ms
runtime_elementwise_chain/f64/broadcast_mul_add/segmented_graph/256x256   146.97 us
runtime_elementwise_chain/f64/broadcast_mul_add/segmented_graph/1024x1024 2.7071 ms
```

The #1472 release build-time checkpoint used Rust/Cargo 1.97.1, a fresh target
directory at `/tmp/tenferro-1472-candidate-target.Y3WFns`, `CARGO_BUILD_JOBS=64`,
`CARGO_INCREMENTAL=0`, and no `RUSTC_WRAPPER`.

The new internal crate also packaged successfully with
`cargo package -p tenferro-internal-cpu-kernels --allow-dirty --no-verify`.
The corresponding `tenferro-cpu` package check still stops on the pre-existing
`t4a-tblis-src` crates.io resolution gap, not on the new internal kernel
dependency. That publish-resolution follow-up is #1487.

Coverage thresholds were updated for the post-U8 execution-path shift and CPU
kernel split: the moved elementwise implementation now lives under
`tenferro-internal-cpu-kernels`, and `Runtime::run_compiled` no longer exercises
the legacy `ExecProgram`/segment interpreter as the primary path. Local
workspace `cargo llvm-cov` plus `scripts/check-coverage.py` passed with the
updated thresholds.

Candidate clean workspace release build:

```text
cargo build --workspace --release --timings
wall: 3m39s
user: 1509.88s
max RSS: 4.48 GiB
timing report: /tmp/tenferro-1472-candidate-target.Y3WFns/cargo-timings/cargo-timing-20260726T151835536Z-dc09497ede236b02.html

Top units:
tenferro-internal-cpu-kernels 213.3s total, 8.3s frontend, 204.9s codegen
tenferro-cpu                  141.5s total, 8.2s frontend, 133.3s codegen
faer                           30.0s total, 29.8s frontend, 0.2s codegen
tenferro-linalg                11.7s total, 3.9s frontend, 7.8s codegen
tenferro-fft                   11.4s total, 0.8s frontend, 10.6s codegen
tenferro-runtime                6.8s total, 3.6s frontend, 3.1s codegen
```

Representative release rebuilds in the same target:

```text
touch crates/tenferro-cpu/src/provider.rs
cargo build -p tenferro-cpu --release --timings
wall: 2m17s
user: 524.16s
max RSS: 3.67 GiB
top unit: tenferro-cpu 133.3s total, 125.6s codegen
observation: tenferro-internal-cpu-kernels was reused and did not recompile.

touch crates/tenferro-internal-cpu-kernels/src/elementwise.rs
cargo build -p tenferro-cpu --release --timings
wall: 3m30s
user: 1200.96s
max RSS: 4.71 GiB
top units: tenferro-internal-cpu-kernels 210.1s total, 202.5s codegen;
           tenferro-cpu 140.6s total, 132.8s codegen
observation: provider/domain edits now reuse kernel artifacts, but kernel
edits still rebuild both the internal kernel unit and the public CPU crate.
```

Focused release test no-run in the same target:

```text
cargo test -p tenferro-cpu --release --lib --no-run --timings
wall: 5m44s
user: 1282.94s
max RSS: 4.75 GiB
timing report: /tmp/tenferro-1472-candidate-target.Y3WFns/cargo-timings/cargo-timing-20260726T153017832Z-dc09497ede236b02.html

Top units:
tenferro-internal-cpu-kernels          203.8s total, 8.2s frontend, 195.6s codegen
tenferro-cpu tenferro_cpu "lib" test   137.4s total
faer                                    30.0s total, 29.8s frontend, 0.2s codegen
criterion                                3.1s total
```

## Residual risks to carry forward

- The scheduled executor currently leases a backend state per instruction and
  uses the unsegmented execution path. This is correct for #1456; #1464 and
  #1473 own evidence-backed fusion/performance restoration.
- Cross-storage transfer currently rewrites the slot in place immediately
  before the downstream operation. This covers the linear fake two-device
  substrate; split-use buffer lifetime and explicit `ScheduledTransfer` nodes
  remain future scheduler work.
- CPU buffer-pool and GEMM analysis caches already expose ownership and limit
  controls, but their event counters remain coarse. This is not a blocker for
  #1478 because the generic runtime extension caches now have bounded retained
  bytes and observable statistics.
- Linalg reverse-mode paths that rely on `linearize` followed by semantic
  transposition still record a local linear fragment so the transpose
  interpreter can traverse it backward. This is no longer used by semantic JVP
  emission, and the numerical/oracle coverage above passed.
- CPU elementwise fusion remains a legacy segment-executor specialization.
  This PR records and slightly generalizes the current `Add`/`Multiply`
  classifier, but does not introduce a broad symbolic optimizer or tune
  `Divide`, `Pow`, ordered ops, broadcast views, or longer chains without
  separate benchmark evidence.
- Cache consolidation is complete for the current PR's runtime/AD/extension
  cache owners (#1478), but performance follow-ups #1484 and #1485 will add or
  reuse FFT/cuTENSOR caches only if they inherit the same owner, bounded
  default, retained-byte accounting, clear/configure API, and stats contract.
- The #1472 split deliberately stopped at the first high-value reuse boundary:
  provider/domain/runtime edits no longer force `elementwise`/`BufferPool`
  re-optimization. The new `tenferro-internal-cpu-kernels` crate remains the
  largest release codegen unit, so a later timing-backed split can separate
  elementwise, reductions/indexing, dot/GEMM helpers, and provider contracts if
  those edit patterns become the dominant bottleneck. That narrower follow-up
  is #1486.
