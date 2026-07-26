# Unification 7-8 tidu retirement and gate status

## Session summary

This session audited the recent branch state after the DeepSeek V4 Pro
implementation, found that Unification 7 was not complete, retired the
remaining production `tidu` dependency from the AD path, and carried
Unification 8 through the U0 terminal performance protocol.

The implementation now routes eager and traced JVP/VJP through semantic
program transforms and removes the old eager tape, linearized graph cache,
primitive transpose, and eager primitive builder modules.

The final U8 gate result is `PASS` from the CPU12 formal r4 campaign:
three complete paired comparisons, 123 parsed comparison records, and zero
threshold violations under the predeclared U0 rules.

## Context read

- Umbrella #1433 and child issues #1454 through #1462.
- Follow-up memo issue #1463 for future benchmark build artifact reuse.
- `AGENTS.md`, `REPOSITORY_RULES.md`, workspace `CODING_RULES.md`, and the
  latest shared Tensor4all rules relevant to Rust, performance, docs/tests,
  and numerical work.
- `docs/worklogs/2026-07-25-unification-0-terminal-performance-gate.md`.
- AD and eager source around `AdContext`, `EagerRuntime`, `EagerTensor`,
  semantic transforms, extension recording, and transform cache ownership.

## Decisions

- Kept `tenferro-internal-ops` as the owner of the primitive AD vocabulary, but
  moved AD error/result/key helper types into tenferro-owned definitions
  instead of importing them from `tidu`.
- Removed the eager backward/JVP/VJP fallback modules instead of preserving
  compatibility shims.
- Made `EagerRuntime` retain `SemanticExtensionRuleSet` from `AdContext`, so
  eager extension AD can use the same semantic rules as traced AD.
- Simplified `AdTransformCache` to semantic-program transform entries only.
- Used CPU12 for the formal U8 protocol after confirming no unrelated active
  Cargo/rustc work and low normalized host load on the 64-core machine.
- Kept the U0 threshold policy unchanged. Microsecond eager dispatch cases
  remain diagnostics unless they satisfy the U0 restart criterion; non-micro
  and domain workload decisions use the predeclared U0 thresholds.
- Marked U8 r3 as failed/inconclusive rather than claiming PASS: one runtime
  case exceeded the +5% upper-bound gate. The final result therefore uses a
  complete unchanged r4 three-pair campaign.

## U8 remediation

The U8 reruns exposed hot-path issues before the final pass:

- `ExtensionCacheStore::get`/`get_mut` did two LRU lookups on every hit and
  updated scoped stats through hash maps in the hot path. The final version
  performs one LRU lookup and stores scoped event counters in small vectors.
- Expanded eager einsum execution cloned `EagerTensor` values for normal
  instructions before dispatching standard ops. The final version builds input
  references directly from slots, avoiding per-instruction eager tensor clone
  overhead.
- Untracked eager standard-op results still entered the `value_records` /
  `value_ptr_records` registry. A red/green regression test now verifies that
  an untracked `dot_general` result stays out of the AD value registry, and
  `new_untracked_result` now constructs an untracked value directly.

These are in addition to the earlier U7/U8 fixes:

- Runtime compiled execution supports prepared graph reuse through
  `prepare_compiled` and `run_prepared`.
- AD prepared derivative cache keys include semantic fingerprint, runtime
  epoch, derivative input index, and shape-resolved input metadata.
- Semantic AD transform cache keys include input metadata with bound shapes.
- Linalg semantic VJP uses the imported triangular-solve primal solution, and
  the linalg extension fast path can execute triangular solve from read-only
  tensors without materializing first.
- U8-only benchmark targets that require candidate bench features now declare
  `required-features`, so default `cargo check --workspace --all-targets`
  does not compile deleted legacy bench API paths while feature-specific bench
  checks still validate the measured candidate harness.

## Verification

Fresh commands run in this worktree:

```console
cargo fmt --all --check
git diff --check
CARGO_BUILD_JOBS=16 cargo check --workspace --all-targets
CARGO_BUILD_JOBS=16 cargo check -p tenferro-runtime --features __bench_unification_run_compiled_api --bench elementwise_fusion
CARGO_BUILD_JOBS=16 cargo check -p tenferro-linalg --features autodiff,__bench_unification_semantic_ad_api --bench linalg_vjp_gate
CARGO_BUILD_JOBS=16 cargo test -p tenferro-ad untracked_standard_op_results_do_not_enter_value_record_registry --lib -- --nocapture
CARGO_BUILD_JOBS=16 cargo test -p tenferro-ad eager_prepared_derivative_cache_reuses_runtime_preparation --lib -- --nocapture
CARGO_BUILD_JOBS=16 cargo test -p tenferro-ad cache_management --test integration -- --nocapture
CARGO_BUILD_JOBS=16 cargo test -p tenferro-runtime extension_cache --lib -- --nocapture
CARGO_BUILD_JOBS=16 cargo test -p tenferro-einsum --lib -- --nocapture
CARGO_BUILD_JOBS=16 cargo test -p tenferro-linalg triangular_solve_semantic_vjp_reuses_imported_primal_solution --lib --features autodiff -- --nocapture
```

All listed commands exited successfully. Before adding benchmark
`required-features`, default `cargo check --workspace --all-targets` failed
because candidate-only U8 bench targets tried to compile removed legacy API
paths. After the metadata fix, the default workspace check and the explicit
feature-specific bench checks both pass.

## U8 final gate

Pinned baseline commit:

```text
c6418eecfe2d38ca09d6e6386760fcb23982691e
```

Formal runner shape:

```console
# baseline save, from the pinned baseline worktree
CARGO_BUILD_JOBS=16 bash scripts/run-unification-performance-gate.sh --mode run --label baseline --cpu 12 --output-dir <baseline-output-dir> -- --save-baseline <criterion-label>

# candidate compare, from the candidate worktree
CARGO_BUILD_JOBS=16 bash scripts/run-unification-performance-gate.sh --mode run --label candidate --cpu 12 --output-dir <candidate-output-dir> -- --baseline <criterion-label>

# reverse pair: candidate save then baseline compare
CARGO_BUILD_JOBS=16 bash scripts/run-unification-performance-gate.sh --mode run --label candidate --cpu 12 --output-dir <candidate-output-dir> -- --save-baseline <criterion-label>
CARGO_BUILD_JOBS=16 bash scripts/run-unification-performance-gate.sh --mode run --label baseline --cpu 12 --output-dir <baseline-output-dir> -- --baseline <criterion-label>
```

The r4 labels were:

- `u8-cpu12-formal-r4-pair1-baseline`
- `u8-cpu12-formal-r4-pair1-candidate`
- `u8-cpu12-formal-r4-pair2-candidate`
- `u8-cpu12-formal-r4-pair2-baseline`
- `u8-cpu12-formal-r4-pair3-baseline`
- `u8-cpu12-formal-r4-pair3-candidate`

Benchmark execution was bound to CPU12. Build steps were left unbound and used
`CARGO_BUILD_JOBS=16`, with separate baseline/candidate `target` directories
and copied Criterion baseline artifacts for comparison. Each copy found 41
Criterion baseline directories. Boundary checks found no unrelated active
Cargo/rustc work and normalized host load below the U0 threshold.

The r3 campaign was retained as evidence but not used as the pass result. It
had one formal violation: pair2 reverse `broadcast_mul_add/1024` converted to
candidate-relative `[+3.626%, +5.721%, +7.984%]`, exceeding the runtime +5%
upper-bound gate. Pair1 and pair3 passed the same case, so this was treated as
a retry trigger, not as a threshold change.

The r4 campaign parsed 123 expected comparison records and found zero
violations. Maximum candidate-relative upper bounds by group:

```text
pair1 eager-micro     +7.086%   (U0 restart threshold +50%)
pair1 eager-small     -4.582%   (threshold +5%)
pair1 runtime         +2.473%   (threshold +5%)
pair1 einsum-prepare  -5.050%   (threshold +10%)
pair1 shape-churn    -56.140%   (threshold +10%)
pair1 linalg-vjp     -12.330%   (threshold +10%)

pair2 eager-micro     +6.334%   (U0 restart threshold +50%)
pair2 eager-small     -1.897%   (threshold +5%)
pair2 runtime         +4.500%   (threshold +5%)
pair2 einsum-prepare  -6.546%   (threshold +10%)
pair2 shape-churn    -55.715%   (threshold +10%)
pair2 linalg-vjp      -8.336%   (threshold +10%)

pair3 eager-micro     +7.935%   (U0 restart threshold +50%)
pair3 eager-small     -1.501%   (threshold +5%)
pair3 runtime         +1.959%   (threshold +5%)
pair3 einsum-prepare  -3.617%   (threshold +10%)
pair3 shape-churn    -55.741%   (threshold +10%)
pair3 linalg-vjp     -13.328%   (threshold +10%)
```

Representative r4 non-micro eager `dot_general` candidate-relative upper
bounds were all improvement-side, including pair1 lazy `dot_general_f64/1`
`-11.178%`, pair2 materialized `dot_general_f64/1` `-9.627%`, and pair3
materialized `dot_general_f64/2` `-9.151%`.

The runtime watch case that failed r3 passed r4:

```text
pair1 broadcast_mul_add/1024 candidate-relative upper +2.473%
pair2 broadcast_mul_add/1024 candidate-relative upper +4.500%
pair3 broadcast_mul_add/1024 candidate-relative upper +1.959%
```

Residual eager dispatch micro noise remains visible in Criterion diagnostics,
but no case met the U0 micro restart criterion, and no non-micro U8 gate
workload violated its predeclared threshold.

## Post Oracle/Cache Diagnostic Rerun

After the crate-local oracle replay harness and prepared-derivative cache
fixes, a candidate-side diagnostic comparison was run on CPU12 against the
existing `u8-cpu12-formal-r4-pair3-baseline` Criterion artifacts:

```console
CARGO_BUILD_JOBS=16 bash scripts/run-unification-performance-gate.sh --mode run --label candidate --cpu 12 --output-dir target/unification-performance-gate/u8-cpu12-after-oracle-cache-fixes-candidate -- --baseline u8-cpu12-formal-r4-pair3-baseline
```

This rerun is not a replacement formal U8 pass. It was useful for checking the
new AD/oracle work, but the runtime steady-state elementwise target produced
non-micro violations:

- `add_mul/65536`: candidate-relative upper `+5.796%`
- `broadcast_mul/256x256`: candidate-relative upper `+6.603%`
- `broadcast_mul/1024x1024`: candidate-relative upper `+7.984%`
- `broadcast_mul_add/256x256`: candidate-relative upper `+8.008%`
- `broadcast_mul_add/1024x1024`: candidate-relative upper `+15.608%`

A focused diagnostic rerun of only `tenferro-runtime/elementwise_fusion` on the
same CPU12 and baseline showed mixed results: the broadcast cases returned to
no-change or improvement, but `add_mul/65536` and `add_mul/1048576` still
exceeded the runtime +5% upper-bound gate (`+8.030%` and `+11.545%`
respectively). During the diagnostic check, host load included long-running
Julia kernels on other cores; no unrelated Cargo/rustc build overlapped the
runtime bench itself.

The directly relevant AD workloads were improvement-side in the full
diagnostic comparison:

- eager backward shape churn: `-51.831%` upper
- linalg triangular solve VJP 8/16: `-3.034%` and `-9.366%` upper
- linalg SVD values VJP 8/16: `-64.627%` and `-61.992%` upper

Conclusion: keep the earlier r4 formal PASS as historical U8 evidence, but do
not claim this post-oracle/cache diagnostic as a fresh U8 PASS. A fresh formal
paired rerun is required before using post-fix runtime numbers as blocking
gate evidence.

## Post runtime fusion repair diagnostics

The post-oracle/cache runtime regression was traced to
`runtime_elementwise_chain/f64/add_mul`. The benchmark graph computes
`(a + b) * a`; this was above the CPU elementwise-fusion threshold for the
65,536 and 1,048,576 element cases but did not match the existing CPU
specialization for `(a * b) + a/b`, so it fell to the generic fused interpreter.

The CPU backend now keeps the change intentionally narrow: two-input,
two-operation, one-output identity-view plans for `(a + b) * a/b` use the same
typed `zip_map2_into` specialization family as the existing `(a * b) + a/b`
cases. This is not a general algebraic optimizer; follow-up issue
https://github.com/tensor4all/tenferro-rs/issues/1464 records the broader
classifier/generalization idea.

Focused correctness and runtime diagnostics after this repair:

```console
CARGO_BUILD_JOBS=48 cargo test -p tenferro-cpu add_then_multiply_identity_fusion_uses_typed_specialization --lib -- --nocapture
CARGO_BUILD_JOBS=48 cargo test -p tenferro-cpu elementwise_fusion --lib -- --nocapture
CARGO_BUILD_JOBS=48 cargo test -p tenferro-runtime shape_infer --lib -- --nocapture
CARGO_BUILD_JOBS=48 cargo test -p tenferro-runtime --test integration runtime_public_api -- --nocapture
```

All four commands exited successfully.

The local oracle replay harness was also updated to run replay records in
parallel via bounded OS worker threads. An attempted outer Rayon pool was
rejected because CPU backend kernels use backend-owned Rayon scopes internally;
wrapping whole oracle records in another Rayon managed scope trips the CPU
backend re-entry guard. The replay entrypoint now lives under the consolidated
`tenferro-linalg` integration test target so the crate still links one
integration-test binary in CI. The full local replay was then run with HVP
enabled:

```console
RUST_TEST_THREADS=1 RUN_ORACLE_REPLAY=1 ORACLE_REPLAY_JOBS=48 CARGO_BUILD_JOBS=48 cargo test -p tenferro-linalg --features autodiff --test integration oracle_replays_supported_db_cases_when_requested -- --nocapture
```

It passed in 79.49s with `9585` total records, `2090` supported-success
records replayed, `2` expected-error records replayed, `7493` unsupported
records classified, and `parallel_jobs: 48`.

A fresh runtime-only paired diagnostic was then run on CPU12 by saving a new
baseline label in the pinned baseline worktree and copying the seven Criterion
runtime baseline directories into the candidate worktree:

```text
baseline label: u8-cpu12-runtime-diagnostic-r5-baseline
candidate log:  target/unification-performance-gate/runtime-paired-diagnostic-r5-candidate/run.log
```

All seven runtime elementwise cases were within the +5% upper-bound gate. The
formerly failing add-mul cases were:

```text
add_mul/4096     upper -6.789%
add_mul/65536    upper -1.684%
add_mul/1048576  upper +1.743%
```

The broadcast cases were improvement-side or no-change:

```text
broadcast_mul/256x256       upper -1.839%
broadcast_mul/1024x1024     upper -3.850%
broadcast_mul_add/256x256   upper -1.226%
broadcast_mul_add/1024x1024 upper +3.491%
```

A subsequent full r5 pair1 attempt is diagnostic only and must not be used as a
formal U8 pass. During that run, unrelated Matrix `rustc` work and several
Julia kernels were active on other cores, which violates the U0 host-noise
validity gate. The candidate run showed impossible drift relative to the
immediately preceding runtime paired diagnostic, including
`add_mul/1048576` upper `+25.869%`; the run was stopped and classified as
invalid/inconclusive.
