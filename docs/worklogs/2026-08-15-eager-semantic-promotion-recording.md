# Issue #1698: eager semantic promotion recording

## Scope

Fix deferred eager AD recording so accepted mixed-dtype primal operations replay
with the same explicit input conversions used by eager execution. This is an
internal correctness fix with no public API, promotion-policy, tolerance,
backend-placement, or transfer change.

## Source and architecture review

Reviewed:

- issue [#1698](https://github.com/tensor4all/tenferro-rs/issues/1698) and the
  standalone F64/C64 Add -> Eigh reproducer;
- owned and borrowed eager dispatch in `crates/tenferro-ad/src/eager_exec.rs`;
- deferred active-edge recording in `crates/tenferro-ad/src/eager.rs`;
- canonical traced binary promotion in `crates/tenferro-runtime/src/traced.rs`;
- core semantic Add JVP/VJP conversion and cotangent projection;
- linalg `LinearizeThenTranspose` seed/local mapping and Eigh rules; and
- the #1692 concatenate exact-shape recording fix.

The linalg fragment mapping was correct. Eager forward dispatch promoted F64 and
C64 operands to C64, but `record_semantic_eager_outputs` appended the original
F64/C64 raw Add. Eigh backward replays that primal residual and reached the
same-dtype CPU Add kernel without the execution-time conversions.

## Design and review gate

The durable design is
`docs/design/eager-semantic-promotion-recording.md`. The reviewer-gpt
pre-implementation gate returned **Correct-to-merge** before implementation.

Rejected alternatives:

- downstream tensor4all casts, which duplicate tenferro promotion policy;
- linalg cotangent-accumulator changes, which target the symptom after the
  invalid primal replay; and
- concrete eager pre-casts, which would repeat forward kernels and copies.

## Implementation

- Added one allocation-free crate-private eager input-promotion descriptor
  covering binary operations, Select, Clamp, Concatenate, Scatter, and
  DynamicUpdateSlice.
- Routed both owned/read eager dispatch promotion sites through that descriptor.
- Deferred semantic recording inserts `TracedTensor::cast` only for inputs whose
  dtype differs from the execution target, before concatenate exact-shape
  recording.
- Unchanged inputs remain borrowed through `Cow`; inputs with no promotion reuse
  their existing semantic traces without added clones or graph nodes.

## Verification and coverage

Focused tests cover:

- every promotion-plan family and an unchanged unary operation;
- exact mixed F64/C64 and F32/C32 Add JVP/VJP behavior in both operand orders
  and both choices of active input, including inactive cotangent absence;
- mixed-dtype Concatenate semantic replay; and
- the end-to-end F64/C64 Add -> complex Eigh eigenvector backward regression.

Verification completed:

- `cargo fmt --all` and `git diff --check` — passed.
- `cargo clippy --workspace --all-targets -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc` — passed.
- `cargo test --release -p tenferro-ad --test integration` — passed (345 tests).
- `cargo test --release -p tenferro-ad eager_input_promotion_plan_covers_all_promoted_families` — passed (1 test, 432 filtered).
- `cargo test --release -p tenferro-linalg --features autodiff --test integration mixed_real_constant_and_tracked_complex_eigh_vector_backward` — passed (1 test, 210 filtered).
- `cargo test --release -p tenferro-linalg --features autodiff --test integration` — passed (211 tests).
- `bash scripts/check-pr-fast.sh --coverage-reviewed --test 'cargo test -p tenferro-linalg --features autodiff --test integration mixed_real_constant_and_tracked_complex_eigh_vector_backward'` — passed, including root and standalone-extension formatting/clippy gates.
- Worktree repository-rules deterministic review and its script tests — passed.

Coverage impact was reviewed for `eager_exec.rs` (every descriptor arm plus
owned/read execution through the integration target), `eager.rs` (mixed Add and
Concatenate semantic replay plus unchanged #1692 stack coverage), and the
linalg residual consumer (Add -> Eigh backward). No threshold or tolerance
changed. The downstream tensor4all #623 GSE regression is rerun after this fix
is merged and all six pins are updated.

## Post-implementation review

Reviewer-gpt round 1 approved promotion/AD semantics but returned **not
Correct-to-merge** for two hot-path regressions and stale worklog evidence: the
first implementation allocated dtype vectors for every eager op and cloned
unchanged semantic traces when another input needed a cast. The promotion plan
was replaced by an allocation-free descriptor, unchanged traces now remain
borrowed through `Cow`, and this verification/coverage record was updated.
Round 2 returned **Correct-to-merge** on the corrected full diff.
