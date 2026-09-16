# Caller-owned payload retention: the registered extension executes

## What changed

The registered-extension path now runs end to end, which closes the last
mechanical gap between registration and execution that
`docs/design/scalar-composition.md` §5.4 recorded.

The ownership path is the retention-record shape from the two candidates the
design doc listed. `AdValueRecord` (`crates/tenferro-ad/src/eager.rs`) and
`RetainedValue` (`crates/tenferro-runtime/src/checkpoint.rs`) each held an
allocation group plus one descriptor slot for every value, so a payload that owns
no pooled storage could not be retained at all. Both now hold a
pooled-or-caller-owned container:

- `RetentionContainer::Pooled { group, slot }` keeps the previous behaviour.
- `RetentionContainer::CallerOwned { tensor }` retains the payload directly. It
  owns no group, so nothing returns to a pool when the record drops; the value
  goes back to its owner.

Three supporting changes make that path sound rather than merely compiling:

- `TensorValue::try_into_group_parts` returns a caller-owned value unchanged
  instead of forcing it into a group. It used to reach the `unreachable!` in
  `Tensor::into_group_parts`, which is what turned the gap into a panic.
- `AllocationGroup::from_tensors` and `append_tensor` report
  `GroupError::InvalidDescriptor` for a caller-owned payload. The panic is now an
  internal invariant behind a typed error on every user-reachable entry point.
- `EagerTensor::duplicate_value` duplicates a caller-owned payload through its own
  read path, which copies the value in its own type and reinterprets no bytes. A
  path that needs a typed descriptor view (`value()`) still fails explicitly, and
  that is deliberate: there is no borrowed descriptor for a payload tenferro did
  not define.

## Evidence

`ext/df64-proof/tests/extension_execution.rs` (no longer ignored):

- `the_extension_operation_runs_through_the_registered_module` runs a registered
  `ExtensionOp` on a caller-owned `Df64` payload through
  `apply_eager_with_extension_session` and checks that the low-order component
  survives (`1 + 2^-80`).
- `the_runtime_retains_and_returns_a_caller_owned_payload` round-trips the payload
  through `EagerTensor` and asserts the value is returned unchanged and still
  external.
- `an_allocation_group_rejects_a_caller_owned_payload_with_a_typed_error` pins the
  typed rejection instead of a panic.
- `the_registered_operation_rejects_a_preset_input` and
  `ordinary_work_shares_the_session_with_the_extension` stay as they were.

Workspace: `cargo check --workspace --all-targets` 0 errors, 0 warnings; clippy
`-D warnings` clean; `cargo test --workspace --no-fail-fast` 5207 passed, 3
failed, and the 3 are the pre-existing `trybuild` environment failures that the
pristine baseline `bca2d54a` also reports. `scripts/check-pr-fast.sh` and
`scripts/repository-rules-review.py` pass.

## What is intentionally not decided here

#1789 still owns pool reuse for external scratch, cross-owner handoffs, and the
accounting a caller-owned payload participates in. This change retains and
returns a payload; it does not pool it.

The unchecked promotion between two distinct external tags is unchanged: 69
`promote_dtype`/`promote_dtypes` call sites are mostly infallible, so the check
belongs at the executing entry points, not in the lattice.

## Residual risk

- The enum carries an inline allocation group, so it is `#[allow(clippy::large_enum_variant)]`
  rather than boxed. Boxing the pooled variant would add one allocation to every
  retained value on the hot path; that trade is not made on measurement.
- `RetainedValue::from_tensor` still uses `expect` for a compact tensor's
  descriptor. Caller-owned payloads now return through the `Err` path instead, so
  the `expect` is reached only by a value that already validated one.

## Follow-on: the required example and directed conversions

With retention working, the parts of #1785 that do not need a view contract are
landed and tested:

- `ext/df64-proof/src/conversion.rs` adds the two directed conversions.
  `to_f64` sums the two components and rounds to nearest with ties to even, so the
  low component participates (`2^-52` reaches the destination, `2^-80` rounds
  away, half an ulp rounds to even) instead of being truncated to the high
  component. `to_df64` is exact and gives a zero low component. Both declare
  range and destination allocation, and both reject a source they do not declare.
- `ext/df64-proof/tests/directed_conversion.rs` walks #1785's required example
  through public boundaries: two `Df64` tensors holding `1` and `2^-80` are added
  in `Df64`, `1` is subtracted in `Df64`, the result is exactly `2^-80`, and the
  same computation in `f64` yields `0`. It also pins the declared rounding of both
  conversions and the typed rejection of an undeclared source.

Workspace after this: 5215 passed, 3 failed (the same pre-existing `trybuild`
failures), clippy `-D warnings` and the strict `missing_errors_doc` /
`missing_panics_doc` pass clean, and all thirteen doctests in the proof crate run.

## Follow-on: the erased payload carries its own view layout

#1785's remaining storage requirement was a view over a caller-owned payload. The
layout lives in the erased value rather than in `TensorView`, so no typed view
variant was needed:

- `ErasedHostTensor` stores shape, element strides, and an element offset beside
  the payload. `permuted(axes)` is metadata only and shares the payload;
  `to_contiguous()` gathers the view into a new dense payload; `duplicate()` copies
  the payload; `payload_element_count()` reports the storage extent a strided view
  may not reach in full.
- The payload became `Arc<dyn ErasedPayload>`, so `Clone` shares storage and
  `duplicate()` copies it. `Tensor::duplicate` and the CPU contiguous-copy path
  now copy through the payload's own entry point instead of rejecting or sharing.
- Typed access applies the layout: `element_at`/`element_at_mut` need the caller to
  be the only holder for mutation, so two live views never yield two mutable
  borrows. `downcast_ref`, `downcast_mut`, `into_typed`, and `as_dense` answer
  `None` for a strided view, which is the "mismatched projection fails safely"
  boundary.
- `tensor_layout` uses the payload's own strides and offset, so an erased tensor
  reports the view it carries instead of assuming a compact layout.

`ext/df64-proof/tests/external_views.rs` covers the requirements: a metadata-only
permutation that preserves every component through typed reads, a mutable view that
writes exactly the element it names, a shared payload that refuses a mutable
borrow, materialization into logical order, a sum reduction whose low-order
component survives (`1 + 2^-80 + 2^-80 = 1 + 2^-79` exactly), and typed rejection
of an invalid permutation.

One consumer test changed with the contract:
`an_external_payload_is_carried_by_the_value_type` asserted that duplication was
rejected, and it now asserts that duplication copies the payload, keeps the element
type, and leaves the original unchanged.

Workspace after this: 5231 passed, 3 failed (the same pre-existing `trybuild`
failures), clippy `-D warnings` clean.

## What the next step needs

#1789's resource decisions (pool reuse for external scratch, cross-owner handoffs,
and the accounting a caller-owned payload participates in) and the numerical work
of #1788/#1793 (first-order Df64 AD, QR) remain. Promotion between two distinct
external tags is still unchecked at the lattice and belongs at the executing entry
points.

## Follow-on: what promotion between two external scalars actually does

The design doc recorded promotion between two distinct external tags as an
unchecked gap. It is now measured, and the conclusion is narrower than the
recording implied:

- The lattice is imprecise by construction: `promote(External(Df64), External(i64))`
  is `External(Df64)` and the reversed pair is `External(i64)`, because a scalar
  tenferro does not declare has no declared relation to another one.
- No executing entry point turns that into a wrong value. A conversion between two
  distinct external tags is rejected in both directions, a conversion between a
  preset and an external tag is rejected in both directions, and a binary operation
  on external tensors does not run at all, matching or not, because no preset kernel
  is instantiated for a caller-owned payload.

`ext/df64-proof/tests/external_mixing.rs` pins all of it, including the
`can_convert_dtype` answers a caller can consult first. The remaining sharp edge is
the dtype a traced graph *reports* for such a pair before execution rejects the
program; closing that needs a checked promotion threaded through the runtime's
infallible inference paths, so it is recorded rather than guessed at.

Workspace after this: 5235 passed, 3 failed (the same pre-existing `trybuild`
failures).

## Follow-on: object-level evidence for the shared kernel

The goal's evidence list requires object or symbol evidence for shared compiled
kernels, and the design doc's sharing claim was measured with a one-off `nm -C`
run. It is now a reproducible probe:
`scripts/check-scalar-composition-kernel-sharing.py` builds the proof crate's
composition test with `--emit=asm`, counts the kernel entries the assembly file
defines, and writes a JSON record to
`docs/design/scalar-composition-kernel-sharing.md`.

On `79092f7c` (release, `rustc 1.97.1`, `x86_64-unknown-linux-gnu`):

| Path | `zip_map2_into` | `zip_map2_parts_into_validated` | Call sites |
| --- | --- | --- | --- |
| preset `f64` | 1 | 2 | 39 |
| external `Df64` | 1 | 3 | 41 |

Every entry on both paths names
`tenferro_internal_cpu_kernels::scalar_ops::scalar_binary_into`, so the preset and
external paths are the same kernel function rather than two implementations, and
every external entry is parameterized by the contribution's `Df64Add`. The probe
exits non-zero if a kernel function is missing on either path, if an instantiation
does not come from the shared crate, if an external instantiation is not
parameterized by the contribution's operation, or if the preset entries are not
reached from more than one place.
