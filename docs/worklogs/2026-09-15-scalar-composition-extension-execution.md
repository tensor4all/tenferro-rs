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
