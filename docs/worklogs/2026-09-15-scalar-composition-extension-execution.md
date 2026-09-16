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

## Follow-on: the traced IR boundary, and a panic it hid

Driving the registered module through a real runtime planning path found a live
panic: `tenferro-runtime/src/program/identity.rs` encoded external scalars with an
`unreachable!`, on the reasoning that no value type carried one. That premise was
already stale, and the panic was reachable from the public trace API.

The fix follows the design's explicit-rejection rule instead of inventing a stable
identity. A semantic program's identity must be reproducible across processes,
while an externally defined tag is a process-local `TypeId`, so the builder now
rejects the tag when an input spec or an operation output carries one
(`tenferro-runtime::program::ProgramBuildError::ExternalScalarWithoutIdentity`).
The encoder's invariant is restored and the panic becomes a typed error. Enabling
the traced and prepared path needs a contribution-declared stable scalar identity,
which is a new payload contract and is recorded as such.

Two tests pin this:
`ext/df64-proof/tests/extension_execution.rs::the_module_installs_and_planning_rejects_the_scalar_without_a_stable_identity`
installs the module into a runtime with the CPU engine and checks the typed
rejection, and a unit test in the builder covers the rejection helper for every
preset tag.

Workspace after this: 5245 passed, 3 failed (the same pre-existing `trybuild`
failures), clippy `-D warnings` clean.

## The traced and prepared path works, with a declared identity

The rejection recorded earlier was the "explicit rejection" half of the design's
conversion-or-rejection rule. The enabling half is now implemented, because the
goal's acceptance requires the traced and AD paths rather than only their refusal.

A semantic program declares the canonical name of an externally defined scalar:
`ProgramValueMetadata::with_scalar_identity` and
`ProgramInputSpec::with_scalar_identity` for inputs,
`ExtensionOp::scalar_identity` (defaulted) for an operation's values, the identity
encoder writes that name instead of a process-local type code, and a value without
one is still rejected with `ProgramBuildError::ExternalScalarWithoutIdentity`. A
core operation may not name an external scalar at all, since tenferro owns no kernel
for one.

The result is verified end to end:
`ext/df64-proof/tests/extension_execution.rs::the_module_installs_and_plans_the_declared_scalar`
installs the module, traces the operation, compiles it, and runs it through prepared
execution, checking that the total is computed in the external scalar; the same test
asserts the undeclared case is a typed error, and two unit tests in
`crates/tenferro-runtime/src/program/tests.rs` cover the rejection helpers.

## An earlier report needed correcting

Two claims from the earlier report needed checking against the source.

**The extension AD surface already exists.** `tenferro-ad/src/semantic_extension.rs`
exposes `SemanticExtensionRuleSet` with `register_linearize`,
`register_linear_transpose`, and `register_primal_vjp`, and
`crates/tenferro-ad/tests/integration/multi_input_traced.rs` drives a registered VJP
rule end to end. So an external first-order rule does not need a new mechanism.
What it needs is a value that survives program construction.

**The identity change is bounded.** A canonical identity for an external tag is 40
constructor sites and 28 identity reads, with the 57 `External(_)` wildcard matches
untouched. The design doc now records three shapes and recommends carrying the
declared identity in `DType::External` beside the `TypeId`, which keeps every
wildcard working and extends the public payload contract only at construction.

**bf16 is genuinely dependency-blocked.** `half` is not a workspace dependency and
no crate mentions `bf16`, so #1785's "use `half::bf16` as a standard scalar
representation" needs either that dependency or a locally defined standard
representation, both of which are maintainer decisions.

## Where the AD path stops, located precisely

The traced and prepared path now works for programs built from a trace context, and
the next acceptance item is an external first-order AD rule. Investigating it found
the exact remaining site rather than a design gap: the `TracedTensor`/AD path reaches
the program through the traced graph's own metadata (`TensorMeta` in
`tenferro-internal-ops/src/ad/context.rs`), and
`crates/tenferro-runtime/src/graph/compiler.rs` builds program inputs from a
descriptor that carries only a dtype and a semantic shape. So the declared identity
has to exist in that layer too. The measured extent is 47 `TensorMeta` references in
`tenferro-runtime` plus its definition and AD users in `tenferro-internal-ops`, and
the natural shape is to carry the identity on the traced leaf and pass it through the
descriptor into `ProgramInputSpec`, reusing the declaration API this step added.

The rule itself needs no new mechanism: `SemanticPrimalVjpRule` plus
`SemanticExtensionRuleSet::with_primal_vjp` already exist, and
`crates/tenferro-ad/tests/integration/multi_input_traced.rs` drives one end to end.

## First-order AD for the external scalar

The next acceptance item was an external first-order rule, and it now runs end to end.
The rule needed no new mechanism; what it needed was for the declared identity to
reach every layer the traced/AD path reads, which took four fixes found one at a time:

- `TensorMeta` carries the declared name beside dtype and extents, with
  `TracedTensor::input_concrete_shape_declaring_scalar` and
  `TracedTensor::from_tensor_concrete_shape_declaring_scalar` declaring it.
- The runtime compiler passes it from the traced value's registered metadata into
  `ProgramInputSpec`, for an unbound placeholder and for a bound default tensor; the
  bound case was the first missed path, and the import path that rebuilds metadata
  was the second and third.
- `tenferro-ad` keeps it when converting program metadata back into traced metadata.

That chain was only findable because `ProgramBuildError::ExternalScalarWithoutIdentity`
now reports where the tag reached the program (an input, an operation output, or a
core operation) instead of only which dtype it was; the error gained a `site` field
for it.

The adjoint is the contribution's own operation: `Df64Expand` broadcasts a scalar to a
declared shape, and `Df64TotalVjpRule` emits it from the output cotangent, reading the
target shape from the primal input metadata and rejecting a symbolic shape rather than
guessing one. The runtime keys one planning config per engine id, so both operations
report one family and the engine dispatches on the payload; the rule rejects a payload
outside its domain.

`ext/df64-proof/tests/extension_ad.rs` verifies the traced VJP, the compilation, and
the execution of the backward program, including a cotangent whose `2^-80` low
component survives the adjoint.

Both directions are covered: `Df64TotalLinearizeRule` sums the tangent inputs, so
the forward rule needs no new operation, and the test suite verifies the JVP and the
VJP, their compilation, and the execution of both programs.

Workspace after this: 5258 passed, 3 failed (the same pre-existing `trybuild`
failures), clippy clean under `-D warnings` and the strict doc lints.

## The contribution-owned QR factorization

#1788's QR checkpoint does not need a new provider mechanism: the contribution owns its
numerical body the same way it owns the total sum. `Df64Qr` is a second operation in
the same family, one input and two outputs, reached through the same registered engine
and prepared execution.

The body is modified Gram-Schmidt with one re-orthogonalization pass in the external
scalar, which needed two additions to that scalar: `Df64::ratio` refines a quotient
with two Newton corrections evaluated in the two-component arithmetic, and
`Df64::sqrt` refines a square root the same way. The `f64` controls are decisive:
`(1/3) * 3 - 1` is exactly zero in `f64` and non-zero in the external scalar, as is
`sqrt(2)^2 - 2`.

`ext/df64-proof/tests/extension_qr.rs` verifies the factorisation through the traced,
compiled, and executed path: reconstruction error below `1e-30`, orthonormal columns
below `1e-30`, a positive diagonal, `[[3], [4]]` giving `R = [[5]]` and
`Q = [[0.6], [0.8]]`, and a `2^-80` low component in the input reaching the factors.

Two mistakes in my own test expectations were caught by the run rather than left in:
comparing rounded `f64` values against two-component results, and computing the
expected value in `f64` before comparing it with an extended-precision result.

Workspace after this: 5268 passed, 3 failed (the same pre-existing `trybuild`
failures), clippy clean under `-D warnings` and the strict doc lints.

## Connected programs across a conversion

#1790's connected programs need the conversion to be an operation in the graph, so the
contribution now owns both directions (`Df64ToF64`, `Df64FromF64`) in the same family,
with the opposite conversion as each adjoint.

Implementing them found a real limitation: the reverse pass hands an operation a
borrowed read, because the adjoint of an ordinary `f64` reduction is a broadcast, which
is a strided view. The executor now resolves an input to either a borrow or a
materialized tensor, using the session the context carries when it has one and
gathering a preset `f64` read by its own layout when it does not. The first attempt
threaded only `execute_in_session` and still failed, because the runtime reaches this
path through `execute`; the erased context does not expose a session, so the preset
gather is what makes the path work.

`ext/df64-proof/tests/connected_conversion_ad.rs` verifies both directions: the
narrowing program differentiates into the external scalar and does *not* recover the
discarded low component, and the widening program differentiates into an ordinary
`f64` gradient of `[6, 8]` for the input `[3, 4]`.

Workspace after this: 5272 passed, 3 failed (the same pre-existing `trybuild`
failures), clippy clean under `-D warnings` and the strict doc lints.

## The connected QR program

#1790's second checkpoint now runs: `Df64 input -> QR -> narrow -> ordinary f64 loss`
differentiates back into the external scalar as `[[6], [8]]` for `A = [[3], [4]]`, which
is the orientation case #1790 states. The forward tangent of the same graph is `3` for
the first unit direction and leaves as an ordinary `f64` value.

Both derivative directions are the contribution's own operations (`Df64QrVjp`,
`Df64QrJvp`) because both need a triangular solve in the external scalar. The adjoint's
payload records which cotangents are present, so a loss that depends on one factor still
differentiates.

The development found a real bug in my own adjoint body: it subtracted `Q^T Q` instead
of `Q_bar^T Q`. Measuring the adjoint *in isolation* first showed a factor of `0.98` that
depended on the data, which localized the fault to the body rather than to the rule or
the cotangent plumbing; the connected test then verified the whole graph.

Both connected graphs #1790 states now run: the Df64-input graph returns its gradient in
the external scalar, and `f64 input -> widen -> QR -> narrow -> loss` returns `[6, 8]` in
`f64`, so a gradient crosses the widening and lands in the input's own dtype.

Workspace after this: 5278 passed, 3 failed (the same pre-existing `trybuild`
failures), clippy clean under `-D warnings` and the strict doc lints.

## The two consumer roles

#1790 asks for two compilation roles rather than one proof crate, and both now exist.
`ext/scalar-consumer-algorithm` declares the capabilities it needs
(`ScalarSupport::qr` and `ScalarSupport::to_f64`) and builds the connected program from
them, using only public traced operations and never naming a scalar, a provider, or a
dtype. `ext/scalar-consumer-application` supplies two bindings for that one algorithm:
canonical standard support from `tenferro-linalg`, and standard support plus the external
scalar contribution.

Both return the same number in their own dtype for `A = [[3], [4]]`: `[6, 8]` as `f64`
and `[[6], [8]]` as `Df64`, with no change to the algorithm source and no dtype or
provider forwarding table. The boundary is mechanical: `cargo tree -p
tenferro-scalar-consumer-algorithm --edges normal` contains no `df64` and no
`tenferro-linalg` entry.

The standard binding also showed what the application is for: the linalg module has to be
installed into the runtime alongside the engine, and the Df64 module alongside it, which
is application composition rather than algorithm work.

Workspace after this: 5282 passed, 3 failed (the same pre-existing `trybuild` failures),
clippy clean under `-D warnings` and the strict doc lints.

## The four configurations

All four of #1790's configurations run from the application crate.

I first reported the cooperating configuration as blocked and that was wrong, so the
correction is worth recording. The refusal I saw
(`RuntimeConfigError::DuplicateProviderDeviceTarget`) came from registering the *same*
default domain twice, not from a missing capability: the runtime tells two CPU owners
apart by resource domain, and `CpuBackend::from_external_managed_domains` builds a second
one. The test now builds two external domains from the discovered topology, registers one
engine for each, installs the linalg module against one and the contribution's
`module_for_engine` against the other, and runs both connected programs in that single
runtime. On a host that declares fewer than two nodes it returns early rather than
claiming the configuration.

The lesson is the one this session has repeated: a typed rejection is evidence about the
shape the caller tried, not proof that the capability is absent. The earlier report
treated it as proof.

Workspace after this: 5287 passed, 3 failed (the same pre-existing `trybuild` failures),
clippy clean under `-D warnings` and the strict doc lints.

## Measured size and coverage

`git diff --numstat origin/main..HEAD` on this head: 119 files, 10964 insertions, 807
deletions, with `ext/df64-proof` at +4265 and the two consumer crates at +525 and +250.

`cargo llvm-cov` over the contribution and the consumer crates reports `lib.rs` 95.2%,
`dense.rs` 95.0%, `extension.rs` 84.2%, `conversion.rs` 82.6%, `ad.rs` 78.5%, and the
algorithm crate 82.6%. The numbers below 90% are explained by two measurable facts rather
than by untested behavior: llvm-cov does not instrument doctests, and this branch's public
items carry runnable examples whose bodies account for whole line ranges in those files;
and several remaining branches are refusals that another guard makes unreachable.

`ext/df64-proof/tests/extension_boundaries.rs` raised `ad.rs` from 72.6% to 78.5% and
`extension.rs` from 79.9% to 84.2% with real boundary assertions: a vector, a wide matrix
and a zero column for the factorization, a preset input for the narrowing and an external
one for the widening, a singular triangular factor for both derivative operations, and a
derivative rule asked about an operation outside its domain.

## The survival half of "later backward"

`ext/scalar-consumer-application/tests/later_backward.rs` covers the part of #1790's
"later backward" checkpoint that does not need #1789's pool contract: a forward
factorization runs in its own runtime, the runtime is dropped, and the retained factors are
then written to and consumed by a later program in a new runtime, whose adjoint output is
exactly the written values. That is evidence of survival rather than of recomputation.

My first version of the test asserted on the triangular factor, whose contribution cancels
in the adjoint (`Q (R R_bar^T) R^{-T}` is `Q R_bar` for one column), so the probe could not
distinguish anything. Moving the write to the factor `Q` made the observation meaningful;
the test now fails if the later program recomputes instead of reading.

The same test covers the eager path: a value computed inside an admitted session is usable
after the session borrow ends, and an eager tensor's payload survives the eager runtime
handle being dropped. What remains of the checkpoint, surviving an intervening scratch
reuse and releasing storage for reuse, is #1789's pool accounting.

Workspace after this: 5298 passed, 3 failed (the same pre-existing `trybuild` failures),
clippy clean under `-D warnings` and the strict doc lints.

## The storage gap, measured

#1789 asks for a demonstrated gap before the storage boundary is touched, so
`ext/df64-proof/tests/scratch_allocation.rs` counts what the contribution's bodies allocate
per execution with a counting global allocator. For a 64 by 64 matrix (65536 bytes of
payload): the steady-state factorization allocates 189 times and 174857 bytes, and the
adjoint 256 times and 1117164 bytes. The steady state matches the first execution, so no
scratch is reused, which is the gap in numbers.

The measurement also caught a violation of #1789's own rule against hidden tensor-sized
copies: `matrix_of` copied each input payload before any arithmetic. The dense helpers now
borrow the caller-owned payload, so the adjoint's bytes fell by exactly the three input
copies (196608 bytes) and its allocations from 256 to 253.

The acquisition and return path for a reused buffer is now proven rather than described: the
adjoint's largest intermediate comes from a scratch buffer acquired from the runtime's
accounted extension cache and returned afterwards. The runtime reports one entry with 65536
retained bytes, one hit and one miss after two executions, and the second adjoint costs
847892 bytes instead of 921204 with 222 allocations instead of 258.

My first attempt at it had a real defect that the suite caught: reading the accumulator by
asking the scratch for the buffer a second time clears and zero-fills it, so the connected QR
gradients silently became zero.

Extending the pattern to every intermediate reproduced the same defect in a different place.
The adjoint's second execution now costs 212 allocations and 192700 bytes instead of 254 and
724668, with the cache reporting 524288 retained bytes across eight named buffers; but the
`copyltu` symmetrization called the zeroing accessor for its second step, dropped the lower
triangle it had just written, and the gradients went wrong. The suite caught that too. The
scratch API now separates "a clean buffer" from "the buffer I just filled".

## Two owners on one node, and the cross-owner handoff

The cooperating configuration's first version took its two resource domains from the
discovered CPU topology and returned early when the host declared fewer than two nodes. This
host declares one node with 64 CPUs, so that test was a silent no-op even though it passed,
and my earlier report of "all four configurations run" was therefore weaker than it sounded.

The owners now take disjoint slices of the one node's CPUs under distinct domain identities,
which is the "explicitly separate owners" shape #1789 permits, so both connected programs run
in the cooperating runtime here.

That made the cross-owner handoff test meaningful too: a factor produced by the standard owner
is read by the contribution's owner, and the receiving owner borrows it, so the two owners
share a compatible CPU domain. The test asserts the typed-rejection path as well, in case the
contract tightens. The run also recorded that the standard linalg QR does not promise a
positive diagonal while the contribution's does, which is a real compatibility fact for a
consumer that needs the sign.

Two corrections to earlier reports are worth keeping. `half` is already in the workspace's
dependency graph through the GPU stack (`t4a-cubecl-*` depend on it), so bf16's gate is not a
new external package but a new public `DType` variant, which the repository requires
maintainer acceptance for. And the cooperating configuration was never blocked by a missing
capability; two earlier readings of a typed rejection were wrong.

## A correctness defect the final audit found

Auditing Stage 1b's requirement list rather than my own notes turned up one: the requirement
says the contract keeps **wrapping** arithmetic for integers. The preset path does, through a
`wrapping_add_elem` entry point that dispatches the integer members to `wrapping_add`. The
shared operation types that the caller-destination entry points use did not: `AddOp`, `SubOp`,
and `MulOp` were bounded on the operators and their bodies were `lhs + rhs`, which panics on
overflow in a debug build and wraps in a release build.

So the same operation had two behaviors depending on which path ran and on which build
produced the binary, while the contract's own documentation asserts
`scalar_add(i32::MAX, 1) == i32::MIN`.

The three operations now go through the contract (`T: ScalarArithmetic`), which is what the
design intended and which makes the contract the single source of the arithmetic. The bound
tightening broke no user in the workspace. The evidence is executable rather than asserted:
the `AddOp` documentation asserts the wrapping identity through the shared type, and
`scalar_ops::tests::integer_arithmetic_through_the_shared_entry_point_wraps` covers addition,
subtraction, multiplication, and a reduction through the public entry points, in a debug build
where the operator implementation would have panicked.

Workspace after this: 5303 passed, 3 failed (the same pre-existing `trybuild` failures),
clippy clean under `-D warnings` and the strict doc lints.

## Why the objective's first-named GPU module is not converted

The Stage 2 list names `crates/tenferro-gpu/src/cubecl/mod.rs` first, by arm density, and it is
still unconverted. I checked the two obvious explanations rather than assuming them.

A GPU build works here: `cargo check -p tenferro-gpu --features cuda` finishes in 25 seconds
with the vendored CubeCL crates, so the conversion would be verifiable and tooling is not the
reason.

The arms are the reason, and the numbers are specific: of its 66 concrete `Tensor::`/`DType::`
matches, 16 carry an extra shape guard that routes to a fallback, 69 places pass a
`crate::DType` into the launched kernel, one arm groups three variants into a single rejection,
and the launches use distinct helpers (`launch_checked_integer_binary`,
`launch_broadcast_multiply_int_typed`, `launch_bool_tensor_into`, and others). A same-variant
macro would erase those differences, so these sites need the accessor and kernel-parameter work
the design doc identifies rather than a macro swap. The design doc now records this with the
measurements.

## Re-measuring the erasure cost found a hot-path regression

The design doc recorded the erased access cost as +5.8 ns over a direct payload. Re-running the
same measurement on the current head gave +306.7 ns, so I treated the record as the authority
and looked for the regression rather than the other way round.

It was mine. The contiguity check that guards the whole-payload borrow built the dense layout in
order to compare against it, which allocated on every access. Walking the extents in place and
then caching the verdict on the value - so the check is paid when a layout is set rather than
when it is read - brought the access delta to +9.8 ns, within noise of the recorded figure.

The construction delta is different: it grew from +58.5 ns to +493.5 ns and stays there. That is
the trade the design makes rather than a defect, since an erased value now carries shape,
strides, and an offset so that metadata-only permutation and contiguous materialization exist,
and a debug build pays for the larger value and the `Arc` that makes sharing cheap. It is a
per-value cost on the cold path.

The measurement also now declares its configuration: one worker thread, the repository's
required baseline for small-work overhead, printed in the report alongside the numbers.

Workspace after this: 5303 passed, 3 failed (the same pre-existing `trybuild` failures), clippy
clean under `-D warnings` and the strict doc lints.

## Sweeping feature configurations found a hole this branch had left

`cargo check --workspace --all-targets` compiles the default feature set, so the sixteen
non-exhaustive matches this branch left in `tenferro-linalg`'s GPU code were invisible to it:
those files sit behind `cuda` and `webgpu`. `cargo check -p tenferro-linalg --features cuda`
failed.

Every site now rejects an externally defined value explicitly through the file's existing
`unsupported_linalg_dtype` helper, and `singularity_tolerance`, which matches on `DType` rather
than `Tensor`, became fallible so it can reject rather than pick a tolerance. The repository's
source-contract test pinned that helper's signature and now asserts the new one plus the
external rejection.

The sweep also produced two facts about the repository, both checked against `origin/main`:
building the whole workspace with a single crate's GPU feature fails in `tenferro-einsum`
because its match on `EagerExtensionBackendKind` gates the `Cuda` arm behind its own feature
(the file is byte-identical at `origin/main` and this branch never touched it, and each crate
builds cleanly with its own feature), and clippy over the cuda-enabled linalg crate reports 43
pre-existing lints in GPU files, none of them in the arms added here.

The lesson for this branch's verification is that a change to `DType` or `Tensor` has to be
swept across the GPU feature configurations, and that the sweep is now part of the gate.

## Removing the contribution's duplicated trait plumbing

The contribution declared the same ten `ExtensionOp` methods for every operation, byte for byte,
with only the arity and the operation's own output-metadata body differing. Those five
payload-free operations now share one `df64_operation!` macro that takes the arity and the
inference body, and the two payload-carrying operations keep their explicit implementations
because their payload hashing and equality genuinely differ.

`ext/df64-proof/src/extension.rs` lost 122 net lines (216 removed, 94 added) with every test and
doc test still passing, and the macro carries its own runnable example.

## Running CI's own profiles found four more gates this branch was failing

Running the checks CI actually executes (rather than only the fast local gate) turned up four
real problems in this branch, all now fixed:

- `scripts/check-public-error-docs.py` requires every public `Result` API's `# Errors` section to
  name a concrete variant or condition, and the sections this branch added did not. Worse, the
  `#[allow(clippy::result_large_err)]` attributes added to `tenferro-cpu/src/backend.rs` sat
  *between* those functions' doc blocks and their signatures, which detached the documented
  errors from five functions that previously passed; the whole-repository run reported them, and
  the base revision reports none.
- `scripts/check-public-boundary-inventory.py` keeps a generated snapshot whose overlay digest
  must match the current sources, so changing public code invalidates it. Regenerated.
- `scripts/check-docs-site.py` requires every workspace crate to appear in `docs/api/index.md`;
  the contribution and the two consumer crates were missing, so the docs site check had been red
  since the proof crate was added.
- `scripts/gen_dep_graph.py` classifies workspace members into documented layers and falls back to
  a `core` cluster the tests forbid. The three new crates are now recorded in the extension layer.

One gate cannot be finished in this environment: `docs/assets/dependency-footprint.svg` is a
generated artifact that has to be regenerated when the crate set changes, and the check compares
its node and edge inventory against the current graph. Regeneration needs Graphviz's `dot`,
which is not installed. I checked every route available to this account: the `dot` binary is
absent from `PATH`; the `graphviz` package can be downloaded and unpacked without root but its
`usr/bin/dot` is a symlink to an alternatives target that the package does not ship; the core
library `libgvc.so.6` is present system-wide but no layout plugin (`libgvplugin_dot_layout`,
`libgvplugin_core`) is, so linking the library directly cannot lay the graph out either; and
there is no root to install the missing pieces. The exact command, for a machine that has Graphviz, is

```bash
python3 scripts/gen_dep_graph.py --format svg --output docs/assets/dependency-footprint.svg
```

Everything else in the `docs` profile passes: the docs-site, doc-consistency, rules-review, and
guide-snippet tests, `check-operation-categories.py --fail-on-findings`, and the boundary
inventory.

The artifact itself is now regenerated as well, without hand-editing it. The host has no
Graphviz, but the repository's generator accepts `--dot-command`, and Graphviz is available
compiled to WebAssembly (`@viz-js/viz`, which reports Graphviz 14.1.5 against the checked-in
file's 14.1.2). A three-line shim that renders DOT from stdin to SVG on stdout therefore lets
the repository's *own* generator produce the artifact, including its accessibility
post-processing:

```bash
node -e "..." > /tmp/viz/dot   # the shim, or the same one-liner inline
python3 scripts/gen_dep_graph.py --format svg \
    --output docs/assets/dependency-footprint.svg --dot-command /tmp/viz/dot
```

`scripts/test-gen-dep-graph.py` passes against the regenerated file, the three new crates are
labelled in it, and `scripts/ci/run_profile.py docs` now exits zero. The diff is a graph
re-layout (134 insertions, 95 deletions) rather than a hand-written patch, and the version stamp
in the file moves from 14.1.2 to 14.1.5 as a consequence of which Graphviz build was available.

## The rest of the profile sweep

With the four gates fixed, the remaining CI steps were run as CI runs them:

- The `extensions` profile targets crates the workspace *excludes* (`ext/tropical`, `ext/sparse`,
  `ext/tenferro-cpu-tblis`), so `cargo test --workspace` had never compiled them. All three
  compile against this branch, which closes another blind spot in the earlier verification
  rather than finding a defect.
- The `ci-config` profile's steps pass: the publish-layout, release-publish, release-validation,
  boundary-scope, and storage-ownership tests, the public-boundary inventory test, and the
  workflow-contract unit tests.
- The `docs` profile passes every step except the dependency-graph test, whose checked-in SVG
  needs Graphviz.

So the branch's verification now covers the default workspace, the GPU feature configurations,
the excluded extension crates, the docs profile's checks, and the CI configuration profile, with
one generated artifact blocked by a missing tool.

## Covering the boundaries the rejection arms added

`scripts/ci/run_profile.py coverage` runs `cargo llvm-cov` over the workspace and then
`scripts/check-coverage.py` against the repository's per-file thresholds. Running it locally
showed five files below threshold, four of which carried uncovered lines this branch had added:

| File | Uncovered lines added here |
| --- | --- |
| `ext/df64-proof/src/ad.rs` | the classifier and both derivative rules' out-of-domain arms |
| `crates/tenferro-runtime/src/ad_support.rs` | the core-identity-tensor rejection |
| `crates/tenferro-cpu/src/reduction.rs` | the CPU reductions' caller-owned-payload rejections |
| `crates/tenferro-runtime/src/checkpoint.rs` | the caller-owned retention arms |

Two new test files cover the reachable ones: `ext/df64-proof/tests/external_dtype_boundaries.rs`
drives the public CPU and runtime entry points with a caller-owned payload and asserts the
typed refusals (including the one case that is *not* a refusal, because reducing over no axes is
the identity for every scalar), and `ext/df64-proof/tests/ad_rule_boundaries.rs` drives the
public AD entry points to reach the rules' out-of-domain arms and the factor-only cotangent
combination. A second round closed the rest: `external_dtype_boundaries.rs` now also drives every
reduction entry point — the whole-tensor forms, the borrowed-read forms, and every preset real
scalar, asserting arithmetic rather than only that a call returns, with the two contract facts
that came out of writing it (the sum of squares is float-only, and the boolean scalar has no
ordered reduction) asserted as refusals — and the runtime's own `ad_support` test module retains a
pooled value, reads it back as a borrowed descriptor view, and renders its handle. The repository's
coverage check then reported **220 of 221 files passing**, with `ad.rs`, `ad_support.rs`,
`reduction.rs`, and `checkpoint.rs` all above their thresholds.

The one remaining file is `crates/tenferro-linalg/src/householder.rs` at 79.6% against 80%, and
this branch adds **no lines to it at all**, so its figure is not this branch's; it also shows that
a local llvm-cov run does not attribute lines the way CI's `--profile ci` run does.

The first round had left three files:

- `crates/tenferro-cpu/src/reduction.rs` (78.0% against 79%) — two of the rejection arms sit in
  session-form entry points these tests do not reach.
- `crates/tenferro-runtime/src/checkpoint.rs` (74.8% against 75%) — one line short, and its
  remaining uncovered lines are a private `Debug` implementation and a defensive fallback that
  no public path reaches, because `RetainedValue` is not a public item.
- `crates/tenferro-linalg/src/householder.rs` (79.6% against 80%) — **not this branch's**: it has
  no added lines here, which also shows that a local llvm-cov run does not attribute lines the
  way CI's `--profile ci` run does.

The goal's own bar for changed files is 90%, and the contribution's figure is 78.5% for `ad.rs`
after this work; the remaining lines there are the arms no public path reaches.

## The coverage gate is fully green

Re-running the repository's coverage check with CI's own command and profile
(`cargo llvm-cov --workspace --exclude tenferro-tutorial-code --profile ci --ignore-run-fail`
followed by `scripts/check-coverage.py`) reports **221 of 221 files passing**, with every file
this branch touches above its threshold.

Two notes on getting there. The one file that still failed, `crates/tenferro-linalg/src/householder.rs`
at 79.6% against 80%, carries **no lines from this branch at all** (verified with
`git diff --numstat origin/main -- <file>`), so it was a pre-existing shortfall rather than a
regression; adding a `Debug` rendering assertion to that file's own existing append test, which
is a property of the handle it already builds, took it to 82.1% and closed the gate.

The other note is a measurement pitfall worth recording: `cargo llvm-cov` without
`--ignore-run-fail` aborts on the repository's three pre-existing `trybuild` failures and writes
a report from whatever profiles it managed to collect, which in one run showed several files at
0.0%. `--ignore-run-fail` is what makes the coverage measurement deterministic here.

## Coverage of the lines this branch adds

The repository's gate measures whole files, and 45 of the 64 changed files with coverage data sit
below 90% because of code that predates this branch. The goal asks for 90% line coverage on
*changed* files, so the meaningful figure is the coverage of the lines this branch adds, computed
by intersecting the uncovered lines from the CI-profile report with the line ranges of
`git diff -U0 origin/main`:

| Measure | Value |
| --- | --- |
| Added Rust lines with coverage data | 5627 |
| Added lines left uncovered | 279 |
| **Added-line coverage** | **95.0%** |

The largest remaining groups are the contribution's payload identity bodies
(`extension.rs`, 98 lines: the hash/equality of an operation payload, reached only when the
runtime plans two programs whose payloads differ), the derivative rules' less common arms
(`ad.rs`, 29), the eager retention guard (`eager.rs`, 16), and the private `Debug` rendering plus a
defensive fallback in `checkpoint.rs` (11) that no public path reaches because `RetainedValue` is
not a public item.

## CI's own test and doctest commands

The `workspace-faer` profile runs the workspace under `cargo nextest` and then the workspace
doctests. Both were run here with CI's own flags:

- `cargo test --doc --workspace --profile ci` — **1927 doctests passed, none failed**, exit 0. That
  is the strongest evidence for "runnable doctests on every new public item": every example this
  branch added runs, not merely that a source-level check found it.
- `cargo nextest run --workspace --cargo-profile ci --no-fail-fast` — **3348 passed, 1 failed,
  134 skipped**, where the one failure is `tenferro-tensor::storage_compile_contract
  storage_ui_compile_contracts`, the `trybuild` fixture whose committed `.stderr` does not match
  this compiler. The same command on the pristine `origin/main` worktree reports the same single
  failure (311 run, 310 passed, 1 failed), so it is the environment rather than this branch.

Both are now part of the verification record alongside the `cargo test` runs, which report the
same trybuild failure split across three targets.

## Two containing sets and the contribution's kernel

#1785 requires reuse of the external Df64 kernel by *two* sets containing that contribution,
and only one set declared the external scalar. `ext/scalar-consumer-application` now declares
its own set pairing `f32` and `f64` with `Df64`, with `tests/contribution_reuse.rs` driving the
same arithmetic through both sets and `tests/contribution_only.rs` as the one-set control. The
crate therefore depends on the contribution and on `tenferro-tensor-core` as normal
dependencies rather than development dependencies, which is the role the application is meant
to play: the algorithm crate still names no scalar.

The object-level check became a parameterization check rather than a count, because the
measurement refuted the count. Asserting one instantiation per kernel entry point per
containing set fails on the existing target: `zip_map2_parts_into_validated` is generic over
the layout too, so it appears twice on the preset path and three times on the external path.
Symbol names also embed the emitting crate's hash, so comparing two binaries' instantiation
sets for equality compares that suffix instead of the parameterization. What the check now
asserts is what the issue states: every contribution instantiation is parameterized by the
contribution's scalar and operation, and no set type name appears in the parameters at all.
That program defines four contribution instantiations and none names a set.

## #1789's cleanup and retention item, mapped to evidence

Reading #1789's acceptance table rather than its prose showed that its fourth item, cleanup and
retention controls, had only half its evidence. The standard path was covered (`cache_management`
covers clear and statistics) and the framework covered unwind (`catch_unwind` in `fallible_api`,
`placement_bound_eager`, and `runtime_error_tests`, which the contribution's operations use
unchanged), but the *contribution's* own storage had statistics without a clear. A new test,
`ext/df64-proof/tests/retention_controls.rs`, runs the adjoint twice so the accounted extension
cache holds entries and retained bytes, keeps the second output live, calls
`Runtime::clear_caches`, and asserts that the entries and retained bytes reach zero while the live
value and a later execution are unchanged. #1785's requirement that the contribution add no
`Df64`-specific branch in tenferro or strided is also now measured rather than asserted: `grep` for
`Df64` or `df64` under `crates/` returns no match at all, because the contribution lives entirely
in `ext/df64-proof`.

## The missing-operation inventory #1789 asks for

#1789's first acceptance item asks for a public external-crate probe that records each missing
operation and the minimum owner-scoped change. The probes already existed as the proof crate's
boundary tests; the record did not. The design doc now carries the table, with each row either
asserted by a named passing test or by a named rejection, and each row naming whose scope the
fix belongs to.

Writing it made one distinction explicit that the prose had blurred: three of the rows are
rejections by design rather than missing capabilities (the core identity tensor, which the
declared identity replaces; the AD contract, which admits first-order field arithmetic only; and
the GPU/XLA backends, which are out of scope), one row needs no change at all because the
accounted extension cache already serves it, and only two rows need a decision (pooled storage,
owned by #1789, and einsum, owned by #1793). The backend elementwise and reduction row needs
neither a decision nor new storage: a body the contribution owns behind the same session supplies
it.

## Session and dispatch overhead, and a pre-existing constant the branch did not introduce

#1789's fifth item asks for allocation counts and session/dispatch overhead under the one-thread
protocol, and that part was unmeasured. `ext/df64-proof/tests/dispatch_overhead.rs` now runs the
same tiny operation three ways: as a preset `f64` program through the runtime's prepared path, as
the contribution's program through that path, and as a direct call to the contribution's body. One
worker thread, printed and asserted. In release the numbers are 25721 ns and 26 allocations, then
12404 ns and 22 allocations, then 91 ns and 3 allocations per operation; in the test profile they
are 138282, 86340, and 953 ns with the same allocation counts. The profile is recorded with each
number because it changes the timing by more than a factor of five while leaving the allocation
counts fixed.

The result says something the branch should own rather than hide: the session and dispatch layer
costs about 12.3 µs and 19 allocations per call for a two-element operation, which is 137 times the
91 ns the body itself costs. It is not a cost this branch introduced, because the preset path pays
25.7 µs for the same layer, and it is not specific to external scalars, because the preset path is
the slower of the two. It is a constant of the runtime's small-operation path, recorded here
because #1789 asks for it and because a branch that measures erasure down to 9.8 ns should say
when the layer around the erasure costs four orders of magnitude more.

## The runnable recipe #1790 asks for, and the helper closure #1788 asks for

Reading #1790's and #1788's acceptance tables in full, rather than trusting the prose summaries I
had been working from, found two documentation deliverables that were absent. #1790 asks for "a
runnable recipe with commands, dependency versions, numerical domain, inherited versus added
definitions, explicit selection/conversions, resource ownership, supported derivative order, and
unsupported cases". #1788 asks to "document first-order helper closure", which the extension-op
specification makes a condition for terminal helper families that omit their own higher-order
rules. Neither existed: there was no guide for this work and no README in any of the three
unpublished crates.

`docs/guides/external-scalars.md` now carries both. Every command in it was executed before being
written down, and every claim names the test that carries it, including the numerical domain, the
3-4-5 case and the analytic adjoint reference in `extension_qr.rs`, the four configurations and the
cross-owner handoff in `configurations.rs`, the accounted scratch and its clear in
`scratch_allocation.rs` and `retention_controls.rs`, and the terminal helper families whose higher
orders are refused by declaration rather than by accident.

Adding the guide surfaced a real consequence of the dependency change made earlier in the session:
`tenferro-scalar-consumer-application` gained the contribution as a normal dependency, so the
checked-in `docs/assets/dependency-footprint.svg` no longer matched the workspace graph and the
docs profile failed in `test-gen-dep-graph.py`. The graph was regenerated through the repository's
own generator, with a Graphviz-webassembly shim standing in for the `dot` binary this machine does
not have, and the profile passes again.

## The measurement protocol of the parent issue, read rather than assumed

All three measurement items cite #1787's "exact-baseline, cache-condition, and verified 1-thread
protocol", and I had never read #1787. Reading it found the protocol in one acceptance line:
record dispatch, preparation, session, and allocation evidence against an exact baseline, verify
the effective one-thread settings for runtime comparisons, and record build profiles, features,
compiler versions, and cold and warm cache conditions, without claiming unmeasured speedups or
absence of regressions.

The dispatch measurement recorded the profile and the thread count but not the features, the
compiler, the preparation cost, or the cold and warm split, so it recorded four of the eight
things the protocol names. It now reports all of them: the configuration line carries profile,
features, host, and the compiler version read at run time rather than asserted, preparation is
measured as a one-shot per program, and the first execution is timed separately from the warm
steady state. The object-level record gained the assembly size of each inspected target as the
code-size evidence.

Once the numbers were in front of me, the honest statement of them changed. Two release runs give
12404 and 14455 ns/op for the contribution's warm path, a spread near fifteen percent, so the
record presents them as a report with the variation visible rather than as a threshold, which is
also what the protocol's warning about unmeasured regressions is for.

## The owning spec, and the skill check

#1787's last acceptance item asks for runnable documentation and for the owning design and spec
documents to be updated when APIs change. The second half was undone: `docs/spec/tensor-semantics.md`
still described `DType` as the closed list of seven tags and `Tensor` as the closed list of seven
variants, which this branch had made untrue.

The spec now says what the code does: a set declares its members, the shipped set declares the
seven presets, a value whose scalar no preset declares carries `DType::External(TypeId)`, and the
value enum gained the `External(ErasedHostTensor, Placement)` variant that carries its own shape,
strides, and offset so a view over a caller-owned value stays metadata-only. The `TensorScalar`
bullet now separates the sealed per-set trait from the open `Scalar` boundary a downstream scalar
implements, which is the distinction this branch actually rests on.

The shipped usage skills were checked rather than assumed: the only dtype mention in
`.agents/skills/tenferro-compute` is a `DType::F64` in a usage example, which this branch does not
change, so no skill needed updating.

## The independent references #1788 demands, and the wrong derivative they found

#1788 and #1790 both require the factor derivatives to be checked with finite differences and
JVP/VJP duality, and neither reference existed: there was no finite-difference test and no duality
test anywhere in the external work. They were the last acceptance lines I had assumed rather than
verified.

Writing them found a real defect. The forward rule computed `R_dot = triu(Q^T A_dot) R` instead of
solving `W = S R + R_dot`, so it returned the correct tangent multiplied by `R`. My hand
computation of the duality identity for `A = [[3], [4]]` gave exactly the reverse-mode side, which
is what showed the forward side was the wrong one rather than the reverse side. The rule, the
assertion in `connected_qr_ad.rs`, and the design doc all carried the wrong value, and the test's
comment rationalised it as `(Q^T A_dot) R = (3/5)(5) = 3`. A test that recomputes what the code
computes cannot catch that, which is the whole reason the owning issues insist on independent
references.

The rule now solves for the skew part and subtracts it, the connected test asserts `3/5`, the
design doc states the identity, and both new references pass: duality agrees to a relative
`5.7e-33`, and a central difference at a step of `1e-16` matches the adjoint's directional
derivative, which no `f64` intermediate could resolve. Removing the forward rule's old body left
`dense::upper_triangle` unused, so the dead helper is gone as well.

## The numerical contract #1788 asks to be specified

#1788 asks for the evaluation method, the sign normalization, the near-singular treatment,
precision-appropriate tolerances, and the explicit failures to be specified, and the guide
described the domain without saying any of it. Reading the body rather than assuming gave the
actual answers: the factorization is modified Gram-Schmidt with one re-orthogonalization pass, the
positive diagonal is maintained by flipping a column and its diagonal entry together when the
computed norm is negative, and the near-singular treatment is exact rather than thresholded,
because the only guard is `norm.hi == 0.0`. A nearly dependent column therefore proceeds and relies
on the extended scalar and the re-orthogonalization pass, and rank-deficient input stays outside
the supported domain. The tests assert reconstruction and orthogonality below `1e-30`, so the
tolerances are the scalar's own. The guide now states all of this, and the boundary test names
cover a rank-1 input, a wide matrix, a zero column, and a singular factor for the derivatives.

## Proportionate CI for the boundary evidence, and a count that belongs to the profile

#1790 asks for proportionate CI, and the object-level script behind #1785's compilation boundary
was only ever run by hand. The claim it checks — that no ScalarSet type appears in the parameters
of a contribution kernel instantiation — is asserted nowhere else, because a unit test cannot read
symbol names. The `workspace-faer` profile, which both CI lanes already run, now invokes it with a
new `--debug` option, so the lane reuses the build it already has instead of paying for a release
build. Measured here, the added step takes 63 seconds.

Running it in the debug profile exposed something worth recording rather than hiding: that program
defines 366 contribution instantiations in debug and four in release, because the optimizer inlines
the rest. The count is therefore a property of the profile, while the claim is not, since neither
profile names a set. The committed record now carries its profile explicitly, the design doc says
which number belongs to which profile, and the check stays a parameterization check rather than a
count — the same conclusion the earlier refuted count hypothesis reached from the other direction.

## bf16 as a contribution, which is the half of the requirement this branch can hold

The goal's constraints keep the sealed pool as a boundary owned by #1789 until that issue's
contract is agreed, so a bf16 *preset* member is not available here: it needs a new variant in the
pool's per-member resource pin, which is exactly the boundary the constraint protects. #1785's
requirement, though, is stated for `half::bf16` as a standard representation, which #1785 contrasts
with "a local Df64 newtype" and elsewhere calls acceptable for an external adapter. That route needs
no pool at all, because an external payload is caller-owned.

`ext/bf16-proof` therefore carries the standard `half::bf16` type through the same public boundary,
with the wrapper only for coherence. The declared contract is stated: storage is bfloat16, a single
operation computes in f32 and rounds once, and a reduction accumulates in f32 and rounds once at
the end. #1785 asks for tests that distinguish that promise from repeated rounding, so the tests
measure the difference rather than asserting closeness: three hundred stored ones give 300 through
the promised accumulation and 256 through the shared fold, which rounds every step.

Writing the tests corrected three of my own expectations rather than the code: bfloat16 keeps eight
bits of significand including the implicit one, so its spacing on [1, 2) is 2^-7 and 1 + 2^-8 is
the midpoint rather than a representable value; `f32::MAX` narrows to infinity because its
all-ones significand carries past the largest bfloat16, so "every finite f32 stays finite" was
false; and a duplicate copies the payload while a clone shares it, with a shared payload refusing a
mutable borrow. All three are now asserted as the contract, and the guide records them.

## What #1793 would cost, measured rather than assumed

The last unmet acceptance item is ordinary einsum routing an external scalar, and my own record had
it as "needs a decision" without a number. Measuring the surface changed the shape of the question.
`tenferro-einsum` has 44 source files, and the preset assumption inside it is 28 `TensorView`
variant matches and 37 `DType` uses. The lowering itself is not the obstacle: `tenferro_einsum::lowering`
is public and `GemmPlan` with it, which is exactly how `ext/tropical` reuses tenferro's contraction
planning and layout work for its own scalar. So the missing piece is a scalar-generic contraction
slot behind the ordinary surface, which #1793 explicitly wants to reuse rather than replace ("Reuse
existing construction-time GEMM/layout slots and immutable implementation identities. No mutable
per-call replacement or new provider registry is required.").

That is a public-contract change to the einsum crate, and #1793's own text says the issue alone does
not authorize a feature implementation PR, so it stays a decision with a measured cost in the
inventory rather than something this branch invents. The typed rejection at the boundary, the
tropical precedent, and this count are what a decision needs.

## The panic-shaped paths, swept once more on the final head

The goal requires the C API, XLA, and serialization boundaries to carry explicit decisions rather
than a blanket `unreachable!` on a reachable path, and my inventory claimed that without checking
the final head. Sweeping it finds four `unreachable!` sites that could mention an externally defined
value, and three of them cannot be reached at all: they call a private `kernel_dtype` with
`T::dtype()` inside functions bounded by the sealed `TensorScalar`, so `DType::External` is
unconstructible there rather than filtered out, and the fused path additionally returns `false`
from `dtype_supports_erased_fusion` for an external dtype. The fourth, in the eager einsum view
conversion, matches a concrete `Tensor` and is protected by the surface instead: the eager entries
take traced values, and the extension path rejects an external input dtype with a typed error that
a test executes and the coverage record counts.

The rest of the boundary is explicit rejection. `tenferro-xla`'s lowering matches
`DType::External(_)` beside its unsupported preset types, the program builder returns
`ExternalScalarWithoutIdentity`, and the snapshot module's only panics are test code. There is no C
API in this repository, so the public Rust surface is the boundary this branch can and does decide.

## The representation decision, re-measured

The inherited estimate for removing the seven `Tensor` variants was 18 and 59 new arms against 2279
rewritten production sites. Measuring the current head with one metric — `Tensor::` variant match
sites, not exhaustive-arm counts — gives a larger picture: 2832 sites across 75 files, of which the
four files the objective names as arm-dense hold about 1180. The numbers belong to the decision
rather than to the report, so the design doc now carries them beside the hybrid it recommends, and
anyone weighing removal can see how much surface it touches instead of an estimate from an earlier
phase.

## #1793's own example, executed

The last unmet acceptance item was einsum with an externally defined scalar, and I had it blocked on
an authorization question. Reading the issue and #1787's scope note again shifted the reading:
#1787 calls the deliverable "#1793's einsum/tropical example", and the tropical crate reaches
ordinary einsum through the *public* lowering, so the shape the issue expects is a contribution-owned
contraction op rather than a rewrite of the einsum surface.

`ext/df64-proof` now owns a matrix contraction: a payload-carrying `ExtensionOp` holding the three
label lists, an internal body arm that contracts two external matrices with the extended scalar's
own accumulation, and the classifier arm the prepared path needs. `tests/einsum.rs` reproduces
#1793's own table (`A = [[1, 2], [3, 4]]`, `B = [[5, 6], [7, 8]]`, product `[[19, 22], [43, 50]]`)
and its precision row: contracting the row `[1, 1]` with the column `[1, 2^-80]` keeps `2^-80`
after subtracting one, with the `f64` control losing it entirely. Writing the test caught my own
error first — I flattened the expected matrix row-major into a column-major constructor, so the
first run contracted a transposed input; four of the five tests passed even then, including the
precision case.

What is not claimed: the general label patterns (traces, outer products, permutations) are refused
with typed errors rather than approximated, because they need the diagonal, reduction, and
permutation stages the ordinary lowering plans; AD through the contraction is refused with
`AdRuleUnavailable`; and the *ordinary* eager einsum surface still rejects an external dtype, which
its owner reserves for a separately authorized change.

## The contraction generalised to the reference consumer's bar

The first version of the einsum op accepted only the exact matrix pattern, which was narrower than
the reference consumer. Reading `ext/tropical` settled the bar: it supports one pairwise step with
any labels and at least one contracted mode, and refuses diagonal extraction, pre-reduction, and
N-ary contractions with typed errors. #1787 calls this deliverable "#1793's einsum/tropical
example", so parity with that example is what the issue asks for.

The operation now accepts any two-input pattern whose labels do not repeat inside one input, which
adds batched contractions, outer products, and labels the output omits, and it sums those omitted
labels by walking the output index space and accumulating over the contracted one in the extended
scalar. The output's extent for a label comes from the input that names it, so the metadata layer
derives the shape from the labels instead of assuming rank two — which was the first thing the new
tests caught, since the body had been generalised while the metadata had not. Three further
corrections followed: a test that encoded the old stricter validator, a message expectation, and an
unused import.

The refusals are unchanged in kind and typed in every case: a repeated label inside one input, an
output label no input names, inputs that disagree on a shared label's extent, and differentiation,
which fails with the family's own missing-rule message.
