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
