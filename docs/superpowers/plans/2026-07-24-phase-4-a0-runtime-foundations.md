# Phase 4 A0 Runtime Foundations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement only the accepted P4-A0 identities, policy and normalized-key values, value-free input signatures, and finite specialization lattice in `tenferro-runtime`.

**Architecture:** Four owner-focused modules share one A0 error module and are re-exported through `tenferro_runtime::runtime` and the crate root. Each slice starts with all of its behavior tests, records the expected RED result before production changes, then implements the accepted API at commit `32d5703e46c37969c2f1123fd5f4ac026ce8945f` plus the reviewed local storage/layout-classification and retry-ownership/YAGNI amendment.

**Tech Stack:** Rust 2021, `Arc`, `NonZeroU64`, `TypeId`, existing `tenferro-tensor` metadata views, `thiserror`, module-local unit tests, integration tests, rustdoc doctests, and repository verification scripts.

---

## Authority, workflow, and stop boundary

The sole design authority is the accepted commit plus the reviewed tracked
local storage/layout-classification and retry-ownership/YAGNI amendment in:

```text
32d5703e46c37969c2f1123fd5f4ac026ce8945f:
docs/superpowers/specs/2026-07-24-phase-4-runtime-preparation-design.md
```

Use these exact accepted ranges rather than inventing parallel declarations:

```bash
sed -n \
  '65,182p;274,313p;337,519p;920,1298p;1619,1631p;1799,1823p' \
  docs/superpowers/specs/2026-07-24-phase-4-runtime-preparation-design.md
```

Rules for every task:

- Write and register every named behavior test before changing production code.
- Run the stated RED command. Confirm failure is caused by the first missing A0
  API or behavior, not a typo, broken fixture, or unrelated error.
- Add no further behavior test during GREEN. If a gap is discovered, stop,
  write the new test, observe RED, and only then resume production work.
- Implement every `pub` type and accessor in the task's API manifest with the
  exact accepted visibility and signature. Constructors used only by later
  runtime nodes remain `pub(crate)` exactly where the accepted design says so.
- Every public type and public function/method receives useful rustdoc with a
  compiling, runnable `# Examples` block. Every public `Result` also receives
  a concrete `# Errors` section. For an intentionally runtime-created-only
  type whose A0 constructor is crate-private, the example must compile and run
  a type-level use (such as an explicit `Debug` bound) without fabricating an
  unsupported public constructor; its value accessors are exercised in a
  module-local test through the accepted crate-private construction path.
- Every public type implements `Debug`. Integration tests compile-check every
  public type/accessor and execute values constructible through public APIs.
  Module-local tests execute accessors on runtime-created values that require
  crate-private constructors and cover bounded manual `Debug` where private
  state must be redacted.
- Do not add a public or private error, validation branch, helper, or test seam
  for a state unreachable through current supported APIs.
- Do not add `Runtime`, a runtime builder, allocation/atomics/exhaustion
  fixtures, snapshots, registrations, capability/provider SPI, prepared
  operations/programs, extension modules, cache owners/state/keys, scheduling,
  execution, resources, buffers, transfers, collectives, events, or Phase 5/6
  fixtures.
- Do not edit `scripts/test-doc-consistency.py`.

Before Task 1, confirm issue #1451 records two distinct acceptances: baseline
commit `32d5703e46c37969c2f1123fd5f4ac026ce8945f`, and the exact 40-character
SHA printed by `git rev-parse HEAD` immediately after committing this amended
design and this forced-added plan. The second issue record must name that SHA,
both file paths, and the A0-only scope. If either record is absent, stop before
production implementation; do not treat a branch name, local diff, plan digest,
or older issue comment as acceptance evidence.

Create only these implementation files:

```text
crates/tenferro-runtime/src/runtime/mod.rs
crates/tenferro-runtime/src/runtime/error.rs
crates/tenferro-runtime/src/runtime/identity.rs
crates/tenferro-runtime/src/runtime/policy.rs
crates/tenferro-runtime/src/runtime/signature.rs
crates/tenferro-runtime/src/runtime/specialization.rs
crates/tenferro-runtime/src/runtime/tests/mod.rs
crates/tenferro-runtime/src/runtime/tests/identity.rs
crates/tenferro-runtime/src/runtime/tests/policy.rs
crates/tenferro-runtime/src/runtime/tests/signature.rs
crates/tenferro-runtime/src/runtime/tests/specialization.rs
crates/tenferro-runtime/tests/integration/runtime_foundations.rs
```

Modify only:

```text
crates/tenferro-runtime/src/lib.rs
crates/tenferro-runtime/tests/integration.rs
```

## Task 1: Opaque identities

**Accepted API:** design lines 65-182 and 274-313.

**Public manifest:** `IdentityKind`, `IdentityError::{kind}`,
`RuntimeId`, `RuntimeEpoch`, `EngineId::{new,as_str}`,
`HardwareClassId::{new,as_str}`,
`RegistrationIdentity::{ordinal}`, and
`ExecutionContextIdentity::{of,type_name}`. Numeric construction/access remains
`pub(crate)` exactly as accepted; the shared string validator is `pub(super)`.

- [ ] **Step 1: Write all identity tests before production**

Add and register these integration tests:

```text
identity_types_are_public_debug_and_context_typed
identity_engine_and_hardware_ids_accept_the_exact_ascii_grammar
identity_engine_and_hardware_ids_reject_the_exact_ascii_grammar
identity_error_redacts_a_unique_rejected_value_and_has_no_source
identity_public_accessors_return_the_stored_values
```

The redaction test must use a unique nonempty rejected value such as
`"caller-SECRET.invalid"` only for the display/debug/source assertions. Test
`""` and `"a"` only as grammar failures; never use substring-redaction
assertions whose needle is empty or appears in the generic error message.

Add and register these module-local tests:

```text
opaque_nonzero_ids_round_trip_only_inside_the_crate
runtime_epoch_checked_next_stops_at_nonzero_max
registration_debug_exposes_ordinal_and_never_issuer
```

- [ ] **Step 2: Observe identity RED**

Run:

```bash
cargo test -p tenferro-runtime --test integration runtime_foundations::identity
cargo test -p tenferro-runtime --lib runtime::tests::identity
```

Expected: compilation fails on unresolved A0 identity imports/types. All named
behavior assertions must already be present before proceeding.

- [ ] **Step 3: Implement the minimal accepted identity slice**

Implement exactly the accepted representations, four `IdentityKind` variants,
ASCII grammar, redacted `IdentityError`, crate-private nonzero operations,
manual bounded `RegistrationIdentity` debug, and type identity. Add no
allocator, atomic, issuer factory, exhaustion error, raw public number API, or
later identity kind. Add compiling examples for every manifest item. Rustdoc
for `RuntimeId`, `RuntimeEpoch`, and `RegistrationIdentity` states that A0
creates them only inside the runtime and that B0 supplies the public creation
path; its runnable example demonstrates the supported type-level contract,
while module-local tests exercise runtime-created values and accessors.

- [ ] **Step 4: Verify identity GREEN**

Run:

```bash
cargo fmt --all --check
cargo test -p tenferro-runtime --test integration runtime_foundations::identity
cargo test -p tenferro-runtime --lib runtime::tests::identity
cargo test -p tenferro-runtime --doc runtime
python3 scripts/check-public-error-docs.py
git diff --check
```

Expected: all pass; invalid values are absent from error display/debug/source,
and every identity accessor and `Debug` contract is exercised.

- [ ] **Step 5: Commit boundary**

```bash
git add crates/tenferro-runtime/src/lib.rs \
  crates/tenferro-runtime/src/runtime \
  crates/tenferro-runtime/tests/integration.rs \
  crates/tenferro-runtime/tests/integration/runtime_foundations.rs
git commit -m "feat(runtime): add A0 runtime identities"
```

Stop if this commit contains any policy, signature, specialization, or
later-node type.

## Task 2: Policy values and normalized keys

**Accepted API:** design lines 337-519 and the complete
`PlacementConstraintError` declaration at lines 1619-1631.

**Public manifest:** `PlacementConstraintError`, `Determinism`,
`StorageClass::{new,as_str}`,
`LayoutClass::{new,as_str}`,
`ProgramPlacementConstraint::{any,new,allowed_engines,storage_class}`,
`ResolvedProgramPlacement::{engine_id,storage_class}`,
`CacheInFlightBehavior::default`,
`ExecutionPolicy::{new,determinism,hard_workspace_limit_bytes,planning_seed}`,
all nine accepted `PrepareOptions` constructor/builder/accessor methods,
`PrepareOptionsKey::{resolved_placement,hard_workspace_limit_bytes,planning_seed}`,
`ResolvedPlanningConfig::{resolve,determinism,hard_workspace_limit_bytes,planning_seed,hardware_class}`,
and all four accepted `ResolvedPlanningKey` accessors. Runtime-created
placement/key constructors remain `pub(crate)`.

- [ ] **Step 1: Write all policy tests before production**

Add and register:

```text
policy_storage_and_layout_ids_share_the_exact_ascii_grammar
policy_placement_constraint_reports_first_duplicate_positions
policy_placement_constraint_preserves_preference_order_and_accessors
policy_resolution_covers_both_determinism_values_all_seed_extremes_and_workspace_lattice
policy_prepare_options_none_inherits_and_some_zero_overrides
policy_prepare_options_builders_and_accessors_round_trip
policy_types_are_debug_and_all_public_accessors_are_callable
policy_normalized_option_key_compares_each_resolved_field
policy_normalized_planning_key_compares_each_resolved_field
policy_cache_in_flight_never_changes_normalized_key_identity
```

The normalized-key unit tests change exactly one of engine, storage class,
workspace, seed, determinism, and hardware class per assertion. They construct
runtime-created values only through crate-private constructors. No public key
constructor or raw-options-to-key conversion is introduced.

- [ ] **Step 2: Observe policy RED**

Run:

```bash
cargo test -p tenferro-runtime --test integration runtime_foundations::policy
cargo test -p tenferro-runtime --lib runtime::tests::policy
```

Expected: compilation fails because the policy types are absent. All listed
normalization assertions must already exist.

- [ ] **Step 3: Implement the minimal accepted policy slice**

Implement the complete public manifest and only these private projections:
`ResolvedProgramPlacement::new`, `PrepareOptionsKey::from_resolved`,
and `ResolvedPlanningKey::from_config`. Use resolved fields exactly as accepted.
`PrepareOptions` must not implement `Hash`; `cache_in_flight` must not enter
any key. Add compiling examples for every public manifest item. Rustdoc for
`ResolvedProgramPlacement`, `PrepareOptionsKey`, and `ResolvedPlanningKey`
states that A0 exposes no public constructor and B0 provides their runtime
creation path; its runnable example demonstrates the supported type-level
contract, while module-local tests exercise runtime-created values/accessors.

- [ ] **Step 4: Verify policy GREEN**

```bash
cargo fmt --all --check
cargo test -p tenferro-runtime --test integration runtime_foundations::policy
cargo test -p tenferro-runtime --lib runtime::tests::policy
cargo test -p tenferro-runtime --doc runtime
cargo check -p tenferro-runtime --all-targets
python3 scripts/check-public-error-docs.py
git diff --check
```

Expected: all pass and every accepted accessor is exercised.

- [ ] **Step 5: Commit boundary**

```bash
git add crates/tenferro-runtime/src/lib.rs \
  crates/tenferro-runtime/src/runtime \
  crates/tenferro-runtime/tests/integration/runtime_foundations.rs
git commit -m "feat(runtime): add A0 policy and normalized keys"
```

Stop if a runtime builder, placement resolver, cache, provider, or prepared
program appears.

## Task 3: Value-free input signatures and truthful alignment

**Test authority:** the full `## Input signatures and finite specialization`
section at design lines 920-1298. Task 3 owns only signature test registration
and RED evidence; Task 4 is the sole production owner of every signature and
specialization declaration.

**Public manifest:** `InputSignatureError`, the A0 `PrepareError::InputSignature`
arm, `InputSignature::{from_reads,new,entries}`, and
`InputSignatureEntry::{new,dtype,shape,placement,layout_class,strides,alignment_log2}`.
Entries retain `Placement` but have no redundant `StorageClass` field.

- [ ] **Step 1: Write all signature tests before production**

Add and register:

```text
signature_entry_rejects_shape_stride_rank_mismatch
signature_entry_rejects_out_of_lattice_alignment
signature_aggregate_is_infallible_and_distinguishes_unknown_from_one_byte
signature_from_reads_copies_only_dtype_shape_strides_placement_layout_and_alignment
signature_nonzero_host_offset_uses_the_actual_logical_pointer_alignment
signature_empty_host_read_uses_type_alignment_without_reading_an_element
signature_backend_read_records_unknown_alignment_and_retains_no_buffer
signature_compact_and_strided_reads_use_exact_layout_classes
signature_tensor_metadata_mapping_source_contract
signature_types_are_debug_and_all_public_accessors_are_callable
signature_error_sources_remain_typed
```

Use the existing public backend buffer API for the weak-lifetime assertion.
`TensorRead::is_col_major_contiguous()` documents a typed metadata-overflow
failure, but A0 has no current valid public tensor/read construction that
reaches it. Preserve the typed `TensorMetadata` mapping and test its source
contract: the module-local named test must inspect the mapping implementation
and prove both `strides()` and `is_col_major_contiguous()` attach their original
input index to `InputSignatureError::TensorMetadata`, never an entry-validation
error. Do not add a fake metadata provider, injectable classifier, or malformed
construction merely to force the branch. Replace or supplement this
source-contract test with a behavioral mapping test only when the owning tensor
API supplies a supported fixture.
Do not add a malformed view, address-overflow fixture, declaration-alignment
fixture, or test-only signature constructor.

- [ ] **Step 2: Observe signature RED**

```bash
cargo test -p tenferro-runtime --test integration runtime_foundations::signature
cargo test -p tenferro-runtime --lib runtime::tests::signature
```

Expected: compilation fails because signature types are absent. The pointer,
empty-host, backend-unknown, lifetime, and source assertions must already be
present.

- [ ] **Step 3: Write all specialization tests before production**

Add and register every specialization test named in Task 4, including the
exact built-in/`Other` storage mapping, non-storage placement, partial-order,
finite-chain, and public-surface tests. The UTF-8 hex test uses fixed empty,
ASCII, and non-ASCII payloads and asserts distinct encodings for distinct
payloads; it does not add a decoder, reverse API, or production test helper.

- [ ] **Step 4: Observe the atomic signature-and-specialization RED**

```bash
cargo test -p tenferro-runtime --test integration runtime_foundations::signature
cargo test -p tenferro-runtime --test integration runtime_foundations::specialization
cargo test -p tenferro-runtime --lib runtime::tests::signature
cargo test -p tenferro-runtime --lib runtime::tests::specialization
```

Expected: all named tests are registered and compilation fails only because the
complete A0 signature/specialization public API is absent. Do not implement any
production item between the signature RED and this combined RED.

## Task 4: Validated finite specialization

**Production authority:** the full `## Input signatures and finite
specialization` section at design lines 920-1298. Task 4 solely owns the
complete atomic signature-plus-specialization production slice, including
requirements, projection, partial order, and storage-class projection behavior.

**Public manifest:** `RankRequirement`,
`InputSpecializationRequirementsError`, `SpecializationError`, the A0
`PrepareError::Specialization` arm, `PlacementSpecialization`,
`LayoutSpecialization`, `InputSpecializationRequirements` and all six
accessors plus `builder`, `InputSpecializationRequirementsBuilder::{new,dtype,rank,concrete_dimensions,placement,layout,alignment_log2,build}`,
`SpecializationRequirements::{polymorphic,new,inputs,strictly_widens,project}`,
`SpecializationProjection::{requirements,inputs}`,
all six `InputSpecializationProjection` accessors, `PlacementProjection`, and
`LayoutProjection`. The public builder has bounded `Debug`.
`RankRequirement` has the exact accepted `Display`.

- [ ] **Step 1: Implement the atomic signature-and-specialization slice**

The tests in Task 3 and the following specialization tests are already
registered and observed RED before this step:

```text
specialization_builder_reports_first_duplicate_axis_positions
specialization_builder_reports_concrete_axis_rank_requirement
specialization_builder_reports_exact_strides_rank_requirement
specialization_builder_reports_out_of_lattice_alignment
specialization_builder_validation_order_is_deterministic
specialization_aggregate_requirements_construction_is_infallible
specialization_projection_reports_wrong_input_count
specialization_projection_reports_axis_outside_actual_rank
specialization_projection_reports_unavailable_alignment
specialization_projection_covers_all_alignment_rows_and_caps_known_alignment
specialization_projection_selects_exact_dtype_rank_dimensions_placement_and_layout
specialization_storage_projection_maps_builtin_memory_kinds_exactly
specialization_storage_projection_uses_other_empty_sentinel
specialization_storage_projection_other_utf8_hex_uses_fixed_utf8_examples_and_distinguishes_payloads
specialization_non_storage_placement_modes_do_not_derive_storage_class
specialization_strict_widening_rejects_equal_lowered_incomparable_and_different_arity_values
specialization_rank_zero_through_sixty_four_finite_chains_terminate
specialization_types_are_debug_and_all_public_accessors_are_callable
```

Every requirement comes from the public per-input builder and every signature
entry comes from `InputSignatureEntry::new`. Projection tests may observe only
`WrongInputCount`, `AxisOutOfRange`, and `AlignmentUnavailable`. Do not create
malformed private requirements, an indexed builder-error mapper, retry-loop
fixture, retry error, or retry accounting helper; A0 owns no retry behavior.

Implement the complete two-arm A0 `PrepareError` and all A0 error types first,
then the signature slice and finally specialization. `InputSignatureEntry::new`
alone validates caller-supplied rank agreement and alignment class;
`InputSignature::new` stores valid entries infallibly. `from_reads` maps only
`strides()`/`is_col_major_contiguous()` failures to indexed `TensorMetadata`,
constructs its already-proven entry directly, derives host/backend alignment as
specified, and records the exact compact/strided layout class. It retains no
storage class, tensor, buffer, pointer, or address.

Implement the exact builder validation order, infallible aggregate, public
accessors, projection, and partial-order widening. Projection trusts builder
invariants and performs only signature-dependent checks. Do not add any other
validation helper, retry accounting, or future retry machinery. Add compiling
examples for every public manifest item.

Only the `PlacementSpecialization::StorageClass` projection derives
`StorageClass` from retained `Placement`: built-in
`MemoryKind::{Device, PinnedHost, UnpinnedHost, Managed}` values map exactly to
`tenferro.storage.device.v1`,
`tenferro.storage.pinned-host.v1`, `tenferro.storage.unpinned-host.v1`, and
`tenferro.storage.managed.v1`. `Other("")` maps to
`tenferro.storage.other-empty.v1`; nonempty `Other(payload)` maps to
`tenferro.storage.other-utf8-<lowercase-full-UTF-8-hex>.v1` using the complete
UTF-8 byte sequence, collision-free and reversible.
`PlacementSpecialization::None` derives nothing, and
`PlacementSpecialization::Device` copies the placement without storage
classification. Any internal grammar-proof construction path or encoder is
introduced here, immediately consumed by projection, and has no validation
fallback, ignored payload, or `expect`.

- [ ] **Step 2: Run the complete atomic A0 GREEN gate**

```bash
cargo fmt --all --check
cargo test -p tenferro-runtime --lib runtime::tests
cargo test -p tenferro-runtime --test integration runtime_foundations
cargo test -p tenferro-runtime --doc runtime
cargo check -p tenferro-runtime --all-targets
python3 scripts/check-public-error-docs.py
bash scripts/check-pr-fast.sh --coverage-reviewed \
  --test 'cargo test -p tenferro-runtime --test integration runtime_foundations'
git diff --check
```

Expected: all pass with every accepted public accessor, `Debug`, `# Examples`,
reachable error, alignment row, and finite-order behavior covered.

- [ ] **Step 3: Audit the standalone A0 boundary**

```bash
rg -n \
  'Atomic|IdentityExhausted|RuntimeConfig|RuntimeState|RegistrationKey|Prepared|Capability|Provider|ExtensionModule|CacheOwner|SingleFlight|NegativeEntry|Subgraph|Einsum' \
  crates/tenferro-runtime/src/runtime
rg -n 'unwrap\\(|expect\\(' \
  crates/tenferro-runtime/src/runtime/{identity,policy,signature,specialization}.rs
git diff -- scripts/test-doc-consistency.py
```

Expected: no matches and no docs-consistency diff. Manually confirm
`IdentityKind` has exactly four A0 variants, `PrepareError` exactly two A0
arms, raw `PrepareOptions` is not hashable, `cache_in_flight` enters no key,
and no stored key contains a tensor, value, buffer, pointer, address, free
memory, scheduler state, diagnostic string, or provider-private value.

- [ ] **Step 4: Commit boundary**

```bash
git add crates/tenferro-runtime/src/lib.rs \
  crates/tenferro-runtime/src/runtime \
  crates/tenferro-runtime/tests/integration/runtime_foundations.rs
git commit -m "feat(runtime): add A0 finite specialization"
```

Stop. P4-A1 and every later node require their own accepted execution plan.

## Plan self-review before execution

- [ ] Every accepted A0 API block maps to exactly one production task; Task 3
  is the test-only RED prelude for Task 4's single atomic production slice.
- [ ] Every public manifest includes all accepted accessors and exact visibility.
- [ ] Every public type/function has a compiling `# Examples`; every public
  `Result` has `# Errors`; every public type has tested `Debug`.
- [ ] Every named behavior test is written and observed RED before its
  production slice.
- [ ] `from_reads` exposes only reachable `TensorMetadata` failure and contains
  no storage classification, `expect`, redundant entry validation, or
  speculative error mapping.
- [ ] Storage classification exists only in `StorageClass` projection, with
  exact built-in IDs and reversible full-UTF-8 `Other` encoding.
- [ ] No placeholder, deferred implementation note, unnamed test, or
  unspecified error handling remains.
- [ ] No A1-E0, cache-owner, Phase 5, or Phase 6 type/fixture appears.
- [ ] Commands, test names, method names, and commit boundaries are internally
  consistent.
