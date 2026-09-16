# Scalar Composition: Staged Plan

This document records the accepted staged plan for letting applications compose
standard scalar support with external scalar types, and for removing the
per-type special casing that the closed `DType` set currently requires.

Parent issue: #1787. Owned by its children: #1785 (storage, views, core
operations, directed conversions), #1788 (Df64 QR and first-order AD), #1789
(CPU resources and lifetimes), #1790 (external application consumer), #1793
(ordinary einsum, providers, shared kernels), #1706 (CPU compilation
boundaries).

## Maintainer decisions this plan must honor

- Custom-dtype AD is in scope, first-order (JVP/VJP) for Df64. Higher-order
  custom AD is out of scope and must fail with a typed error, never as a zero
  gradient. Existing standard AD keeps working.
- Arithmetic is assumed to follow the ordinary real/complex field rules
  (associative, distributive, ordinary zero and one). Non-field scalars such as
  tropical/min-plus, boolean, saturating, or general semiring arithmetic must be
  rejected at the shared AD paths with a typed failure. The existing `tropical`
  extension owns its own semiring rules and is neither a precedent for nor a
  requirement on the shared model.
- Preset scalar types must not be special-cased. `f32`, `f64`, `i32`, `i64`,
  `bool`, `Complex32`, and `Complex64` must go through the same mechanism as an
  external scalar, and the resulting change must reduce source, not add a
  second parallel path.

## 1. Why two stages

The end state implied by #1785 is `Tensor<S: ScalarSet = DefaultScalars>`, where
the preset seven types are ordinary members of one set and a downstream crate
can add its own. Reaching that in one step is not reviewable: the erased tag
reaches backend dispatch, runtime metadata and IR, cache identity, typed errors,
and the C API / XLA / serialization boundaries. #1785 itself requires a proven
cross-crate scalar path before a broad representation rewrite.

Stage 1 therefore unifies the scalar abstraction and the numerical layers,
deletes the duplicated preset machinery, and proves an external scalar running
through the same kernels. Stage 2 opens the erased layer.

```
+---------------------------------------+     +---------------------------------------+
| Stage 1: uniform abstraction          |     | Stage 2: open the erased layer        |
| typed and numerical layers only       | --> | Tensor<S: ScalarSet = DefaultScalars> |
| one scalar trait, one preset table    |     | per-set tag, promotion lattice        |
| net reduction in source               |     | boundary conversion or rejection      |
+---------------------------------------+     +---------------------------------------+
```

## 2. Measured baseline

Measured on `origin/main` at `28dfc7e3`. These numbers decide several questions
that were previously argued from intuition.

| Measurement | Value |
| --- | --- |
| `TensorScalar::dtype()` and `.dtype()` call sites | 605 in crate source, 830 including tests |
| Files containing at least five concrete `Tensor::Variant(..)` arms | 88 files, 2,488 arm lines |
| Files referencing `DType::` | 239 files, 2,741 lines |
| `supported_representation` | 10 lines, already minimal: `source == target` plus four real/complex pairs |
| Duplicate tag enum | `tenferro_tensor_core::DType` (`tenferro-tensor-core/src/lib.rs:106`) and `tenferro_tensor::DType` (`tenferro-tensor/src/types.rs:3451`), identical variants and identical derives |
| Duplicate scalar machinery in one file | `tenferro-tensor/src/types.rs:3451-3746`, about 296 lines: second `DType`, second `TensorScalar`, second `private::Sealed`, second `impl_tensor_scalar!` |
| Hand-written cross-layer tag maps | `core_dtype()` defined in 4 places (`tenferro-tensor/src/lib.rs:101`, `tenferro-runtime/src/error.rs:919`, `tenferro-runtime/src/typed_tensor.rs:343`, `tenferro-gpu/src/cubecl/tests/mod.rs:114`), 44 call sites |
| Elementwise dispatch | already delegates to generic typed kernels such as `typed_add_view_with_pool<T, L, R>`; the erased side is a per-op macro invoked once per preset variant |
| Numeric replay bounds | `replay_binary`, `replay_scalar_left`, and `replay_scalar_right` in `tenferro-internal-cpu-kernels/src/elementwise.rs` required the sealed `PoolScalar` on the element type even though the inner `strided-kernel` `zip_map2_into` / `map_into` calls they wrap are element-type agnostic and need only `Copy` |

The duplication is the concrete form of the problem: the same seven scalars are
enumerated twice, and every boundary between the two layers carries a
hand-written seven-arm conversion.

Two consequences follow, and both were used to reject earlier proposals:

- `TensorScalar::dtype() -> DType` stays. Removing it would touch 605 call sites
  in crate source alone, which is not a reviewable first step. What changes is
  where the tag comes from, not whether it exists.
- `supported_representation` and the existing generic typed kernels are already
  compact. A first stage that only adds an open trait beside them would be a
  second path and would add lines. The net reduction has to come from deleting
  the duplicated preset machinery and the per-op dispatch duplication.

## 3. Principle

One scalar abstraction, and preset scalars are ordinary members of it.

- No preset-only numerical path. A kernel is generic over the scalar it
  computes with; `f64` and an external scalar are different instantiations of one
  source definition.
- No per-layer re-listing of the seven types. The preset list exists once, as a
  single table.
- No hand-written tag-to-tag maps between layers. One tag type, re-exported.
- Adding a scalar means implementing one trait. It must not require editing N
  dispatch sites per operation family.
- `ScalarSet`-style membership is not introduced in stage 1. It belongs to the
  value and dispatch adapters of stage 2, outside the numerical bodies.

## 4. Stage 1: uniform scalar abstraction on the typed and numerical layers

### 4.1 Deliverables

1. A single open scalar abstraction in `tenferro-tensor-core`, implemented once
   for the preset seven through one table macro, and implementable by an
   external crate for its own scalar.
2. Numeric kernels for the bounded slice expressed generically over that
   abstraction, with the erased dispatch collapsing to one shared dispatch
   mechanism per crate instead of one macro per operation.
3. Deletion of the duplicated preset machinery: the second `DType`, the second
   `TensorScalar`, the second `private::Sealed`, and the second `impl_*` macro in
   `tenferro-tensor`, together with the `core_dtype()` copies that exist only
   because the tag exists twice.
4. An external crate in the workspace that defines its own scalar and runs it
   through the same public kernels, proving that the preset path is not a
   precondition for execution.

### 4.2 Public surface

Additive:

- The open scalar abstraction and its field contract in `tenferro-tensor-core`.
- Explicit precision-reducing conversion for a scalar that opts into it. The
  conversion is a separate capability, not a requirement of the scalar
  abstraction.
- Generic entry points for the bounded slice. Names must not collide with the
  existing `add`, `sum`, `mul`, and `sub` surfaces in `tenferro-cpu` and
  `tenferro-internal-cpu-kernels`.
- A typed error for rejecting unsupported differentiation.

Changed:

- The sealed `TensorScalar` keeps its method set, including `dtype()`, and gains
  the open abstraction as its supertrait.
- `tenferro_tensor::DType` becomes a re-export of `tenferro_tensor_core::DType`.
  The variants, derives, and `as u8` behavior are identical, so existing matches
  and casts are unaffected.
- Kernels in the bounded slice become generic over the scalar abstraction. This
  is internal to their owning crates.

Removed:

- The duplicate tag enum and the duplicate scalar trait machinery in
  `tenferro-tensor`.
- The `core_dtype()` helper copies and their call sites. With one tag type,
  those conversions are the identity and disappear.

Source compatibility: the seven tag variants keep their names, paths, and
derives; `Tensor` keeps its variants; `TypedTensor`, `HostTensor`, and the
sealed `TensorScalar` method set are unchanged. Expected breakage is limited to
glob imports of the two former tag paths, which now resolve to one type.

### 4.3 Bounded numeric slice

The slice is the elementwise `add` / `sub` / `mul` same-dtype path plus the `sum`
reduction, in `tenferro-internal-cpu-kernels` and `tenferro-cpu`. It is chosen
because it already routes through generic typed kernels, its shapes are the
simplest, it reaches the unbounded public `HostTensor<T>` layer, and it is the
cheapest place to demonstrate that the preset path is not special.

The concrete seam found during implementation: the reusable numerical step is
the pair of destination-writing helpers `replay_binary`, `replay_scalar_left`,
and `replay_scalar_right`, which wrap `strided-kernel`'s `zip_map2_into` and
`map_into`. Those wrapped calls are element-type agnostic, but their callers
required the sealed `PoolScalar` bound on the element type, which restricted the
whole numerical body to the seven presets. Removing that bound leaves the
allocation step (`PooledUninitOutput`, which genuinely needs the sealed typed
pool) as the only pool-coupled part, so the numerical body becomes shared
between a pooled preset destination and a caller-provided external destination.
That is the mechanism the external proof crate exercises; the sealed pool stays
at the resource boundary and is not opened, which remains #1789 work.

Stage 1 is delivered in two parts so that the mechanical simplification is not
blocked by the external proof:

| Part | Content | Evidence |
| --- | --- | --- |
| 1a | Unify the dtype tag across `tenferro-tensor-core` and `tenferro-tensor`, delete the duplicate tag and the four `core_dtype()` copies, and drop the unnecessary `PoolScalar` bound from the elementwise replay helpers | Workspace `check --all-targets` clean, existing tests unchanged, measured net line reduction |
| 1b | The open scalar contract, the single preset table, the shared erased dispatch, and the external proof crate | The external type constructs, borrows, mutates, adds, reduces, and converts through public APIs, with the low-order retention case |

Part 1b lands incrementally on the same branch. Landed so far: the
caller-provided-destination entry points `scalar_binary_into` and `scalar_fold`
in `tenferro-internal-cpu-kernels` (re-exported from `tenferro-cpu`), which run
the same `zip_map2_into` / `reduce` bodies the preset pool path uses, and the
`ext/df64-proof` crate, whose own two-`f64` scalar exercises construction,
borrowed and mutable access, an elementwise operation, a reduction, and an
explicit `f64` narrowing through those public functions. Landed next: the open scalar contract (`Scalar`, `ScalarArithmetic`,
`ScalarDomain`) and `ad_admission` in `tenferro-tensor-core`, with the seven
presets declared once in a single table that expands into every contract they
implement, and the proof crate's scalar implementing `Scalar` and
`ScalarArithmetic` for itself so the contract has an external consumer. The
external scalar reaches `ad_admission` and is rejected explicitly
(`AdRuleUnavailable` at first order, `UnsupportedAdOrder` above it) rather than
receiving a zero gradient. The shared erased dispatch followed: `elementwise.rs` now declares the
variant-to-kernel dispatch once (`dispatch_read_same_variant`,
`dispatch_read_real_complex_scalar`) and the preset variant list once
(`dispatch_read_presets`), so `add`, `sub`, and `mul` supply only their
kernel rather than three copies of the matching code. Stage 1b is complete;
Stage 2 is next.

The slice must exercise: typed construction, shared and mutable borrowing, one
binary operation, one reduction, and one explicit precision-reducing conversion.
A contraction is deliberately excluded: it reaches rank, layout, and provider
selection, which belongs to #1793.

The remaining conversion path for the other 88 files is ordered by arm density
and recorded in section 5.

### 4.4 AD boundary

No differentiation is implemented in stage 1. One admission function is added
that answers whether the shared AD paths may differentiate a given scalar, and
it returns typed errors:

- an order other than one is rejected as unsupported order;
- a scalar whose arithmetic is not the ordinary real/complex field is rejected
  as an unsupported scalar;
- a first-order field scalar whose rules do not exist yet is rejected as
  unavailable rather than silently producing a zero gradient.

Existing AD for the preset set is not routed through this function and keeps
working unchanged. #1788 opens first-order admission when real JVP/VJP rules
exist.

### 4.5 Non-goals

- No `Tensor<S>` parameterization, no `ScalarSet`, no `define_scalar_set!`, no
  membership or promotion lattice.
- No new `DType` or `Tensor` variant. In particular `half::bf16` is deferred and
  the `half` dependency is not added.
- No reinterpretation for an external scalar. `host_slice_as` keeps its sealed
  boundary, and its safety comment continues to justify itself from the sealed
  preset set.
- No provider registry, no change to provider slots, `KernelDType`, or the
  `strided-rs` pin. No accumulator type axis. No hidden sequential CPU loop that
  bypasses the existing execution scope.
- No GPU, no C API, no XLA, no serialization change.
- No machine-code sharing claim. Stage 1 establishes that one source definition
  is shared; distinct scalar types remain distinct instantiations, and whether
  and where machine code is duplicated is #1706 work.

### 4.6 Acceptance

- The existing workspace suite passes without modification, and the seven preset
  types keep their exact numerical behavior.
- An external crate defines its own real scalar and, through public APIs only,
  constructs it, borrows it, mutates it through a mutable borrow, adds it,
  reduces it, and converts it explicitly to `f64`.
- Low-order information survives that path: for the external type,
  `sum([1, 2^-80]) - 1 == 2^-80` evaluated in that type, with an `f64` control
  showing the loss.
- Errors are typed and explicit: shape mismatch, empty input, unsupported
  scalar, and unsupported differentiation order.
- The changed files show a net reduction in source. Deletions and additions are
  reported as measured numbers in the pull request, not as a claim.
- Every new public item has a runnable doc example, changed files meet the
  coverage target, and the local gate passes.

### 4.6.1 The operation is a type, and the sharing is measured

The first version of the entry points took the arithmetic as a closure. Symbol
inspection of the proof test binary showed the consequence: `strided-kernel`'s
`zip_map2_into` was instantiated once per call site, because a closure is part of
a generic function's type parameters. Three `f64` instantiations existed for two
call sites, which is exactly the set-induced duplication #1793 forbids and which
linker deduplication must not be relied on to repair.

The entry points now take a named operation (`BinaryScalarOp` with `AddOp`,
`SubOp`, and `MulOp`), and an external contribution supplies its own operation
type for its own scalar. Re-measured on the rebuilt proof test binary with
`nm -C`, counting distinct `zip_map2_into` instantiations:

| Element type | Instantiations | Call sites in the test binary |
| --- | --- | --- |
| `f64` | 1 | 2, in two different sets |
| external `Df64` | 1 | 3 |

So the heavy kernel body is keyed on the element type and the operation, and two
sets that contain the same scalar reach one compiled body. This is a structural
property of the signature rather than a linker effect. It is measured on a
debug test binary in one crate and is not yet the full #1706/#1790 protocol.

### 4.7 Honest limit

Stage 1 does not prove that an external scalar traverses the erased `Tensor`
layer, provider dispatch, or the AD graph. Those are exactly what stage 2 opens.
A stage 1 pull request must state this limit instead of implying completion of
the #1787 model.

## 5. Stage 2: open the erased layer

### 5.1 Shape

The set is generated by `define_scalar_set!`, which emits the tag enum, the value
enum whose variants hold a `HostTensor` of each member, and the `ScalarSet`
implementation. The set is represented by that value enum rather than by a
wrapper struct, so tenferro's own set reads as `pub use DefaultScalars as Tensor;`.

A fixed set of enum variants cannot host a member tenferro does not know. The
runtime value type is `pub enum Tensor { F32(TypedTensor<f32>), ... }` with seven
variants and `#[derive(Debug)]` only; adding a member would add a variant, which
is what stage 2 exists to stop. Parameterizing the payload (`Tensor<Payload>` with
a GAT mapping each member to its storage type) keeps the variants fixed, so it
does not admit an external member either. The stage 2 end state is therefore a
tag plus a single erased payload rather than a widened variant list: the value
carries `S::Tag` and an erased payload whose concrete type the accessors
recover, the seven variants disappear last, and a downstream set member is
representable because it is never a variant. That shape needs its own soundness
contract for erasure, layout, and reinterpretation, which is #1785 design
question 2 and #1789's resource boundary; it is recorded here as the decision
stage 2 must make rather than left implicit in the notation.

Prototype landed: `ErasedHostTensor` carries a host tensor whose element type is
recovered at run time. The payload keeps its own concrete type and the value
stores its `TypeId` explicitly, so identity is the actual Rust type; recovery by
the wrong type returns nothing and no bytes are ever reinterpreted. The external
proof crate builds one container holding both a preset `f64` value and its own
two-component scalar, runs the shared reduction on the external member, mutates
it in place through the erased value, and confirms that a mismatched recovery
returns nothing. That establishes the shape: an externally defined member needs
no variant.

The pool-backed payload is measured rather than assumed: `TypedTensor<T>`,
`Tensor`, and `TensorView` are `Send + Sync + 'static`, asserted in
`crates/tenferro-tensor/src/tests/types_tests.rs`, so an erased container can hold
a pool-backed payload with the same identity-based recovery, and the concrete
payload releases its storage when it is dropped. What remains is a cost and
ownership choice, not a type-system blocker, and the cost is now measured.
`ext/df64-proof/tests/erasure_cost.rs` compares the direct host payload with the
erased one, single-threaded, host-only, fastest of nine rounds of 200,000
iterations over a 64-element `f64` payload:

| Operation | Direct | Erased | Delta |
| --- | --- | --- | --- |
| Element-type recovery plus access | 12.0 ns | 17.8 ns | +5.8 ns |
| Construction and drop | 602.7 ns | 661.2 ns | +58.5 ns |

Erasure therefore costs about 6 ns per access that recovers the element type and
about 59 ns per tensor for the extra allocation, roughly a tenth of the host
construction cost for this payload size. That is the input for choosing between a
tag plus an erased payload for every member and a fast path for the preset
members with erasure only for externally defined ones. The measurement is
host-only and single-threaded; the pool-backed runtime payload and its release to
the originating owner (#1789) are not covered by it.

The promotion lattice is generated too. Each member declares its arithmetic kind,
its rank within that kind, and its component width, and `define_scalar_set!`
derives `promote` from those facts: a boolean yields to anything, two members of
one kind keep the higher rank, an integer yields to the widest member of the
float or complex kind, and a float with a complex takes the narrowest complex that
still holds both. The hand-written table in
`crates/tenferro-tensor/src/validate/mod.rs` is deleted and `promote_dtype`
delegates to the set, so the lattice belongs to whichever set declares the
members.

Measured: the derived lattice reproduces the recorded hand-written table for all
49 pairs (`crates/tenferro-tensor/src/validate/tests.rs`), and the external set in
`ext/df64-proof` promotes within its own lattice without touching tenferro's.

Landed and measured: `DType` and `DefaultScalars` are now generated from one
declaration in `tenferro-tensor-core`, `ScalarSet` is the open membership
contract, and `ext/df64-proof` declares its own two-member set with the same
macro without touching tenferro's set. With `Tensor` aliased to the generated
set, `cargo check --workspace --all-targets` reports **zero errors and zero
warnings** across every crate, which verifies #1785's source-compatibility claim
for the in-tree surface instead of assuming it.

The import forms are measured too. A plain type alias breaks `use Tensor::F64;`
and variant glob imports, which #1785 predicted, so `Tensor` is a re-export
(`pub use DefaultScalars as Tensor;`) instead: `Tensor::F64(value)`,
`use Tensor::F64;`, and `use Tensor::*;` all compile, and
`crates/tenferro-tensor-core/tests/scalar_set_import_forms.rs` keeps them
compiling. The promotion lattice is still described rather than generated, and
generating it needs a decision, because the preset rule is not a lattice
property: `i32 + f32` promotes to `f64`, a deliberate widening rather than a
structural join.

### 5.2 Ordered conversion path

Each module is converted to tag-based dispatch with a single erased payload, in
descending arm density, so the largest boilerplate is removed first. The ledger
below counts every occurrence of `Tensor::<variant>(..)`, test files included, and
was measured on this branch; test files hold 1057 of the arms in 103 files, and
the remaining 2279 arms sit in 62 production files.

Counting the arms that sit in plain code rather than inside a `macro_rules!`
definition separates the real conversion targets from the arms that are already
declared once per crate. Of the 2279 production arms, **2083 are plain and 196 sit
inside macro definitions**, so the density table below is close to the real
workload and a file's raw count is not misleading.

| Order | Module | Plain arms | Inside macros |
| --- | --- | --- | --- |
| — | all files | 3298 in 165 files | — |
| — | production files only | 2083 in 62 files | 196 |
| 1 | `tenferro-gpu/src/cubecl/mod.rs` | 476 | 1 |
| 2 | `tenferro-internal-cpu-kernels/src/elementwise.rs` | 323 | 8 |
| 3 | `tenferro-linalg/src/cpu/backend.rs` | 267 | 10 |
| 4 | `tenferro-linalg/src/gpu/linalg.rs` | 253 | 0 |
| 5 | `tenferro-tensor/src/types.rs` | 119 | 0 |
| 6 | `tenferro-cpu/src/reduction.rs` | 85 | 0 |
| 7 | `tenferro-cpu/src/structural.rs` | 56 | 35 |
| 8 | `tenferro-linalg/src/gpu/mod.rs` | 44 | 0 |
| 9 | `tenferro-cpu/src/analytic.rs` | 31 | 0 |
| 10 | `tenferro-cpu/src/dot_runtime.rs` | 25 | 0 |
| — | remaining production files | 304 | 142 |

The seven `Tensor` variants are removed last.

### 5.2.1 The conversion pattern, demonstrated

The order in section 5.2 converts each module to tag-based dispatch. The pattern
was demonstrated on the highest-density CPU file that can be verified locally,
`crates/tenferro-linalg/src/cpu/backend.rs` (225 arm lines):

- `same_variant_pair!` declares the four supported real and complex arms and the
  unsupported-pair error once. A call site now supplies only the typed kernel it
  calls, for example
  `same_variant_pair!("full_piv_lu_solve", a, &rhs, |a, b| linalg::faer::full_piv_lu_solve(ctx, buffers, a, b, transpose_a))`.
- Six call sites (the `full_piv_lu_solve`, `triangular_solve`, and `solve` paths
  for the faer and blas providers) were converted: **102 lines deleted net**, with
  `cargo test -p tenferro-linalg` passing (127 + 163 + 1 + 144 tests) and the
  `cpu-blas` feature path still compiling.
- What this buys is not only the deletion: a call site no longer names the
  variants, so it does not change again when the value type stops being a closed
  enum. That is why this happens before the representation change.

What remains in the same file, and why: 26 single-tensor sites were attempted
with a second macro (`same_variant_unary!`) plus the generic constructor
`TensorScalar::typed_tensor_into_tensor`, so that a call site would not need to
name the variant its result lands in. Eight sites converted and the tests passed,
but the file's net change went from 102 deleted lines to 22, because the two macro
definitions cost more than the sites they removed. That is the opposite of the
simplification this work is for, so the unary conversion was reverted and the
measured outcome is recorded here instead of being carried.

The reason is structural, not cosmetic: the remaining sites are not uniform. Their
real and complex arms use *different* conversions, for example
`.map(|outputs| outputs.into_iter().map(Tensor::F32).collect())` for the real arm
against `.and_then(svd_c32_outputs_to_public_tensors)` for the complex one. A
variant-agnostic wrapper cannot express that difference, so those sites need the
representation change (one erased payload with accessors) rather than a mechanical
rewrite, and converting them before it would only move the same branching around.
The pair sites did convert cleanly because their arms differ only in the variant.

So the order in section 5.2 is refined: convert the sites whose arms differ only
in the variant (measured, landed), and leave the heterogeneous real/complex sites
for the representation change.

The mechanically convertible set is now exhausted. A scan of the workspace for
matches whose whole body is a same-variant dispatch finds **two** remaining
sites, both in the householder-QR factor import path, and they are left alone
because the dispatched value is a `CompactQrResult` rather than a `Tensor`, so
they would need a macro of their own for four arms. Every other such site in
production code is converted.

The same limit applies to the remaining large clusters, which was measured after
the first conversions. The biggest ones are not whole matches: the
`tenferro-internal-cpu-kernels` owned-tensor operations (`add_with_pool` and its
siblings, 42 arms in 8 sites) carry six same-variant arms followed by four
real-with-complex scalar-mixing arms and a fallback, so a macro that replaces a
whole match cannot take them, and one that pastes the extra arms would be as long
as the match it removes. The shared dispatch macros therefore cover the sites
whose whole match is the variant dispatch, and the rest wait for accessors.

### 5.3 Boundary decisions, with the current state as evidence

Audited on `origin/main` rather than assumed:

| Boundary | Current state | Decision |
| --- | --- | --- |
| XLA lowering | already explicit: `crates/tenferro-xla/src/lowering/types.rs:61` and `program.rs:658` return `Error::UnsupportedDType { dtype, context }`, mapped to `ErrorKind::Unsupported` in `src/error.rs:86` | Stays default-set-only. An unmapped member is rejected at lowering, never converted implicitly. Stage 2 must not widen this path. |
| C API | no C API crate and no exported `#[no_mangle] extern "C"` surface exist in this workspace; the `extern "C"` uses are bindings to BLAS and system libraries | Nothing to change here. A binding layer outside this repository converts or rejects; tenferro's own types stay default-set-only. |
| Serialization | no graph serialization surface exists: `serialize`, `to_bytes`, `from_bytes`, `encode`, and `decode` have no definition in `tenferro-runtime` or `tenferro-ad` | Default-set-only if one is added, and any other member must be rejected explicitly rather than written with a guessed tag. |
| Runtime metadata and IR | dtype is carried concretely: `crates/tenferro-runtime/src/runtime/execution.rs:208` (`PreparedExecution` metadata), `runtime/signature.rs:37`, and `graph/compiler.rs:40` with binding validation at `:315` | Metadata keeps a concrete identity, but once a value type is set-parameterized the carried identity must be the actual Rust scalar, not only the default-set tag. |
| Cache identity | dtype already participates: `crates/tenferro-linalg/src/extension.rs:1791` hashes a seven-arm `hash_dtype`, and runtime extension metadata carries `dtype` | This is a real gap for stage 2. A per-tag integer hash cannot distinguish two different scalars that share a tag, so two external members with the same tag would collide in the prepared-execution cache. The cache key must carry the actual scalar identity. |

Storage and reinterpretation remain the open contract: the host prototype proves
identity-based recovery with no byte reinterpretation, but the pool-backed
runtime payload's erasure, release, and provider retirement are #1789's decision.
Custom-dtype AD follows the stage 1 admission contract.

### 5.4 Exit criteria

The acceptance criteria of #1785 (Df64 and bf16 through storage, views, core
operations, and directed conversions, in Df64-only and mixed configurations,
with two sets sharing compiled kernels evidenced by object or symbol inspection),
#1793 (ordinary einsum with explicit providers and standard/Df64 first-order AD),
#1788 (Df64 QR with first-order AD), #1789 (resource ownership and lifetimes),
and #1790 (external application consumer) are met.

## 6. Risks and open questions

- Naming: the open abstraction must not be confused with the existing
  `strided-traits::ScalarBase`, `PoolScalar`, `OrderedElem`, or
  `ContractionScalar`. The final names are chosen during implementation.
- Deduplicating the two scalar traits is not a pure re-export. The core trait
  produces the host-only model and the runtime trait produces the erased tensor.
  The target is one preset table and one tag; the erased extension is defined
  once on top of it.
- The `xprec` crate used by the external proof is version 0.2.2, MIT licensed.
  Its MSRV, `Copy` behavior, and rounding API are confirmed during
  implementation.
- Stage 1's net reduction is a measured outcome, not a guarantee, and the
  measurement is now available. The mechanical simplification removed 254 lines:
  68 in part 1a (the duplicate tag and the four `core_dtype()` copies) and 186 in
  `elementwise.rs` (the tripled dispatch macros and the repeated preset variant
  lists). The stage as a whole measures 751 added and 130 deleted lines against
  `origin/main`, because the open scalar contract, the caller-provided-destination
  entry points, the `ad_admission` query, and the external proof crate are new
  code with no counterpart to delete. The 100-line target is therefore met by the
  simplification and not by the stage total, which is the honest reading: the
  deleted lines are the duplicated preset machinery, and the remaining
  per-variant lists in the other 88 files are removed by stage 2, where the
  closed enums themselves disappear. A later reader should not treat the stage
  total as evidence that the simplification failed, nor treat the simplification
  as evidence that nothing was added.
- The `trybuild` storage UI fixtures (`crates/tenferro-tensor/tests/ui/storage`)
  report 10 of 14 mismatches in this worktree both with and without these
  changes, because the expected `.stderr` files were generated elsewhere than the
  `/kache/...` worktree prefix. That is a pre-existing environment artifact, not
  a consequence of this change.

## 7. What this plan does not claim

It does not claim that an external scalar is fully supported, that the erased
layer is open, that Df64 QR or its derivatives run, or that set-induced
recompilation is prevented. It records the order in which those become true and
the evidence each stage must produce.
