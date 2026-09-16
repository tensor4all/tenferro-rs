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
property of the signature rather than a linker effect.

The measurement is reproducible and now recorded as an artifact:
`scripts/check-scalar-composition-kernel-sharing.py` builds the proof crate's
composition test with `--emit=asm`, counts the kernel entries the assembly file
*defines*, and writes its JSON record to
[`scalar-composition-kernel-sharing.md`](./scalar-composition-kernel-sharing.md).
On `79092f7c` (release build, `rustc 1.97.1`, `x86_64-unknown-linux-gnu`) the
counts are:

| Path | `zip_map2_into` entries | `zip_map2_parts_into_validated` entries | Call sites |
| --- | --- | --- | --- |
| preset `f64` | 1 | 2 | 39 |
| external `Df64` | 1 | 3 | 41 |

Every entry on both paths names
`tenferro_internal_cpu_kernels::scalar_ops::scalar_binary_into`, so the two paths
are the same kernel function rather than two implementations, and every external
entry is parameterized by the contribution's own `Df64Add` operation. The preset
entries are reached from 39 places, so the body is shared between call sites
rather than duplicated per call site.

This is measured on one release test binary in one crate and is not yet the full
#1706/#1790 protocol.

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

The cost of each shape was measured rather than assumed, by adding the variant in
a disposable worktree and compiling the workspace:

| Change | Exhaustive matches that need an arm |
| --- | --- |
| Add an erased variant to `Tensor` | 18, all inside `tenferro-tensor` |
| Add an external variant to `DType` | 59, across about thirty files |
| Make `dtype()` return `Option<DType>` instead | 605 call sites |

So the cheap way to open the tag is a `DType` variant, not a fallible accessor, and
admitting an external member costs roughly 77 explicit arms in total. The tag
variant's 59 sites break down as 7 in test files, 40 in production code in
statement position where an arm can reject directly, and 12 in production code in
value position such as `let eps = match dtype { .. }`, where the enclosing function
has to start returning a result. Those 12 are why this is a deliberate change
rather than a sweep.

Both variants are public type changes: `DType` gains a variant and `Tensor` gains
one, which is a semver break and, under this repository's rules, a change that
needs maintainer acceptance before a feature pull request can carry it.

The value half landed on the third attempt. `Tensor` carries
`External(ErasedHostTensor, Placement)`: the value type reports the payload's own
element identity as its dtype, its shape, its placement, and its element count,
while views, allocation groups, storage identity, layout offsets, and duplication
reject it or document an invariant, because a caller-owned payload has none of
those in this crate.

Every crate gained an explicit arm - about ninety sites the compiler enumerated
across `tenferro-tensor`, the CPU kernel crate, `tenferro-cpu`,
`tenferro-cpu-fused`, `tenferro-runtime`, `tenferro-ad`, `tenferro-einsum`,
`tenferro-linalg`, `tenferro-gpu`, and the FFT and XLA edges. A result-returning
path rejects with a typed error, a cleanup helper that tags or reclaims pooled
storage treats a caller-owned payload as a no-op, and `einsum` rejects an
externally defined input dtype before borrowing a preset-only view, which is what
makes its internal invariant sound.

The first two attempts failed for a reason worth recording before the next change
of this shape: the sites are not uniform, several live inside shared dispatch
macros that the compiler reports at the call site rather than at the match, and a
blanket `unreachable!` would put a panic on a path a caller can reach.

The tag half is now implemented and measured. `DType` carries
`External(TypeId)`, 56 sites gained an explicit arm, and the workspace compiles
and tests exactly as before. The measured price is that `DType` grew from **1 byte
to 24 bytes**, because the variant carries a `TypeId`; the tag is embedded in
runtime metadata, cache keys, and error types, so that growth tripped
`clippy::result_large_err` in nine `tenferro-cpu` functions and enlarged those
error types. Three consequences follow and they are the reason this stays a
decision rather than a detail:

- A unit `External` variant would keep the tag at one byte but would drop the
  identity that distinguishes two external scalars, so a `DType`-keyed cache would
  collide. The linalg cache key now hashes the carried identity for exactly that
  reason.
- A small registered id (`External(u32)` plus a process-local map) would keep the
  tag at eight bytes at the price of runtime registration, which the original
  proposal rejected.
- Both alternatives trade identity or an id registry against the 23 bytes, and the
  choice should be made against the tag's real footprint in the runtime, which has
  not been measured. That is an
order of magnitude below the 2267 production pattern sites that removing the
seven variants rewrites, which is the opposite of what was assumed before the
measurement: the hybrid shape is not the expensive one.

Two shapes can admit an external member, and the difference is where the cost
lands. A tag plus one erased payload for every member removes the variant list but
charges the measured erasure cost to the preset members as well. Keeping the seven
variants and adding one erased variant charges it only to external members, and
because the same-variant dispatch sites already end in a fallback arm, most of
them would reject an external member through their existing error path without
being rewritten. The catch is the same in both shapes and it is not in the tensor
layer: a preset payload is pool-owned through `RootResourcePin::Host*` in
`crates/tenferro-tensor/src/storage/root.rs`, and the pool is typed per preset
through the sealed `PoolScalar`, so an external payload has to be caller-owned
rather than pool-owned. That is #1789's ownership contract, and it is why this
decision cannot be made from the tensor layer alone.

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
erased one, on one worker thread, host-only, fastest of nine rounds of 200,000
iterations over a 64-element `f64` payload. The numbers below are re-measured after
the erased value gained a layout, and the earlier record is kept beside them:

| Operation | Direct | Erased | Delta | Earlier delta |
| --- | --- | --- | --- | --- |
| Element-type recovery plus access | 11.7 ns | 21.5 ns | +9.8 ns | +5.8 ns |
| Construction and drop | 571.2 ns | 1064.7 ns | +493.5 ns | +58.5 ns |

Re-measuring paid for itself twice. The access cost had grown from +5.8 ns to +306.7 ns
because the contiguity check that guards the whole-payload borrow built the dense layout to
compare against it, which allocated on every access; walking the extents in place and then
caching the verdict on the value brought the access delta back to +9.8 ns. The construction
delta is a real trade rather than a defect: an erased value now carries shape, strides, and an
offset so that a metadata-only permutation and a contiguous materialization exist, and a debug
build pays for the larger value plus the `Arc` that makes sharing cheap. It is a per-value cost
on the cold path, not a per-element one.

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

One question this leaves is why the objective's *first*-named target,
`crates/tenferro-gpu/src/cubecl/mod.rs`, is among the remaining sites, since a GPU feature
builds in this checkout
What stops the GPU modules is narrower than "the macros cannot express them", which was my
first guess and is wrong: `same_variant_unary!` takes a caller-supplied wrap closure, so the
result's dtype derivation is expressible, and `same_variant_pair!` re-wraps the result itself.
The actual limits are specific and checkable. Both macros cover only the four float and complex
arms, so the integer and boolean arms stay outside them, which matters here because the GPU
linalg file rejects integer input and the elementwise file launches a different
`launch_checked_integer_binary` with a `crate::DType` argument for the integer pairs. The
sixteen shape-guard arms must remain pre-checks ahead of any macro. And the conversions left in
those files are structural deduplication in a backend whose kernels cannot run in this
environment, so the change would ship as compile-only verification. That is why the sites are
recorded as outstanding work with their reasons rather than converted here.
 (`cargo check -p tenferro-gpu --features cuda` succeeds in 25 seconds
with the vendored CubeCL crates, so verification is not what stops it). The answer is measured:
its 66 concrete `Tensor::`/`DType::` matches are not a uniform same-variant dispatch. 16 arms
carry an extra shape guard that routes to a fallback, 69 places pass a `crate::DType` into the
kernel they launch, one arm groups three variants into a single rejection, and the launches go
through distinct helpers such as `launch_checked_integer_binary`,
`launch_broadcast_multiply_int_typed`, and `launch_bool_tensor_into`. Rewriting those with a
same-variant macro would change behavior, so they need the accessor and kernel-parameter work
this section identifies as the stopping point, not a macro swap.

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

## 5.4 What is left, and why it needs its own step

The gaps between the landed foundation and the acceptance criteria of #1785,
#1788, #1789, #1790, and #1793 are now in the numerical layer and in #1789's
resource decisions rather than in the boundary: the extension boundary exists and
executes, and a caller-owned payload carries its own view layout.

**Executing an external scalar needs the extension boundary.** The tag and the
value type can name and carry an external scalar, but every CPU kernel rejects one
with a typed error, because tenferro owns no implementation for it. Running one
means an extension-owned operation reached through the runtime's `ExtensionOp`,
`ExtensionModule`, and prepared-execution path, which is what #1785 and #1790 own.
The pieces that step needs are landed: the payload answers its element identity,
shape, and element count, `ScalarSet::promote` accepts an external tag, and the
cache-key identity carries the actual scalar rather than a shared code.

The registered-extension path now executes end to end. The extension op family,
its planning config, engine, prepared operation, executor, and module all live in
the downstream crate, the family rejects a preset input explicitly, and
`ext/df64-proof/tests/extension_execution.rs` runs the operation on a
caller-owned payload through the registered module and checks that the low-order
component survives.

What that required is the ownership path, and it is the first of the two candidate
shapes recorded earlier, now implemented:

- `AdValueRecord` in `tenferro-ad` and `RetainedValue` in `tenferro-runtime` each
  hold a pooled-or-caller-owned container. A caller-owned payload is retained
  directly, so it needs no allocation group and nothing returns to a pool when the
  record drops.
- A caller-owned payload has no typed descriptor view, so a path that needs one
  (`value()`) fails with a typed runtime error rather than borrowing the payload as
  bytes; the read, duplicate, and consume paths serve it instead.
- `TensorValue::try_into_group_parts` returns a caller-owned value unchanged rather
  than forcing it into a group, which is what lets the runtime's retention model
  accept it.
- `to_contiguous_read` and duplication clone a caller-owned payload through its own
  entry point, which keeps its element type and reinterprets no bytes.

#1789 still owns what this deliberately does not decide: pool reuse for external
scratch, cross-owner handoffs, and the accounting a caller-owned payload
participates in. The payload here is retained and returned, not pooled.

Session composition is also demonstrated on its own.
`ext/df64-proof/tests/session_composition.rs` carries an external payload as a
runtime `Tensor`, enters `with_backend_session`, runs an ordinary addition on `f64`
tensors in that session, then runs the extension's own Df64 kernel on the carried
payloads through the public caller-destination entry point, and checks after the
session that the low-order component survived. So an external scalar coexists with
ordinary tensor work inside one admitted session and inherits its admission and
thread budget.

**The required extended-precision example and directed conversions run.**
`ext/df64-proof/tests/directed_conversion.rs` walks #1785's example through public
boundaries: `Df64` tensors holding `1` and `2^-80` are added in `Df64`, `1` is
subtracted in `Df64`, the result is exactly `2^-80`, and the same computation in
`f64` yields `0`. `conversion::to_f64` and `conversion::to_df64` declare their
rounding, range, and allocation behaviour: the low component participates in the
sum and rounds to nearest with ties to even (a low component of `2^-52` reaches the
destination, `2^-80` rounds away, and half an ulp rounds to even), the `f64` to
`Df64` direction is exact with a zero low component, and neither direction coerces
a source it does not declare.

**A caller-owned payload carries its own view layout.** #1785's permutation,
mutable-view, and materialization requirements are met without a variant in the
typed view types, because the layout lives in the erased value instead:

- `ErasedHostTensor` stores shape, element strides, and an element offset beside
  the payload. `new` is dense column-major, `permuted(axes)` changes only that
  metadata and shares the payload, `to_contiguous()` gathers the view into a new
  dense payload, and `duplicate()` copies the payload.
- The payload is shared through an `Arc`, so `Clone` means "share the storage" and
  `duplicate()` means "copy it" — the same distinction the pooled path draws
  between a view and `Tensor::duplicate`.
- A typed read applies the layout (`element_at`/`element_at_mut`), and mutable
  access requires the caller to be the only holder, so two live views cannot
  produce two mutable borrows of one element.
- `downcast_ref`, `downcast_mut`, `into_typed`, and `as_dense` answer `None` for a
  strided view, so no caller reads the payload as if it were the view. That is
  #1785's "mismatched projection fails safely" boundary.
- `Tensor::duplicate` now copies the payload through its own entry point instead of
  rejecting it, and the erased layout feeds `Tensor::shape`/`layout_summary`, so
  the erased tensor type reports the view it carries.

`ext/df64-proof/tests/external_views.rs` covers the requirements: a permutation
that shares its payload and preserves every component through typed reads, a
mutable view that writes exactly the element it names, a shared payload that
refuses a mutable borrow, materialization into logical order, and a sum reduction
whose low-order component survives (`1 + 2^-80 + 2^-80 = 1 + 2^-79` exactly).

**Promotion between two distinct external scalars is checked at execution, not
in the lattice.** `promote(lhs, rhs)` is derived from declared facts and cannot
relate a scalar tenferro does not declare to anything, so for two distinct
external tags it returns one of its two inputs. That is now measured rather than
assumed: `promote(External(Df64), External(i64))` is `External(Df64)` and the
reversed pair is `External(i64)`.

What matters is that no executing entry point can turn that imprecise answer into a
wrong value, and that is verified in
`ext/df64-proof/tests/external_mixing.rs`:

- A conversion between two distinct external tags is rejected in both directions,
  as is a conversion between a preset and an external tag. The promotion answer can
  therefore only ever reach an explicit rejection; it never selects one payload's
  kernel for the other's elements.
- A binary operation on two external tensors does not run at all, whether the tags
  match or not, because no preset kernel is instantiated for a caller-owned
  payload. The supported route for external arithmetic is the registered
  extension operation.
- `can_convert_dtype` agrees, so a caller can ask before executing.

The remaining sharp edge is the inferred dtype a *traced* graph reports for such a
pair: it names one operand's tag before execution rejects the program. Closing that
would need a checked promotion threaded through
`tenferro-runtime/src/shape_infer.rs` (16 call sites on infallible inference paths)
and `tenferro-ad/src/eager_exec.rs`, which is a signature change rather than a
missing check, so it stays recorded as a known imprecision with execution-time
rejection instead of being guessed at.

**The runtime's traced IR rejects an external scalar explicitly.** A semantic
program's identity must be reproducible across processes, and an externally
defined tag is a process-local `TypeId`, so
`tenferro-runtime/src/program/identity.rs` had no encoding for one and reached an
`unreachable!`. A traced program that carried an external scalar therefore
panicked on a user-reachable path.

Resolving it follows the design's "explicit conversion-or-rejection" rule rather
than inventing a stable identity: the semantic-program builder now rejects the
tag when an input spec or an operation output carries one
(`ProgramBuildError::ExternalScalarWithoutIdentity`), which restores the identity
encoder's invariant and turns the panic into a typed error. The eager path — the
route the proof crate uses — is unaffected.

What would enable the traced and prepared path is a contribution-declared stable
scalar identity (a name that survives across processes and builds), which is a new
payload contract rather than a mechanical encoding, so it is recorded here instead
of being guessed at. `ext/df64-proof/tests/extension_execution.rs` pins the
rejection, and installing the module into a runtime with the CPU engine is
verified in the same test.

### 5.5 The traced and prepared path, and the identity it needs

A semantic program's identity must be reproducible across processes, and an
externally defined tag is a process-local `TypeId`, so the traced and prepared path
had no identity for one. This is now resolved by declaring the identity instead of
guessing an encoding, in 228 added lines across seven files:

- `ProgramValueMetadata::with_scalar_identity` and
  `ProgramInputSpec::with_scalar_identity` let a program declare the canonical name
  of an input's externally defined scalar.
- `ExtensionOp::scalar_identity` is a defaulted method, so an operation whose values
  are externally defined declares the name once and the runtime stamps it onto that
  operation's external value metadata.
- The identity encoder writes the declared name for an external value
  (`DType::External` code 7) rather than a bare type code, so two processes agree.
- A value that carries an external tag without a declared name is rejected with
  `ProgramBuildError::ExternalScalarWithoutIdentity`, and a *core* operation may not
  name an external scalar at all, because tenferro owns no kernel for it. Carrying
  one through a program is what an extension operation is for.

`ext/df64-proof/tests/extension_execution.rs` verifies both halves: the undeclared
case is a typed error, and the declared case runs the registered operation through
trace, compilation, and prepared execution
(`the_module_installs_and_plans_the_declared_scalar`). The repository's extension AD
surface already exists — `SemanticExtensionRuleSet` with `register_linearize`,
`register_linear_transpose`, and `register_primal_vjp`, exercised by
`crates/tenferro-ad/tests/integration/multi_input_traced.rs` — so an external
first-order rule needs no new mechanism; it needs the graph to be plannable, which
this change provides for programs built from a trace context.

**The AD path now carries the identity too, so an external scalar can be
differentiated.** The identity travels to the places the traced/AD path reads:

- `TensorMeta` (`tenferro-internal-ops/src/ad/context.rs`) carries the declared name
  with its dtype and extents, and `TracedTensor::input_concrete_shape_declaring_scalar`
  and `TracedTensor::from_tensor_concrete_shape_declaring_scalar` declare it.
- The runtime's compiler passes it from the traced value's registered metadata into
  `ProgramInputSpec` for both an unbound placeholder and a bound default tensor, and
  the import path keeps the declared name on the values it rebuilds.
- `tenferro-ad` keeps it when it converts program metadata back into traced metadata.

The adjoint is the contribution's own operation. `Df64Expand` broadcasts a scalar to
a declared shape — a preset broadcast is not available for a scalar tenferro does not
declare — and `Df64TotalVjpRule` emits it from the output cotangent, reading the
target shape from the primal input metadata and rejecting a symbolic shape rather
than guessing one. Both directions run: `Df64TotalLinearizeRule` sums the tangent inputs (the sum is
linear, so the linearization is the same operation) and `Df64TotalVjpRule` emits the
broadcast. `ext/df64-proof/tests/extension_ad.rs` verifies the traced VJP and JVP,
their compilation, and the execution of both programs, including cases whose
cotangent and tangent carry a `2^-80` low component that survives.

One structural fact this needed: the runtime keys one planning config per engine id,
so a contribution owns one *family* of operations and distinguishes them by payload.
Both `Df64Total` and `Df64Expand` therefore report the same family, the engine
dispatches on the payload, and the VJP rule rejects a payload outside its domain
instead of pretending to handle it.

`ProgramBuildError::ExternalScalarWithoutIdentity` also reports *where* the tag
reached the program (an input, an operation output, or a core operation), which is
what located the remaining plumbing when this step was implemented.

### 5.6 The contribution-owned QR factorization

#1788's QR checkpoint does not need a new provider mechanism. The contribution owns
its numerical body the same way it owns the total sum: `Df64Qr` is a second
operation in the same family, with one input and two outputs, and the runtime reaches
it through the same registered engine and prepared execution.

The body is modified Gram-Schmidt with one re-orthogonalization pass, computed in the
external scalar, which is why the factors keep its precision. That needed division
and square root in the scalar: `Df64::ratio` refines the quotient with two Newton
corrections evaluated in the two-component arithmetic, and `Df64::sqrt` refines a
square root the same way, so `(1/3) * 3 - 1` and `sqrt(2)^2 - 2` are non-zero in the
external scalar while the `f64` computation reports zero.

`ext/df64-proof/tests/extension_qr.rs` verifies the factorization through the traced,
compiled, and executed path: a square factorisation reconstructs its input with an
error below `1e-30`, its columns are orthonormal below `1e-30`, its diagonal is
positive, `[[3], [4]]` gives `R = [[5]]` and `Q = [[0.6], [0.8]]`, and a `2^-80` low
component in the input reaches the factors instead of being narrowed away.

What is still open for #1788/#1790 is the *differentiated* QR: the reverse rule needs
the adjoint of the factorization (a triangular solve, so it needs the same division
and a solve in the external scalar), and the conversions in the connected graphs need
their own rules. Those are the next step, not a missing mechanism.

### 5.7 Connected programs across a conversion

The connected programs of #1790 need the conversion to be an operation in the graph
rather than a call between graphs, so the contribution now owns both directions:
`Df64ToF64` and `Df64FromF64`, in the same family as the total sum, the broadcast and
the factorization. Their adjoints are the opposite conversions, so the reverse rule
emits one op and no residual.

The reverse pass can hand an operation a borrowed read: the adjoint of an ordinary
`f64` reduction is a broadcast, which is a strided view rather than an owned tensor.
The executor therefore resolves an input to either a borrow or a materialized tensor,
using the session the context carries when it has one, and gathering a preset `f64`
read by its own layout when it does not. That removed a real limitation rather than
working around it: before this, a connected reverse pass failed with a typed error
instead of running.

`ext/df64-proof/tests/connected_conversion_ad.rs` verifies both directions:

- `Df64 -> to_f64 -> sum of squares` differentiates back into the external scalar, and
  the gradient is `2x` evaluated at the *narrowed* values: the low component the
  narrowing discarded is not recovered by widening it back, which is the convention
  #1788 asks to check.
- `f64 -> from_f64 -> to_f64 -> sum of squares` differentiates back into an ordinary
  `f64` gradient of `[6, 8]` for the input `[3, 4]`, so the graph's dtype boundary is
  respected in both directions.

### 5.8 The differentiated factorization and the connected QR program

#1790's second checkpoint is the connected program `Df64 input -> QR -> f64 loss` with
the gradient flowing back into the external scalar. It runs.

The reverse rule needs the adjoint of `A = Q R`, which is
`A_bar = (Q_bar + Q copyltu(R R_bar^T - Q_bar^T Q)) R^{-T}`, and the forward rule needs
`R_dot = triu(Q^T A_dot) R` with `Q_dot = (A_dot - Q R_dot) R^{-1}`. Both need a
triangular solve in the external scalar, so both are operations in the contribution's
family (`Df64QrVjp`, `Df64QrJvp`) rather than graphs of preset operations, which could
not execute for a scalar tenferro does not declare. The adjoint's payload records which
cotangents are present, because a loss need not depend on both factors, and an absent
cotangent is the zero cotangent.

`ext/df64-proof/tests/extension_qr.rs` checks the adjoint in isolation against the
analytic derivative, and `ext/df64-proof/tests/connected_qr_ad.rs` runs the connected
program end to end:

- `A -> QR -> R -> f64 -> R^2` with `A = [[3], [4]]` differentiates back into the
  external scalar as `[[6], [8]]`, which is #1790's orientation case
  (`R = [[5]]`, `L = 25`, `dL/dA = 2 R A / |A|`).
- The forward tangent of the same graph is `3` for the first unit direction, which is
  `(Q^T A_dot) R = (3/5)(5)`, and it leaves the graph as an ordinary `f64` value.
- The second connected graph, `f64 input -> widen -> QR -> narrow -> loss`, also
  differentiates: the gradient crosses the widening and lands in the input's own dtype as
  `[6, 8]`.

Implementing this found one real bug in my own body: the adjoint subtracted `Q^T Q`
instead of `Q_bar^T Q`, which scales the gradient by an amount that depends on the
data. The isolated adjoint test measured that factor (`0.98`) before the connected test
was touched, which is why the two tests exist separately.

### 5.9 The two consumer roles

#1790 asks for two compilation roles rather than one proof crate: an algorithm crate
that states the scalar properties and operation capabilities it needs, and a final
application that composes a canonical support with an optional scalar contribution. Both
exist and both tests pass:

- `ext/scalar-consumer-algorithm` declares
  `ScalarSupport { fn qr(&self, &TracedTensor) -> (Q, R); fn to_f64(&self, &TracedTensor) -> f64 }`
  and builds the connected program from it, using only public traced operations and never
  naming a scalar, a provider, or a dtype. `factor_norm_gradient` is the whole algorithm:
  factor, present the factor as ordinary `f64`, form the squared norm, and take the
  reverse pass.
- `ext/scalar-consumer-application` supplies two bindings for that one algorithm:
  canonical standard support (`tenferro_linalg`'s factorization and the identity
  presentation) and standard support plus the external scalar contribution
  (`Df64Qr` and `Df64ToF64`). The application is the only role that names a scalar, an
  identity, or a provider, and it installs both modules into the same runtime.

Both cases return the same number in their own dtype: `[6, 8]` as `f64` for the standard
binding and `[[6], [8]]` as `Df64` for the contribution, for the same input
`A = [[3], [4]]` and the same algorithm source. The boundary is mechanical, not a
convention: `cargo tree -p tenferro-scalar-consumer-algorithm --edges normal` contains no
`df64` and no `tenferro-linalg` entry, so the algorithm cannot reach the contribution or
the provider even by accident.

### 5.10 The four configurations of #1790

All four run, and both connected graphs execute in the mixed and cooperating ones:

| Configuration | Evidence |
| --- | --- |
| Standard, assembled from reusable support | the linalg module alone, `[6, 8]` as `f64` |
| Df64-only numerical support | the contribution's module alone with no linalg installed, `[[6], [8]]` as `Df64` |
| Mixed | both modules on one engine, both gradients in one runtime |
| Cooperating | two CPU resource domains under distinct engine identities in one runtime, the standard family on one and the contribution's on the other, both gradients |

An earlier assessment called the cooperating configuration blocked, and it was wrong:
the runtime tells two CPU owners apart by *resource domain*
(`CpuBackend::from_external_managed_domains`), so two engines whose provider is the same
still have distinct provider/device identities when their domains differ. The first
attempt failed with `RuntimeConfigError::DuplicateProviderDeviceTarget` only because it
registered the same default domain twice.

The second version of that test was also weaker than it looked: it took the two domains from
the discovered topology and returned early when the host declared fewer than two nodes, which
this host does (one node, 64 CPUs), so the configuration was silently skipped. The test now
gives each owner its own *disjoint slice of the node's CPUs* under its own domain identity,
which is the "explicitly separate owners" shape #1789 allows and works on a single-node host.
Both connected programs therefore run in the cooperating runtime here rather than on a
machine that happens to have two NUMA nodes.

**Cross-owner handoff.** #1789 requires a value produced under one owner to reach another
either through a contract that permits the borrow or through an explicit transfer or typed
rejection, never by relabeling. `a_value_from_one_owner_reaches_the_other_owner_explicitly`
runs the standard factorization on one owner and reads its factor on the other: the receiving
owner borrows the produced value, so the two owners share a compatible CPU domain, and the
test keeps the typed-rejection path asserted in case the contract ever tightens. The same run
recorded a compatibility fact between the two factorizations: the standard linalg QR does not
promise a positive diagonal while this contribution's does, so a consumer that needs the
positive sign has to say so.

The contribution needs one thing for this shape, and it has it:
`extension::module_for_engine` binds the contribution's operations to a caller-selected
engine, and each family is routed to the engine that registered it.

### 5.11 Measured size and coverage of the branch

The stage measurements asked for by the goal, taken on this head with
`git diff --numstat origin/main..HEAD`: 119 files, 10964 insertions, 807 deletions, of
which the largest areas are

| Area | Added | Removed |
| --- | --- | --- |
| `ext/df64-proof` (the contribution) | 4265 | 0 |
| `crates/tenferro-tensor-core` | 1827 | 38 |
| `docs` | 1691 | 3 |
| `crates/tenferro-internal-cpu-kernels` | 542 | 333 |
| `ext/scalar-consumer-application` | 525 | 0 |
| `crates/tenferro-runtime` | 448 | 72 |
| `crates/tenferro-tensor` | 368 | 101 |
| `ext/scalar-consumer-algorithm` | 250 | 0 |
| `scripts` | 201 | 0 |
| `crates/tenferro-linalg` | 179 | 193 |

Coverage of the contribution and the consumer crates, measured with
`cargo llvm-cov -p tenferro-df64-proof -p tenferro-scalar-consumer-algorithm -p
tenferro-scalar-consumer-application --json`:

| File | Line coverage |
| --- | --- |
| `ext/df64-proof/src/dense.rs` | 99.2% |
| `ext/df64-proof/src/lib.rs` | 95.2% |
| `ext/df64-proof/src/conversion.rs` | 93.5% |
| `ext/df64-proof/src/extension.rs` | 86.7% |
| `ext/df64-proof/src/ad.rs` | 78.5% |
| `ext/scalar-consumer-algorithm/src/lib.rs` | 78.3% |

Two facts explain why the smaller numbers are not untested behavior. First, llvm-cov does
not instrument doctests, and this branch's public items carry runnable examples: whole
line ranges of `ad.rs`, `extension.rs`, `lib.rs`, and the algorithm crate are example
bodies that the doc tests execute. Second, several branches are deliberate refusals that
another guard makes unreachable, such as the conversion bodies rejecting a payload the
operation entry point already validated.

Boundary tests raised `conversion.rs` from 82.6% to 93.5%, `ad.rs` from 72.6% to 78.5%, and
`extension.rs` from 79.9% to 86.7%, by covering real refusals and edge cases: a vector, a wide
matrix and a zero column for the factorization; a preset input for the narrowing and an
external one for the widening; a payload of another element type for both conversions; a
singular triangular factor for both derivative operations; a derivative rule asked about an
operation outside its domain; and an algorithm whose loss does not depend on its input. Those
are `ext/df64-proof/tests/extension_boundaries.rs`, `tests/directed_conversion.rs`, and the
algorithm crate's own tests. What keeps `ad.rs` and `extension.rs` below the goal's 90% is the
doctest bodies, which llvm-cov does not instrument, plus arms another guard makes unreachable;
covering those would mean testing examples twice or padding defensives, which the repository's
coverage policy forbids.

### 5.12 The survival half of #1790's "later backward"

`ext/scalar-consumer-application/tests/later_backward.rs` covers the half of that
checkpoint that does not need #1789's pool contract:

- A forward factorization runs in its own runtime, which is then dropped, so nothing can
  recompute the factors. The retained `Q` and `R` are written to and then consumed by a
  *later* program in a *new* runtime, whose adjoint output is exactly the written values.
  The result therefore comes from the retained factors rather than from a fresh
  factorization, and the external scalar's caller-owned storage is what made that possible:
  a pooled value could not outlive the runtime that owned its group.
- The same holds on the eager path: a value computed inside an admitted backend session is
  still usable after the session borrow ends, and an eager tensor's payload survives the
  eager runtime handle being dropped.

What is left of that checkpoint is #1789's own scope: surviving an *intervening scratch
reuse* and releasing storage so it can be reused need the pool accounting that issue owns.

### 5.13 The storage gap, measured before it is extended

#1789 prescribes trying the existing mechanisms first and extending the owning boundary only
for a demonstrated gap. `ext/df64-proof/tests/scratch_allocation.rs` measures what the
contribution's bodies allocate per execution with a counting global allocator, on a 64 by 64
matrix whose payload is 65536 bytes:

| Execution | Allocations | Bytes |
| --- | --- | --- |
| first factorization | 228 | 183381 |
| steady-state factorization | 189 | 174857 |
| adjoint | 256 | 1117164 |
| adjoint, after the copies were removed | 253 | 920556 |

Two findings came out of it, and the second is already fixed:

- **The bodies allocate fresh scratch per execution.** The steady-state factorization costs
  about the same as the first, so nothing is reused; the adjoint costs roughly seventeen
  matrix payloads. That is the demonstrated gap a reusable workspace would have to close,
  and it is now a number rather than an assumption.
- **The bodies made a hidden tensor-sized copy of every input.** `matrix_of` copied each
  payload into its own buffer before any arithmetic, which #1789 forbids ("no ... hidden
  tensor-sized copy"). The dense helpers now borrow the caller-owned payload
  (`dense::Matrix<'a>` holds a `Cow`), so the adjoint's bytes dropped by exactly the three
  input copies, 196608 bytes, and 256 allocations became 253.

**The acquisition and return path is now proven.** The adjoint's largest intermediate comes
from a scratch buffer the body acquires from the runtime's accounted extension cache
(`ExtensionCacheStore::put` with a `retained_bytes` figure) and returns afterwards. The
runtime's own statistics show the path working after two executions of one program: one
entry, 65536 retained bytes, one hit and one miss. The measured effect on the same 64 by 64
case is

| Execution | Allocations | Bytes |
| --- | --- | --- |
| adjoint, first | 258 | 921204 |
| adjoint, second (reused) | 222 | 847892 |

so the reuse is a number rather than an intention, and it is accounted rather than an
extension-private cache. Extending the same path to the remaining intermediates is
mechanical: each one needs a slot in the scratch and a `retained_bytes` figure the cache
already collects.

**Every intermediate of the adjoint now comes from that entry**, which the same measurement
shows:

| Execution | Allocations | Bytes |
| --- | --- | --- |
| adjoint, first | 254 | 724668 |
| adjoint, second (all intermediates reused) | 212 | 192700 |

The runtime reports the entry as 524288 retained bytes across the eight buffers, which is why
the second execution allocates almost nothing: what remains is the two factors it returns.

That work also found the same defect twice, and the second time is the more instructive. The
first version asked the scratch for the accumulator a second time in order to read it, which
clears and zero-fills the buffer, and the connected QR gradients silently became zero; the fix
was to read it without touching it. Extending the reuse then reproduced the bug in `copyltu`:
the second step of the symmetrization called the zeroing accessor again, dropped the lower
triangle it had just written, and the gradients went wrong again. The suite caught both. The
lesson is that a scratch API should make "give me a clean buffer" and "let me extend the buffer
I just filled" different operations rather than one accessor with a clearing side effect.

### 5.14 A correctness defect the Stage 1b audit found

Stage 1b requires the scalar contract to keep **wrapping** arithmetic for the integer
members. The preset pool path does: it has a `wrapping_add_elem` entry point that dispatches
`i32` and `i64` to `wrapping_add`. The shared operation types introduced for the
caller-destination entry points did not: `AddOp`, `SubOp`, and `MulOp` were bounded on the
operators (`T: core::ops::Add<Output = T>`), so their bodies were `lhs + rhs`, which **panics
on overflow in a debug build and wraps in a release build**.

That is two defects in one: a behavior difference between the two paths for the same
operation, and a behavior difference between builds of the same path, in a contract whose own
documentation asserts `scalar_add(i32::MAX, 1) == i32::MIN`.

The fix routes the three operations through the contract
(`T: ScalarArithmetic`, delegating to `scalar_add`/`scalar_sub`/`scalar_mul`), so the shared
path and the preset path now agree on wrapping in every build, and the contract is the single
source of the arithmetic. Tightening the bound broke no user in the workspace, which is
evidence that the contract is universal here rather than a subset.

The evidence is executable: `AddOp`'s documentation now asserts
`<AddOp as BinaryScalarOp<i32>>::apply(i32::MAX, 1) == i32::MIN`, which a doc test runs in a
debug build and which therefore fails against the operator implementation, and
`scalar_ops::tests::integer_arithmetic_through_the_shared_entry_point_wraps` covers addition,
subtraction, multiplication, and a reduction through the public entry points.

### 5.15 A hole in this branch's own verification, found by sweeping features

Opening `DType` and `Tensor` to an externally defined member left sixteen non-exhaustive
matches in `tenferro-linalg`'s GPU code (`crates/tenferro-linalg/src/gpu/linalg.rs` and
`gpu/linalg/rank_revealing_qr.rs`). They were invisible to this branch's own verification
because `cargo check --workspace --all-targets` compiles the default feature set, and those
files sit behind `cuda` and `webgpu`. `cargo check -p tenferro-linalg --features cuda` failed
with sixteen errors.

Every site now carries an explicit typed rejection through the file's existing
`unsupported_linalg_dtype` helper, and the one site that matches on `DType` rather than `Tensor`
(`singularity_tolerance`) became fallible so it can reject instead of picking a tolerance for a
scalar tenferro does not declare. The repository's own source-contract test pinned that
helper's old signature and was updated to the new one, with an added assertion that the
external tag is rejected.

The sweep that found this also turned up two facts about the repository rather than about this
branch, and both are checked against `origin/main`:

- Building the *whole workspace* with one crate's GPU feature enabled
  (`--features tenferro-linalg/cuda`) fails in `tenferro-einsum`, whose match on
  `EagerExtensionBackendKind` gates the `Cuda` arm behind its own feature. The file is
  byte-identical at `origin/main` and this branch never touched it, and each crate builds
  cleanly with its own GPU feature, so the mismatch is a pre-existing feature-interaction hole.
- Clippy over `tenferro-linalg --features cuda` reports 43 pre-existing lints in the GPU files
  (unneeded `Ok(..?)`, too many arguments, complex types). None is in the arms this branch
  added.

What this changes for the audit is that "the workspace check is clean" is only true of the
default feature set, and that a change to `DType` or `Tensor` has to be swept across the GPU
feature configurations too. The verification now includes that sweep, and the configurations
that are clean on this host are:

| Configuration | Result |
| --- | --- |
| `cargo check --workspace --all-targets` (default set) | clean |
| `-p tenferro-gpu --features cuda` and `webgpu` | clean |
| `-p tenferro-linalg --features cuda` and `webgpu` | clean |
| `-p tenferro-einsum --features cuda` and `webgpu` | clean |
| `-p tenferro-fft --features cuda` and `webgpu` | clean |
| `-p tenferro-xla --features pjrt` | clean |
| `-p tenferro-cpu --features cpu-faer` | clean |
| `--all-features` on any of these | not applicable: it pulls `accelerate-src`, an Apple-only framework, on every platform this branch can reach |
| the whole workspace with one crate's GPU feature (`--features tenferro-linalg/cuda`) | fails in `tenferro-einsum`, pre-existing and unrelated to this branch |
| `--no-default-features` | intended failure: the crates that need a backend reject it with `compile_error!` ("enable at least one CPU backend"); `tenferro-tensor-core`, `tenferro-internal-cpu-kernels`, `tenferro-df64-proof`, and the consumer crates build clean |
| `cargo doc --workspace --no-deps` | succeeds, with four pre-existing unresolved links in `runtime/snapshot.rs` and `fft/backend.rs`, files this branch did not touch |

The `--no-default-features` sweep also found a pre-existing test-gate bug outside this branch's
files: `tenferro-internal-ops/src/tests/input_key_tests.rs` imports the input key behind
`autodiff` but leaves the test itself ungated, so the crate's test target does not compile
without that feature. The file is untouched here and the defect is recorded rather than bundled
into this branch.

### 5.16 The contribution's own duplication

The contribution declared `ExtensionOp`'s ten methods for every operation with identical bodies,
differing only in arity and in the operation's output metadata. Five payload-free operations now
share one macro that takes those two things; the two payload-carrying operations keep explicit
implementations because their payload hashing and equality differ. The file lost 122 net lines
and every test and doc test still passes, which is the "reduce source where possible" direction
the repository asks changes to take.

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
