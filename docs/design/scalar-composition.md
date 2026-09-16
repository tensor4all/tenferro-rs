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
explicit `f64` narrowing through those public functions. Still outstanding in
1b: the open scalar contract and single preset table, the shared erased
dispatch, and `ad_admission`.

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

### 4.7 Honest limit

Stage 1 does not prove that an external scalar traverses the erased `Tensor`
layer, provider dispatch, or the AD graph. Those are exactly what stage 2 opens.
A stage 1 pull request must state this limit instead of implying completion of
the #1787 model.

## 5. Stage 2: open the erased layer

### 5.1 Shape

`Tensor<S: ScalarSet = DefaultScalars>`, with the set generated by a macro that
emits the enum, the per-set tag, the promotion lattice, and the dispatch helper.
The default type parameter keeps ordinary users concrete, and variant paths
continue to resolve through the alias so existing matches keep compiling.

### 5.2 Ordered conversion path

Each module is converted to tag-based dispatch with a single erased payload, in
descending arm density, so the largest boilerplate is removed first:

| Order | Module | Arm lines |
| --- | --- | --- |
| 1 | `tenferro-gpu/src/cubecl/mod.rs` | 317 |
| 2 | `tenferro-linalg/src/cpu/backend.rs` | 225 |
| 3 | `tenferro-internal-cpu-kernels/src/elementwise.rs` | 205 |
| 4 | `tenferro-linalg/src/gpu/linalg.rs` | 147 |
| 5 | remaining files with five or more variant arms | 88 files total |

The seven `Tensor` variants are removed last.

### 5.3 Boundary work outside the tensor layer

Storage and reinterpretation contracts, runtime metadata and IR, cache identity,
typed errors, and the default-only C API / XLA / serialization boundaries each
need an explicit conversion-or-rejection decision. Custom-dtype AD follows the
stage 1 admission contract.

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
- Stage 1's net reduction is a measured outcome, not a guarantee. Part 1a
  measured a net reduction of 68 lines across the changed Rust sources (45 added,
  113 deleted) before the scalar contract is introduced; the 100-line target for
  the whole of stage 1 remains unproven and part 1b still adds the open contract
  and its tests. If the total does not reach the target, the slice is widened
  within the same operation boundary before the pull request is opened, and the
  measured numbers are reported either way.
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
