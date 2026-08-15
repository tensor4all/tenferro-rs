# Eager Concatenate Exact-Shape Recording

## Status

Implementation design for [#1692](https://github.com/tensor4all/tenferro-rs/issues/1692).
This is a bug fix to existing eager AD behavior. It adds no public API, changes
no AD formula, and does not expand traced symbolic-shape support.

## Problem

Eager tensor values always have concrete runtime shapes, but their semantic AD
leaves intentionally use symbolic shape metadata so derivative programs can be
reused. `Concatenate` requires equal non-concatenated dimensions and currently
checks those dimensions by structural `DimExpr` equality. Distinct symbolic
leaves therefore fail semantic AD compilation even when their concrete eager
shapes are compatible.

The failure affects:

- direct eager concatenate of separately tracked inputs; and
- eager stack with mixed tracked and untracked rows, because stack lowers to
  reshape plus concatenate and active-edge recording lazily creates a symbolic
  constant leaf for the untracked row.

Forward execution has already validated the concrete shapes and produced the
correct value. VJP/JVP compilation later fails with `input ... dimension ...
does not match the first input`.

## Design

Shape-specialize only the deferred eager semantic carrier at the concatenate
boundary.

1. Keep active-edge recording and lazy untracked constants unchanged.
2. In `record_semantic_eager_outputs`, after collecting each semantic input and
   before appending a `Concatenate` op, ensure every semantic input has the
   current eager input's exact concrete shape.
3. If a semantic input is already concrete-shaped, reuse it. Otherwise append a
   semantic `Reshape` to `EagerTensor::shape()` and feed that result to the
   concatenate op.
4. The reshape exists only in the deferred semantic graph. It adds no
   additional eager tensor copy, upload, materialization, or execution kernel.
   Existing tracked n-ary forward execution still obtains owned tensors through
   `to_tensor()`, and lazy untracked semantic constants still retain an owned
   tensor value; removing those existing boundaries is outside this bug fix.

This is valid because an `EagerTensor` value has a fixed concrete shape for its
lifetime. The exact reshape extents intentionally enter the semantic program
fingerprint, while both transform and prepared-derivative cache keys also carry
bound metadata. Concatenate VJP needs concrete concatenated-axis extents to
slice the cotangent, so the eager derivative graph is inherently
shape-specialized at that operation. The implementation must update the stale
`analyze_deferred_semantic_trace` comment: concrete leaf extents remain binding
data, but explicit operation parameters such as this reshape are valid semantic
identity.

## Scope boundary

Public traced symbolic behavior is unchanged. In particular, this PR does not:

- make rank-known symbolic `TracedTensor::concatenate` accept unresolved
  equality obligations;
- change symbolic stack's current concrete-shape requirement;
- add core shape constraints, guard namespaces, import remapping, or compiler
  staging behavior.

Those changes require a separate accepted design because they affect reusable
traced-program contracts. The current fix is owned by eager semantic recording,
where concrete shapes are already known and validated.

## Rejected alternatives

- **Core symbolic concatenate constraints.** Correct as a broader traced feature,
  but it requires changes across metadata analysis, semantic builders, graph
  import guard remapping, staging, and strict inference callers. It exceeds
  #1692's eager regression scope.
- **Concrete lazy constants only.** Fixes mixed stack but leaves direct eager
  concatenate of distinct tracked symbolic leaves broken.
- **Downstream tensor4all workaround.** Temporary tracked leaves or pad/add
  substitutes add copies or kernels and repair the wrong abstraction layer.
- **Skip equality validation.** This would weaken reusable traced-program shape
  safety.

## Verification ownership matrix

| Owner | Coverage |
|---|---|
| `tenferro-ad` eager recording | Direct concatenate with distinct tracked leaves; mixed tracked/untracked inputs in both orders; unequal concatenate-axis extents |
| `tenferro-ad` shape packing | Mixed stack in both orders and insertion axes `0` and `-1` |
| Reverse AD | Direct concatenate and mixed stack exact weighted f64 and Complex64 cotangents, including offsets across inactive inputs |
| Forward AD | Direct concatenate and mixed stack JVPs with exact tangent oracles |
| Higher-order composition | One concatenate or mixed-stack VJP→JVP or JVP→VJP regression proves derivative traces remain composable |
| Cache identity | Same runtime, different compatible concatenate shapes do not reuse an incorrect transform or prepared derivative |
| Existing active-edge contract | Existing untracked-constant-feeds-tracked-AD and all-tracked tests remain green |
| Scope boundary | Existing symbolic traced stack rejection remains unchanged |
| Source contract | `analyze_deferred_semantic_trace` distinguishes symbolic leaf metadata from explicit shape-bearing operation semantics |

The implementation must add a curated work log under `docs/worklogs/` with the
source/architecture review, the rejected core-constraint expansion, focused
commands, coverage review, and residual traced-symbolic limitation.

No tolerance changes are involved. The new metadata work is
`O(input_count)` for concatenate and adds no additional tensor-sized copy or
execution kernel.
