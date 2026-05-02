# Primitive DType And Shape Semantics Dispatch Design

**Date:** 2026-05-02

**Status:** Proposed dispatch spec

## Issues

Primary:

- #775: BUG: DynamicTruncate shape inference returns pre-truncation shape
- #776: BUG: Add/Mul shape inference returns LHS shape instead of broadcast shape
- #778: BUG: scalar_size_value rejects I64 for DynamicTruncate
- #780: BUG: convert() silently overflows on narrowing casts
- #784: BUG: eps parameter ignored in Svd/Eigh execution path
- #785: BUG: eig_output_dtype maps I64 to I64
- #786: BUG: scale_real truncates for I64 instead of rounding

Related follow-up candidates:

- #790: PERF: CPU elementwise add/mul limited to scalar broadcasting
- #794: BUG: elementwise div lacks real-to-complex scalar promotion
- #811: DESIGN: No dtype promotion anywhere, apply_binary uses lhs.dtype
- #818: BUG: convert and embed_diagonal API inconsistencies

## Goal

Align compile-time metadata and small runtime scalar/dtype decisions with the
actual tensor semantics, without attempting a full dtype-promotion redesign.

## Scope

This dispatch covers:

- shape inference for `DynamicTruncate` and scalar/NumPy-style binary
  broadcasting metadata,
- I64 scalar size inputs for `DynamicTruncate`,
- `svd_with_eps` and `eigh_with_eps` execution plumbing,
- output dtype selection for eig,
- I64 `scale_real` rounding behavior,
- CPU convert semantics only if the API can be adjusted narrowly.

It does not cover:

- full dtype promotion policy (#811),
- full NumPy broadcasting implementation for CPU kernels (#790),
- real-to-complex binary promotion (#794),
- broad structural API redesign (#818),
- AD rule changes.

## Acceptance Specification

### Shape inference

Shape inference must report the same output rank and dimensions that runtime
execution will produce for:

- `DynamicTruncate`,
- binary add/mul with scalar operands,
- binary add/mul with compatible non-scalar broadcast shapes.

If runtime add/mul does not yet support full NumPy broadcasting, shape inference
must not claim support beyond what execution can perform. In that case, stop
and report the runtime/metadata mismatch rather than encoding optimistic
metadata.

### DynamicTruncate scalar sizes

`DynamicTruncate` size inputs may be I64 scalar tensors. Float scalar inputs may
keep the existing rounding policy if that is current public behavior.

Invalid scalar size tensors must return `Err`, not panic.

### Linalg eps parameters

`svd_with_eps` and `eigh_with_eps` must pass their `eps` value through execution
or the public API must stop accepting an unused value. The preferred outcome is
to plumb `eps` through the traced execution path and keep eager behavior
explicit.

### DType decisions

`eig_output_dtype` must not report I64 outputs for integer inputs. Since current
backends reject I64 eig, acceptable outcomes are:

- return a complex output dtype for real integer metadata, or
- reject I64 eig metadata construction before execution.

`scale_real` on I64 must use the same explicit rounding policy used elsewhere
for scalar size conversion, or reject non-integer scale factors if rounding is
not mathematically meaningful.

### Convert semantics

Do not silently convert lossy narrowing casts if the conversion API can return
`Result` without broad churn. If changing `convert` from `Tensor` to
`Result<Tensor>` requires a wide public API migration, document that migration
and stop before making a partial inconsistent change.

## Design

Prefer small semantic helpers over ad hoc fixes:

- a shared broadcast-shape helper for shape inference,
- a scalar-size extraction helper that accepts I64 and validates scalar shape,
- a single dtype-output helper for eig-like operations,
- one documented rounding policy for real-to-I64 scalar conversion.

Any helper added here should live near the existing owner of the semantic
decision. Do not leak internal tensor crate details into the `tenferro` facade
to solve metadata issues.

## Testing

Required tests:

- shape inference for `f32[3,1] + f32[1,4]` if runtime supports that case, or a
  test proving unsupported broadcasting is rejected consistently,
- `DynamicTruncate` with I64 scalar size,
- `DynamicTruncate` metadata reflects the truncated dimension,
- `svd_with_eps` or `eigh_with_eps` dispatch observes the configured `eps` or
  rejects unused `eps`,
- `eig_output_dtype` no longer maps I64 to I64,
- I64 `scale_real` follows the documented rounding/rejection rule.

Run at least:

```bash
cargo test -p tenferro shape
cargo test -p tenferro linalg
cargo fmt --all --check
```

## Dispatch Prompt

```text
Implement the primitive dtype and shape semantics dispatch from
docs/plans/2026-05-02-primitive-dtype-shape-design.md.

Fix only the listed metadata, scalar extraction, eps plumbing, eig dtype, and
I64 scale semantics. Do not implement a full dtype-promotion system, full CPU
NumPy broadcasting, or broad convert API redesign unless the existing signatures
already support it locally. Stop and report if a correct fix requires a wide
public API migration.
```

## Review Checklist

- Shape inference and runtime behavior agree.
- I64 scalar sizes are accepted only for scalar size tensors.
- No accepted public parameter is silently ignored.
- No new lossy cast is added.
- Any deferred convert or promotion work is explicitly called out.

## Stop Conditions

Stop and report if:

- fixing broadcast metadata correctly requires implementing runtime broadcast
  kernels first,
- `eps` cannot be plumbed without backend trait changes,
- `convert` requires a workspace-wide signature migration.
