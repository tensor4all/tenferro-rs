# API Convention Specification

**Date:** 2026-06-10
**Parent:** [`../index.md`](../index.md)
**Related:** [`tensor-semantics.md`](tensor-semantics.md), [`backend-contract.md`](backend-contract.md), [`extension-op.md`](extension-op.md), [`../design/api-and-convention-freeze.md`](../design/api-and-convention-freeze.md)

---

## Purpose

This document is the normative specification for public API naming, module
shape, feature naming, and documentation-surface checks. It owns the fixed
rules implemented by `scripts/check-api-consistency.py`.

Release cleanup rationale and migration order live in
[`../design/api-and-convention-freeze.md`](../design/api-and-convention-freeze.md).
This file owns the rules that remain after the cleanup is complete.

---

## Public API Strata

Every exported item belongs to one stratum:

| Stratum | Meaning | Requirement |
| --- | --- | --- |
| Release API | Intended for downstream users. | Public rustdoc contract and runnable example when the item is callable. |
| Owner-scoped bridge | Required only because Rust crate boundaries separate owning implementation units. | Narrow API, no guide promotion, and documented owner. |
| Experimental or unsupported | Visible because an incomplete capability is intentionally exposed. | Explicit unsupported behavior and error contract. |
| Internal | Tests, planning, lowering, dispatch, cache plumbing, provider selection, or backend glue. | Private or `pub(crate)` unless a documented owner-scoped bridge is required. |

The default stratum is internal.

## Naming Rules

1. There is no root `tenferro` facade crate. User-facing docs and examples must
   import direct crates such as `tenferro_runtime`, `tenferro_ad`,
   `tenferro_gpu`, `tenferro_einsum`, `tenferro_linalg`, and `tenferro_fft`.
2. Tensor operation names use unsuffixed names for owned compact tensor inputs.
3. `_read` is reserved for APIs that explicitly accept borrowed read-oriented
   inputs such as `TensorRead`.
4. `_view` is reserved for metadata-only layout operations. Operations that
   allocate, canonicalize, execute kernels, transfer data, or materialize data
   must not use `_view`.
5. Scalar constructors use generic `TensorScalar`-bounded entry points instead
   of dtype-specific public functions such as `constant_f64`.
6. Traced tensor method and free-function names do not use a `traced_` prefix.
7. Public module namespaces may use `traced_tensor` when they identify the
   traced tensor surface as a peer of `eager_tensor`. The namespace is a
   surface stratum, not an operation-name prefix.
8. User-facing backend features use concrete backend family names such as
   `cuda` and `rocm`. Public crates must not expose a vague `gpu` feature.
9. Optional operation-specific AD support belongs behind an `autodiff` feature
   in the owning operation crate.

## API Shape Rules

1. Unary single-output traced ops are methods on `TracedTensor`.
2. Binary single-output ops use operator overloads when the operator is natural;
   otherwise they use methods.
3. Multi-output linalg and decomposition ops are free functions.
4. Einsum is a free function owned by `tenferro-einsum`.
5. Standard operation families are first-class crates, not modules under a
   broad facade.

## Documentation Checks

1. `README.md`, `docs/guides/`, and `docs/getting-started/` must not reference
   internal crates, internal graph/IR vocabulary, or deleted public paths.
2. User-facing examples must use direct public crates and must not rely on a
   root facade path.
3. Flat-buffer constructors, exports, examples, and FFI contracts must state or
   encode column-major layout expectations.
4. Rustdoc examples for release APIs must compile and run as doctests unless
   they intentionally use `compile_fail`.

## Checker Mapping

`scripts/check-api-consistency.py` implements these automated checks:

| Check category | Spec rule |
| --- | --- |
| `traced_prefix` | Naming rule 6, limited to public function and method names. |
| `read_suffix_without_read_input` | Naming rule 3. |
| `per_dtype_constructor` | Naming rule 5. |
| `public_gpu_feature` | Naming rule 8, limited to published crates. |
| `facade_path_in_user_docs` | Naming rule 1 and documentation check 2. |
| `internal_jargon_in_user_docs` | Documentation check 1. |

The concept-family matrices emitted by the checker are review aids. A matrix
difference becomes a finding only when the relevant spec, design doc, or
rustdoc contract does not explain the difference.

The release-freeze convention gate is:

```bash
python3 scripts/check-api-consistency.py --fail-on-findings
```

This command must exit `0` before the API convention freeze is considered
green. Concept-family matrices remain review aids until a specific matrix rule
is promoted into this spec.
