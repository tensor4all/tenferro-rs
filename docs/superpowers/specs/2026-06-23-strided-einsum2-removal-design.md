# Strided-Einsum2 Removal Design

## Context

`tenferro-cpu` currently depends on `strided-einsum2` for the non-conjugated
Faer `dot_general` path. This blocks the tenferro v0.1 publish flow because
Cargo package validation requires every optional dependency to exist on
crates.io, even when the feature is disabled.

The dependency is narrow: `tenferro-einsum` owns parsing, planning, and public
einsum APIs. `strided-einsum2` is used only as a CPU backend adapter for
`dot_general` over strided tensor/view inputs.

## Goal

Remove the `strided-einsum2` dependency from tenferro without changing the
performance algorithm for the affected `dot_general` path.

The replacement must preserve the current optimized structure:

1. Convert `DotGeneralConfig` into left-free, right-free, contraction, and
   batch axis groups.
2. Reorder operands through metadata-only view permutation into GEMM canonical
   order:
   - lhs: `[left_free..., contract..., batch...]`
   - rhs: `[contract..., right_free..., batch...]`
   - output: `[left_free..., right_free..., batch...]`
3. Fuse each GEMM dimension group when the strided layout permits it.
4. Avoid copying operands whose fused groups can be passed directly to Faer.
5. Copy only the operand groups that cannot be represented as strided GEMM
   matrices.
6. Execute batched Faer GEMM with strided pointers, using a fast pointer-step
   loop when batch strides are contiguous.

## Non-Goals

- Do not port the public `strided-einsum2` crate, generic binary einsum API,
  trace-axis reduction API, BLAS provider layer, or scalar extension surface.
- Do not replace optimized GEMM with naive contraction loops.
- Do not make new public tenferro APIs.
- Do not change `tenferro-einsum` parsing or contraction semantics.

## Design

Move the dot-general-specific preparation algorithm into `tenferro-cpu` under
the existing `gemm` ownership boundary.

The implementation should add an internal Faer preparation path that mirrors
the relevant `strided-einsum2` behavior while using tenferro-owned tensor
types, buffer pools, errors, and `CpuContext`.

The path should produce a small internal plan containing:

- output shape `[lhs_free..., rhs_free..., batch...]`
- `m`, `n`, `k`, and batch dimensions
- per-operand row, column, and batch strides
- optional pooled temporary buffers for operands that need col-major copies

For each input operand, the planner checks the two inner groups:

- lhs: left-free group and contraction group
- rhs: contraction group and right-free group

If both groups are fusable in canonical col-major logical order, the plan keeps
the original host pointer, offset, and strides. If either group is not fusable,
the path copies that operand into a pooled col-major temporary and records the
temporary layout. This preserves the current `strided-einsum2` performance
property: non-contiguous cases copy only what is necessary instead of forcing a
full fallback materialization at the public `TensorRead` boundary.

The owned compact tensor path and the read/view path should share the same
preparation logic. The existing `typed_faer_gemm` direct path may remain as a
fast cached case, but it must no longer be the only optimized path once
`strided-einsum2` is removed.

The conjugated Faer path should continue to use Faer conjugation flags through
`FaerGemm::strided_gemm_with_conj`. The non-conjugated and conjugated paths
should share the same preparation and batched GEMM execution where practical.

## Error Handling

Use tenferro CPU errors, not `strided-einsum2` errors. Publicly reachable
validation remains in the tenferro `dot_general` validation path:

- axis bounds
- duplicate axes
- contracting and batch role conflicts
- lhs/rhs contracting size mismatches
- lhs/rhs batch size mismatches
- checked products and pointer-offset arithmetic

Backend-buffer inputs should continue to return `Ok(None)` from read-direct
helpers where the caller already materializes or routes through the normal
backend path.

## Dependency Changes

Remove `strided-einsum2` from:

- workspace dependencies
- `tenferro-cpu` dependencies
- `cpu-faer` and BLAS provider feature forwarding
- provider feature contract tests
- publish-readiness documentation

Keep the existing `strided-kernel` dependency where tenferro still uses it for
elementwise, broadcast, and copy kernels.

## Testing

Add or update tests to lock the algorithmic contract:

- `tenferro-cpu` no longer mentions `strided-einsum2` in manifests or provider
  feature contract expectations.
- Faer `dot_general` produces the same results for compact owned tensors.
- Faer `dot_general_read` handles a transposed host view directly without
  falling back to public `TensorRead` materialization.
- Faer handles a non-canonical contraction by copying only the non-fusable
  operand group and still executing through GEMM.
- Complex non-conjugated and conjugated cases remain correct.
- Zero-size or zero-contraction cases preserve current zero-fill semantics.

Verification for the release branch should include:

```bash
cargo fmt --all --check
cargo test -p tenferro-cpu --features cpu-faer
cargo package -p tenferro-cpu --allow-dirty
```

If time permits, compare `dot_general_overhead` or an existing einsum benchmark
before and after the change for representative compact, transposed-view, and
non-canonical-layout cases.

## Success Criteria

- `rg "strided-einsum2" Cargo.toml crates/tenferro-cpu` returns no dependency
  or code references.
- `tenferro-cpu` packages without requiring `strided-einsum2` on crates.io.
- The optimized Faer algorithm remains GEMM-based and operand-local-copy based,
  not public-boundary materialization based.
- Tests cover the layout cases that previously depended on
  `strided-einsum2::dot_general_with_backend_into`.
