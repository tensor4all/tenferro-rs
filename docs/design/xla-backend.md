# XLA Backend Design

Date: 2026-06-14
Status: Experimental implementation
Related issue: https://github.com/tensor4all/tenferro-rs/issues/984

## Summary

The XLA path is a peer executor over `GraphProgram` lowering views. It is
implemented in `tenferro-xla`, not in `tenferro-runtime`, and it does not
implement `TensorBackend`.

```text
GraphProgram lowering view
  |
  v
tenferro_xla::lower_to_stablehlo()
  |
  v
StableHLO MLIR text
  |
  v
OpenXLA execution check or runtime-loaded PJRT plugin
```

The native path remains:

```text
GraphProgram -> GraphExecutor<B: TensorBackend>
```

## Runtime Dependency Boundary

XLA and PJRT are optional runtime dependencies. The `pjrt` feature enables
dynamic loading with `libloading`, but tenferro does not link a PJRT library at
compile time.

The loader contract is:

- `TENFERRO_PJRT_PLUGIN` points at the default PJRT plugin shared library.
- `TENFERRO_PJRT_GPU_PLUGIN` is available for GPU-specific scripts.
- The plugin must export `GetPjrtApi`.
- Missing variables, empty variables, missing files, and missing symbols return
  explicit errors.

This keeps the XLA boundary out of `tenferro-runtime` and avoids introducing
XLA as a transitive dependency for users who only need native execution.

## StableHLO Subset

The first lowering implementation supports exact static shapes, `F32`, `F64`,
and this operation set:

- `Constant`
- `Add`
- `Multiply`
- `Negate`
- `Convert`
- `Reshape`
- `BroadcastInDim`
- `Transpose`
- `ReduceSum`
- `DotGeneral`

Unsupported dtypes, non-exact extents, extension operations without a
fixed-shape standard-op lowering, and unsupported `ExecOp` variants fail
before PJRT compilation.

Operation-family APIs can still reach this path when they build or expose a
fixed-shape lowering to supported standard operations. For example,
fixed-shape N-ary einsum plans that `tenferro-einsum` expands into
`dot_general` operations lower like any other standard traced graph.

## Shape and Layout

StableHLO tensor types encode logical dimension order, not physical host memory
order. tenferro's host tensors are compact column-major. PJRT host transfers may
expect or return C-contiguous host bytes, so `tenferro-xla` owns explicit
conversion helpers at the boundary.

`dot_general` also has a logical ordering mismatch. StableHLO batched
`dot_general` produces `[batch..., lhs_free..., rhs_free...]`, while tenferro's
`DotGeneralConfig` produces `[lhs_free..., rhs_free..., batch...]`. The lowering
emits a StableHLO `transpose` after batched `dot_general` so the result shape
matches tenferro's existing contract.

## Verification

The implementation has three verification layers:

- Rust lowering tests for emitted StableHLO structure and unsupported errors.
- `pjrt` feature tests for runtime plugin env-var loading and `dlopen` errors.
- An environment-gated execution test that writes generated StableHLO to a
  temporary file and runs OpenXLA's `run_hlo_module` when
  `TENFERRO_XLA_RUN_HLO_MODULE` is set. This includes a fixed-shape N-ary
  einsum module after extension standard-op lowering.

The external execution test is intentionally environment-gated. Normal CI does
not require a local OpenXLA checkout or GPU, but a configured developer machine
can run the same generated StableHLO through `Host` or `CUDA` platforms.

## Future Work

- Bind the remaining PJRT C API calls needed for compile, upload, execute, and
  download directly from Rust.
- Add executable cache entries keyed by StableHLO fingerprint, plugin identity,
  platform, compiler options, and layout policy.
- Extend dtype and operation coverage after each new lowering is verified by
  OpenXLA execution.
- Add direct native CPU-vs-XLA value comparisons once the Rust PJRT execution
  wrappers are complete.
