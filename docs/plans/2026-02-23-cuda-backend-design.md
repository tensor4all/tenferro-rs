# CUDA Backend Design (cuTENSOR via cudarc + libloading)

## Context

Issue #183: Add GPU support to tenferro-einsum via cuTENSOR.
tenferro-prims has CpuBackend fully implemented; CudaBackend is a stub.

## Key Design Decisions

1. **CUDA runtime**: cudarc crate (dlopen internally, no compile-time CUDA SDK)
2. **cuTENSOR**: libloading to dlopen `libcutensor.so`, vtable of function pointers
3. **Trait signature**: `TensorPrims::execute` takes `&[&Tensor<T>]` instead of `&[&StridedView<T>]`
   - PyTorch-aligned: operations receive Tensor directly
   - strided-rs is internal to CpuBackend only
   - `T: Scalar` (was `T: ScalarBase`; Tensor requires Scalar)
4. **DataBuffer::Gpu**: keeps raw `*mut T` pointer + release callback (no cudarc dep in tenferro-tensor)
5. **Feature gate**: `cfg(feature = "cuda")` on tenferro-prims; CI does not run GPU tests
6. **Scope**: `Standard<f64/f32>` only (tropical GPU is separate path)

## Architecture

```
einsum::<T, Alg, CudaBackend>(ctx, "ij,jk->ik", &[&a_gpu, &b_gpu])
  └─ execute_pairwise_contraction
       └─ Backend::has_extension_for::<T>(Contract) → true
            └─ CudaBackend::plan(Contract descriptor)
            └─ CudaBackend::execute(&[&tensor_a, &tensor_b], &mut tensor_c)
                 └─ extract device_ptr from each Tensor
                 └─ cutensorContract via vtable
```

## PrimDescriptor → cuTENSOR v2 Mapping

| PrimDescriptor      | cuTENSOR API              | Notes                        |
|---------------------|---------------------------|------------------------------|
| Contract            | cutensorContract          | einsum primary path          |
| Permute             | cutensorPermute           |                              |
| Reduce              | cutensorReduce            |                              |
| Trace               | cutensorReduce + stride   |                              |
| BatchedGemm         | Contract subset           |                              |
| MakeContiguous      | noop                      | cuTENSOR handles strides     |
| AntiTrace/AntiDiag  | Contract(eye, dC)         | AD backward                  |
| ElementwiseUnary    | cutensorElementwiseTrinary|                              |
| ElementwiseMul      | cutensorElementwiseBinary |                              |

## Module Structure (after split)

```
tenferro-prims/src/
    lib.rs       — TensorPrims trait, PrimDescriptor, PlanCache, helpers
    cpu.rs       — CpuBackend (uses strided-rs internally)
    cuda.rs      — CudaBackend (cfg(feature = "cuda"))
    cuda_ffi.rs  — CutensorVtable, FFI types (cfg(feature = "cuda"))
    registry.rs  — BackendRegistry
```

## Implementation Phases

### Phase 0: Module split (PR, issue #173)
Split tenferro-prims/src/lib.rs (2030 lines) into cpu.rs, gpu_stubs.rs, registry.rs.

### Phase 1: TensorPrims signature change (PR)
Change execute to take `&[&Tensor<T>]` + `&mut Tensor<T>`.
Update all backends and callers (einsum, capi, tropical, tests).

### Phase 2: Feature flag + cudarc + cuTENSOR FFI (PR)
Add cuda feature, cudarc dependency, CutensorVtable, real CudaBackend skeleton.

### Phase 3: GPU memory + core primitives (PR)
Implement Contract, Permute, Reduce via cuTENSOR. GPU alloc/transfer helpers.

### Phase 4: Remaining primitives + einsum integration (PR)
Trace, AntiTrace, AntiDiag, ElementwiseUnary/Mul, BatchedGemm.
End-to-end einsum on GPU with integration tests.
