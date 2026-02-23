# CudaBackend Implementation Design (Issue #183)

## Context

PR #184 established the CudaBackend skeleton (Phase 0-4). This document
designs the real implementation: cuTENSOR v2 API calls for all PrimDescriptor
operations, even without GPU access, to verify correctness.

Reference: `../omeinsum-rs` cuTENSOR integration patterns.

## Key Design Decisions

1. **Library loading**: libloading runtime dlopen with CutensorVtable (20 typed function pointers)
2. **GPU memory**: DataBuffer::Gpu variant (already exists in tenferro-tensor)
3. **Resource management**: RAII wrappers for all cuTENSOR handles (Drop calls destroy)
4. **Scope**: All core + extended PrimDescriptor operations

## PrimDescriptor → cuTENSOR v2 Mapping

| PrimDescriptor      | cuTENSOR Operation             | Notes                          |
|---------------------|--------------------------------|--------------------------------|
| Contract            | cutensorContract               | Primary einsum path            |
| BatchedGemm         | cutensorContract               | Special case of Contract       |
| Trace               | cutensorContract (with eye)    | Diagonal sum via contraction   |
| AntiTrace           | cutensorContract (with eye)    | AD backward of trace           |
| AntiDiag            | cutensorContract (with eye)    | AD backward of diag            |
| Permute             | cutensorPermute                |                                |
| MakeContiguous      | cutensorPermute                | Identity permutation           |
| Reduce              | cutensorReduce                 | Sum/Max/Min reduction          |
| ElementwiseUnary    | cutensorElementwiseTrinary     | C = alpha * op(A) + beta * C   |
| ElementwiseMul      | cutensorElementwiseBinary      | C = alpha * A * B + beta * C   |

## Module Structure

```
tenferro-prims/src/
    cuda_ffi.rs  — FFI types, enums, opaque handles, CutensorVtable (20 fn ptrs)
    cuda.rs      — RAII wrappers, CudaContext, CudaPlan, CudaBackend, CutensorType trait
```

## cuda_ffi.rs — FFI Types and Vtable

### Enums

```rust
#[repr(u32)]
pub enum CutensorDataType { R_32F = 0, R_64F = 1, C_32F = 2, C_64F = 3 }

#[repr(i32)]
pub enum CutensorAlgo { Default = -1 }

#[repr(i32)]
pub enum CutensorJitMode { None = 0 }

#[repr(i32)]
pub enum CutensorWorksizePref { Min = 1, Recommended = 2, Max = 3 }

#[repr(i32)]
pub enum CutensorOperator { Identity = 1, Conj = 2 }

#[repr(i32)]
pub enum CutensorOpReduceOp { Add = 3, Max = 5, Min = 6 }
```

### Opaque Handles

All `*mut c_void`:
- `cutensorHandle_t`
- `cutensorTensorDescriptor_t`
- `cutensorOperationDescriptor_t`
- `cutensorPlanPreference_t`
- `cutensorPlan_t`

Constant pointer: `cutensorComputeDescriptor_t = *const c_void`

### CutensorVtable (20 function pointers)

```
Handle lifecycle (2):
  create, destroy

Tensor descriptor (2):
  create_tensor_descriptor, destroy_tensor_descriptor

Operation descriptors (6):
  create_contraction, create_permutation, create_reduction,
  create_elementwise_binary, create_elementwise_trinary,
  destroy_operation_descriptor

Plan (5):
  create_plan_preference, destroy_plan_preference,
  estimate_workspace_size, create_plan, destroy_plan

Execution (5):
  contract, permute, reduce,
  elementwise_binary_execute, elementwise_trinary_execute
```

### CutensorVtable::load()

```rust
impl CutensorVtable {
    pub unsafe fn load(lib: &libloading::Library) -> Result<Self> {
        // Load each symbol by name: lib.get(b"cutensorCreate\0")?
    }
}
```

## cuda.rs — RAII Wrappers and Backend

### RAII Wrappers

Each cuTENSOR resource gets a wrapper holding raw handle + Arc<CutensorVtable>:

- `CutensorHandleWrapper` — Drop calls `vtable.destroy(raw)`
- `CutensorTensorDescWrapper` — Drop calls `vtable.destroy_tensor_descriptor(raw)`
- `CutensorOpDescWrapper` — Drop calls `vtable.destroy_operation_descriptor(raw)`
- `CutensorPlanPrefWrapper` — Drop calls `vtable.destroy_plan_preference(raw)`
- `CutensorPlanWrapper` — Drop calls `vtable.destroy_plan(raw)`

### CutensorType Trait

Maps Rust scalar types to cuTENSOR types:

```rust
trait CutensorType {
    fn data_type() -> CutensorDataType;
    fn compute_descriptor() -> cutensorComputeDescriptor_t;
}

// f32 → R_32F, COMPUTE_DESC_32F
// f64 → R_64F, COMPUTE_DESC_64F
// Complex<f32> → C_32F, COMPUTE_DESC_32F
// Complex<f64> → C_64F, COMPUTE_DESC_64F
```

### CudaContext

```rust
pub struct CudaContext {
    handle: CutensorHandleWrapper,
    stream: Arc<cudarc::driver::CudaStream>,
    vtable: Arc<CutensorVtable>,
    workspace: cudarc::driver::CudaSlice<u8>,
    plan_cache: PlanCache,
}
```

### CudaPlan<T>

```rust
pub struct CudaPlan<T: Scalar> {
    plan: CutensorPlanWrapper,     // compiled cuTENSOR plan (RAII)
    desc: PrimDescriptor,          // for cache key matching
    workspace_size: u64,
    _marker: PhantomData<T>,
}
```

### CudaBackend

```rust
pub struct CudaBackend {
    _lib: libloading::Library,
}
```

### plan() Flow

For Contract/BatchedGemm/Trace/AntiTrace/AntiDiag:
1. Create tensor descriptors (A, B, C) from shapes + strides
2. Map conjugated flag → CutensorOperator::Conj or Identity
3. Create contraction operation descriptor
4. Create plan preference (default algo, no JIT)
5. Estimate workspace size
6. Grow workspace buffer if needed
7. Create plan
8. Intermediate RAII wrappers (tensor descs, op desc, plan pref) auto-dropped

For Permute/MakeContiguous:
1. Create tensor descriptors (A, B)
2. Create permutation operation descriptor
3. Same plan preference → estimate → create plan flow

For Reduce:
1. Create tensor descriptors (A, C)
2. Map ReduceOp → CutensorOpReduceOp
3. Create reduction operation descriptor
4. Same plan flow

For ElementwiseUnary:
1. Create tensor descriptors (A, C)
2. Map UnaryOp → CutensorOperator
3. Create elementwise trinary operation descriptor (unary case)
4. Same plan flow

For ElementwiseMul:
1. Create tensor descriptors (A, B, C)
2. Create elementwise binary operation descriptor
3. Same plan flow

### execute() Flow

1. Extract `device_ptr` from each `Tensor<T>.buffer().as_device_ptr()`
2. Cast alpha/beta to `*const c_void`
3. Dispatch on PrimDescriptor variant → call appropriate vtable function
4. Check cuTENSOR status code → return Result
5. (Future: create CompletionEvent, attach to output tensor)

### has_extension_for()

Returns `true` for Contract and ElementwiseMul (all scalar types).

## Error Handling

cuTENSOR status codes → tenferro_device::Error mapping:
- `CUTENSOR_STATUS_SUCCESS (0)` → Ok(())
- Other → `Error::DeviceError(format!("cuTENSOR error: {status}"))`

Helper: `fn check_cutensor_status(status: cutensorStatus_t) -> Result<()>`

## Testing Strategy

- No GPU tests in CI (feature-gated)
- Compile-time verification: `cargo build -p tenferro-prims --features cuda`
- Unit tests for CutensorType trait mapping
- Unit tests for mode label → cuTENSOR mode conversion
- Integration tests require GPU hardware (manual / future CI)
