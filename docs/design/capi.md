# C-API (FFI)

C-API for Julia, Python (JAX, PyTorch), and other languages. Exposes
tensor lifecycle, einsum, SVD (with AD rules), and DLPack interop.

---

## Position in Workspace Architecture

`tenferro-capi` sits at Layer 5, the topmost layer:

```
Layer 5: tenferro-capi          ← this crate
Layer 4: tenferro-einsum, tenferro-linalg
Layer 3: tenferro-prims
Layer 2: tenferro-tensor
```

Dependencies: `tenferro-device`, `tenferro-tensor`, `tenferro-einsum`,
`tenferro-linalg`.

Tropical extension: `tenferro-ext-tropical-capi` (separate shared library)
reuses `TfeTensorF64` handles and adds tropical einsum functions.

**Error mapping for unsupported AD modes:** `AutodiffError::ModeNotSupported`
maps to `TFE_INVALID_ARGUMENT` in the C-API. No `_frule_<algebra>_f64`
functions are provided for tropical einsum (consistent with issue #66);
if a caller were to invoke such a path internally, it would surface as
`TFE_INVALID_ARGUMENT` with a descriptive message, not a panic or
`TFE_INTERNAL_ERROR`.

---

## Design Principles

1. **Opaque pointers.** `TfeTensorF64` is an opaque `#[repr(C)]` handle.
   Host languages never see Rust internals.

2. **Status codes.** Every function takes `*mut tfe_status_t` as its
   last argument. Rust panics are caught with `catch_unwind` and
   converted to `TFE_INTERNAL_ERROR`.

3. **Stateless AD only.** Only `rrule` (VJP) and `frule` (JVP) are
   exposed. `TrackedValue` / `DualValue` / `Tape` are Rust-internal.
   Host languages manage their own AD tapes (ChainRules.jl, PyTorch
   autograd, JAX custom_vjp).

4. **f64 only** in POC phase. All functions carry the `_f64` suffix.

5. **DLPack v1.0.** Zero-copy tensor exchange across language boundaries
   (NumPy, PyTorch, JAX, DLPack.jl). Export preserves CPU/GPU logical memory
   spaces; import currently accepts only `KDLCPU` with `device_id = 0`.

6. **Copy semantics for convenience.** `tfe_tensor_f64_from_data` copies
   caller data. For zero-copy, use DLPack.

---

## Status Codes

```rust
pub type tfe_status_t = i32;

pub const TFE_SUCCESS: tfe_status_t = 0;
pub const TFE_INVALID_ARGUMENT: tfe_status_t = -1;
pub const TFE_SHAPE_MISMATCH: tfe_status_t = -2;
pub const TFE_INTERNAL_ERROR: tfe_status_t = -3;
```

---

## API Surface

### Tensor Lifecycle

| Function | Description |
|----------|-------------|
| `tfe_tensor_f64_from_data(data, len, shape, ndim, status)` | Create tensor from data (copy) |
| `tfe_tensor_f64_zeros(shape, ndim, status)` | Create zero-filled tensor |
| `tfe_tensor_f64_clone(tensor, status)` | Deep-copy (not Arc clone) |
| `tfe_tensor_f64_release(tensor)` | Free tensor (null-safe) |
| `tfe_tensor_f64_ndim(tensor)` | Query rank |
| `tfe_tensor_f64_shape(tensor, out_shape)` | Query shape into buffer |
| `tfe_tensor_f64_len(tensor)` | Query total elements |
| `tfe_tensor_f64_data(tensor)` | Raw data pointer (read-only) |

### DLPack Interop

| Function | Description |
|----------|-------------|
| `tfe_tensor_f64_to_dlpack(tensor, status)` | Export (consumes handle) |
| `tfe_tensor_f64_from_dlpack(managed, status)` | Import (takes ownership) |

**Export**: tensor handle is consumed — do not call `release` on it.
The returned `DLManagedTensorVersioned` must be consumed by the host
language, which calls the `deleter` callback when done.

**Import**: takes ownership of `DLManagedTensorVersioned`. The deleter
is called exactly once, either when the returned tensor is released or
immediately if import validation rejects the input.

### Einsum

| Function | Description |
|----------|-------------|
| `tfe_einsum_f64(subscripts, operands, n, status)` | Einsum (returns new tensor) |
| `tfe_einsum_rrule_f64(subscripts, operands, n, cotangent, grads_out, status)` | Reverse-mode AD |
| `tfe_einsum_frule_f64(subscripts, primals, n, tangents, status)` | Forward-mode AD |

rrule: `grads_out` is a caller-provided array of `n` pointers. Each
returned gradient tensor must be released by the caller.

frule: `tangents` elements may be null (zero tangent for that operand).

### SVD

| Function | Description |
|----------|-------------|
| `tfe_svd_f64(tensor, left, left_len, right, right_len, max_rank, cutoff, u_out, s_out, vt_out, status)` | SVD with dimension indices |
| `tfe_svd_rrule_f64(tensor, left, left_len, right, right_len, max_rank, cutoff, cot_u, cot_s, cot_vt, status)` | Reverse-mode AD |
| `tfe_svd_frule_f64(tensor, left, left_len, right, right_len, max_rank, cutoff, tangent, u_out, s_out, vt_out, status)` | Forward-mode AD |

SVD takes `left`/`right` dimension indices (not a pre-matricized matrix).
The capi handles permute+reshape+contiguous internally. Set `max_rank=0`
for no rank limit, `cutoff<0` for no cutoff.

rrule cotangent pointers may be null (zero cotangent for that output).
frule tangent pointer may be null.

---

## Memory Ownership

| Allocation | Freed by |
|-----------|----------|
| Tensor from `_from_data` / `_zeros` / `_clone` | `tfe_tensor_f64_release` |
| Tensor from `_from_dlpack` | `tfe_tensor_f64_release` (calls DLPack deleter) |
| Output tensor (via `**_out`) | `tfe_tensor_f64_release` |
| Gradient tensor (rrule output) | `tfe_tensor_f64_release` |
| `grads_out` array (einsum rrule) | Caller provides buffer |
| Input `data` pointer | Caller (data is copied) |
| `DLManagedTensorVersioned` from `_to_dlpack` | Consumer calls `deleter` |

---

## DLPack v1.0 Types

```rust
#[repr(C)]
pub struct DLPackVersion { pub major: u32, pub minor: u32 }

#[repr(C)]
pub struct DLDevice { pub device_type: i32, pub device_id: i32 }

#[repr(C)]
pub struct DLDataType { pub code: u8, pub bits: u8, pub lanes: u16 }

#[repr(C)]
pub struct DLTensor {
    pub data: *mut c_void,
    pub device: DLDevice,
    pub ndim: i32,
    pub dtype: DLDataType,
    pub shape: *mut i64,
    pub strides: *mut i64,   // NULL = row-major contiguous
    pub byte_offset: u64,
}

#[repr(C)]
pub struct DLManagedTensorVersioned {
    pub version: DLPackVersion,
    pub manager_ctx: *mut c_void,
    pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensorVersioned)>,
    pub flags: u64,
    pub dl_tensor: DLTensor,
}
```

Device type constants:

| Constant | Value | Description |
|----------|-------|-------------|
| `KDLCPU` | 1 | CPU |
| `KDLCUDA` | 2 | CUDA GPU memory |
| `KDLCUDA_HOST` | 3 | Pinned CUDA CPU memory |
| `KDLROCM` | 10 | ROCm GPU memory |
| `KDLROCM_HOST` | 11 | Pinned ROCm CPU memory |
| `KDLCUDA_MANAGED` | 13 | CUDA managed/unified memory |

DLPack device types map to `LogicalMemorySpace`:

| DLPack | LogicalMemorySpace |
|--------|-------------------|
| `KDLCPU` | `MainMemory` |
| `KDLCUDA_HOST`, `KDLROCM_HOST` | `PinnedMemory` |
| `KDLCUDA`, `KDLROCM` | `GpuMemory { device_id }` |
| `KDLCUDA_MANAGED` | `ManagedMemory` |

Current import support is narrower than the full mapping above:
`tfe_tensor_f64_from_dlpack` only accepts `KDLCPU` with `device_id = 0`
and `float64` dtype today.

---

## Tropical Extension (tenferro-ext-tropical-capi)

Separate shared library reusing `TfeTensorF64` handles. Algebra is
selected by function name, not tensor type (`MaxPlus<f64>` is
`#[repr(transparent)]` over `f64`).

| Function | Algebra |
|----------|---------|
| `tfe_tropical_einsum_maxplus_f64` | MaxPlus (⊕=max, ⊗=+) |
| `tfe_tropical_einsum_minplus_f64` | MinPlus (⊕=min, ⊗=+) |
| `tfe_tropical_einsum_maxmul_f64` | MaxMul (⊕=max, ⊗=×) |

Each has a corresponding `_rrule_<algebra>_f64` function (same signature as
standard einsum rrule). No `_frule_<algebra>_f64` is provided for tropical
einsum: tropical semirings (max/min) are not smooth, so JVP is not
well-defined. Only VJP (rrule via argmax) is supported.

The **smallest linear index wins** tie-break rule applies to all tropical rrule
functions: when multiple elements share the maximum (or minimum) value, the
gradient is routed to the element with the smallest linear index. This is
consistent across CPU and GPU backends.

---

## Design Decisions

1. **SVD takes dimension indices, not matrices.** Unlike `tenferro-linalg`
   (which takes pre-matricized `(m, n, *)` tensors), the C-API takes
   `left`/`right` dimension index arrays and handles matricization
   internally. This is more ergonomic for host languages that work with
   high-rank tensors.

2. **No QR/LU/eigen/solve in C-API.** Only SVD is exposed in the POC
   because it is the primary operation for tensor network applications.
   Other decompositions can be added as needed.

3. **No tape/tracked types exposed.** Host languages already have their
   own AD systems. Exposing stateless rrule/frule lets each host
   integrate naturally (Julia ChainRules, PyTorch custom_vjp, JAX
   custom_vjp).

4. **Separate tropical shared library.** Avoids bloating the core
   `tenferro-capi` with tropical code. Users who don't need tropical
   algebra don't link it.

5. **`catch_unwind` for panic safety.** All extern "C" functions wrap
   their body in `catch_unwind` to prevent Rust panics from unwinding
   through C frames (undefined behavior).

---

## Implementation Phases

### Phase 1: Tensor Lifecycle + Status Plumbing
- Implement: `tfe_tensor_f64_from_data`, `_zeros`, `_clone`, `_release`, `_ndim`, `_shape`, `_len`, `_data`
- Status plumbing: `catch_unwind` wrapper, `tfe_status_t` return convention
- Exit criteria: All tensor lifecycle functions pass null-safety and round-trip tests

### Phase 2: DLPack Import/Export
- Implement: `tfe_tensor_f64_to_dlpack`, `_from_dlpack`
- Exit criteria: Round-trip test (Rust -> DLPack -> Rust) preserves data, shape, strides; deleter is called exactly once

### Phase 3: Einsum + SVD + AD
- Implement: `tfe_einsum_f64`, `_rrule`, `_frule`; `tfe_svd_f64`, `_rrule`, `_frule`
- Exit criteria: Gradient check passes for einsum rrule/frule; SVD round-trip U*S*Vt ~= A

### Phase 4: Tropical C-API
- Implement: `tfe_tropical_einsum_maxplus_f64` etc. + `_rrule` variants
- No `_frule` for tropical (rrule-only policy)
- Exit criteria: Tropical einsum matches CPU reference; rrule gradient check passes

---

## Error Mapping

| Rust Error | `tfe_status_t` | Notes |
|-----------|---------------|-------|
| `tenferro_device::Error::InvalidArgument` | `TFE_INVALID_ARGUMENT` (-1) | |
| `tenferro_device::Error::ShapeMismatch` | `TFE_SHAPE_MISMATCH` (-2) | |
| `tenferro_device::Error::RankMismatch` | `TFE_SHAPE_MISMATCH` (-2) | Shape/rank grouped |
| `tenferro_device::Error::DeviceError` | `TFE_INTERNAL_ERROR` (-3) | |
| `tenferro_device::Error::NoCompatibleComputeDevice` | `TFE_INTERNAL_ERROR` (-3) | |
| `tenferro_device::Error::CrossMemorySpaceOperation` | `TFE_INVALID_ARGUMENT` (-1) | |
| `tenferro_device::Error::StrideError` | `TFE_INVALID_ARGUMENT` (-1) | |
| `chainrules_core::AutodiffError::ModeNotSupported` | `TFE_INVALID_ARGUMENT` (-1) | Tropical frule/hvp |
| `chainrules_core::AutodiffError::*` (other) | `TFE_INTERNAL_ERROR` (-3) | |
| Rust panic (caught by `catch_unwind`) | `TFE_INTERNAL_ERROR` (-3) | |

Status message: human-readable, non-empty. Written to an internal thread-local buffer accessible via future `tfe_last_error_message()` API.

---

## C-API Test Matrix

| Category | Test Cases |
|----------|-----------|
| NULL safety | NULL tensor pointer to all query functions; NULL status pointer |
| Invalid shape | Zero-dim shape, empty data with non-zero shape, mismatched len |
| Einsum errors | Invalid subscript string, shape mismatch between operands |
| DLPack | Unsupported dtype, device mismatch, deleter called exactly once |
| Ownership | Double release (must not crash), release after to_dlpack (must not crash) |
| Panic safety | Internal panic caught and converted to TFE_INTERNAL_ERROR |
| AD error paths | Tropical frule returns TFE_INVALID_ARGUMENT; NULL cotangent in rrule |

See [testing.md](./testing.md) for the workspace-level testing strategy.

---

## ABI Policy

### Header Generation
C headers are generated via `cbindgen` from the two FFI crates:

```bash
cbindgen --config cbindgen.toml --crate tenferro-capi --output tenferro.h
cbindgen \
  --config extension/tenferro-ext-tropical-capi/cbindgen.toml \
  --crate tenferro-ext-tropical-capi \
  --output tenferro_ext_tropical.h
```

`tenferro_ext_tropical.h` is the extension header and includes `tenferro.h`.

### Symbol Naming
All public symbols use the `tfe_` prefix. Tropical extension uses `tfe_tropical_` prefix.

### Versioning
- Shared library versioned as `libtenferro.so.{major}.{minor}.{patch}`
- ABI compatibility: patch releases are backward-compatible; minor releases may add symbols; major releases may break ABI
- Version query: `tfe_version()` returns `(major, minor, patch)` tuple

### Backward Compatibility
- Existing function signatures are frozen after 1.0
- New functions added with new names (no overloading)
- Deprecated functions kept for one major version cycle
