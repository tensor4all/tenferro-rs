# tenferro-internal-device

Device abstraction and shared error types for the tenferro workspace.

---

## Position in Workspace Architecture

`tenferro-internal-device` is a **shared foundation crate** depended upon by every other
tenferro crate. It has no tensor or algebra dependency — only `thiserror`. Its
role is to:

- Define the vocabulary for *where data lives* (`LogicalMemorySpace`) and
  *which hardware computes* (`ComputeDevice`).
- Classify tensor operations for capability queries (`OpKind`).
- Provide the single `Error` / `Result` pair used across the entire workspace.

---

## Core Types

### LogicalMemorySpace

Describes where tensor buffer data physically resides. Separates the concept
of data location from execution hardware: a tensor in `MainMemory` can be
processed by any CPU, while a tensor in `GpuMemory { device_id: 1 }` can only
be processed by a device with access to that GPU.

| Variant | Fields | DLPack `device_type` |
|---------|--------|----------------------|
| `MainMemory` | — | `kDLCPU` (1) |
| `PinnedMemory` | — | `kDLCUDAHost` (3) / `kDLROCMHost` (11) |
| `GpuMemory` | `device_id: usize` | `kDLCUDA` (2) / `kDLROCM` (10) |
| `ManagedMemory` | — | `kDLCUDAManaged` (13) |

`GpuMemory.device_id` matches the DLPack `device_id` field (zero-based GPU
index). For `MainMemory`, `PinnedMemory`, and `ManagedMemory` the DLPack
`device_id` is always 0.

### ComputeDevice

Identifies the hardware unit that executes tensor kernels. A compute device is
independent of where the data resides — the backend is responsible for
validating that it can access the required memory space.

| Variant | Fields | `Display` format |
|---------|--------|-----------------|
| `Cpu` | `device_id: usize` | `cpu:<id>` |
| `Cuda` | `device_id: usize` | `cuda:<id>` |
| `Rocm` | `device_id: usize` | `rocm:<id>` |

`device_id = 0` is the default device for each variant.

```rust
use tenferro_device::ComputeDevice;

let dev = ComputeDevice::Cpu { device_id: 0 };
assert_eq!(format!("{dev}"), "cpu:0");
```

### OpKind

Classifies a tensor operation for device-capability filtering. Passed to
`preferred_compute_devices` so callers can query which hardware can execute a
specific class of work on a given memory space.

| Variant | Description |
|---------|-------------|
| `Contract` | General tensor contraction |
| `BatchedGemm` | Batched matrix-matrix multiply |
| `Reduce` | Reduction (sum, max, min) over one or more modes |
| `Trace` | Diagonal contraction of paired modes |
| `MakeContiguous` | Materialize a view into a contiguous dense tensor |
| `ElementwiseMul` | Element-wise multiplication |

---

## Device Selection

### `preferred_compute_devices`

```rust
pub fn preferred_compute_devices(
    space: LogicalMemorySpace,
    op_kind: OpKind,
) -> Result<Vec<ComputeDevice>>
```

Returns a list of compute devices capable of executing `op_kind` on data
residing in `space`, ordered from most preferred to least preferred.

**Selection rules (planned, not yet implemented):**

| Memory space | Typically preferred devices |
|---|---|
| `MainMemory` | `Cpu { device_id: 0 }` |
| `PinnedMemory` | `Cuda { device_id: 0 }` first, then `Cpu { device_id: 0 }` |
| `GpuMemory { device_id }` | `Cuda { device_id }` or `Rocm { device_id }` |
| `ManagedMemory` | `Cuda { device_id: 0 }` first, then `Cpu { device_id: 0 }` |

Returns `Error::NoCompatibleComputeDevice` if no registered backend can handle
the requested combination.

---

## Error Types

### Error enum

The single `Error` type used by every tenferro crate. Upstream crates map their
own errors into this enum's variants.

| Variant | Fields | `Display` format |
|---------|--------|-----------------|
| `ShapeMismatch` | `expected: Vec<usize>`, `got: Vec<usize>` | `shape mismatch: expected [...], got [...]` |
| `RankMismatch` | `expected: usize`, `got: usize` | `rank mismatch: expected N, got M` |
| `DeviceError` | `String` | `device error: <msg>` |
| `NoCompatibleComputeDevice` | `space: LogicalMemorySpace`, `op: OpKind` | `no compatible compute device for <op> on <space>` |
| `CrossMemorySpaceOperation` | `left: LogicalMemorySpace`, `right: LogicalMemorySpace` | `cross-memory-space operation between <left> and <right>` |
| `InvalidArgument` | `String` | `invalid argument: <msg>` |
| `StrideError` | `String` | `stride error: <msg>` |

`CrossMemorySpaceOperation` carries both operand spaces so diagnostics can show
exactly which spaces were mixed. `StrideError` wraps a plain `String` rather
than using `#[from] strided_view::StridedError` to avoid a dependency on
`strided-view` in this crate.

```rust
use tenferro_device::Error;

let err = Error::InvalidArgument("bad index".into());
assert!(err.to_string().contains("bad index"));
```

### Relationship with `tenferro-internal-tensor::effective_compute_devices`

`Tensor<T>` carries an optional `preferred_compute_device: Option<ComputeDevice>`
field. When resolving which device to use for an operation:

1. If `preferred_compute_device` is `Some(dev)`, that device is used directly
   (no call to `preferred_compute_devices`).
2. If it is `None`, the tensor's `LogicalMemorySpace` and the `OpKind` are
   passed to `preferred_compute_devices` to obtain the ranked list.

This lets callers pin a specific GPU without going through the global selection
logic, which is useful when constructing tensors on a non-default device.

---

## Dependencies

- `thiserror` only — no `strided-view`, no `strided-traits`, no algebra crates.

---

## Design Decisions

### 1. LogicalMemorySpace vs ComputeDevice separation

Keeping "where data lives" distinct from "which hardware computes" mirrors the
DLPack model and avoids conflating two orthogonal concerns. It allows, for
example, a `MainMemory` tensor to be processed by `Cpu { device_id: 0 }` while
a `GpuMemory { device_id: 1 }` tensor is processed by `Cuda { device_id: 1 }`,
with explicit transfer required to cross boundaries.

### 2. StrideError(String) instead of #[from] StridedError

Using `#[from] strided_view::StridedError` would add `strided-view` as a
dependency of `tenferro-internal-device`, pulling it into every downstream crate even
when only error types are needed. A plain `StrideError(String)` avoids this
coupling: callers that use `strided-view` convert errors at the boundary with
`.map_err(|e| Error::StrideError(e.to_string()))`.

### 3. CrossMemorySpaceOperation carries left/right fields for diagnostics

An error message that reports both operand memory spaces (`left` and `right`)
makes it immediately obvious which tensor is on which space, without the caller
having to re-inspect the tensors. This is especially useful in multi-step
einsum trees where the errant pair may be two levels deep.
