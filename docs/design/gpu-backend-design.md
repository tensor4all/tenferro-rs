# GPU Backend Design

This document details the GPU backend design for tenferro-rs, covering
both CUDA (cuTENSOR) and ROCm (hipTENSOR) backends.

**Purpose**: Detect potential problems before implementation begins.
Define the types, modules, and API mapping needed for GPU support.

See [tensor-prims.md](./tensor-prims.md) for the semiring/scalar/analytic
primitive families and the plan-cache key policy that applies to both CPU and
GPU backends.

---

## Design Principles

1. **Two separate backends** — `CudaBackend` and `RocmBackend` as distinct
   types in `tenferro-prims`. Subtle API differences and future custom
   kernel needs justify separate implementations over a unified vtable.
2. **Runtime dlopen** — `libloading` loads cuTENSOR/hipTENSOR shared
   libraries at runtime. No compile-time GPU SDK dependency. Caller
   (Julia/Python) provides the `.so` path.
3. **Layer 1 core infrastructure** — GPU backends are not extensions
   (unlike `tenferro-ext-tropical`). They live in `tenferro-prims` alongside
   `CpuBackend`.
4. **Family-native primitives** — each GPU backend implements
   `TensorSemiringCore<Standard<S>>`, `TensorSemiringFastPath<Standard<S>>`,
   and additional families as coverage grows. `einsum` remains backend-agnostic
   via `EinsumBackend`.
5. **Prims basic + Contract first** — the minimum set for einsum to work.
6. **cuTENSOR v2 API** — describe → plan → execute pattern, matching
   tenferro's family-descriptor → plan → execute model.
7. **Naming convention** — platform names (`Cuda`, `Rocm`), not API
   names (`Hip`). `ComputeDevice::Hip` renamed to `ComputeDevice::Rocm`.

---

## Module Structure

```
tenferro-prims/src/
    lib.rs          — family traits, descriptors, shared types
    cpu/mod.rs      — CpuBackend, CpuContext, CpuPlan
    cpu/planning.rs      — semiring-core / fast-path planning
    cpu/execution.rs     — semiring execution dispatch
    cpu/batched_gemm.rs  — CPU GEMM execution paths
    cpu/contract.rs      — contract fallback + GEMM specialization
    cpu/reduction.rs     — reduce/trace/anti-trace/anti-diag kernels
    cpu/gemm_support.rs  — shared GEMM helpers and dtype dispatch
    cpu/scratch.rs       — BLAS scratch-pool management
    cuda/mod.rs     — CudaBackend, CudaContext, CudaPlan
    cuda/planning.rs     — cuTENSOR descriptor/plan builders
    cuda/execution.rs    — family dispatch for plan/execute
    cuda/scalar_type.rs  — scalar dtype/compute-descriptor mapping
    cuda/wrappers.rs     — RAII wrappers for cuTENSOR handles/descriptors/plans
    rocm.rs         — RocmBackend, RocmContext, RocmPlan, RocmTensorVtable
    registry.rs     — BackendRegistry
```
The current CPU backend is already split under `src/cpu/` for the same
reason as CUDA: planning, execution, scratch management, and contract/GEMM
specialization should not collapse back into one dispatcher file.

---

## Key Types

### CudaBackend / RocmBackend

```rust
// cuda/mod.rs
pub struct CudaBackend {
    vtable: CutensorVtable,
    handle: *mut c_void,    // cutensorHandle_t
    _lib: libloading::Library,
}

pub struct CudaContext {
    stream: *mut c_void,    // cudaStream_t
    workspace: Vec<u8>,     // GPU workspace (resizable)
    plan_cache: PlanCache,
}

pub enum CudaPlan<T: ScalarBase> {
    Contract { plan_handle: *mut c_void, workspace_size: usize, _marker: PhantomData<T> },
    Reduce { plan_handle: *mut c_void, workspace_size: usize, _marker: PhantomData<T> },
    Trace { plan_handle: *mut c_void, workspace_size: usize, _marker: PhantomData<T> },
    AntiTrace { _marker: PhantomData<T> },  // Composed via Contract(eye, ∂C)
    AntiDiag { _marker: PhantomData<T> },   // Composed via Contract(eye, ∂C)
    ElementwiseUnary { _marker: PhantomData<T> },
    ElementwiseMul { _marker: PhantomData<T> },
    BatchedGemm { _marker: PhantomData<T> },  // Via Contract subset
    MakeContiguous { _marker: PhantomData<T> },  // explicit materialization path
}

impl<S: ScalarBase> TensorSemiringCore<Standard<S>> for CudaBackend {
    type Plan = CudaPlan<S>;
    type Context = CudaContext;
    // ...
}
```

`RocmBackend` follows the same structure with `RocmTensorVtable`,
`RocmContext`, `RocmPlan`.

### BackendRegistry

```rust
// registry.rs
pub struct BackendRegistry {
    pub cpu: CpuBackend,
    pub cuda: Option<CudaBackend>,
    pub rocm: Option<RocmBackend>,
}

impl BackendRegistry {
    pub fn new() -> Self { /* CPU only */ }
    pub fn load_cutensor(&mut self, path: &str) -> Result<()> { ... }
    pub fn load_hiptensor(&mut self, path: &str) -> Result<()> { ... }
}
```

---

## cuTENSOR / hipTENSOR API Mapping

### Family Descriptor → GPU API

| Family descriptor | cuTENSOR v2 API | hipTENSOR API | Notes |
|---|---|---|---|
| `SemiringFastPathDescriptor::Contract` | `cutensorContract` | `hiptensorContraction` | 最優先。einsum動作に必須 |
| Tensor view `permute` + optional materialize | metadata-only + optional `cutensorPermute`/copy | 同左 | `Permute` prim は削除済み |
| `SemiringCoreDescriptor::ReduceAdd` | `cutensorReduce` | `hiptensorReduction` | |
| `SemiringCoreDescriptor::Trace` | `cutensorReduce` on diagonal | `hiptensorReduction` on diagonal | stride trick + reduce |
| `SemiringCoreDescriptor::BatchedGemm` | Contract subset (mode制限) | 同左 | Contract経由 |
| `SemiringCoreDescriptor::MakeContiguous` | 不要 | 不要 | GPU はstrideをネイティブに受け付け |
| `SemiringCoreDescriptor::AntiTrace` | Contract(eye, ∂C) | 同左 | コアprimだがContract合成で実装 |
| `SemiringCoreDescriptor::AntiDiag` | Contract(eye, ∂C) | 同左 | 同上 |
| `ScalarPrimsDescriptor::*` | `cutensorElementwiseTrinary` / reduction APIs | 同等API | scalar family |
| `SemiringFastPathDescriptor::ElementwiseBinary` | `cutensorElementwiseBinary` | `hiptensorElementwiseBinary` | semiring fast path |

### AntiTrace / AntiDiag の GPU 実装

AntiTrace と AntiDiag はコアプリミティブ（全バックエンド実装必須）だが、
GPU ではカスタムカーネルを書かずに Contract で合成できる。

```
anti_trace: ∂C[j,k] → ∂A[i,j,i',k] = eye[i,i'] × ∂C[j,k]
anti_diag:  ∂C[i,j] → ∂A[i,i',j]   = eye[i,i'] × ∂C[i,j]
```

どちらも `Contract(eye(I), ∂C)` で表現可能。cuTENSOR/hipTENSOR が
ネイティブに処理する。CPU 側は単純ループで実装し、strided-einsum2 に
依存しない。

### Vtable 設計

```rust
// CutensorVtable — cuTENSOR v2 function pointers
struct CutensorVtable {
    // Handle lifecycle
    create: Symbol<unsafe extern "C" fn(*mut cutensorHandle_t) -> cutensorStatus_t>,
    destroy: Symbol<unsafe extern "C" fn(cutensorHandle_t) -> cutensorStatus_t>,

    // Tensor descriptor
    create_tensor_descriptor: Symbol<unsafe extern "C" fn(...) -> cutensorStatus_t>,
    destroy_tensor_descriptor: Symbol<unsafe extern "C" fn(...) -> cutensorStatus_t>,

    // Contraction (highest priority)
    create_contraction_descriptor: Symbol<unsafe extern "C" fn(...) -> cutensorStatus_t>,
    create_contraction_plan: Symbol<unsafe extern "C" fn(...) -> cutensorStatus_t>,
    contract: Symbol<unsafe extern "C" fn(...) -> cutensorStatus_t>,

    // Permutation
    create_permutation: Symbol<unsafe extern "C" fn(...) -> cutensorStatus_t>,
    permute: Symbol<unsafe extern "C" fn(...) -> cutensorStatus_t>,

    // Reduction
    create_reduction: Symbol<unsafe extern "C" fn(...) -> cutensorStatus_t>,
    reduce: Symbol<unsafe extern "C" fn(...) -> cutensorStatus_t>,

    // Element-wise
    elementwise_binary: Symbol<unsafe extern "C" fn(...) -> cutensorStatus_t>,
}
```

`RocmTensorVtable` も同構造。関数名プレフィックスが `hiptensor` に変わる。

---

## Tensor Clone / Conj Strategy

### 方針: Arc 共有 + PyTorch パターン

`DataBuffer<T>` は `Arc<BufferInner<T>>` で内部ストレージを共有する。
PyTorch と同じセマンティクス:

| 操作 | PyTorch | tenferro | コスト |
|------|---------|----------|--------|
| 浅いコピー | `tensor.clone()` (view) | `tensor.clone()` | O(1), Arc refcount++ |
| 深いコピー | `tensor.detach().clone()` | `MakeContiguous` / explicit materialize | O(n) |
| 共役 | `tensor.conj()` (lazy) | `tensor.conj()` / `tensor.into_conj()` | O(1), flag flip |
| 共役実体化 | `torch.resolve_conj()` | `Backend::resolve_conj()` | O(n) |
| 共役チェック | `tensor.is_conj()` | `tensor.is_conjugated()` | O(1) |

### 設計

```rust
// tenferro-tensor: DataBuffer は Arc で共有
pub struct DataBuffer<T> {
    inner: Arc<BufferInner<T>>,
}

// clone() = Arc refcount++（浅いコピー）
// conj() = Arc refcount++ + conjugated flag flip（lazy）
// as_mut_slice() は Arc::get_mut で排他性チェック
```

### 利点

- `Differentiable::Tangent: Clone` が自然に満たされる
- `conj()` が CPU/GPU 統一的に lazy（PyTorch 準拠）
- `clone_tensor()` が不要（浅い clone は Tensor で完結）
- 深いコピーは既存の prims 操作で表現可能
- tenferro-prims → tenferro-tensor 依存は `resolve_conj()` のみ

### 依存関係への影響

`resolve_conj()` が `Tensor<T>` を引数に取るため、`tenferro-prims` →
`tenferro-tensor` の依存が追加される。

```
tenferro-algebra
    ├────────────────────┐
    ↓                    ↓
tenferro-prims ──→ tenferro-tensor   (prims depends on tensor)
    │                    │
    └──────────┬─────────┘
               ↓
          tenferro-einsum
```

---

## GPU Memory and DataBuffer

### DataBuffer GPU variant

```rust
// tenferro-tensor/src/buffer.rs
enum BufferInner<T> {
    Owned(Vec<T>),                    // CPU, Rust-owned
    External { ptr, len, release },   // CPU, externally-owned (DLPack)
    Gpu {                             // GPU device memory
        device_ptr: *mut T,
        len: usize,
        space: LogicalMemorySpace,    // GpuMemory { device_id }
        release: Option<Box<dyn FnOnce() + Send>>,
    },
}
```

- `as_slice()` — CPU バッファ用。GPU variant ではエラーを返す。
  この関数はそもそも CPU バッファのデータアクセスが目的。
- `as_device_ptr()` — GPU バッファ用。デバイスポインタを返す。
  cuTENSOR/hipTENSOR FFI に渡す。
- GPU メモリの alloc/free は vtable 経由（`cudaMalloc`/`cudaFree` 等）。
  release callback に GPU free 関数を設定。

### CompletionEvent

```rust
// tenferro-tensor/src/completion_event.rs
pub struct CompletionEvent {
    inner: CompletionEventInner,
}

enum CompletionEventInner {
    Noop,                           // CPU (既存)
    Cuda { event: *mut c_void },    // cudaEvent_t
    Rocm { event: *mut c_void },    // hipEvent_t
}
```

- `record(stream)`: イベントを stream に記録
- `wait()`: CPU 側で完了を待機
- `is_complete()`: 非ブロッキング完了チェック
- stream/event の create/destroy も vtable 経由

---

## Conjugation on GPU

`Tensor::conj()` は現在すでに lazy で、`Arc` 共有のまま
`conjugated` フラグを反転するだけである。

### 対応方針

cuTENSOR は tensor descriptor に `CUTENSOR_OP_CONJ` を指定でき、
lazy conjugation をネイティブにサポートする。

1. **Tensor の conjugated フラグを維持** — メタデータのみ（zero-cost）
2. **plan() 時に descriptor に CONJ op を設定** — cuTENSOR が内部処理
3. **materialize が必要な場合** — `resolve_conj()` などの explicit path を使う

standalone `conj()` 自体は CPU/GPU とも lazy のままでよい。実データ化が
必要な場面だけ explicit materialization path を通す。

---

## POC Skeleton Changes

| 変更 | ファイル | 内容 |
|------|---------|------|
| `ComputeDevice::Hip` → `Rocm` | tenferro-device/src/lib.rs | enum variant 名変更、doc 更新 |
| CudaBackend stub | tenferro-prims/src/lib.rs | 型定義 + `todo!()` |
| RocmBackend stub | tenferro-prims/src/lib.rs | 型定義 + `todo!()` |
| BackendRegistry stub | tenferro-prims/src/lib.rs | 型定義 |
| CpuBackend 分離 | tenferro-prims/src/cpu/ | planning / execution / GEMM / scratch を分離 |
| DataBuffer Gpu variant | tenferro-tensor/src/buffer.rs | BufferInner に Gpu 追加 |
| CompletionEvent 拡張 | tenferro-tensor/src/completion_event.rs | Cuda/Rocm variant 追加 |
| libloading 依存追加 | workspace Cargo.toml | workspace dependency |
| Tensor conjugated flag | tenferro-tensor/src/tensor/mod.rs | lazy conjugation 用 |
| Arc\<BufferInner\> | tenferro-tensor/src/buffer.rs | DataBuffer を Arc 共有に変更。clone() は浅い、conj() は lazy |
| resolve_conj stubs | tenferro-prims/src/lib.rs | 各バックエンドに resolve_conj() 追加 |
| `ScalarUnaryOp::Conj` | tenferro-prims scalar family | resolve_conj 用の scalar unary op |
| prims → tensor dep | tenferro-prims/Cargo.toml | resolve_conj が Tensor<T> を扱うため |

---

## Tropical GPU Path: Custom Kernels

cuTENSOR and hipTENSOR implement standard arithmetic only (`+`, `*`, `f32`,
`f64`, complex). They have no mechanism for user-defined semiring operations
such as `(max, +)` or `(max, ×)`. Tropical algebra GPU support therefore
requires a **separate custom kernel path**, independent of the cuTENSOR path.

### Boundary between the two paths

| Algebra | GPU path | Library |
|---------|----------|---------|
| `Standard<f32/f64/Complex>` | cuTENSOR / hipTENSOR | dlopen at runtime |
| `MaxPlus<T>`, `MinPlus<T>`, `MaxMul<T>` | custom kernels | separate integration target |
| User-defined algebra | custom kernels (user-provided) | user crate |

The `CudaBackend` / `RocmBackend` types implement the standard primitive
families (`TensorSemiringCore<Standard<S>>`, `TensorSemiringFastPath<Standard<S>>`,
and standard scalar/analytic families) only. They do not implement the same
families for `MaxPlus<S>` or any other
non-standard algebra — that would be a type error at compile time.

### Tropical GPU implementation target

For tropical argmax-capable GPU flows the integration target is a library
such as `tropical-gemm-cuda` (or an equivalent CUDA/HIP kernel library).
The expected integration shape is:

```rust
// tenferro-ext-tropical (future GPU support)
pub struct TropicalCudaBackend {
    _lib: libloading::Library,   // tropical-gemm-cuda.so, loaded at runtime
}

impl TensorSemiringCore<MaxPlus<f64>> for TropicalCudaBackend {
    // delegates to tropical-gemm-cuda kernels, not cuTENSOR
    type Plan = TropicalCudaPlan<f64>;
    …
}
```

This backend lives in `tenferro-ext-tropical`, not in `tenferro-prims`, preserving
the same separation used for the CPU tropical path.

### Argmax state for AD

Tropical argmax-capable variants (e.g. max-plus Viterbi) need to store the
argmax index tensor alongside the result so that the backward pass can
reconstruct the gradient path. This is algebra-specific state that cuTENSOR
cannot carry; it must be allocated and managed by the custom kernel. When
designing the tropical GPU backend, the plan type must accommodate an optional
argmax buffer:

**Tie-break contract**: The argmax buffer must store the smallest linear index
when ties occur. Custom GPU kernels must implement this deterministically
(e.g., via `atomicMin` on the index when values are equal).

```rust
enum TropicalCudaPlan<T: ScalarBase> {
    Contract {
        // argmax buffer allocated on GPU alongside the output
        argmax_device_ptr: Option<*mut u64>,
        …
        _marker: PhantomData<T>,
    },
    …
}
```

---

## Problem Catalog

| ID | Problem | Severity | Resolution |
|---|---|---|---|
| G1 | GPU メモリ管理 | Medium | DataBuffer に Gpu variant 追加。`as_slice()` は CPU 用（GPU ではエラー）。`as_device_ptr()` を追加。alloc/free は vtable 経由 |
| G2 | Workspace 管理 | Medium | GpuContext 内に可変長 workspace。`plan()` が必要サイズ報告、`execute()` が自動拡張 |
| G3 | Stream/Event 同期 | Medium | CompletionEvent に Cuda/Rocm variant。vtable 経由で record/wait |
| G4 | Plan cache key に stride 含む | Low | cuTENSOR plan は stride 依存。PlanCacheKey に shapes + strides 含める |
| G5 | FFI エラー処理 | Medium | vtable 呼び出しの status code を `Error::DeviceError` に map |
| G6 | cuTENSOR v1→v2 API 差異 | Low | v2 ベース（describe → plan → execute）で統一 |
| G7 | hipTENSOR 成熟度 | Medium | JIT 未対応、block-sparse 未対応。制限事項として記録 |
| G8 | カスタムカーネル (将来) | Low | AntiTrace/AntiDiag は Contract 合成で対応。将来カーネルが必要になれば別クレート |
| G9 | Multi-GPU | Low | GpuContext = per-device。複数 GPU = 複数 Context |
| G10 | `ComputeDevice::Hip` → `Rocm` rename | Low | POC 修正で対応 |
| G11 | Conjugation on GPU | Medium | cuTENSOR は `CUTENSOR_OP_CONJ` でlazy conjugation をサポート。Tensor に conjugated フラグ追加。standalone `conj()` は CPU 転送が必要 |
| G12 | strided-einsum2 / omeinsum-rs 廃止 | — | 両方廃止予定。アルゴリズムは参考元として参照するが依存しない。tenferro-prims / tenferro-einsum に自前実装 |
| G13 | Tensor Clone / Conj | Medium | DataBuffer を `Arc<BufferInner>` で共有。clone() は浅い（refcount++）。conj() は lazy（flag flip + refcount++）。深いコピーは `MakeContiguous` などの materialize path 経由。resolve_conj() を各バックエンドに提供。PyTorch 準拠 |
