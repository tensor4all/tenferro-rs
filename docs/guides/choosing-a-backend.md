# Choosing A Backend

Choose a backend in two steps. First choose the execution backend that owns
the device and its synchronization boundary. If that backend is CPU, choose a
dense CPU provider separately. The important choice is control: a provider
that owns its own workers can deliver peak library performance, but tenferro
cannot make the same affinity, nesting, and thread-budget guarantees around
it.

## Execution Backend

| Backend | Device | Hard requirements | Cargo feature | Threading and synchronization | Good fit |
| --- | --- | --- | --- | --- | --- |
| CPU | Host CPUs | At least one CPU provider; `native` is the default | `tenferro-cpu` `native` or `blas` | `CpuBackend` admits an operation to its selected domain. faer and native kernels use that domain; CPU calls return synchronously. | Controlled CPU and NUMA execution, dense linalg, and host pipelines |
| CUDA | NVIDIA GPU | CUDA runtime plus the NVIDIA library stack required by the operation, including cuBLAS/cuSOLVER/cuTENSOR where applicable; missing libraries are typed errors | `tenferro-gpu` `cuda` | The CUDA runtime owns the stream and device work. Eager launches are asynchronous until an explicit synchronize, download, or host inspection. | NVIDIA production workloads and CUDA library-backed dense operations |
| WebGPU | A WebGPU adapter/device and queue | A working wgpu/WebGPU implementation; coverage is experimental and operation-specific | `tenferro-gpu` `webgpu` | wgpu owns the device queue. Submissions are asynchronous; queue synchronization or download establishes a host-visible boundary. | Portable GPU experiments and Apple shared CPU/Metal paths |
| XLA/PJRT | A PJRT addressable device | A PJRT plugin shared library; the `pjrt` loader returns a typed error when it is absent or invalid | `tenferro-xla` `pjrt` | The PJRT plugin owns device execution and intra-op scheduling. tenferro controls lowering and invocation, not the plugin's worker pool. | Static traced programs and deployments already using XLA plugins |

CUDA does not use a native CubeCL kernel as a silent fallback when a required
NVIDIA library is unavailable. The operation returns a typed library/provider
error instead. XLA likewise rejects an unavailable plugin or unsupported
lowering instead of changing execution backends behind the caller's back. See
the [CUDA and NVIDIA library compatibility matrix](devices-and-gpu.md#cuda-and-nvidia-library-compatibility)
for supported and CI-tested CUDA stacks, [Devices and GPU](devices-and-gpu.md)
for operation coverage, and [XLA and PJRT](xla.md) for PJRT setup.

The hardware-gated two-CUDA-device identity and execution evidence can be run
with:

```text
cargo test -p tenferro-gpu --features cuda --lib \
  cuda_registration_preserves_two_caller_selected_engine_ids_and_devices -- \
  --ignored --nocapture
```

The test uses discovered device IDs, registers both caller-selected engines in
one runtime, and reports an explicit `SKIP` with the detected count only when
successful discovery finds fewer than two visible devices. A discovery error
fails the test with the original typed error.

## CPU Provider Choice

`CpuBackend::new()` uses the default provider compiled into the application.
The faer provider is the default because it gives tenferro the most control:
tenferro can inject `Par::Seq` or `Par::rayon(n)` from the selected backend
context, and a managed CPU domain can own the corresponding Rayon executor and
affinity contract.

External BLAS providers own their worker pools. Provider workers inherit the
calling thread's mask — the domain CPU set for a managed domain — but tenferro
cannot bound the provider's fan-out, and their environment variables are
generally process-wide, so a provider call can oversubscribe an outer tenferro
fan-out unless its provider thread count is reduced, often to one. Choose an
external provider when its peak dense-kernel performance is worth giving up
tenferro's per-operation thread-count control.

| Provider family | Required library | Cargo feature | Thread ownership and scope | Guidance |
| --- | --- | --- | --- | --- |
| faer / cpueinsum / tprims / tlinalg | None beyond the Rust dependencies | `native` | tenferro-owned Rayon pool; the budget is per selected CPU domain | Default choice when placement, reproducibility, and predictable nesting matter |
| OpenBLAS | OpenBLAS and its CBLAS/LAPACK entry points | `blas`, `blas-openblas` | OpenBLAS-owned pool; settings such as `OPENBLAS_NUM_THREADS` are provider/process-wide | Use for peak BLAS/LAPACK throughput; use `CpuPlacement::Auto` and avoid outer oversubscription |
| Intel MKL | MKL and its BLAS/LAPACK entry points | `blas`, `blas-mkl` | MKL-owned pool; `MKL_NUM_THREADS` and related OpenMP settings are provider/process-wide | Use when the deployment already standardizes on MKL; tenferro confines its own workers to the domain CPU set but cannot bound the provider's fan-out |
| Apple Accelerate | Apple Accelerate BLAS/LAPACK | `blas`, `blas-accelerate` | Accelerate-owned pool; `VECLIB_MAXIMUM_THREADS` is provider/process-wide | Use for Apple-native deployments; explicit NUMA placement is not a tenferro guarantee |
| BLIS | BLIS | Planned in [#1334](https://github.com/tensor4all/tenferro-rs/issues/1334) | Provider-owned; the final scope and controls are not part of the current API | Do not rely on a BLIS feature until the planned provider contract lands |

The CPU backend is selected at compile time, not per backend handle:
`native` is the default and `blas` adds the vendor adapters. Both expose the
same tenferro-owned placement API, so a build chooses the numerical provider
and the application only chooses a thread count and CPU set:

<!-- snippet-source: crates/tenferro-cpu/examples/choosing_cpu_provider.rs -->
```rust
use tenferro_cpu::{CpuBackend, CpuId, CpuSet};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The CPU backend is chosen at compile time: `native` (the default) or
    // `blas`. Both expose the same tenferro-owned placement API.
    let default_backend = CpuBackend::new();
    let pinned = CpuBackend::builder()
        .cpus(CpuSet::new([CpuId::new(0)])?)
        .threads(1)?
        .build()?;
    assert!(default_backend.num_threads() >= 1);
    assert_eq!(pinned.num_threads(), 1);
    Ok(())
}
```
<!-- end-snippet-source -->

This requires both `native` and `blas` in the resolved Cargo feature
set. It does not make OpenBLAS and MKL independently selectable at runtime:
the concrete external BLAS implementation is selected by the process's build
and linked library configuration. Its worker environment remains a
provider-owned process-wide concern.

## Nested Parallelism

The two common nesting models are intentionally different:

```text
Panel A: faer (tenferro-managed)
caller thread
  -> CpuOperationEntry::enter (permit + owned pool entry, budget N fixed)
    -> with_native_parallelism (ExecutionPolicy::Rayon{max_threads: N})
      -> strided-basic / strided-kernel / strided-fused fanout
         (<= N partitions on SAME owned pool; nested strided ops inside
         partition sequential by fanout guard)
      -> faer ops (Par::rayon(N) same budget/pool)

Panel B: external BLAS (provider-owned)
caller thread
  -> CpuOperationEntry::enter (outer permit, tenferro budget N)
    -> outer tenferro fanout (N partitions)
      -> provider call -> provider-owned pool
         possible workers: N partitions * OPENBLAS_NUM_THREADS
  remedy: set the provider thread count to 1 when outer fanout is active
```

The faer model keeps outer and inner work inside one admitted domain and
prevents a child operation from recursively fanning out on the same pool. The
external model cannot provide that invariant because the library owns the
inner pool. Do not infer a strict tenferro thread budget from a provider
environment variable.

## Backend Capability Catalog

The catalog uses the same vocabulary as runtime engine registration. A future
`EngineRegistration` can expose these properties as capabilities without
changing the user-facing decision structure.

| Backend/family | Control tier | Outer/inner parallelism and nesting | Worker, stream, or queue owner and scope | Managed or ExternalManaged | Synchronization | Storage and event domain | Unsupported behavior and dtype/op coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CPU/native kernels | Tenferro-managed | The lower library owns lanes and nesting inside the one parallelism token tenferro passes | tenferro-owned Rayon pool, one engine per resolved CPU placement | Always `Managed`; tenferro never borrows an application pool | Direct CPU operations complete before return | Host storage classes; the current CPU runtime uses an immediate event domain | Typed unsupported errors; see [CPU execution](cpu-execution.md) and [linear algebra](linear-algebra.md) |
| CPU/BLAS or LAPACK | Vendor-owned inner parallelism | The vendor owns its own worker team; tenferro neither bounds nor schedules it | Linked provider pool and environment; usually process-wide | The tenferro engine is still `Managed`; the vendor call itself is called from the coordinator thread | Provider call returns before the synchronous CPU API returns; worker synchronization is vendor-owned | Host storage classes; CPU event domain | Unsupported operations are typed errors; `blas` keeps a native fallback for declined small steps; see [CPU execution](cpu-execution.md) |
| CUDA | Backend and NVIDIA-library managed | Device kernels use provider-selected grid/stream parallelism; host-side nesting is limited to explicit backend submissions | CUDA runtime stream, handles, and backend plan caches owned by the CUDA runtime/backend instance | GPU runtime managed; CPU `ExternalManaged` labels do not describe CUDA worker ownership | Launches may be asynchronous; synchronize, download, or host inspection is the boundary | CUDA device storage and CUDA stream/event domain | Unsupported CUDA operation or dtype, and missing NVIDIA library, return typed errors; see [GPU coverage](devices-and-gpu.md) |
| WebGPU | Queue/provider managed | Device parallelism is inside the submitted shader; independent work may share the queue, with no tenferro CPU-style nested budget | wgpu device and queue owned by the WebGPU runtime/backend | GPU runtime managed; no CPU NUMA contract | Queue submission is asynchronous; explicit runtime synchronization or download makes results host-visible | WebGPU storage and queue/event domain | Unsupported operation or dtype returns an error; no implicit CPU fallback; see [GPU coverage](devices-and-gpu.md) |
| XLA/PJRT | Plugin-managed | XLA decides intra-op decomposition; tenferro does not add a second hidden inner pool | PJRT plugin owns its client, device, streams, and worker scope | Plugin-managed; not a tenferro CPU `Managed` or `ExternalManaged` domain | PJRT execution and transfers define the host-visible boundary | PJRT device buffers and plugin event domain | Unsupported lowering, dtype, shape, or missing plugin is rejected; see [XLA subset](xla.md) |

The dtype and operation links above are the sources of truth for coverage.
This page describes ownership and failure behavior, not a second operation
catalog.

## Ownership Contracts

`Managed` and `ExternalManaged` are resource-domain contracts, not performance
labels:

- `Managed` means tenferro constructs and owns the executor, can enforce the
  declared worker budget and supported affinity guarantee, and owns shutdown
  for that executor. The faer path is the primary CPU example.
- `ExternalManaged` means the caller or provider owns the executor and its
  shutdown. tenferro retains the owner for dependent work and arbitrates the
  declared domain, but it does not repin external workers or claim live OS
  affinity verification. External BLAS uses this conservative contract.

GPU runtimes and PJRT plugins have their own stream, queue, and client
lifetimes. They should be described by explicit engine capabilities rather
than being forced into a CPU NUMA label.

## Placement

An ordinary tensor is single-device. Placement selects one registered engine
and storage class for an operation; it does not create a distributed tensor.
The current multi-device work schedules independent single-device tensors and
inserts explicit transfers. Cross-device transfer providers, event domains,
and asynchronous submission are tracked in [execution substrate issue
#1471](https://github.com/tensor4all/tenferro-rs/issues/1471). Distributed tensors
and collectives are outside this contract.

When one process needs two CPU engine registrations, use distinct engine IDs.
The canonical helper keeps `tenferro-cpu.default.v1` for the usual single
backend case; the caller-selected helper avoids an ID collision:

<!-- snippet-source: crates/tenferro-cpu/examples/multiple_cpu_engines.rs -->
```rust
use tenferro_cpu::{runtime_engine_registration_with_id, CpuBackend, CpuId, CpuSet};
use tenferro_runtime::{EngineId, Runtime};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cpus = CpuSet::new([CpuId::new(0)])?;
    let primary = CpuBackend::builder()
        .cpus(cpus.clone())
        .threads(1)?
        .build()?;
    let secondary = CpuBackend::builder().cpus(cpus).threads(1)?.build()?;

    let mut builder = Runtime::builder();
    builder.register_engine(runtime_engine_registration_with_id(
        &primary,
        EngineId::new("example.cpu.primary.v1")?,
    )?)?;
    builder.register_engine(runtime_engine_registration_with_id(
        &secondary,
        EngineId::new("example.cpu.secondary.v1")?,
    )?)?;
    let runtime = builder.build()?;
    assert_eq!(runtime.snapshot()?.engine_count(), 2);
    Ok(())
}
```
<!-- end-snippet-source -->

Compile this example with both `native` and `blas`. If the second
backend uses OpenBLAS, MKL, or Accelerate, its library and thread settings are
still process-wide provider settings. Separate processes are required when
the application needs mutually incompatible external BLAS installations or
independent provider environment pools.

## Policy Selection

Use `CpuBackend::with_threads`, `CpuBackend::with_kind`, and explicit placement
when tenferro should own the policy. Use a provider bundle or an external
domain only when the provider's own controls are part of the application's
deployment contract. In either case, an unsupported request must produce a
typed error; a backend must not silently switch to another provider or device.

See [Parallelism and Caching](parallelism-and-caching.md) for thread-budget
mechanics, provider environment variables, cache lifetime, and
oversubscription diagnostics.
