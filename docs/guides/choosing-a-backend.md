# Choosing A Backend

Choose a backend in two steps. First choose the execution backend that owns the
device and its synchronization boundary. If that backend is CPU, choose a dense
CPU provider separately. In every case the important choice is the same one:
**the caller owns the engine**. tenferro never builds threads, never opens a
hidden session, and never moves a tensor to another device on its own. You build
the pool or open the device, hold the engine, and pass it to every operation.

## Execution Backend

| Backend | Device | Hard requirements | Cargo feature | Threading and synchronization | Good fit |
| --- | --- | --- | --- | --- | --- |
| CPU | Host CPUs | At least one CPU provider; `native` is the default | `tenferro-cpu` `native` or `blas` | The caller's Rayon pool (or no pool at all for `CpuEngine::serial`). CPU calls return synchronously. | Controlled CPU execution, dense linalg, and host pipelines |
| CUDA | NVIDIA GPU | CUDA runtime plus the NVIDIA library stack required by the operation, including cuBLAS/cuSOLVER/cuTENSOR/cuFFT where applicable; missing libraries are typed errors | `tenferro-gpu` `cuda` | `CudaEngine::open` owns one device, its stream and its caches. Launches are asynchronous until an explicit synchronize, download, or host inspection. | NVIDIA production workloads and CUDA library-backed dense operations |
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

The CPU numerical provider is selected at compile time, never per engine
handle. `native` (the default) links the tenferro-owned faer/cpueinsum/
tcontract/tlinalg kernels; `blas` adds the vendor adapters.

The faer provider is the default because it gives tenferro the most control:
the engine derives `Par::Seq` or `Par::rayon(n)` from the pool the caller
handed it, and every worker runs inside that pool.

External BLAS providers own their worker pools. Provider workers inherit the
calling thread's mask, but tenferro cannot bound the provider's fan-out, and
their environment variables are generally process-wide, so a provider call can
oversubscribe an outer tenferro fan-out unless its provider thread count is
reduced, often to one. Choose an external provider when its peak dense-kernel
performance is worth giving up tenferro's per-operation thread-count control.

| Provider family | Required library | Cargo feature | Thread ownership and scope | Guidance |
| --- | --- | --- | --- | --- |
| faer / cpueinsum / tcontract / tlinalg | None beyond the Rust dependencies | `native` | The caller's Rayon pool; the engine never creates workers | Default choice when reproducibility and predictable nesting matter |
| OpenBLAS | OpenBLAS and its CBLAS/LAPACK entry points | `blas`, `blas-openblas` | OpenBLAS-owned pool; settings such as `OPENBLAS_NUM_THREADS` are provider/process-wide | Use for peak BLAS/LAPACK throughput; keep the outer fan-out small to avoid oversubscription |
| Intel MKL | MKL and its BLAS/LAPACK entry points | `blas`, `blas-mkl` | MKL-owned pool; `MKL_NUM_THREADS` and related OpenMP settings are provider/process-wide | Use when the deployment already standardizes on MKL; tenferro cannot bound the provider's fan-out |
| Apple Accelerate | Apple Accelerate BLAS/LAPACK | `blas`, `blas-accelerate` | Accelerate-owned pool; `VECLIB_MAXIMUM_THREADS` is provider/process-wide | Use for Apple-native deployments |
| BLIS | BLIS | Planned in [#1334](https://github.com/tensor4all/tenferro-rs/issues/1334) | Provider-owned; the final scope and controls are not part of the current API | Do not rely on a BLIS feature until the planned provider contract lands |

A build chooses the numerical provider; the application chooses the pool it
hands to `CpuEngine`:

<!-- snippet-source: crates/tenferro-cpu/examples/choosing_cpu_provider.rs -->
```rust
use tenferro_cpu::{thread_pool_builder, CpuEngine};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The caller builds the pool (and pins its workers if it wants to); the
    // engine computes on it. A serial engine has no pool at all.
    let serial = CpuEngine::serial();
    let pooled = CpuEngine::new(thread_pool_builder().num_threads(2).build()?);
    println!(
        "provider={} serial_threads={} pooled_threads={}",
        tenferro_cpu::cpu_provider_id(),
        serial.num_threads(),
        pooled.num_threads()
    );
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

The two common nesting models are intentionally different, and both are decided
by the caller's pool:

```text
Panel A: faer (caller-owned pool)
caller thread
  -> CpuEngine operation (one operation at a time; &mut borrow)
    -> operation starts on the calling thread
      -> enters the caller's pool only when a provider chose > 1 worker
         (<= pool width partitions on that pool; nested strided ops inside
         a partition stay sequential)
      -> faer ops (Par::rayon(pool width) on the same pool)

Panel B: external BLAS (provider-owned pool)
caller thread
  -> CpuEngine operation (borrowed for the call)
    -> outer tenferro fanout (pool width partitions)
      -> provider call -> provider-owned pool
         possible workers: pool width * OPENBLAS_NUM_THREADS
  remedy: set the provider thread count to 1 when outer fanout is active
```

The faer model keeps outer and inner work inside one caller-owned pool and
prevents a child operation from recursively fanning out on the same pool. The
external model cannot provide that invariant because the library owns the inner
pool. Do not infer a strict tenferro thread budget from a provider environment
variable.

Small operations never enter the pool at all: an operation starts on the
calling thread and only fans out after a provider has chosen more than one
worker. `CpuEngine::serial` has no pool, so it is safe to use from inside a pool
the caller already owns.

## Backend Capability Catalog

| Backend/family | Outer/inner parallelism and nesting | Worker, stream, or queue owner and scope | Synchronization | Storage and event domain | Unsupported behavior and dtype/op coverage |
| --- | --- | --- | --- | --- | --- |
| CPU/native kernels | The engine starts on the calling thread and enters the caller's pool only for a multi-worker provider choice | The caller's Rayon pool, one engine per pool | Direct CPU operations complete before return | Host storage classes; the CPU runtime uses an immediate event domain | Typed unsupported errors; see [CPU execution](cpu-execution.md) and [linear algebra](linear-algebra.md) |
| CPU/BLAS or LAPACK | The vendor owns its own worker team; tenferro neither bounds nor schedules it | Linked provider pool and environment; usually process-wide | The provider call returns before the synchronous CPU API returns; worker synchronization is vendor-owned | Host storage classes; CPU event domain | Unsupported operations are typed errors; `blas` keeps a native fallback for declined small steps; see [CPU execution](cpu-execution.md) |
| CUDA | Device kernels use provider-selected grid/stream parallelism; host-side nesting is limited to explicit engine calls | One `CudaEngine` owns its `CudaRuntime`, stream, handles and plan caches | Launches may be asynchronous; synchronize, download, or host inspection is the boundary | CUDA device storage and CUDA stream/event domain | Unsupported CUDA operation or dtype, and missing NVIDIA library, return typed errors; see [GPU coverage](devices-and-gpu.md) |
| WebGPU | Device parallelism is inside the submitted shader; independent work may share the queue, with no tenferro CPU-style nested budget | wgpu device and queue owned by the WebGPU backend | Queue submission is asynchronous; explicit synchronization or download makes results host-visible | WebGPU storage and queue/event domain | Unsupported operation or dtype returns an error; no implicit CPU fallback; see [GPU coverage](devices-and-gpu.md) |
| XLA/PJRT | XLA decides intra-op decomposition; tenferro does not add a second hidden inner pool | PJRT plugin owns its client, device, streams, and worker scope | PJRT execution and transfers define the host-visible boundary | PJRT device buffers and plugin event domain | Unsupported lowering, dtype, shape, or missing plugin is rejected; see [XLA subset](xla.md) |

The dtype and operation links above are the sources of truth for coverage.
This page describes ownership and failure behavior, not a second operation
catalog.

## Ownership Contracts

Every engine is caller-held, and that is the whole contract:

- **The caller owns execution resources.** You build the Rayon pool or open the
  CUDA device, you keep the engine, and you decide when it is dropped. The
  engine owns the plan cache, the buffer pool and the extension caches that
  belong to those resources.
- **tenferro owns nothing it cannot see.** No background threads, no hidden
  pools, no implicit device, no implicit transfer.
- **External providers keep their own workers.** A BLAS/LAPACK library is
  constructed and shut down by the process, not by tenferro; the engine simply
  calls into it.

GPU runtimes and PJRT plugins have their own stream, queue, and client
lifetimes. They are described by explicit engine capabilities rather than being
forced into a CPU ownership label.

## Placement

An ordinary tensor is single-device. An engine selects one device and storage
class for an operation; it does not create a distributed tensor. The current
multi-device work schedules independent single-device tensors and inserts
explicit transfers. Cross-device transfer providers, event domains, and
asynchronous submission are tracked in [execution substrate issue
#1471](https://github.com/tensor4all/tenferro-rs/issues/1471). Distributed tensors
and collectives are outside this contract.

A runtime accepts **one registration per provider/device pair**. For the CPU
that pair is `(cpu provider, "host")`, so two CPU engines are two runtimes, not
two registrations in one runtime. What you can choose is the engine *id*: the
canonical helper keeps `tenferro-cpu.default.v1` for the usual single-engine
case, and the caller-selected helper lets the registration and the engine agree
on a different id (extension modules install against that id):

<!-- snippet-source: docs/tutorial-code/src/bin/execution_snippets.rs#choosing_a_backend_engine_id -->
```rust
use tenferro_cpu::{runtime_engine_registration_with_id, thread_pool_builder, CpuEngine};
use tenferro_runtime::{EngineId, ExecutionEngine, Runtime};

// A caller-chosen engine id must agree on both sides: extension modules are
// installed against the id the runtime registered.
let engine_id = EngineId::new("example.cpu.primary.v1")?;
let engine = CpuEngine::new(thread_pool_builder().num_threads(1).build()?)
    .with_runtime_engine_id(engine_id.clone());

let mut builder = Runtime::builder();
builder.register_engine(runtime_engine_registration_with_id(&engine, engine_id)?)?;
let runtime = builder.build()?;
assert_eq!(runtime.snapshot()?.engine_count(), 1);
```
<!-- end-snippet-source -->

If two pools must run in one process, give each its own `CpuEngine` and its own
`Runtime`. Compile either example with both `native` and `blas`; if an engine
uses OpenBLAS, MKL, or Accelerate, its library and thread settings are still
process-wide provider settings. Separate processes are required when
the application needs mutually incompatible external BLAS installations or
independent provider environment pools.

## Policy Selection

Build the policy into the engine you hand to tenferro: the pool's width and
affinity, `CpuEngine::with_batch_policy`, or the CUDA device ordinal. Use a
provider bundle or an external domain only when the provider's own controls are
part of the application's deployment contract. In either case, an unsupported
request must produce a typed error; an engine must not silently switch to
another provider or device.

See [Parallelism and Caching](parallelism-and-caching.md) for thread-budget
mechanics, provider environment variables, cache lifetime, and
oversubscription diagnostics.

<!-- ponytail: the WebGPU/Apple backend is not on the engine model yet
(TODO(#2206 webgpu)); this page describes its ownership shape only and must be
rechecked when that migration lands. -->
