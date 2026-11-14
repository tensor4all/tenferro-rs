# Choosing A Backend

Choose the execution backend that owns the device and its synchronization
boundary. A CPU build has no provider to pick: it links exactly one faer stack.
In every case the important choice is the same one:
**the caller owns the engine**. tenferro never builds threads, never opens a
hidden session, and never moves a tensor to another device on its own. You build
the pool or open the device, hold the engine, and pass it to every operation.

## Execution Backend

| Backend | Device | Hard requirements | Cargo feature | Threading and synchronization | Good fit |
| --- | --- | --- | --- | --- | --- |
| CPU | Host CPUs | None beyond the Rust dependencies (faer is compiled in) | none | The caller's `Pool` (or no pool at all for `CpuEngine::serial`); one placement token per operation. CPU calls return synchronously. | Controlled CPU execution, dense linalg, and host pipelines |
| CUDA | NVIDIA GPU | CUDA runtime plus the NVIDIA library stack required by the operation, including cuBLAS/cuSOLVER/cuTENSOR/cuFFT where applicable; missing libraries are typed errors | `tenferro-gpu` `cuda` | `CudaEngine::open` owns one device, its stream and its caches. Launches are asynchronous until an explicit synchronize, download, or host inspection. | NVIDIA production workloads and CUDA library-backed dense operations |
| XLA/PJRT | A PJRT addressable device | A PJRT plugin shared library; the `pjrt` loader returns a typed error when it is absent or invalid | `tenferro-xla` `pjrt` | The PJRT plugin owns device execution and intra-op scheduling. tenferro controls lowering and invocation, not the plugin's worker pool. | Static traced programs and deployments already using XLA plugins |

CUDA does not use a native kernel as a silent fallback when a required
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

## CPU Provider

The CPU numerical provider is not selectable: every build links the
tenferro-owned faer/cpueinsum/tcontract/tlinalg kernels, and
`tenferro-cpu` declares no backend feature.

That stack gives tenferro the most control: the operation's placement token
carries the engine's `threadteam::Team`, which faer runs on as `Par::Team` (or
`Par::Seq` for a serial token), and every worker belongs to the pool the caller
handed the engine. Nothing goes through Rayon.

External CPU providers remain possible only as provider-bundle slots (for
example a replacement `dot_general`); they do not replace the linalg path, and
tenferro offers no wrapper that gives a vendor library a per-lane width.

The application chooses the pool it hands to `CpuEngine`:

<!-- snippet-source: crates/tenferro-cpu/examples/choosing_cpu_provider.rs -->
```rust
use tenferro_cpu::{CpuEngine, Pool};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The caller builds the pool (and pins its workers if it wants to); the
    // engine computes on it. A serial engine has no pool at all.
    let serial = CpuEngine::serial();
    let pooled = CpuEngine::new(Pool::builder().threads(2).build()?);
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

`cpu_provider_id()` is fixed for the build.

## Nested Parallelism

The two common nesting models are intentionally different, and both are decided
by the caller's pool:

```text
Panel A: faer (caller-owned pool)
caller thread
  -> CpuEngine operation (one operation at a time; &mut borrow)
    -> operation starts on the calling thread
      -> enters the caller's pool only when an entry splits into > 1 task
         (one token: at most its budget of tasks at once; a fork gives each
         child a share of the budget)
      -> faer ops (Par::Team on the engine's team, Par::Seq for a serial token)

```

The faer model keeps outer and inner work inside one caller-owned pool and
prevents a child operation from recursively fanning out on the same pool.

Small operations never enter the pool at all: an operation starts on the
calling thread and only fans out after a provider has chosen more than one
worker. `CpuEngine::serial` has no pool, so it is safe to use from inside a pool
the caller already owns.

## Backend Capability Catalog

| Backend/family | Outer/inner parallelism and nesting | Worker, stream, or queue owner and scope | Synchronization | Storage and event domain | Unsupported behavior and dtype/op coverage |
| --- | --- | --- | --- | --- | --- |
| CPU/faer kernels | The engine starts on the calling thread and enters the caller's pool only for a multi-worker provider choice | The caller's `Pool`, one engine per pool | Direct CPU operations complete before return | Host storage classes; the CPU runtime uses an immediate event domain | Typed unsupported errors; see [CPU execution](cpu-execution.md) and [linear algebra](linear-algebra.md) |
| CUDA | Device kernels use provider-selected grid/stream parallelism; host-side nesting is limited to explicit engine calls | One `CudaEngine` owns its `CudaRuntime`, stream, handles and plan caches | Launches may be asynchronous; synchronize, download, or host inspection is the boundary | CUDA device storage and CUDA stream/event domain | Unsupported CUDA operation or dtype, and missing NVIDIA library, return typed errors; see [GPU coverage](devices-and-gpu.md) |
| XLA/PJRT | XLA decides intra-op decomposition; tenferro does not add a second hidden inner pool | PJRT plugin owns its client, device, streams, and worker scope | PJRT execution and transfers define the host-visible boundary | PJRT device buffers and plugin event domain | Unsupported lowering, dtype, shape, or missing plugin is rejected; see [XLA subset](xla.md) |

The dtype and operation links above are the sources of truth for coverage.
This page describes ownership and failure behavior, not a second operation
catalog.

## Ownership Contracts

Every engine is caller-held, and that is the whole contract:

- **The caller owns execution resources.** You build the `Pool` or open the
  CUDA device, you keep the engine, and you decide when it is dropped. The
  engine owns the plan cache, the buffer pool and the extension caches that
  belong to those resources.
- **tenferro owns nothing it cannot see.** No background threads, no hidden
  pools, no implicit device, no implicit transfer.
- **A provider bundle keeps its own workers.** An external provider slot is
  constructed and shut down by whoever installed it, not by tenferro; the engine
  simply calls into it.

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
use tenferro_cpu::{runtime_engine_registration_with_id, Pool, CpuEngine};
use tenferro_runtime::{EngineId, ExecutionEngine, Runtime};

// A caller-chosen engine id must agree on both sides: extension modules are
// installed against the id the runtime registered.
let engine_id = EngineId::new("example.cpu.primary.v1")?;
let engine = CpuEngine::new(Pool::builder().threads(1).build()?)
    .with_runtime_engine_id(engine_id.clone());

let mut builder = Runtime::builder();
builder.register_engine(runtime_engine_registration_with_id(&engine, engine_id)?)?;
let runtime = builder.build()?;
assert_eq!(runtime.snapshot()?.engine_count(), 1);
```
<!-- end-snippet-source -->

If two pools must run in one process, give each its own `CpuEngine` and its own
`Runtime`. Every engine of a build shares that build's single CPU provider.

## Policy Selection

Build the policy into the engine you hand to tenferro: the pool's width and
affinity, `CpuEngine::with_batch_policy`, or the CUDA device ordinal. Use a
provider bundle or an external domain only when the provider's own controls are
part of the application's deployment contract. In either case, an unsupported
request must produce a typed error; an engine must not silently switch to
another provider or device.

See [Parallelism and Caching](parallelism-and-caching.md) for thread-budget
mechanics, cache lifetime, and oversubscription diagnostics.

