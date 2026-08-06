//! Compiled documentation snippets for issue #1609.

// INVARIANT: Independent documentation examples intentionally leave some imports,
// variables, and helper mains unused when compiled as one family binary.
#![allow(dead_code, unused_imports, unused_variables, unused_mut)]

#[rustfmt::skip]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    snippet_parallelism_and_caching_1()?;

    // snippet source: docs/guides/parallelism-and-caching.md:27
    fn snippet_parallelism_and_caching_1() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:parallelism_and_caching_1
use tenferro_cpu::CpuBackend;

let backend = CpuBackend::with_threads(4)?;
assert_eq!(backend.num_threads(), 4);
        // snippet-end:parallelism_and_caching_1
        Ok(())
    }

    snippet_parallelism_and_caching_2()?;

    // snippet source: docs/guides/parallelism-and-caching.md:92
    fn snippet_parallelism_and_caching_2() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:parallelism_and_caching_2
use tenferro_cpu::CpuBackend;
use tenferro_tensor::{TypedTensor};
use tenferro_tensor::backend::TensorViewCanonicalization;

let tensor = TypedTensor::<f64>::from_vec_col_major(
    vec![2, 3],
    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
)?;
let transposed = tensor.as_view().transpose_view([1, 0])?;
let mut backend = CpuBackend::with_threads(4)?;
let compact = backend.to_contiguous(&transposed)?;

assert_eq!(compact.shape(), &[3, 2]);
assert_eq!(compact.as_slice()?, &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
        // snippet-end:parallelism_and_caching_2
        Ok(())
    }

    snippet_parallelism_and_caching_3()?;

    // snippet source: docs/guides/parallelism-and-caching.md:166
    fn snippet_parallelism_and_caching_3() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:parallelism_and_caching_3
use tenferro_cpu::CpuBackend;

let backend = CpuBackend::with_threads(1)?;
        // snippet-end:parallelism_and_caching_3
        Ok(())
    }

    snippet_parallelism_and_caching_4()?;

    // snippet source: docs/guides/parallelism-and-caching.md:194
    fn snippet_parallelism_and_caching_4() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:parallelism_and_caching_4
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{GraphCompiler, Runtime};

let mut compiler = GraphCompiler::new();
let backend = CpuBackend::with_threads(4)?;
let mut builder = Runtime::builder();
builder.register_engine(tenferro_cpu::runtime_engine_registration(&backend)?)?;
let runtime = builder.build()?;
        // snippet-end:parallelism_and_caching_4
        Ok(())
    }

    snippet_parallelism_and_caching_5()?;

    // snippet source: docs/guides/parallelism-and-caching.md:215
    fn snippet_parallelism_and_caching_5() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:parallelism_and_caching_5
use std::num::NonZeroUsize;
use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::extension::ExtensionCacheLimits;

let eager = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
eager.set_extension_cache_limits(ExtensionCacheLimits::new(
    NonZeroUsize::new(128).ok_or("positive cache-entry limit")?,
).with_max_retained_bytes(
    NonZeroUsize::new(64 * 1024 * 1024).ok_or("positive cache-byte limit")?,
))?;
        // snippet-end:parallelism_and_caching_5
        Ok(())
    }

    snippet_parallelism_and_caching_6()?;

    // snippet source: docs/guides/parallelism-and-caching.md:246
    fn snippet_parallelism_and_caching_6() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:parallelism_and_caching_6
use tenferro_cpu::CpuBackend;

let mut backend = CpuBackend::new();
backend.set_buffer_pool_limit_bytes(32 * 1024 * 1024)?;
        // snippet-end:parallelism_and_caching_6
        Ok(())
    }

    snippet_parallelism_and_caching_7()?;

    // snippet source: docs/guides/parallelism-and-caching.md:259
    fn snippet_parallelism_and_caching_7() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:parallelism_and_caching_7
use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{GraphCompiler, Runtime};

fn main() -> Result<(), Box<dyn std::error::Error>> {
let eager = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
let runtime = Runtime::builder().build()?;
let mut compiler = GraphCompiler::new();

runtime.clear_prepared_cache()?;
runtime.clear_caches()?;
compiler.clear_caches();
eager.clear_caches()?;
Ok(())
}
main()?;
        // snippet-end:parallelism_and_caching_7
        Ok(())
    }

    snippet_parallelism_and_caching_8()?;

    // snippet source: docs/guides/parallelism-and-caching.md:279
    fn snippet_parallelism_and_caching_8() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:parallelism_and_caching_8
use tenferro_cpu::CpuBackend;

let mut backend = CpuBackend::new();
backend.reset_buffer_pool()?;
        // snippet-end:parallelism_and_caching_8
        Ok(())
    }

    snippet_troubleshooting_9()?;

    // snippet source: docs/guides/troubleshooting.md:37
    fn snippet_troubleshooting_9() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:troubleshooting_9
use tenferro_gpu::cuda::{cuda_devices, upload_tensor, CudaBackend};
use tenferro_tensor::{Tensor, TensorBackend};

let devices = cuda_devices()?;
let device = devices.first().ok_or("no CUDA device is visible")?;
let backend = CudaBackend::new(device.id())?;
let x = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
let gpu_x = upload_tensor(backend.runtime(), &x)?;
assert_eq!(gpu_x.shape(), &[2]);
        // snippet-end:troubleshooting_9
        Ok(())
    }

    snippet_troubleshooting_10()?;

    // snippet source: docs/guides/troubleshooting.md:54
    fn snippet_troubleshooting_10() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:troubleshooting_10
use tenferro_gpu::cuda::{cuda_devices, download_tensor, upload_tensor, CudaBackend};
use tenferro_tensor::{Tensor, TensorBackend};

let devices = cuda_devices()?;
let device = devices.first().ok_or("no CUDA device is visible")?;
let backend = CudaBackend::new(device.id())?;
let x = Tensor::from_vec_col_major(vec![1], vec![3.0_f64])?;
let gpu_x = upload_tensor(backend.runtime(), &x)?;
let cpu_x = download_tensor(backend.runtime(), &gpu_x)?;
assert_eq!(cpu_x.as_slice::<f64>()?, &[3.0]);
        // snippet-end:troubleshooting_10
        Ok(())
    }





    Ok(())
}
