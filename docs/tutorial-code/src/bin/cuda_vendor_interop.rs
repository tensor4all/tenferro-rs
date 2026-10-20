//! Direct cuBLAS call on tenferro-owned CUDA buffers (issue #1940).
//!
//! A downstream crate calls a vendor library on tenferro's device memory and
//! tenferro's execution stream through the public raw CUDA session only:
//! `CudaEngine::with_raw` binds the stream, `tensor` / `tensor_mut` give checked
//! device pointers, `with_cublas_handle` hands out the runtime's stream-bound
//! cuBLAS handle, `stream().raw_handle()` gives the native stream, and
//! `synchronize` is the host barrier. The cuBLAS bindings come from the `cudarc`
//! that tenferro-gpu re-exports, so the vendor binding cannot disagree with
//! tenferro's CUDA version selection.
//!
//! `cuda_tutorial` also runs [`run`], which is how the GPU CI lane executes this
//! example.

use tenferro_gpu::cuda::{cuda_devices, download_tensor, gpu_available, upload_tensor, CudaEngine};
use tenferro_runtime::TensorSessionOpsExt;
use tenferro_tensor::{Error, Tensor, TypedTensor};

const OP: &str = "tutorial.cublas_dgemm";

// snippet-start:cuda-vendor-cublas
use tenferro_gpu::cuda::cudarc::cublas::{result as cublas, sys as cublas_sys};

/// `C = A * B` with cuBLAS `dgemm` on tenferro device tensors.
///
/// `a` is `m x k` and `b` is `k x n`; both are compact column-major CUDA
/// tensors resident on `engine`. The output is allocated by tenferro and
/// written by cuBLAS on tenferro's stream.
fn cublas_dgemm(
    engine: &mut CudaEngine,
    a: &TypedTensor<f64>,
    b: &TypedTensor<f64>,
) -> tenferro_tensor::Result<TypedTensor<f64>> {
    let (m, k, n) = (a.shape()[0], a.shape()[1], b.shape()[1]);
    // cuBLAS takes LP64 `int` dimensions; convert instead of casting.
    let dim = |value: usize| {
        i32::try_from(value)
            .map_err(|_| Error::invalid_argument(OP, "shape", "dimension exceeds i32"))
    };
    let (m32, k32, n32) = (dim(m)?, dim(k)?, dim(n)?);

    engine.with_raw(OP, |session| {
        let a_ref = session.tensor(a)?;
        let b_ref = session.tensor(b)?;
        let mut c = session.alloc_output::<f64>(&[m, n])?;
        let c_ref = session.tensor_mut(&mut c)?;
        let (alpha, beta) = (1.0_f64, 0.0_f64);

        // The handle is the runtime's own, already bound to this
        // session's stream, so a raw scope pays neither `cublasCreate`
        // nor `cublasDestroy` (tenferro-rs #2083) and never has to
        // re-bind the stream. It is usable only inside this closure.
        session.with_cublas_handle(
            OP,
            cublas_sys::cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST,
            |handle| {
                // SAFETY: the three device pointers come from checked
                // tensor references that stay borrowed until the
                // `synchronize` below; A is m x k (lda = m), B is k x n
                // (ldb = k) and C is m x n (ldc = m) in compact
                // column-major storage, which is cuBLAS's native layout.
                // C is the only buffer written and does not alias A or B.
                unsafe {
                    cublas::dgemm(
                        handle,
                        cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                        cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                        m32,
                        n32,
                        k32,
                        &alpha,
                        a_ref.raw_ptr().cast(),
                        m32,
                        b_ref.raw_ptr().cast(),
                        k32,
                        &beta,
                        c_ref.raw_ptr().cast(),
                        m32,
                    )
                }
                .map_err(|err| Error::backend_source(OP, err))
            },
        )?;
        // `synchronize` is the only host barrier: the GEMM was merely
        // enqueued on the session's stream.
        session.synchronize()?;
        Ok(c)
    })
}
// snippet-end:cuda-vendor-cublas

/// Run the example on `engine` and check it against tenferro's own matmul.
pub(crate) fn run(engine: &mut CudaEngine) -> Result<(), Box<dyn std::error::Error>> {
    // A = [[1,2,3],[4,5,6]] (2x3), B = [[7,8],[9,10],[11,12]] (3x2), column-major.
    let a = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0])?;
    let b = Tensor::from_vec_col_major(vec![3, 2], vec![7.0_f64, 9.0, 11.0, 8.0, 10.0, 12.0])?;
    let gpu_a = upload_tensor(engine.runtime(), &a)?;
    let gpu_b = upload_tensor(engine.runtime(), &b)?;
    let typed_a = gpu_a.as_typed::<f64>().ok_or("A is f64")?;
    let typed_b = gpu_b.as_typed::<f64>().ok_or("B is f64")?;

    let vendor = cublas_dgemm(engine, typed_a, typed_b)?;
    let vendor = download_tensor(engine.runtime(), &Tensor::from_typed(vendor))?;
    let tenferro = gpu_a.matmul(&gpu_b, engine)?;
    let tenferro = download_tensor(engine.runtime(), &tenferro)?;

    // C = [[58,64],[139,154]], stored column-major. Small integers are exact,
    // so the vendor and tenferro results must agree bit for bit.
    assert_eq!(vendor.as_slice::<f64>()?, &[58.0, 139.0, 64.0, 154.0]);
    assert_eq!(vendor.as_slice::<f64>()?, tenferro.as_slice::<f64>()?);
    println!("cuda_vendor_interop: cuBLAS dgemm on tenferro buffers matches matmul");
    Ok(())
}

#[allow(dead_code)] // `cuda_tutorial` includes this file as a module and calls `run`.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let require_cuda = std::env::var("TENFERRO_REQUIRE_CUDA").is_ok_and(|value| value == "1");
    let device = if gpu_available() {
        cuda_devices()?.into_iter().next()
    } else {
        None
    };
    let Some(device) = device else {
        if require_cuda {
            return Err("cuda_vendor_interop requires a usable CUDA device".into());
        }
        eprintln!("TENFERRO_TUTORIAL_SKIP: cuda_vendor_interop skipped: no usable CUDA device");
        return Ok(());
    };
    let mut engine = CudaEngine::open(device.id())?;
    run(&mut engine)
}
