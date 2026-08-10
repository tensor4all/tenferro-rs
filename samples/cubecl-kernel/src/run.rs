//! End-to-end run of the fixture kernel through the public session seam.

use cubecl::prelude::{ArrayArg, CubeCount};
use cubecl_cuda::CudaRuntime as CubeclCudaRuntime;
use tenferro_gpu::cuda::{
    download_tensor, upload_tensor, with_cuda_exec_session, CudaBackend, CudaExecSession,
    CudaRuntime,
};
use tenferro_tensor::backend::BackendSessionHost;
use tenferro_tensor::{Tensor, TypedTensor};

use crate::kernel::scale;

/// Run `out = in * 2.0` on the first CUDA device and assert the result.
///
/// Uploads an f32 tensor, launches [`scale`] with a one-dimensional
/// domain sized by the session's `cube_count_1d` helper, synchronizes
/// explicitly, downloads, and numerically asserts. Uses only the public
/// `cuda::cubecl::Session` surface.
///
/// Hardware-gated: `cuda_devices` returns an error when no CUDA device is
/// reachable, so this is safe to call from a hardware-gated test.
///
/// # Examples
///
/// Hardware-gated end-to-end run; without a CUDA device this returns an
/// error, which the example ignores.
///
/// ```
/// let _ = cubecl_kernel_sample::run_scale_check();
/// ```
///
/// # Errors
///
/// Returns [`tenferro_tensor::Error::runtime_state`] when no CUDA device is
/// reachable, the CUDA backend or session cannot be created, the kernel
/// launch or explicit synchronization fails, or a downloaded value mismatches
/// the expected scaled output.
pub fn run_scale_check() -> tenferro_tensor::Result<()> {
    const OP: &str = "cubecl-kernel-sample.scale";
    let devices = tenferro_gpu::cuda::cuda_devices()
        .map_err(|err| tenferro_tensor::Error::runtime_state(OP, err.to_string()))?;
    let device = devices
        .first()
        .ok_or_else(|| tenferro_tensor::Error::runtime_state(OP, "no CUDA device"))?;
    let mut backend = CudaBackend::new(device.id())
        .map_err(|err| tenferro_tensor::Error::runtime_state(OP, err.to_string()))?;

    let n = 8192usize;
    let host_input =
        TypedTensor::<f32>::from_vec_col_major(vec![n], (0..n).map(|i| i as f32).collect())?;
    let host_output = TypedTensor::<f32>::from_vec_col_major(vec![n], vec![0.0; n])?;

    backend
        .with_backend_session(|session| {
            with_cuda_exec_session(session, |exec| {
                run_in_cuda_session(OP, exec, host_input, host_output, n)
            })
        })
        .ok_or_else(|| tenferro_tensor::Error::runtime_state(OP, "CUDA session rejected backend"))?
}

/// Perform the full upload → launch → sync → download → assert flow.
///
/// The runtime handle is cloned out of the session first so uploads do not
/// hold a borrow of `exec` while the mutable `with_cubecl`/`synchronize`
/// methods run.
fn run_in_cuda_session(
    op: &'static str,
    exec: &mut CudaExecSession<'_>,
    host_input: TypedTensor<f32>,
    host_output: TypedTensor<f32>,
    n: usize,
) -> tenferro_tensor::Result<()> {
    let runtime = exec.runtime().clone();
    let input = upload_tensor(&runtime, &Tensor::F32(host_input))?;
    let output = upload_tensor(&runtime, &Tensor::F32(host_output))?;

    exec.with_cubecl(op, |cubecl| {
        let Tensor::F32(input_typed) = &input else {
            unreachable!("f32 upload")
        };
        let Tensor::F32(output_typed) = &output else {
            unreachable!("f32 upload")
        };

        let input_binding: ArrayArg<CubeclCudaRuntime> = cubecl.array_arg(input_typed, op)?;
        let output_binding: ArrayArg<CubeclCudaRuntime> = cubecl.array_arg(output_typed, op)?;

        let cube_count = cubecl.cube_count_1d(n)?;
        let CubeCount::Static(x, y, z) = cube_count else {
            return Err(tenferro_tensor::Error::runtime_state(
                op,
                "expected static cube count",
            ));
        };
        let cube_dim = cubecl.cube_dim_1d();

        // Sole unsafe call: vendor launch entry with a fully formed typed
        // arg list. SAFETY: the kernel reads `input[i]` and writes
        // `output[i]` only for `i < n`, with the domain sized to exactly
        // `n` elements above (no bounds elision by the launcher).
        unsafe {
            scale::launch_unchecked::<CubeclCudaRuntime>(
                cubecl.client(),
                CubeCount::Static(x, y, z),
                cube_dim,
                input_binding,
                output_binding,
                2.0f32,
            );
        }
        Ok(())
    })?;

    // Explicit host barrier, then download and assert.
    exec.synchronize()?;
    let downloaded = download_tensor(&runtime, &output)?;
    let Tensor::F32(host) = downloaded else {
        unreachable!("f32 download")
    };
    let (shape, values): (Vec<usize>, Vec<f32>) = host.into_vec_col_major()?;
    assert_eq!(shape, vec![n]);
    for (i, value) in values.iter().enumerate() {
        let expected = (i as f32) * 2.0;
        if (value - expected).abs() > 1e-3 {
            return Err(tenferro_tensor::Error::runtime_state(
                op,
                format!("scale mismatch at {i}: got {value}, expected {expected}"),
            ));
        }
    }
    Ok(())
}

// Keep the runtime type visible for the seam contract without adding an unused
// import lint: `CudaRuntime` is referenced above via `runtime: CudaRuntime`
// inference on `exec.runtime().clone()`.
#[allow(dead_code)]
fn _cuda_runtime_type_check(_: &CudaRuntime) {}
