//! End-to-end runs of the fixture kernels through the public session seam.

use std::time::{Duration, Instant};

use cubecl::client::Client;
use cubecl::prelude::{BufferArg, CubeCount, CubeDim};
use tenferro_gpu::cuda::{
    download_tensor, upload_tensor, with_cuda_exec_session, CudaBackend, CudaExecSession,
};
use tenferro_tensor::backend::BackendSessionHost;
use tenferro_tensor::{Error, Tensor, TypedTensor};

use crate::kernel::{gather, reverse_tile, scale};

/// Open the first CUDA device as a tenferro backend.
fn first_backend(op: &'static str) -> tenferro_tensor::Result<CudaBackend> {
    let devices = tenferro_gpu::cuda::cuda_devices()
        .map_err(|err| Error::runtime_state(op, err.to_string()))?;
    let device = devices
        .first()
        .ok_or_else(|| Error::runtime_state(op, "no CUDA device"))?;
    CudaBackend::new(device.id()).map_err(|err| Error::runtime_state(op, err.to_string()))
}

/// Run `out = in * 2.0` on the first CUDA device, print it, and assert it.
///
/// This is the canonical downstream flow: upload the input, **allocate the
/// output through the session**, launch a downstream `#[cube]` kernel through
/// `Session::client()`, synchronize explicitly, download, and check the values.
/// It uses only the public `cuda::cubecl::Session` surface.
///
/// Hardware-gated: `cuda_devices` returns an error when no CUDA device is
/// reachable, so this is safe to call from a hardware-gated test.
///
/// # Examples
///
/// Hardware-gated: `cudarc` panics when the CUDA driver library is absent, so
/// the call is wrapped in `catch_unwind` (the same guard `gpu_available`
/// uses). On CUDA hardware this runs the full upload → launch → sync →
/// download → assert flow and returns `Ok(Ok(()))`; elsewhere it returns
/// `Ok(Err(..))` or `Err(panic)`.
///
/// ```
/// let outcome = std::panic::catch_unwind(|| cubecl_kernel_sample::run_scale_check());
/// let _ = outcome;
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
    let n = 8192usize;
    let host_input =
        TypedTensor::<f32>::from_vec_col_major(vec![n], (0..n).map(|i| i as f32).collect())?;

    let mut backend = first_backend(OP)?;
    let downloaded = backend
        .with_backend_session(|session| {
            with_cuda_exec_session(session, |exec| {
                run_scale_in_session(OP, exec, host_input, n)
            })
        })?
        .ok_or_else(|| Error::runtime_state(OP, "CUDA session rejected backend"))??;

    let (shape, values): (Vec<usize>, Vec<f32>) = downloaded
        .into_vec_col_major()
        .map_err(|failure| failure.into_parts().1)?;
    assert_eq!(shape, vec![n]);
    // Printed evidence that the kernel ran: endpoints plus a checksum.
    let checksum: f32 = values.iter().sum();
    let expected_checksum: f32 = (0..n).map(|i| (i as f32) * 2.0).sum();
    println!(
        "{OP}: n={n} out[0]={} out[n-1]={} checksum={checksum}",
        values[0],
        values[n - 1]
    );
    for (i, value) in values.iter().enumerate() {
        let expected = (i as f32) * 2.0;
        if (value - expected).abs() > 1e-3 {
            return Err(Error::runtime_state(
                OP,
                format!("scale mismatch at {i}: got {value}, expected {expected}"),
            ));
        }
    }
    assert!(
        (checksum - expected_checksum).abs() <= 1e-3 * expected_checksum.abs(),
        "checksum {checksum} does not match {expected_checksum}"
    );
    Ok(())
}

// snippet-start:downstream-cube-launch
/// Upload `input`, allocate an output on the session, launch [`scale`], and
/// download the result.
///
/// The three launch inputs the CubeCL API does not make obvious are all named
/// here: the output comes from `Session::alloc_output`, the launch domain from
/// `Session::cube_count_1d` plus `Session::cube_dim_1d`, and the kernel
/// arguments are the `BufferArg` values `Session::array_arg` hands back —
/// `KernelArguments` is never built by hand.
///
/// The runtime handle is cloned out of the session first so uploads do not
/// hold a borrow of `exec` while the mutable `with_cubecl`/`synchronize`
/// methods run.
///
/// `n` must be a multiple of the 256-unit cube dim: `cube_count_1d` rounds the
/// domain up, and `scale` has no `ABSOLUTE_POS < n` guard, so only an exact
/// domain keeps it in bounds.
fn run_scale_in_session(
    op: &'static str,
    exec: &mut CudaExecSession<'_>,
    host_input: TypedTensor<f32>,
    n: usize,
) -> tenferro_tensor::Result<TypedTensor<f32>> {
    let runtime = exec.runtime().clone();
    let input = upload_tensor(&runtime, &Tensor::from_typed::<f32>(host_input))?;
    let input = input.into_typed::<f32>().expect("f32 upload stays f32");

    let output = exec.with_cubecl(op, |cubecl| {
        // SAFETY: both bindings are created for tensors this call owns, and the
        // kernel writes only `output`.
        let input_arg: BufferArg = unsafe { cubecl.array_arg(&input, op)? };
        // Session-owned output allocation: no host staging tensor is uploaded.
        let output = cubecl.alloc_output::<f32>(&[n])?;
        // SAFETY: as above; `output` is exclusively owned here.
        let output_arg: BufferArg = unsafe { cubecl.array_arg(&output, op)? };

        let CubeCount::Static(x, y, z) = cubecl.cube_count_1d(n)? else {
            return Err(Error::runtime_state(op, "expected static cube count"));
        };
        // SAFETY: the kernel reads `input[i]` and writes `output[i]` only for
        // `i < n`; `n` is a multiple of the cube dim, so the rounded-up domain
        // above covers exactly `n` elements and never overruns either buffer.
        unsafe {
            scale::launch_unchecked(
                cubecl.client(),
                CubeCount::Static(x, y, z),
                cubecl.cube_dim_1d(),
                input_arg,
                output_arg,
                2.0f32,
            );
        }
        Ok(output)
    })?;

    // Explicit host barrier, then download.
    exec.synchronize()?;
    Ok(download_tensor(&runtime, &Tensor::from_typed(output))?
        .into_typed::<f32>()
        .expect("f32 download"))
}
// snippet-end:downstream-cube-launch

/// Run the typing-rule kernels ([`gather()`] and [`reverse_tile()`]) and verify
/// their values.
///
/// The two kernels pin the `#[cube]` index rules the guide documents: an `i32`
/// buffer element needs `usize::cast_from` (the cast accepts a signed source on
/// this pin), and the `u32` topology constants (`UNIT_POS`) need the cast while
/// `ABSOLUTE_POS` and index literals do not.
///
/// Hardware-gated like [`run_scale_check`].
///
/// # Examples
///
/// Hardware-gated: the call is wrapped in `catch_unwind` because `cudarc`
/// panics when the CUDA driver library is absent.
///
/// ```
/// let outcome = std::panic::catch_unwind(|| cubecl_kernel_sample::run_typing_rules_check());
/// let _ = outcome;
/// ```
///
/// # Errors
///
/// Returns [`tenferro_tensor::Error::runtime_state`] when no CUDA device is
/// reachable, a session or launch fails, or a downloaded value is wrong.
pub fn run_typing_rules_check() -> tenferro_tensor::Result<()> {
    const OP: &str = "cubecl-kernel-sample.typing";
    const N: usize = 8;
    let input: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let indices: Vec<i32> = vec![7, 0, 3, 6, 1, 5, 2, 4];

    let mut backend = first_backend(OP)?;
    let (gathered, reversed) = backend
        .with_backend_session(|session| {
            with_cuda_exec_session(session, |exec| {
                run_typing_rules_in_session(exec, &input, &indices, OP)
            })
        })?
        .ok_or_else(|| Error::runtime_state(OP, "CUDA session rejected backend"))??;

    let values = |tensor: TypedTensor<f32>| -> tenferro_tensor::Result<Vec<f32>> {
        let (_, values) = tensor
            .into_vec_col_major()
            .map_err(|failure| failure.into_parts().1)?;
        Ok(values)
    };
    let gathered = values(gathered)?;
    let reversed = values(reversed)?;

    let expected: Vec<f32> = indices.iter().map(|&i| input[i as usize]).collect();
    let expected_reversed: Vec<f32> = input.iter().rev().copied().collect();
    println!("{OP}: gather       = {gathered:?}");
    println!("{OP}: reverse_tile = {reversed:?}");
    assert_eq!(gathered, expected);
    assert_eq!(reversed, expected_reversed);
    Ok(())
}

/// Upload the three inputs, launch the typing-rule kernels in one session
/// scope, and download the three outputs.
fn run_typing_rules_in_session(
    exec: &mut CudaExecSession<'_>,
    input: &[f32],
    indices: &[i32],
    op: &'static str,
) -> tenferro_tensor::Result<(TypedTensor<f32>, TypedTensor<f32>)> {
    let n = input.len();
    let runtime = exec.runtime().clone();
    let host_input = TypedTensor::<f32>::from_vec_col_major(vec![n], input.to_vec())?;
    let host_indices = TypedTensor::<i32>::from_vec_col_major(vec![n], indices.to_vec())?;
    let input = upload_tensor(&runtime, &Tensor::from_typed::<f32>(host_input))?;
    let indices = upload_tensor(&runtime, &Tensor::from_typed::<i32>(host_indices))?;
    let input = input.into_typed::<f32>().expect("f32 upload stays f32");
    let indices = indices.into_typed::<i32>().expect("i32 upload stays i32");

    let outputs = exec.with_cubecl(op, |cubecl| {
        // SAFETY: each binding is created for a tensor this call owns, and no
        // kernel writes an input.
        let input_arg = unsafe { cubecl.array_arg(&input, op)? };
        // A second binding for the second launch: `BufferArg` is not `Clone`.
        let input_arg_again = unsafe { cubecl.array_arg(&input, op)? };
        let indices_arg = unsafe { cubecl.array_arg(&indices, op)? };
        let gathered = cubecl.alloc_output::<f32>(&[n])?;
        let reversed = cubecl.alloc_output::<f32>(&[n])?;
        // SAFETY: as above; the two outputs are exclusively owned here.
        let gathered_arg = unsafe { cubecl.array_arg(&gathered, op)? };
        let reversed_arg = unsafe { cubecl.array_arg(&reversed, op)? };

        // One cube of exactly `n` units for both launches: `cube_count_1d(n)`
        // would round the domain up to whole 256-unit cubes, and neither kernel
        // guards `ABSOLUTE_POS < n`. `n <= 1024` keeps the shared tile valid.
        let client = cubecl.client();

        // SAFETY: the domain is exactly `n` units, every stored index lies in
        // `0..n`, the gather writes `n` elements, and the reversal runs one cube
        // of `n` units over a shared tile of at most 1024 elements.
        unsafe {
            gather::launch_unchecked(
                client,
                CubeCount::new_single(),
                CubeDim::new_1d(n as u32),
                input_arg,
                indices_arg,
                gathered_arg,
            );
            reverse_tile::launch_unchecked(
                client,
                CubeCount::new_single(),
                CubeDim::new_1d(n as u32),
                input_arg_again,
                reversed_arg,
                n as u32,
            );
        }
        Ok((gathered, reversed))
    })?;

    exec.synchronize()?;
    Ok((
        download_tensor(&runtime, &Tensor::from_typed(outputs.0))?
            .into_typed::<f32>()
            .expect("f32 download"),
        download_tensor(&runtime, &Tensor::from_typed(outputs.1))?
            .into_typed::<f32>()
            .expect("f32 download"),
    ))
}

// snippet-start:downstream-cube-cache
/// Launch the same kernel from successive `with_cubecl` scopes.
///
/// The compiled kernel belongs to CubeCL's per-device server, which the tenferro
/// runtime's client already reaches, so nothing has to be cached downstream: the
/// first launch JIT-compiles, and every later scope reuses that artifact. The
/// host cost of each scope is returned as the evidence.
fn launches_across_scopes(
    exec: &mut CudaExecSession<'_>,
    input: &TypedTensor<f32>,
    n: usize,
    launches: usize,
) -> tenferro_tensor::Result<(Vec<Duration>, Vec<TypedTensor<f32>>)> {
    let mut costs = Vec::with_capacity(launches);
    let mut outputs = Vec::with_capacity(launches);
    for _ in 0..launches {
        let started = Instant::now();
        let output = exec.with_cubecl("downstream.scale.cache", |cubecl| {
            // SAFETY: this call owns `input` and the fresh `output`; the kernel
            // writes only `output`.
            let input_arg = unsafe { cubecl.array_arg(input, "downstream.scale.cache")? };
            let output = cubecl.alloc_output::<f32>(&[n])?;
            // SAFETY: as above.
            let output_arg = unsafe { cubecl.array_arg(&output, "downstream.scale.cache")? };
            let CubeCount::Static(x, y, z) = cubecl.cube_count_1d(n)? else {
                return Err(Error::runtime_state(
                    "downstream.scale.cache",
                    "expected static cube count",
                ));
            };
            // SAFETY: the domain covers exactly `n` elements, and `scale` is
            // in-bounds for both buffers.
            unsafe {
                scale::launch_unchecked(
                    cubecl.client(),
                    CubeCount::Static(x, y, z),
                    cubecl.cube_dim_1d(),
                    input_arg,
                    output_arg,
                    2.0f32,
                );
            }
            Ok(output)
        })?;
        // The launch is asynchronous: the allocation is kept alive, and the
        // caller's single barrier covers all of them.
        costs.push(started.elapsed());
        outputs.push(output);
    }
    exec.synchronize()?;
    Ok((costs, outputs))
}
// snippet-end:downstream-cube-cache

/// Show that the compiled kernel is reused across `with_cubecl` scopes.
///
/// Launches [`scale()`] from `LAUNCHES` separate session scopes on one backend and
/// then from a second, freshly constructed [`CudaBackend`] for the same device,
/// printing the host cost of every scope. The first scope pays the JIT compile;
/// the rest are launch-only. Every launch is verified after a single barrier.
///
/// Hardware-gated like [`run_scale_check`].
///
/// # Examples
///
/// Hardware-gated: the call is wrapped in `catch_unwind` because `cudarc`
/// panics when the CUDA driver library is absent.
///
/// ```
/// let outcome = std::panic::catch_unwind(|| cubecl_kernel_sample::run_kernel_cache_check());
/// let _ = outcome;
/// ```
///
/// # Errors
///
/// Returns [`tenferro_tensor::Error::runtime_state`] when no CUDA device is
/// reachable, a session or launch fails, or a downloaded value is wrong.
pub fn run_kernel_cache_check() -> tenferro_tensor::Result<()> {
    const OP: &str = "cubecl-kernel-sample.cache";
    const LAUNCHES: usize = 6;
    let n = 1024usize;
    let input: Vec<f32> = (0..n).map(|i| i as f32).collect();

    for backend_index in 0..2 {
        // A fresh backend per round: the second one proves the compiled kernel
        // is not owned by the first backend object.
        let mut backend = first_backend(OP)?;
        let host_input = TypedTensor::<f32>::from_vec_col_major(vec![n], input.clone())?;
        let costs = backend
            .with_backend_session(|session| {
                with_cuda_exec_session(session, |exec| {
                    let runtime = exec.runtime().clone();
                    let uploaded = upload_tensor(&runtime, &Tensor::from_typed::<f32>(host_input))?;
                    let uploaded = uploaded.into_typed::<f32>().expect("f32 upload stays f32");
                    // The second backend runs one launch only: its cost shows
                    // whether the artifact outlives the first backend.
                    let launches = if backend_index == 0 { LAUNCHES } else { 1 };
                    let (costs, outputs) = launches_across_scopes(exec, &uploaded, n, launches)?;
                    for (index, output) in outputs.into_iter().enumerate() {
                        let (_, values) = download_tensor(&runtime, &Tensor::from_typed(output))?
                            .into_typed::<f32>()
                            .expect("f32 download")
                            .into_vec_col_major()
                            .map_err(|failure| failure.into_parts().1)?;
                        for (i, value) in values.iter().enumerate() {
                            let expected = (i as f32) * 2.0;
                            if (value - expected).abs() > 1e-3 {
                                return Err(Error::runtime_state(
                                    OP,
                                    format!(
                                        "launch {index} mismatch at {i}: {value} != {expected}"
                                    ),
                                ));
                            }
                        }
                    }
                    Ok(costs)
                })
            })?
            .ok_or_else(|| Error::runtime_state(OP, "CUDA session rejected backend"))??;

        println!("{OP}: backend {backend_index} per-scope host cost:");
        for (index, cost) in costs.iter().enumerate() {
            println!("  launch {index}: {cost:?}");
        }
    }
    Ok(())
}

#[allow(dead_code)]
// compile-checked source of the guide's ordering snippet
// snippet-start:custom_cuda_cubecl_ordering
/// Launch [`scale`], then run raw work that depends on its output.
///
/// The fence is what orders the raw continuation after the launch; see the
/// external-kernel guide for the mechanism.
fn launch_then_raw(
    exec: &mut CudaExecSession<'_>,
    client: &Client,
    input: BufferArg,
    output: BufferArg,
) -> tenferro_tensor::Result<()> {
    exec.with_raw("downstream.hybrid", |raw| {
        // SAFETY: the caller owns both bindings, the launch domain matches their
        // element counts, and the kernel writes only `output`.
        unsafe {
            scale::launch_unchecked(
                client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                input,
                output,
                2.0f32,
            );
        }
        // Dispatch the launch onto this session's stream before the raw work
        // that depends on it: without this fence the raw kernel below can reach
        // the stream first (#2092).
        raw.flush_cubecl()?;
        // The raw continuation reads `output` here: an NVRTC/PTX launch, a
        // vendor call, or a transfer.
        Ok(())
    })
}
// snippet-end:custom_cuda_cubecl_ordering
