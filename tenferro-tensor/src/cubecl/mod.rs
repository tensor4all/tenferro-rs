//! CubeCL-based GPU backend for tenferro tensors.
//!
//! This module provides GPU acceleration via [CubeCL](https://github.com/tracel-ai/cubecl)
//! running on NVIDIA CUDA devices. It is gated behind the `cubecl` feature flag and
//! requires **CUDA 12+** with a compatible NVIDIA GPU.
//!
//! # Enabling the feature
//!
//! Add to your `Cargo.toml`:
//!
//! ```toml
//! tenferro-tensor = { version = "...", features = ["cubecl"] }
//! ```
//!
//! You must also enable a CPU backend (`cpu-faer` or `cpu-blas`); the CubeCL backend
//! complements the CPU path but does not replace it.
//!
//! # Prerequisites
//!
//! - NVIDIA GPU with CUDA compute capability ≥ 7.0
//! - CUDA Toolkit 12.x installed (provides NVRTC for JIT kernel compilation)
//! - cuTENSOR, cuSOLVER, cuBLAS shared libraries available on `LD_LIBRARY_PATH`
//!
//! ## Environment variables
//!
//! | Variable | Purpose |
//! |----------|---------|
//! | `CUDA_PATH` | CUDA toolkit root (e.g. `/usr/local/cuda-12.0`) |
//! | `CUBECL_DEBUG_LOG` | Set to `0` to suppress verbose JIT logs |
//! | `TENFERRO_CUTENSOR_PATH` | Override cuTENSOR library search path |
//! | `TENFERRO_CUSOLVER_PATH` | Override cuSOLVER library search path |
//! | `TENFERRO_CUBLAS_PATH` | Override cuBLAS library search path |
//!
//! # Basic usage
//!
//! GPU tensors must be explicitly uploaded before use on the device and downloaded
//! back to the host afterwards (no implicit CPU↔GPU transfer, following the PyTorch
//! convention).
//!
//! ```ignore
//! use tenferro_tensor::cubecl::{CubeclBackend, upload_tensor, download_tensor};
//! use tenferro_tensor::{Tensor, TensorBackend, TypedTensor};
//!
//! fn main() -> tenferro_tensor::Result<()> {
//! // 1. Create the GPU backend (device ordinal 0)
//! let mut backend = CubeclBackend::new(0)?;
//!
//! // 2. Create tensors on the CPU
//! let a = Tensor::F64(TypedTensor::from_vec(vec![2], vec![1.0, 2.0]));
//! let b = Tensor::F64(TypedTensor::from_vec(vec![2], vec![3.0, 4.0]));
//!
//! // 3. Upload to GPU
//! let gpu_a = upload_tensor(backend.runtime(), &a)?;
//! let gpu_b = upload_tensor(backend.runtime(), &b)?;
//!
//! // 4. Compute on GPU
//! let gpu_c = backend.add(&gpu_a, &gpu_b)?;
//!
//! // 5. Download result back to CPU
//! let cpu_c = download_tensor(backend.runtime(), &gpu_c)?;
//! assert_eq!(cpu_c.shape(), &[2]);
//! Ok(())
//! }
//! ```
//!
//! # Running GPU tests
//!
//! All GPU tests are marked `#[ignore]` so that `cargo test --features cubecl`
//! passes on machines without a GPU. To actually run them:
//!
//! ```sh
//! CUBECL_DEBUG_LOG=0 \
//! CUDA_PATH=/usr/local/cuda-12.0 \
//! cargo test -p tenferro-tensor --features cubecl -- --ignored
//! ```

use std::cell::OnceCell;

use cubecl::client::ComputeClient;
use cubecl::features::AtomicUsage;
use cubecl::prelude::{
    ArrayArg, Complex as CubeComplex, CubeElement, CubePrimitive, Float as CubeFloat,
};
use cubecl::prelude::{Int as CubeInt, StorageType, TensorBinding, Type};
use cubecl_cuda::CudaRuntime;
use num_complex::{Complex32, Complex64};
use tenferro_cubecl::reduce::{self as cubecl_reduce, ReduceStrategy};
use tenferro_cubecl::{diagonal, elementwise, indexing, structural};

use crate::backend::TensorBackend;
use crate::config::{
    CompareDir, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
};
use crate::{Buffer, Tensor, TypedTensor};

mod dispatch;
mod ffi;
mod fusion;
mod gemm;
mod linalg;
mod memory;
mod runtime;

use dispatch::{
    alloc_output, comptime_sequence, cube_count_for_len, cube_dim_1d, dtype_mismatch,
    ensure_axes_unique, ensure_axis, ensure_rank, ensure_resident_on_runtime, launch_binary,
    launch_binary_tensor, launch_nullary_into, launch_ternary, launch_unary, launch_unary_tensor,
    launch_unary_tensor_into, ternary_dtype_mismatch, typed_tensor_array_arg,
    typed_tensor_array_arg_as, typed_tensor_binding,
};

pub use memory::{device_ptr, download_tensor, upload_tensor};
pub use runtime::{gpu_available, CubeclRuntime};

fn unsupported_dtype(op: &'static str, dtype: crate::DType) -> crate::Error {
    crate::Error::BackendFailure {
        op,
        message: format!("unsupported dtype {dtype:?}"),
    }
}

fn ensure_atomic_add_supported<T: CubePrimitive>(
    client: &ComputeClient<CudaRuntime>,
    op: &'static str,
) -> crate::Result<()> {
    let elem = T::as_type_native_unchecked().elem_type();
    let atomic_ty = Type::new(StorageType::Atomic(elem));
    if client
        .properties()
        .atomic_type_usage(atomic_ty)
        .contains(AtomicUsage::Add)
    {
        Ok(())
    } else {
        Err(crate::Error::BackendFailure {
            op,
            message: format!("CubeCL runtime does not support atomic add for {elem:?}"),
        })
    }
}

fn checked_dim_product(
    op: &'static str,
    role: &'static str,
    shape: &[usize],
) -> crate::Result<usize> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| crate::Error::BackendFailure {
                op,
                message: format!("{role} product overflow for shape {shape:?}"),
            })
    })
}

fn scatter_update_len(meta: &ScatterLaunchMeta) -> crate::Result<usize> {
    let batch_len = checked_dim_product("scatter", "batch shape", &meta.batch_shape)?;
    let window_len =
        checked_dim_product("scatter", "window update shape", &meta.window_shape_updates)?;
    batch_len
        .checked_mul(window_len)
        .ok_or_else(|| crate::Error::BackendFailure {
            op: "scatter",
            message: format!(
                "scatter update domain product overflow for batch {:?} and window {:?}",
                meta.batch_shape, meta.window_shape_updates
            ),
        })
}

/// CubeCL-based GPU backend.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::cubecl::CubeclBackend;
///
/// let _ctor: fn(usize) -> tenferro_tensor::Result<CubeclBackend> = CubeclBackend::new;
/// ```
pub struct CubeclBackend {
    rt: CubeclRuntime,
    cutensor: OnceCell<crate::Result<ffi::cutensor::CutensorHandle>>,
    linalg: OnceCell<crate::Result<ffi::cusolver::CudaLinalgHandles>>,
}

impl CubeclBackend {
    /// Create a new CubeCL backend for the given CUDA device ordinal.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::cubecl::CubeclBackend;
    ///
    /// let _ctor: fn(usize) -> tenferro_tensor::Result<CubeclBackend> = CubeclBackend::new;
    /// ```
    pub fn new(device_ordinal: usize) -> crate::Result<Self> {
        Ok(Self {
            rt: CubeclRuntime::new(device_ordinal)?,
            cutensor: OnceCell::new(),
            linalg: OnceCell::new(),
        })
    }

    /// Borrow the underlying CubeCL runtime.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::cubecl::{CubeclBackend, CubeclRuntime};
    ///
    /// let _runtime: fn(&CubeclBackend) -> &CubeclRuntime = CubeclBackend::runtime;
    /// ```
    pub fn runtime(&self) -> &CubeclRuntime {
        &self.rt
    }

    fn i64_indices_as_f64(&self, indices: &TypedTensor<i64>) -> crate::Result<TypedTensor<f64>> {
        let host_values = match &indices.buffer {
            Buffer::Host(data) => data.clone(),
            Buffer::Cubecl(_) => {
                let downloaded = download_tensor(self.runtime(), &Tensor::I64(indices.clone()))?;
                downloaded
                    .as_slice::<i64>()
                    .expect("downloaded I64 tensor")
                    .to_vec()
            }
            Buffer::Backend(_) => {
                return Err(crate::Error::BackendFailure {
                    op: "index_tensor",
                    message: "backend buffers are not supported for CubeCL index conversion".into(),
                })
            }
        };
        let converted = Tensor::F64(TypedTensor::from_vec(
            indices.shape.clone(),
            host_values.into_iter().map(|value| value as f64).collect(),
        ));
        match &indices.buffer {
            Buffer::Cubecl(_) => match upload_tensor(self.runtime(), &converted)? {
                Tensor::F64(tensor) => Ok(tensor),
                _ => unreachable!("upload preserves dtype"),
            },
            _ => match converted {
                Tensor::F64(tensor) => Ok(tensor),
                _ => unreachable!("constructed F64 tensor"),
            },
        }
    }

    fn cutensor_handle(&self) -> crate::Result<&ffi::cutensor::CutensorHandle> {
        match self
            .cutensor
            .get_or_init(ffi::cutensor::CutensorHandle::load)
        {
            Ok(handle) => Ok(handle),
            Err(err) => Err(err.clone()),
        }
    }

    fn linalg_handles(&self) -> crate::Result<&ffi::cusolver::CudaLinalgHandles> {
        match self
            .linalg
            .get_or_init(ffi::cusolver::CudaLinalgHandles::load)
        {
            Ok(handles) => Ok(handles),
            Err(err) => Err(err.clone()),
        }
    }

    fn transpose_typed<T>(
        &self,
        input: &TypedTensor<T>,
        perm: &[usize],
    ) -> crate::Result<TypedTensor<T>>
    where
        T: CubeElement + CubePrimitive + Clone,
    {
        validate_permutation("transpose", perm, input.shape.len())?;
        let output_shape: Vec<usize> = perm.iter().map(|&axis| input.shape[axis]).collect();
        launch_unary_tensor(
            self.runtime(),
            input,
            &output_shape,
            "transpose",
            |client, count, dim, out, input_arg| unsafe {
                structural::transpose_kernel::launch_unchecked::<T, CudaRuntime>(
                    client,
                    count,
                    dim,
                    out.into_tensor_arg(),
                    input_arg.into_tensor_arg(),
                    comptime_sequence(perm),
                );
            },
        )
    }

    fn broadcast_typed<T>(
        &self,
        input: &TypedTensor<T>,
        shape: &[usize],
        dims: &[usize],
    ) -> crate::Result<TypedTensor<T>>
    where
        T: CubeElement + CubePrimitive + Clone,
    {
        validate_broadcast_in_dim(input.shape.as_slice(), shape, dims)?;
        launch_unary_tensor(
            self.runtime(),
            input,
            shape,
            "broadcast_in_dim",
            |client, count, dim, out, input_arg| unsafe {
                structural::broadcast_in_dim_kernel::launch_unchecked::<T, CudaRuntime>(
                    client,
                    count,
                    dim,
                    out.into_tensor_arg(),
                    input_arg.into_tensor_arg(),
                    comptime_sequence(dims),
                    shape.len(),
                );
            },
        )
    }

    fn reverse_typed<T>(
        &self,
        input: &TypedTensor<T>,
        axes: &[usize],
    ) -> crate::Result<TypedTensor<T>>
    where
        T: CubeElement + CubePrimitive + Clone,
    {
        ensure_axes_unique("reverse", "axes", axes, input.shape.len())?;
        launch_unary_tensor(
            self.runtime(),
            input,
            &input.shape,
            "reverse",
            |client, count, dim, out, input_arg| unsafe {
                structural::reverse_kernel::launch_unchecked::<T, CudaRuntime>(
                    client,
                    count,
                    dim,
                    out.into_tensor_arg(),
                    input_arg.into_tensor_arg(),
                    comptime_sequence(axes),
                    input.shape.len(),
                );
            },
        )
    }

    fn convert_float_to_float<In, Out>(
        &self,
        input: &TypedTensor<In>,
    ) -> crate::Result<TypedTensor<Out>>
    where
        In: CubeElement + CubeFloat + Clone,
        Out: CubeElement + CubeFloat + Clone,
    {
        launch_unary(
            self.runtime(),
            input,
            &input.shape,
            "convert",
            |client, count, dim, out, input_arg| unsafe {
                structural::convert_float_to_float::launch_unchecked::<Out, In, CudaRuntime>(
                    client, count, dim, out, input_arg,
                );
            },
        )
    }

    fn convert_f32_to_c32(
        &self,
        input: &TypedTensor<f32>,
    ) -> crate::Result<TypedTensor<Complex32>> {
        self.convert_float_to_complex_raw::<f32, Complex32, f32>(input, |client, out, input, n| {
            unsafe {
                // SAFETY: `convert_float_to_complex_raw` validated that
                // `input` has `n` elements and `out` has `2 * n` scalar
                // components. The kernel launches exactly `n` logical input
                // positions and guards with `ABSOLUTE_POS < input.len()`.
                structural::convert_f32_to_c32_raw::launch_unchecked::<CudaRuntime>(
                    client,
                    cube_count_for_len(n),
                    cube_dim_1d(),
                    out,
                    input,
                );
            }
        })
    }

    fn convert_f32_to_c64(
        &self,
        input: &TypedTensor<f32>,
    ) -> crate::Result<TypedTensor<Complex64>> {
        self.convert_float_to_complex_raw::<f32, Complex64, f64>(input, |client, out, input, n| {
            unsafe {
                // SAFETY: `convert_float_to_complex_raw` validated that
                // `input` has `n` elements and `out` has `2 * n` scalar
                // components. The kernel launches exactly `n` logical input
                // positions and guards with `ABSOLUTE_POS < input.len()`.
                structural::convert_f32_to_c64_raw::launch_unchecked::<CudaRuntime>(
                    client,
                    cube_count_for_len(n),
                    cube_dim_1d(),
                    out,
                    input,
                );
            }
        })
    }

    fn convert_f64_to_c32(
        &self,
        input: &TypedTensor<f64>,
    ) -> crate::Result<TypedTensor<Complex32>> {
        self.convert_float_to_complex_raw::<f64, Complex32, f32>(input, |client, out, input, n| {
            unsafe {
                // SAFETY: `convert_float_to_complex_raw` validated that
                // `input` has `n` elements and `out` has `2 * n` scalar
                // components. The kernel launches exactly `n` logical input
                // positions and guards with `ABSOLUTE_POS < input.len()`.
                structural::convert_f64_to_c32_raw::launch_unchecked::<CudaRuntime>(
                    client,
                    cube_count_for_len(n),
                    cube_dim_1d(),
                    out,
                    input,
                );
            }
        })
    }

    fn convert_f64_to_c64(
        &self,
        input: &TypedTensor<f64>,
    ) -> crate::Result<TypedTensor<Complex64>> {
        self.convert_float_to_complex_raw::<f64, Complex64, f64>(input, |client, out, input, n| {
            unsafe {
                // SAFETY: `convert_float_to_complex_raw` validated that
                // `input` has `n` elements and `out` has `2 * n` scalar
                // components. The kernel launches exactly `n` logical input
                // positions and guards with `ABSOLUTE_POS < input.len()`.
                structural::convert_f64_to_c64_raw::launch_unchecked::<CudaRuntime>(
                    client,
                    cube_count_for_len(n),
                    cube_dim_1d(),
                    out,
                    input,
                );
            }
        })
    }

    /// Generic float-to-complex conversion via raw interleaved kernel.
    ///
    /// The kernel writes `(re, 0, re, 0, ...)` into a raw float buffer that
    /// is then reinterpreted as complex.
    fn convert_float_to_complex_raw<InFloat, OutComplex, OutFloat>(
        &self,
        input: &TypedTensor<InFloat>,
        launch: impl FnOnce(
            &cubecl::client::ComputeClient<CudaRuntime>,
            ArrayArg<CudaRuntime>,
            ArrayArg<CudaRuntime>,
            usize,
        ),
    ) -> crate::Result<TypedTensor<OutComplex>>
    where
        InFloat: CubeElement + Clone,
        OutComplex: CubeElement + Clone,
        OutFloat: CubeElement + Clone,
    {
        let n = input.n_elements();
        let output = alloc_output::<OutComplex>(self.runtime(), &input.shape);
        if n == 0 {
            return Ok(output);
        }
        let output_part_len = n
            .checked_mul(2)
            .ok_or_else(|| crate::Error::BackendFailure {
                op: "convert",
                message: "complex output part length overflow".into(),
            })?;
        let output_parts =
            typed_tensor_array_arg_as::<OutComplex, OutFloat>(&output, output_part_len, "convert")?;
        let input_arg = typed_tensor_array_arg(input, "convert")?;
        // SAFETY: The checked raw-array helpers prove that `input_arg` covers
        // exactly the dense input shape and `output_parts` covers the complete
        // real/imaginary scalar representation of the output allocation.
        launch(self.runtime().client(), output_parts, input_arg, n);
        Ok(output)
    }

    fn convert_c32_to_f32(
        &self,
        input: &TypedTensor<Complex32>,
    ) -> crate::Result<TypedTensor<f32>> {
        launch_unary(
            self.runtime(),
            input,
            &input.shape,
            "convert",
            |client, count, dim, out, input_arg| unsafe {
                structural::convert_c32_to_f32::launch_unchecked::<CudaRuntime>(
                    client, count, dim, out, input_arg,
                );
            },
        )
    }

    fn convert_c32_to_f64(
        &self,
        input: &TypedTensor<Complex32>,
    ) -> crate::Result<TypedTensor<f64>> {
        launch_unary(
            self.runtime(),
            input,
            &input.shape,
            "convert",
            |client, count, dim, out, input_arg| unsafe {
                structural::convert_c32_to_f64::launch_unchecked::<CudaRuntime>(
                    client, count, dim, out, input_arg,
                );
            },
        )
    }

    fn convert_c64_to_f32(
        &self,
        input: &TypedTensor<Complex64>,
    ) -> crate::Result<TypedTensor<f32>> {
        launch_unary(
            self.runtime(),
            input,
            &input.shape,
            "convert",
            |client, count, dim, out, input_arg| unsafe {
                structural::convert_c64_to_f32::launch_unchecked::<CudaRuntime>(
                    client, count, dim, out, input_arg,
                );
            },
        )
    }

    fn convert_c64_to_f64(
        &self,
        input: &TypedTensor<Complex64>,
    ) -> crate::Result<TypedTensor<f64>> {
        launch_unary(
            self.runtime(),
            input,
            &input.shape,
            "convert",
            |client, count, dim, out, input_arg| unsafe {
                structural::convert_c64_to_f64::launch_unchecked::<CudaRuntime>(
                    client, count, dim, out, input_arg,
                );
            },
        )
    }

    fn convert_complex_to_complex<In, Out>(
        &self,
        input: &TypedTensor<In>,
    ) -> crate::Result<TypedTensor<Out>>
    where
        In: CubeElement + CubeComplex + Clone,
        Out: CubeElement + CubeComplex + Clone,
    {
        launch_unary(
            self.runtime(),
            input,
            &input.shape,
            "convert",
            |client, count, dim, out, input_arg| unsafe {
                structural::convert_complex_to_complex::launch_unchecked::<Out, In, CudaRuntime>(
                    client, count, dim, out, input_arg,
                );
            },
        )
    }

    fn extract_diagonal_typed<T>(
        &self,
        input: &TypedTensor<T>,
        axis_a: usize,
        axis_b: usize,
    ) -> crate::Result<TypedTensor<T>>
    where
        T: CubeElement + CubePrimitive + Clone,
    {
        let (output_shape, diag_output_axis) =
            extract_diagonal_shape(input.shape.as_slice(), axis_a, axis_b)?;
        launch_unary_tensor(
            self.runtime(),
            input,
            &output_shape,
            "extract_diagonal",
            |client, count, dim, out, input_arg| unsafe {
                diagonal::extract_diagonal_kernel::launch_unchecked::<T, CudaRuntime>(
                    client,
                    count,
                    dim,
                    out.into_tensor_arg(),
                    input_arg.into_tensor_arg(),
                    axis_a,
                    axis_b,
                    diag_output_axis,
                    input.shape.len(),
                    output_shape.len(),
                );
            },
        )
    }

    fn embed_diagonal_typed<T>(
        &self,
        input: &TypedTensor<T>,
        axis_a: usize,
        axis_b: usize,
    ) -> crate::Result<TypedTensor<T>>
    where
        T: CubeElement + CubePrimitive + Clone,
    {
        let output_shape = embed_diagonal_shape(input.shape.as_slice(), axis_a, axis_b)?;
        let output = alloc_output::<T>(self.runtime(), &output_shape);
        launch_nullary_into(
            self.runtime(),
            &output,
            "embed_diagonal",
            cube_count_for_len(output.n_elements()),
            cube_dim_1d(),
            |client, count, dim, out| unsafe {
                structural::fill_zero_kernel::launch_unchecked::<T, CudaRuntime>(
                    client, count, dim, out,
                );
            },
        )?;
        launch_unary_tensor_into(
            self.runtime(),
            &output,
            input,
            "embed_diagonal",
            cube_count_for_len(input.n_elements()),
            cube_dim_1d(),
            |client, count, dim, out, input_arg| unsafe {
                diagonal::embed_diagonal_copy_kernel::launch_unchecked::<T, CudaRuntime>(
                    client,
                    count,
                    dim,
                    out.into_tensor_arg(),
                    input_arg.into_tensor_arg(),
                    axis_a,
                    axis_b,
                    input.shape.len(),
                    output_shape.len(),
                );
            },
        )?;
        Ok(output)
    }

    fn tril_typed<T>(&self, input: &TypedTensor<T>, k: i64) -> crate::Result<TypedTensor<T>>
    where
        T: CubeElement + CubePrimitive + Clone,
    {
        if input.shape.len() < 2 {
            return Err(crate::Error::RankMismatch {
                op: "tril",
                expected: 2,
                actual: input.shape.len(),
            });
        }
        launch_unary_tensor(
            self.runtime(),
            input,
            &input.shape,
            "tril",
            |client, count, dim, out, input_arg| unsafe {
                diagonal::tril_kernel::launch_unchecked::<T, CudaRuntime>(
                    client,
                    count,
                    dim,
                    out.into_tensor_arg(),
                    input_arg.into_tensor_arg(),
                    k,
                );
            },
        )
    }

    fn triu_typed<T>(&self, input: &TypedTensor<T>, k: i64) -> crate::Result<TypedTensor<T>>
    where
        T: CubeElement + CubePrimitive + Clone,
    {
        if input.shape.len() < 2 {
            return Err(crate::Error::RankMismatch {
                op: "triu",
                expected: 2,
                actual: input.shape.len(),
            });
        }
        launch_unary_tensor(
            self.runtime(),
            input,
            &input.shape,
            "triu",
            |client, count, dim, out, input_arg| unsafe {
                diagonal::triu_kernel::launch_unchecked::<T, CudaRuntime>(
                    client,
                    count,
                    dim,
                    out.into_tensor_arg(),
                    input_arg.into_tensor_arg(),
                    k,
                );
            },
        )
    }

    fn launch_reduce_axis_typed<T>(
        &self,
        input: &TypedTensor<T>,
        axis: usize,
        op: &'static str,
        launch: impl FnOnce(
            &ComputeClient<CudaRuntime>,
            TensorBinding<CudaRuntime>,
            TensorBinding<CudaRuntime>,
        ) -> tenferro_cubecl::Result<()>,
    ) -> crate::Result<TypedTensor<T>>
    where
        T: CubeElement + Clone,
    {
        let output_shape = reduction_keepdims_shape(&input.shape, axis);
        let output = alloc_output::<T>(self.runtime(), &output_shape);
        if output.n_elements() == 0 {
            return Ok(output);
        }

        let input_binding = typed_tensor_binding(input, op)?;
        let output_binding = typed_tensor_binding(&output, op)?;
        launch(self.runtime().client(), input_binding, output_binding).map_err(|err| {
            crate::Error::BackendFailure {
                op,
                message: err.to_string(),
            }
        })?;
        Ok(output)
    }

    fn reduce_axes_typed<T>(
        &self,
        input: &TypedTensor<T>,
        axes: &[usize],
        op: &'static str,
        mut launch_axis: impl FnMut(&Self, &TypedTensor<T>, usize) -> crate::Result<TypedTensor<T>>,
    ) -> crate::Result<TypedTensor<T>>
    where
        T: CubeElement + Clone,
    {
        ensure_axes_unique(op, "axes", axes, input.shape.len())?;
        if axes.is_empty() {
            return Ok(input.clone());
        }

        let final_shape = reduction_output_shape(input.shape.as_slice(), axes);
        let mut sorted_axes = axes.to_vec();
        sorted_axes.sort_unstable();

        let mut current = input.clone();
        for axis in sorted_axes {
            current = launch_axis(self, &current, axis)?;
        }

        cubecl_reshape_metadata(current, final_shape, op)
    }

    fn reduce_sum_float_typed<F: CubeElement + CubeFloat + Clone>(
        &self,
        input: &TypedTensor<F>,
        axes: &[usize],
    ) -> crate::Result<TypedTensor<F>> {
        self.reduce_axes_typed(input, axes, "reduce_sum", |backend, current, axis| {
            backend.launch_reduce_axis_typed(
                current,
                axis,
                "reduce_sum",
                |client, input, output| {
                    cubecl_reduce::launch_sum_float::<CudaRuntime, F>(
                        client,
                        input,
                        output,
                        axis,
                        ReduceStrategy::Auto,
                    )
                },
            )
        })
    }

    fn reduce_sum_complex_typed<C: CubeElement + CubeComplex + Clone>(
        &self,
        input: &TypedTensor<C>,
        axes: &[usize],
    ) -> crate::Result<TypedTensor<C>> {
        self.reduce_axes_typed(input, axes, "reduce_sum", |backend, current, axis| {
            backend.launch_reduce_axis_typed(
                current,
                axis,
                "reduce_sum",
                |client, input, output| {
                    cubecl_reduce::launch_sum_complex::<CudaRuntime, C>(
                        client,
                        input,
                        output,
                        axis,
                        ReduceStrategy::Auto,
                    )
                },
            )
        })
    }

    fn reduce_sum_int_typed<I: CubeElement + CubeInt + Clone>(
        &self,
        input: &TypedTensor<I>,
        axes: &[usize],
    ) -> crate::Result<TypedTensor<I>> {
        self.reduce_axes_typed(input, axes, "reduce_sum", |backend, current, axis| {
            backend.launch_reduce_axis_typed(
                current,
                axis,
                "reduce_sum",
                |client, input, output| {
                    cubecl_reduce::launch_sum_int::<CudaRuntime, I>(
                        client,
                        input,
                        output,
                        axis,
                        ReduceStrategy::Auto,
                    )
                },
            )
        })
    }

    fn reduce_prod_float_typed<F: CubeElement + CubeFloat + Clone>(
        &self,
        input: &TypedTensor<F>,
        axes: &[usize],
    ) -> crate::Result<TypedTensor<F>> {
        self.reduce_axes_typed(input, axes, "reduce_prod", |backend, current, axis| {
            backend.launch_reduce_axis_typed(
                current,
                axis,
                "reduce_prod",
                |client, input, output| {
                    cubecl_reduce::launch_prod_float::<CudaRuntime, F>(
                        client,
                        input,
                        output,
                        axis,
                        ReduceStrategy::Auto,
                    )
                },
            )
        })
    }

    fn reduce_prod_complex_typed<C: CubeElement + CubeComplex + Clone>(
        &self,
        input: &TypedTensor<C>,
        axes: &[usize],
    ) -> crate::Result<TypedTensor<C>> {
        self.reduce_axes_typed(input, axes, "reduce_prod", |backend, current, axis| {
            backend.launch_reduce_axis_typed(
                current,
                axis,
                "reduce_prod",
                |client, input, output| {
                    cubecl_reduce::launch_prod_complex::<CudaRuntime, C>(
                        client,
                        input,
                        output,
                        axis,
                        ReduceStrategy::Auto,
                    )
                },
            )
        })
    }

    fn reduce_prod_int_typed<I: CubeElement + CubeInt + Clone>(
        &self,
        input: &TypedTensor<I>,
        axes: &[usize],
    ) -> crate::Result<TypedTensor<I>> {
        self.reduce_axes_typed(input, axes, "reduce_prod", |backend, current, axis| {
            backend.launch_reduce_axis_typed(
                current,
                axis,
                "reduce_prod",
                |client, input, output| {
                    cubecl_reduce::launch_prod_int::<CudaRuntime, I>(
                        client,
                        input,
                        output,
                        axis,
                        ReduceStrategy::Auto,
                    )
                },
            )
        })
    }

    fn reduce_max_typed<F: CubeElement + CubeFloat + Clone>(
        &self,
        input: &TypedTensor<F>,
        axes: &[usize],
    ) -> crate::Result<TypedTensor<F>> {
        self.reduce_axes_typed(input, axes, "reduce_max", |backend, current, axis| {
            backend.launch_reduce_axis_typed(
                current,
                axis,
                "reduce_max",
                |client, input, output| {
                    cubecl_reduce::launch_max_float::<CudaRuntime, F>(
                        client,
                        input,
                        output,
                        axis,
                        ReduceStrategy::Auto,
                    )
                },
            )
        })
    }

    fn reduce_min_typed<F: CubeElement + CubeFloat + Clone>(
        &self,
        input: &TypedTensor<F>,
        axes: &[usize],
    ) -> crate::Result<TypedTensor<F>> {
        self.reduce_axes_typed(input, axes, "reduce_min", |backend, current, axis| {
            backend.launch_reduce_axis_typed(
                current,
                axis,
                "reduce_min",
                |client, input, output| {
                    cubecl_reduce::launch_min_float::<CudaRuntime, F>(
                        client,
                        input,
                        output,
                        axis,
                        ReduceStrategy::Auto,
                    )
                },
            )
        })
    }

    fn slice_typed<T>(
        &self,
        input: &TypedTensor<T>,
        config: &SliceConfig,
    ) -> crate::Result<TypedTensor<T>>
    where
        T: CubeElement + CubePrimitive + Clone,
    {
        let output_shape = validate_slice(input.shape.as_slice(), config)?;
        launch_unary_tensor(
            self.runtime(),
            input,
            &output_shape,
            "slice",
            |client, count, dim, out, input_arg| unsafe {
                indexing::slice_kernel::launch_unchecked::<T, CudaRuntime>(
                    client,
                    count,
                    dim,
                    out.into_tensor_arg(),
                    input_arg.into_tensor_arg(),
                    comptime_sequence(&config.starts),
                    comptime_sequence(&config.strides),
                );
            },
        )
    }

    fn dynamic_slice_typed<T, I>(
        &self,
        input: &TypedTensor<T>,
        starts: &TypedTensor<I>,
        slice_sizes: &[usize],
    ) -> crate::Result<TypedTensor<T>>
    where
        T: CubeElement + CubePrimitive + Clone,
        I: CubeElement + CubeFloat + Clone,
    {
        ensure_rank("dynamic_slice", input.shape.len(), slice_sizes.len())?;
        ensure_rank("dynamic_slice", 1, starts.shape.len())?;
        if starts.shape[0] != input.shape.len() {
            return Err(crate::Error::RankMismatch {
                op: "dynamic_slice",
                expected: input.shape.len(),
                actual: starts.shape[0],
            });
        }
        for (axis, (&window, &dim)) in slice_sizes.iter().zip(&input.shape).enumerate() {
            if window > dim {
                return Err(crate::Error::InvalidConfig {
                    op: "dynamic_slice",
                    message: format!("slice size exceeds dimension on axis {axis}"),
                });
            }
        }
        launch_binary_tensor(
            self.runtime(),
            input,
            starts,
            slice_sizes,
            "dynamic_slice",
            |client, count, dim, out, input_arg, starts_arg| unsafe {
                indexing::dynamic_slice_kernel::launch_unchecked::<T, I, CudaRuntime>(
                    client,
                    count,
                    dim,
                    out.into_tensor_arg(),
                    input_arg.into_tensor_arg(),
                    starts_arg.into_tensor_arg(),
                    comptime_sequence(slice_sizes),
                );
            },
        )
    }

    fn pad_typed<T>(
        &self,
        input: &TypedTensor<T>,
        config: &PadConfig,
    ) -> crate::Result<TypedTensor<T>>
    where
        T: CubeElement + CubePrimitive + Clone,
    {
        let output_shape = pad_output_shape(input.shape.as_slice(), config)?;
        launch_unary_tensor(
            self.runtime(),
            input,
            &output_shape,
            "pad",
            |client, count, dim, out, input_arg| unsafe {
                indexing::pad_kernel::launch_unchecked::<T, CudaRuntime>(
                    client,
                    count,
                    dim,
                    out.into_tensor_arg(),
                    input_arg.into_tensor_arg(),
                    comptime_sequence(&config.edge_padding_low),
                    comptime_sequence(&config.interior_padding),
                );
            },
        )
    }

    fn concatenate_typed<T>(
        &self,
        inputs: &[&TypedTensor<T>],
        axis: usize,
    ) -> crate::Result<TypedTensor<T>>
    where
        T: CubeElement + CubePrimitive + Clone,
    {
        let output_shape = concatenate_output_shape(inputs, axis)?;
        let output = alloc_output::<T>(self.runtime(), &output_shape);
        let mut offset = 0usize;
        for input in inputs {
            launch_unary_tensor_into(
                self.runtime(),
                &output,
                input,
                "concatenate",
                cube_count_for_len(input.n_elements()),
                cube_dim_1d(),
                |client, count, dim, out, input_arg| unsafe {
                    structural::concatenate_copy_kernel::launch_unchecked::<T, CudaRuntime>(
                        client,
                        count,
                        dim,
                        out.into_tensor_arg(),
                        input_arg.into_tensor_arg(),
                        axis,
                        offset,
                        input.shape.len(),
                    );
                },
            )?;
            offset += input.shape[axis];
        }
        Ok(output)
    }

    fn gather_typed<T, I>(
        &self,
        operand: &TypedTensor<T>,
        start_indices: &TypedTensor<I>,
        config: &GatherConfig,
    ) -> crate::Result<TypedTensor<T>>
    where
        T: CubeElement + CubePrimitive + Clone,
        I: CubeElement + CubeFloat + Clone,
    {
        let meta = gather_launch_meta(&operand.shape, &start_indices.shape, config)?;
        launch_binary_tensor(
            self.runtime(),
            operand,
            start_indices,
            &meta.output_shape,
            "gather",
            |client, count, dim, out, operand_arg, indices_arg| unsafe {
                indexing::gather_kernel::launch_unchecked::<T, I, CudaRuntime>(
                    client,
                    count,
                    dim,
                    out.into_tensor_arg(),
                    operand_arg.into_tensor_arg(),
                    indices_arg.into_tensor_arg(),
                    comptime_sequence(&meta.batch_shape),
                    comptime_sequence(&meta.window_dims),
                    comptime_sequence(&config.offset_dims),
                    comptime_sequence(&config.start_index_map),
                    comptime_sequence(&config.slice_sizes),
                    config.index_vector_dim,
                    operand.shape.len(),
                    meta.output_shape.len(),
                    start_indices.shape.len(),
                );
            },
        )
    }

    fn scatter_float_typed<T, I>(
        &self,
        operand: &TypedTensor<T>,
        scatter_indices: &TypedTensor<I>,
        updates: &TypedTensor<T>,
        config: &ScatterConfig,
    ) -> crate::Result<TypedTensor<T>>
    where
        T: CubeElement + CubeFloat + Clone,
        I: CubeElement + CubeFloat + Clone,
    {
        let meta = scatter_launch_meta(
            &operand.shape,
            &scatter_indices.shape,
            &updates.shape,
            config,
        )?;
        let output = alloc_output::<T>(self.runtime(), &operand.shape);
        if output.n_elements() == 0 {
            return Ok(output);
        }

        launch_unary_tensor_into(
            self.runtime(),
            &output,
            operand,
            "scatter",
            cube_count_for_len(output.n_elements()),
            cube_dim_1d(),
            |client, count, dim, out_arg, operand_arg| unsafe {
                indexing::scatter_copy_kernel::launch_unchecked::<T, CudaRuntime>(
                    client,
                    count,
                    dim,
                    out_arg.into_tensor_arg(),
                    operand_arg.into_tensor_arg(),
                );
            },
        )?;

        let update_len = scatter_update_len(&meta)?;
        if update_len == 0 {
            return Ok(output);
        }
        let client = self.runtime().client();
        ensure_atomic_add_supported::<T>(client, "scatter")?;
        let output_arg = typed_tensor_binding(&output, "scatter")?;
        let operand_arg = typed_tensor_binding(operand, "scatter")?;
        let scatter_arg = typed_tensor_binding(scatter_indices, "scatter")?;
        let updates_arg = typed_tensor_binding(updates, "scatter")?;
        unsafe {
            // SAFETY: `scatter_launch_meta` validates the scatter/update
            // shapes and dimension-number mappings. `typed_tensor_binding`
            // validates every logical tensor buffer length. The launch domain
            // is `scatter_update_len(meta)`, and the kernel maps each launched
            // update through the validated metadata before indexing.
            indexing::scatter_float_kernel::launch_unchecked::<T, I, CudaRuntime>(
                client,
                cube_count_for_len(update_len),
                cube_dim_1d(),
                output_arg.into_tensor_arg(),
                operand_arg.into_tensor_arg(),
                scatter_arg.into_tensor_arg(),
                updates_arg.into_tensor_arg(),
                comptime_sequence(&meta.batch_shape),
                comptime_sequence(&meta.window_dims),
                comptime_sequence(&config.update_window_dims),
                comptime_sequence(&config.scatter_dims_to_operand_dims),
                config.index_vector_dim,
                comptime_sequence(&meta.window_shape_updates),
                operand.shape.len(),
                updates.shape.len(),
                scatter_indices.shape.len(),
            );
        }
        Ok(output)
    }

    fn scatter_complex_typed<T, F, I>(
        &self,
        operand: &TypedTensor<T>,
        scatter_indices: &TypedTensor<I>,
        updates: &TypedTensor<T>,
        config: &ScatterConfig,
    ) -> crate::Result<TypedTensor<T>>
    where
        T: CubeElement + CubeComplex + Clone,
        F: CubeElement + CubeFloat + Clone,
        I: CubeElement + CubeFloat + Clone,
    {
        let meta = scatter_launch_meta(
            &operand.shape,
            &scatter_indices.shape,
            &updates.shape,
            config,
        )?;
        let output = alloc_output::<T>(self.runtime(), &operand.shape);
        if output.n_elements() == 0 {
            return Ok(output);
        }

        launch_unary_tensor_into(
            self.runtime(),
            &output,
            operand,
            "scatter",
            cube_count_for_len(output.n_elements()),
            cube_dim_1d(),
            |client, count, dim, out_arg, operand_arg| unsafe {
                indexing::scatter_copy_kernel::launch_unchecked::<T, CudaRuntime>(
                    client,
                    count,
                    dim,
                    out_arg.into_tensor_arg(),
                    operand_arg.into_tensor_arg(),
                );
            },
        )?;

        let update_len = scatter_update_len(&meta)?;
        if update_len == 0 {
            return Ok(output);
        }
        let client = self.runtime().client();
        ensure_atomic_add_supported::<F>(client, "scatter")?;
        let output_part_len =
            output
                .n_elements()
                .checked_mul(2)
                .ok_or_else(|| crate::Error::BackendFailure {
                    op: "scatter",
                    message: "complex output part length overflow".into(),
                })?;
        let update_part_len =
            updates
                .n_elements()
                .checked_mul(2)
                .ok_or_else(|| crate::Error::BackendFailure {
                    op: "scatter",
                    message: "complex update part length overflow".into(),
                })?;
        // num_complex::Complex<T> is repr(C) as { re: T, im: T }, so the
        // complex buffers can be viewed as real scalar parts for atomic add.
        let output_parts = typed_tensor_array_arg_as::<T, F>(&output, output_part_len, "scatter")?;
        let update_parts = typed_tensor_array_arg_as::<T, F>(updates, update_part_len, "scatter")?;
        let operand_arg = typed_tensor_binding(operand, "scatter")?;
        let scatter_arg = typed_tensor_binding(scatter_indices, "scatter")?;
        let updates_arg = typed_tensor_binding(updates, "scatter")?;
        unsafe {
            // SAFETY: `scatter_launch_meta` validates the scatter/update
            // shapes and dimension-number mappings. `typed_tensor_binding`
            // validates logical tensor buffers, while `typed_tensor_array_arg_as`
            // proves complex real/imaginary part arrays stay within their
            // backing allocations. The launch domain is
            // `scatter_update_len(meta)` and the kernel indexes via the
            // validated metadata.
            indexing::scatter_complex_kernel::launch_unchecked::<T, F, I, CudaRuntime>(
                client,
                cube_count_for_len(update_len),
                cube_dim_1d(),
                output_parts,
                operand_arg.into_tensor_arg(),
                scatter_arg.into_tensor_arg(),
                updates_arg.into_tensor_arg(),
                update_parts,
                comptime_sequence(&meta.batch_shape),
                comptime_sequence(&meta.window_dims),
                comptime_sequence(&config.update_window_dims),
                comptime_sequence(&config.scatter_dims_to_operand_dims),
                config.index_vector_dim,
                comptime_sequence(&meta.window_shape_updates),
                operand.shape.len(),
                updates.shape.len(),
                scatter_indices.shape.len(),
            );
        }
        Ok(output)
    }
}

impl TensorBackend for CubeclBackend {
    fn add(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
        match (lhs, rhs) {
            (Tensor::F32(lhs), Tensor::F32(rhs)) => launch_binary(
                self.runtime(),
                lhs,
                rhs,
                &lhs.shape,
                "add",
                |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                    elementwise::add_float::launch_unchecked::<f32, CudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg,
                    );
                },
            )
            .map(Tensor::F32),
            (Tensor::F64(lhs), Tensor::F64(rhs)) => launch_binary(
                self.runtime(),
                lhs,
                rhs,
                &lhs.shape,
                "add",
                |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                    elementwise::add_float::launch_unchecked::<f64, CudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg,
                    );
                },
            )
            .map(Tensor::F64),
            (Tensor::C32(lhs), Tensor::C32(rhs)) => launch_binary(
                self.runtime(),
                lhs,
                rhs,
                &lhs.shape,
                "add",
                |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                    elementwise::add_complex::launch_unchecked::<Complex32, CudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg,
                    );
                },
            )
            .map(Tensor::C32),
            (Tensor::C64(lhs), Tensor::C64(rhs)) => launch_binary(
                self.runtime(),
                lhs,
                rhs,
                &lhs.shape,
                "add",
                |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                    elementwise::add_complex::launch_unchecked::<Complex64, CudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg,
                    );
                },
            )
            .map(Tensor::C64),
            _ => Err(dtype_mismatch("add", lhs, rhs)),
        }
    }

    fn mul(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
        match (lhs, rhs) {
            (Tensor::F32(lhs), Tensor::F32(rhs)) => launch_binary(
                self.runtime(),
                lhs,
                rhs,
                &lhs.shape,
                "mul",
                |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                    elementwise::mul_float::launch_unchecked::<f32, CudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg,
                    );
                },
            )
            .map(Tensor::F32),
            (Tensor::F64(lhs), Tensor::F64(rhs)) => launch_binary(
                self.runtime(),
                lhs,
                rhs,
                &lhs.shape,
                "mul",
                |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                    elementwise::mul_float::launch_unchecked::<f64, CudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg,
                    );
                },
            )
            .map(Tensor::F64),
            (Tensor::C32(lhs), Tensor::C32(rhs)) => launch_binary(
                self.runtime(),
                lhs,
                rhs,
                &lhs.shape,
                "mul",
                |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                    elementwise::mul_complex::launch_unchecked::<Complex32, CudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg,
                    );
                },
            )
            .map(Tensor::C32),
            (Tensor::C64(lhs), Tensor::C64(rhs)) => launch_binary(
                self.runtime(),
                lhs,
                rhs,
                &lhs.shape,
                "mul",
                |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                    elementwise::mul_complex::launch_unchecked::<Complex64, CudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg,
                    );
                },
            )
            .map(Tensor::C64),
            _ => Err(dtype_mismatch("mul", lhs, rhs)),
        }
    }

    fn neg(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        match input {
            Tensor::F32(tensor) => launch_unary(
                self.runtime(),
                tensor,
                &tensor.shape,
                "neg",
                |client, count, dim, out, input_arg| unsafe {
                    elementwise::neg_float::launch_unchecked::<f32, CudaRuntime>(
                        client, count, dim, out, input_arg,
                    );
                },
            )
            .map(Tensor::F32),
            Tensor::F64(tensor) => launch_unary(
                self.runtime(),
                tensor,
                &tensor.shape,
                "neg",
                |client, count, dim, out, input_arg| unsafe {
                    elementwise::neg_float::launch_unchecked::<f64, CudaRuntime>(
                        client, count, dim, out, input_arg,
                    );
                },
            )
            .map(Tensor::F64),
            Tensor::C32(tensor) => launch_unary(
                self.runtime(),
                tensor,
                &tensor.shape,
                "neg",
                |client, count, dim, out, input_arg| unsafe {
                    elementwise::neg_complex::launch_unchecked::<Complex32, CudaRuntime>(
                        client, count, dim, out, input_arg,
                    );
                },
            )
            .map(Tensor::C32),
            Tensor::C64(tensor) => launch_unary(
                self.runtime(),
                tensor,
                &tensor.shape,
                "neg",
                |client, count, dim, out, input_arg| unsafe {
                    elementwise::neg_complex::launch_unchecked::<Complex64, CudaRuntime>(
                        client, count, dim, out, input_arg,
                    );
                },
            )
            .map(Tensor::C64),
            Tensor::I64(_) => Err(unsupported_dtype("neg", input.dtype())),
        }
    }

    fn conj(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        match input {
            Tensor::F32(tensor) => {
                ensure_resident_on_runtime(self.runtime(), tensor, "conj")?;
                Ok(Tensor::F32(tensor.clone()))
            }
            Tensor::F64(tensor) => {
                ensure_resident_on_runtime(self.runtime(), tensor, "conj")?;
                Ok(Tensor::F64(tensor.clone()))
            }
            Tensor::I64(_) => Err(unsupported_dtype("conj", input.dtype())),
            Tensor::C32(tensor) => launch_unary(
                self.runtime(),
                tensor,
                &tensor.shape,
                "conj",
                |client, count, dim, out, input_arg| unsafe {
                    elementwise::conj_complex::launch_unchecked::<Complex32, CudaRuntime>(
                        client, count, dim, out, input_arg,
                    );
                },
            )
            .map(Tensor::C32),
            Tensor::C64(tensor) => launch_unary(
                self.runtime(),
                tensor,
                &tensor.shape,
                "conj",
                |client, count, dim, out, input_arg| unsafe {
                    elementwise::conj_complex::launch_unchecked::<Complex64, CudaRuntime>(
                        client, count, dim, out, input_arg,
                    );
                },
            )
            .map(Tensor::C64),
        }
    }

    fn div(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
        match (lhs, rhs) {
            (Tensor::F32(lhs), Tensor::F32(rhs)) => launch_binary(
                self.runtime(),
                lhs,
                rhs,
                &lhs.shape,
                "div",
                |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                    elementwise::div_float::launch_unchecked::<f32, CudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg,
                    );
                },
            )
            .map(Tensor::F32),
            (Tensor::F64(lhs), Tensor::F64(rhs)) => launch_binary(
                self.runtime(),
                lhs,
                rhs,
                &lhs.shape,
                "div",
                |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                    elementwise::div_float::launch_unchecked::<f64, CudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg,
                    );
                },
            )
            .map(Tensor::F64),
            (Tensor::C32(lhs), Tensor::C32(rhs)) => launch_binary(
                self.runtime(),
                lhs,
                rhs,
                &lhs.shape,
                "div",
                |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                    elementwise::div_complex::launch_unchecked::<Complex32, CudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg,
                    );
                },
            )
            .map(Tensor::C32),
            (Tensor::C64(lhs), Tensor::C64(rhs)) => launch_binary(
                self.runtime(),
                lhs,
                rhs,
                &lhs.shape,
                "div",
                |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                    elementwise::div_complex::launch_unchecked::<Complex64, CudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg,
                    );
                },
            )
            .map(Tensor::C64),
            _ => Err(dtype_mismatch("div", lhs, rhs)),
        }
    }

    fn abs(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        match input {
            Tensor::F32(tensor) => launch_unary(
                self.runtime(),
                tensor,
                &tensor.shape,
                "abs",
                |client, count, dim, out, input_arg| unsafe {
                    elementwise::abs_float::launch_unchecked::<f32, CudaRuntime>(
                        client, count, dim, out, input_arg,
                    );
                },
            )
            .map(Tensor::F32),
            Tensor::F64(tensor) => launch_unary(
                self.runtime(),
                tensor,
                &tensor.shape,
                "abs",
                |client, count, dim, out, input_arg| unsafe {
                    elementwise::abs_float::launch_unchecked::<f64, CudaRuntime>(
                        client, count, dim, out, input_arg,
                    );
                },
            )
            .map(Tensor::F64),
            Tensor::I64(_) | Tensor::C32(_) | Tensor::C64(_) => Err(crate::Error::BackendFailure {
                op: "abs",
                message: format!("unsupported dtype {:?}", input.dtype()),
            }),
        }
    }

    fn sign(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        match input {
            Tensor::F32(tensor) => launch_unary(
                self.runtime(),
                tensor,
                &tensor.shape,
                "sign",
                |client, count, dim, out, input_arg| unsafe {
                    elementwise::sign_float::launch_unchecked::<f32, CudaRuntime>(
                        client, count, dim, out, input_arg,
                    );
                },
            )
            .map(Tensor::F32),
            Tensor::F64(tensor) => launch_unary(
                self.runtime(),
                tensor,
                &tensor.shape,
                "sign",
                |client, count, dim, out, input_arg| unsafe {
                    elementwise::sign_float::launch_unchecked::<f64, CudaRuntime>(
                        client, count, dim, out, input_arg,
                    );
                },
            )
            .map(Tensor::F64),
            Tensor::I64(_) | Tensor::C32(_) | Tensor::C64(_) => Err(crate::Error::BackendFailure {
                op: "sign",
                message: format!("unsupported dtype {:?}", input.dtype()),
            }),
        }
    }

    fn maximum(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
        match (lhs, rhs) {
            (Tensor::F32(lhs), Tensor::F32(rhs)) => launch_binary(
                self.runtime(),
                lhs,
                rhs,
                &lhs.shape,
                "maximum",
                |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                    elementwise::maximum_float::launch_unchecked::<f32, CudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg,
                    );
                },
            )
            .map(Tensor::F32),
            (Tensor::F64(lhs), Tensor::F64(rhs)) => launch_binary(
                self.runtime(),
                lhs,
                rhs,
                &lhs.shape,
                "maximum",
                |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                    elementwise::maximum_float::launch_unchecked::<f64, CudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg,
                    );
                },
            )
            .map(Tensor::F64),
            (Tensor::C32(_), Tensor::C32(_)) | (Tensor::C64(_), Tensor::C64(_)) => {
                Err(crate::Error::BackendFailure {
                    op: "maximum",
                    message: format!("unsupported dtype {:?}", lhs.dtype()),
                })
            }
            _ => Err(dtype_mismatch("maximum", lhs, rhs)),
        }
    }

    fn minimum(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
        match (lhs, rhs) {
            (Tensor::F32(lhs), Tensor::F32(rhs)) => launch_binary(
                self.runtime(),
                lhs,
                rhs,
                &lhs.shape,
                "minimum",
                |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                    elementwise::minimum_float::launch_unchecked::<f32, CudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg,
                    );
                },
            )
            .map(Tensor::F32),
            (Tensor::F64(lhs), Tensor::F64(rhs)) => launch_binary(
                self.runtime(),
                lhs,
                rhs,
                &lhs.shape,
                "minimum",
                |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                    elementwise::minimum_float::launch_unchecked::<f64, CudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg,
                    );
                },
            )
            .map(Tensor::F64),
            (Tensor::C32(_), Tensor::C32(_)) | (Tensor::C64(_), Tensor::C64(_)) => {
                Err(crate::Error::BackendFailure {
                    op: "minimum",
                    message: format!("unsupported dtype {:?}", lhs.dtype()),
                })
            }
            _ => Err(dtype_mismatch("minimum", lhs, rhs)),
        }
    }

    fn compare(&mut self, lhs: &Tensor, rhs: &Tensor, dir: &CompareDir) -> crate::Result<Tensor> {
        match (lhs, rhs) {
            (Tensor::F32(lhs), Tensor::F32(rhs)) => launch_binary(
                self.runtime(),
                lhs,
                rhs,
                &lhs.shape,
                "compare",
                |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                    elementwise::compare_float::launch_unchecked::<f32, CudaRuntime>(
                        client,
                        count,
                        dim,
                        out,
                        lhs_arg,
                        rhs_arg,
                        dispatch::compare_mode(dir),
                    );
                },
            )
            .map(Tensor::F32),
            (Tensor::F64(lhs), Tensor::F64(rhs)) => launch_binary(
                self.runtime(),
                lhs,
                rhs,
                &lhs.shape,
                "compare",
                |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                    elementwise::compare_float::launch_unchecked::<f64, CudaRuntime>(
                        client,
                        count,
                        dim,
                        out,
                        lhs_arg,
                        rhs_arg,
                        dispatch::compare_mode(dir),
                    );
                },
            )
            .map(Tensor::F64),
            (Tensor::C32(_), Tensor::C32(_)) | (Tensor::C64(_), Tensor::C64(_)) => {
                Err(crate::Error::BackendFailure {
                    op: "compare",
                    message: format!("unsupported dtype {:?}", lhs.dtype()),
                })
            }
            _ => Err(dtype_mismatch("compare", lhs, rhs)),
        }
    }

    fn select(
        &mut self,
        pred: &Tensor,
        on_true: &Tensor,
        on_false: &Tensor,
    ) -> crate::Result<Tensor> {
        match (pred, on_true, on_false) {
            (Tensor::F32(pred), Tensor::F32(on_true), Tensor::F32(on_false)) => launch_ternary(
                self.runtime(),
                pred,
                on_true,
                on_false,
                &pred.shape,
                "select",
                |client, count, dim, out, pred_arg, true_arg, false_arg| unsafe {
                    elementwise::select_float::launch_unchecked::<f32, CudaRuntime>(
                        client, count, dim, out, pred_arg, true_arg, false_arg,
                    );
                },
            )
            .map(Tensor::F32),
            (Tensor::F64(pred), Tensor::F64(on_true), Tensor::F64(on_false)) => launch_ternary(
                self.runtime(),
                pred,
                on_true,
                on_false,
                &pred.shape,
                "select",
                |client, count, dim, out, pred_arg, true_arg, false_arg| unsafe {
                    elementwise::select_float::launch_unchecked::<f64, CudaRuntime>(
                        client, count, dim, out, pred_arg, true_arg, false_arg,
                    );
                },
            )
            .map(Tensor::F64),
            (Tensor::C32(_), Tensor::C32(_), Tensor::C32(_))
            | (Tensor::C64(_), Tensor::C64(_), Tensor::C64(_)) => {
                Err(crate::Error::BackendFailure {
                    op: "select",
                    message: format!("unsupported dtype {:?}", pred.dtype()),
                })
            }
            _ => Err(ternary_dtype_mismatch("select", pred, on_true, on_false)),
        }
    }

    fn clamp(&mut self, input: &Tensor, lower: &Tensor, upper: &Tensor) -> crate::Result<Tensor> {
        match (input, lower, upper) {
            (Tensor::F32(input), Tensor::F32(lower), Tensor::F32(upper)) => launch_ternary(
                self.runtime(),
                input,
                lower,
                upper,
                &input.shape,
                "clamp",
                |client, count, dim, out, input_arg, lower_arg, upper_arg| unsafe {
                    elementwise::clamp_float::launch_unchecked::<f32, CudaRuntime>(
                        client, count, dim, out, input_arg, lower_arg, upper_arg,
                    );
                },
            )
            .map(Tensor::F32),
            (Tensor::F64(input), Tensor::F64(lower), Tensor::F64(upper)) => launch_ternary(
                self.runtime(),
                input,
                lower,
                upper,
                &input.shape,
                "clamp",
                |client, count, dim, out, input_arg, lower_arg, upper_arg| unsafe {
                    elementwise::clamp_float::launch_unchecked::<f64, CudaRuntime>(
                        client, count, dim, out, input_arg, lower_arg, upper_arg,
                    );
                },
            )
            .map(Tensor::F64),
            (Tensor::C32(_), Tensor::C32(_), Tensor::C32(_))
            | (Tensor::C64(_), Tensor::C64(_), Tensor::C64(_)) => {
                Err(crate::Error::BackendFailure {
                    op: "clamp",
                    message: format!("unsupported dtype {:?}", input.dtype()),
                })
            }
            _ => Err(ternary_dtype_mismatch("clamp", input, lower, upper)),
        }
    }

    fn exp(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        match input {
            Tensor::F32(tensor) => launch_unary(
                self.runtime(),
                tensor,
                &tensor.shape,
                "exp",
                |client, count, dim, out, input_arg| unsafe {
                    elementwise::exp_float::launch_unchecked::<f32, CudaRuntime>(
                        client, count, dim, out, input_arg,
                    );
                },
            )
            .map(Tensor::F32),
            Tensor::F64(tensor) => launch_unary(
                self.runtime(),
                tensor,
                &tensor.shape,
                "exp",
                |client, count, dim, out, input_arg| unsafe {
                    elementwise::exp_float::launch_unchecked::<f64, CudaRuntime>(
                        client, count, dim, out, input_arg,
                    );
                },
            )
            .map(Tensor::F64),
            Tensor::I64(_) | Tensor::C32(_) | Tensor::C64(_) => Err(crate::Error::BackendFailure {
                op: "exp",
                message: format!("unsupported dtype {:?}", input.dtype()),
            }),
        }
    }

    fn log(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        match input {
            Tensor::F32(tensor) => launch_unary(
                self.runtime(),
                tensor,
                &tensor.shape,
                "log",
                |client, count, dim, out, input_arg| unsafe {
                    elementwise::log_float::launch_unchecked::<f32, CudaRuntime>(
                        client, count, dim, out, input_arg,
                    );
                },
            )
            .map(Tensor::F32),
            Tensor::F64(tensor) => launch_unary(
                self.runtime(),
                tensor,
                &tensor.shape,
                "log",
                |client, count, dim, out, input_arg| unsafe {
                    elementwise::log_float::launch_unchecked::<f64, CudaRuntime>(
                        client, count, dim, out, input_arg,
                    );
                },
            )
            .map(Tensor::F64),
            Tensor::I64(_) | Tensor::C32(_) | Tensor::C64(_) => Err(crate::Error::BackendFailure {
                op: "log",
                message: format!("unsupported dtype {:?}", input.dtype()),
            }),
        }
    }

    fn sin(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        match input {
            Tensor::F32(tensor) => launch_unary(
                self.runtime(),
                tensor,
                &tensor.shape,
                "sin",
                |client, count, dim, out, input_arg| unsafe {
                    elementwise::sin_float::launch_unchecked::<f32, CudaRuntime>(
                        client, count, dim, out, input_arg,
                    );
                },
            )
            .map(Tensor::F32),
            Tensor::F64(tensor) => launch_unary(
                self.runtime(),
                tensor,
                &tensor.shape,
                "sin",
                |client, count, dim, out, input_arg| unsafe {
                    elementwise::sin_float::launch_unchecked::<f64, CudaRuntime>(
                        client, count, dim, out, input_arg,
                    );
                },
            )
            .map(Tensor::F64),
            Tensor::I64(_) | Tensor::C32(_) | Tensor::C64(_) => Err(crate::Error::BackendFailure {
                op: "sin",
                message: format!("unsupported dtype {:?}", input.dtype()),
            }),
        }
    }

    fn cos(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        match input {
            Tensor::F32(tensor) => launch_unary(
                self.runtime(),
                tensor,
                &tensor.shape,
                "cos",
                |client, count, dim, out, input_arg| unsafe {
                    elementwise::cos_float::launch_unchecked::<f32, CudaRuntime>(
                        client, count, dim, out, input_arg,
                    );
                },
            )
            .map(Tensor::F32),
            Tensor::F64(tensor) => launch_unary(
                self.runtime(),
                tensor,
                &tensor.shape,
                "cos",
                |client, count, dim, out, input_arg| unsafe {
                    elementwise::cos_float::launch_unchecked::<f64, CudaRuntime>(
                        client, count, dim, out, input_arg,
                    );
                },
            )
            .map(Tensor::F64),
            Tensor::I64(_) | Tensor::C32(_) | Tensor::C64(_) => Err(crate::Error::BackendFailure {
                op: "cos",
                message: format!("unsupported dtype {:?}", input.dtype()),
            }),
        }
    }

    fn tanh(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        match input {
            Tensor::F32(tensor) => launch_unary(
                self.runtime(),
                tensor,
                &tensor.shape,
                "tanh",
                |client, count, dim, out, input_arg| unsafe {
                    elementwise::tanh_float::launch_unchecked::<f32, CudaRuntime>(
                        client, count, dim, out, input_arg,
                    );
                },
            )
            .map(Tensor::F32),
            Tensor::F64(tensor) => launch_unary(
                self.runtime(),
                tensor,
                &tensor.shape,
                "tanh",
                |client, count, dim, out, input_arg| unsafe {
                    elementwise::tanh_float::launch_unchecked::<f64, CudaRuntime>(
                        client, count, dim, out, input_arg,
                    );
                },
            )
            .map(Tensor::F64),
            Tensor::I64(_) | Tensor::C32(_) | Tensor::C64(_) => Err(crate::Error::BackendFailure {
                op: "tanh",
                message: format!("unsupported dtype {:?}", input.dtype()),
            }),
        }
    }

    fn sqrt(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        match input {
            Tensor::F32(tensor) => launch_unary(
                self.runtime(),
                tensor,
                &tensor.shape,
                "sqrt",
                |client, count, dim, out, input_arg| unsafe {
                    elementwise::sqrt_float::launch_unchecked::<f32, CudaRuntime>(
                        client, count, dim, out, input_arg,
                    );
                },
            )
            .map(Tensor::F32),
            Tensor::F64(tensor) => launch_unary(
                self.runtime(),
                tensor,
                &tensor.shape,
                "sqrt",
                |client, count, dim, out, input_arg| unsafe {
                    elementwise::sqrt_float::launch_unchecked::<f64, CudaRuntime>(
                        client, count, dim, out, input_arg,
                    );
                },
            )
            .map(Tensor::F64),
            Tensor::I64(_) | Tensor::C32(_) | Tensor::C64(_) => Err(crate::Error::BackendFailure {
                op: "sqrt",
                message: format!("unsupported dtype {:?}", input.dtype()),
            }),
        }
    }

    fn rsqrt(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        match input {
            Tensor::F32(tensor) => launch_unary(
                self.runtime(),
                tensor,
                &tensor.shape,
                "rsqrt",
                |client, count, dim, out, input_arg| unsafe {
                    elementwise::rsqrt_float::launch_unchecked::<f32, CudaRuntime>(
                        client, count, dim, out, input_arg,
                    );
                },
            )
            .map(Tensor::F32),
            Tensor::F64(tensor) => launch_unary(
                self.runtime(),
                tensor,
                &tensor.shape,
                "rsqrt",
                |client, count, dim, out, input_arg| unsafe {
                    elementwise::rsqrt_float::launch_unchecked::<f64, CudaRuntime>(
                        client, count, dim, out, input_arg,
                    );
                },
            )
            .map(Tensor::F64),
            Tensor::I64(_) | Tensor::C32(_) | Tensor::C64(_) => Err(crate::Error::BackendFailure {
                op: "rsqrt",
                message: format!("unsupported dtype {:?}", input.dtype()),
            }),
        }
    }

    fn pow(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
        match (lhs, rhs) {
            (Tensor::F32(lhs), Tensor::F32(rhs)) => launch_binary(
                self.runtime(),
                lhs,
                rhs,
                &lhs.shape,
                "pow",
                |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                    elementwise::pow_float::launch_unchecked::<f32, CudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg,
                    );
                },
            )
            .map(Tensor::F32),
            (Tensor::F64(lhs), Tensor::F64(rhs)) => launch_binary(
                self.runtime(),
                lhs,
                rhs,
                &lhs.shape,
                "pow",
                |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                    elementwise::pow_float::launch_unchecked::<f64, CudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg,
                    );
                },
            )
            .map(Tensor::F64),
            (Tensor::C32(_), Tensor::C32(_)) | (Tensor::C64(_), Tensor::C64(_)) => {
                Err(crate::Error::BackendFailure {
                    op: "pow",
                    message: format!("unsupported dtype {:?}", lhs.dtype()),
                })
            }
            _ => Err(dtype_mismatch("pow", lhs, rhs)),
        }
    }

    fn expm1(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        match input {
            Tensor::F32(tensor) => launch_unary(
                self.runtime(),
                tensor,
                &tensor.shape,
                "expm1",
                |client, count, dim, out, input_arg| unsafe {
                    elementwise::expm1_float::launch_unchecked::<f32, CudaRuntime>(
                        client, count, dim, out, input_arg,
                    );
                },
            )
            .map(Tensor::F32),
            Tensor::F64(tensor) => launch_unary(
                self.runtime(),
                tensor,
                &tensor.shape,
                "expm1",
                |client, count, dim, out, input_arg| unsafe {
                    elementwise::expm1_float::launch_unchecked::<f64, CudaRuntime>(
                        client, count, dim, out, input_arg,
                    );
                },
            )
            .map(Tensor::F64),
            Tensor::I64(_) | Tensor::C32(_) | Tensor::C64(_) => Err(crate::Error::BackendFailure {
                op: "expm1",
                message: format!("unsupported dtype {:?}", input.dtype()),
            }),
        }
    }

    fn log1p(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        match input {
            Tensor::F32(tensor) => launch_unary(
                self.runtime(),
                tensor,
                &tensor.shape,
                "log1p",
                |client, count, dim, out, input_arg| unsafe {
                    elementwise::log1p_float::launch_unchecked::<f32, CudaRuntime>(
                        client, count, dim, out, input_arg,
                    );
                },
            )
            .map(Tensor::F32),
            Tensor::F64(tensor) => launch_unary(
                self.runtime(),
                tensor,
                &tensor.shape,
                "log1p",
                |client, count, dim, out, input_arg| unsafe {
                    elementwise::log1p_float::launch_unchecked::<f64, CudaRuntime>(
                        client, count, dim, out, input_arg,
                    );
                },
            )
            .map(Tensor::F64),
            Tensor::I64(_) | Tensor::C32(_) | Tensor::C64(_) => Err(crate::Error::BackendFailure {
                op: "log1p",
                message: format!("unsupported dtype {:?}", input.dtype()),
            }),
        }
    }

    fn transpose(&mut self, input: &Tensor, perm: &[usize]) -> crate::Result<Tensor> {
        match input {
            Tensor::F32(t) => self.transpose_typed(t, perm).map(Tensor::F32),
            Tensor::F64(t) => self.transpose_typed(t, perm).map(Tensor::F64),
            Tensor::I64(t) => self.transpose_typed(t, perm).map(Tensor::I64),
            Tensor::C32(t) => self.transpose_typed(t, perm).map(Tensor::C32),
            Tensor::C64(t) => self.transpose_typed(t, perm).map(Tensor::C64),
        }
    }

    fn reshape(&mut self, input: &Tensor, shape: &[usize]) -> crate::Result<Tensor> {
        let old_n: usize = input.shape().iter().product();
        let new_n: usize = shape.iter().product();
        if old_n != new_n {
            return Err(crate::Error::ShapeMismatch {
                op: "reshape",
                lhs: input.shape().to_vec(),
                rhs: shape.to_vec(),
            });
        }
        match input {
            Tensor::F32(t) => Ok(Tensor::F32(TypedTensor {
                buffer: t.buffer.clone(),
                shape: shape.to_vec(),
                placement: t.placement.clone(),
            })),
            Tensor::F64(t) => Ok(Tensor::F64(TypedTensor {
                buffer: t.buffer.clone(),
                shape: shape.to_vec(),
                placement: t.placement.clone(),
            })),
            Tensor::I64(t) => Ok(Tensor::I64(TypedTensor {
                buffer: t.buffer.clone(),
                shape: shape.to_vec(),
                placement: t.placement.clone(),
            })),
            Tensor::C32(t) => Ok(Tensor::C32(TypedTensor {
                buffer: t.buffer.clone(),
                shape: shape.to_vec(),
                placement: t.placement.clone(),
            })),
            Tensor::C64(t) => Ok(Tensor::C64(TypedTensor {
                buffer: t.buffer.clone(),
                shape: shape.to_vec(),
                placement: t.placement.clone(),
            })),
        }
    }

    fn broadcast_in_dim(
        &mut self,
        input: &Tensor,
        shape: &[usize],
        dims: &[usize],
    ) -> crate::Result<Tensor> {
        match input {
            Tensor::F32(t) => self.broadcast_typed(t, shape, dims).map(Tensor::F32),
            Tensor::F64(t) => self.broadcast_typed(t, shape, dims).map(Tensor::F64),
            Tensor::I64(t) => self.broadcast_typed(t, shape, dims).map(Tensor::I64),
            Tensor::C32(t) => self.broadcast_typed(t, shape, dims).map(Tensor::C32),
            Tensor::C64(t) => self.broadcast_typed(t, shape, dims).map(Tensor::C64),
        }
    }

    fn convert(&mut self, input: &Tensor, to: crate::DType) -> crate::Result<Tensor> {
        match (input, to) {
            (Tensor::F32(t), crate::DType::F32) => Ok(Tensor::F32(t.clone())),
            (Tensor::F32(t), crate::DType::F64) => {
                self.convert_float_to_float::<f32, f64>(t).map(Tensor::F64)
            }
            (Tensor::F32(_), crate::DType::I64) => Err(unsupported_dtype("convert", to)),
            (Tensor::F32(t), crate::DType::C32) => self.convert_f32_to_c32(t).map(Tensor::C32),
            (Tensor::F32(t), crate::DType::C64) => self.convert_f32_to_c64(t).map(Tensor::C64),
            (Tensor::F64(t), crate::DType::F32) => {
                self.convert_float_to_float::<f64, f32>(t).map(Tensor::F32)
            }
            (Tensor::F64(t), crate::DType::F64) => Ok(Tensor::F64(t.clone())),
            (Tensor::F64(_), crate::DType::I64) => Err(unsupported_dtype("convert", to)),
            (Tensor::F64(t), crate::DType::C32) => self.convert_f64_to_c32(t).map(Tensor::C32),
            (Tensor::F64(t), crate::DType::C64) => self.convert_f64_to_c64(t).map(Tensor::C64),
            (Tensor::I64(_), crate::DType::I64) => Ok(input.clone()),
            (Tensor::I64(_), _) => Err(unsupported_dtype("convert", input.dtype())),
            (Tensor::C32(t), crate::DType::F32) => self.convert_c32_to_f32(t).map(Tensor::F32),
            (Tensor::C32(t), crate::DType::F64) => self.convert_c32_to_f64(t).map(Tensor::F64),
            (Tensor::C32(_), crate::DType::I64) => Err(unsupported_dtype("convert", to)),
            (Tensor::C32(t), crate::DType::C32) => Ok(Tensor::C32(t.clone())),
            (Tensor::C32(t), crate::DType::C64) => self
                .convert_complex_to_complex::<Complex32, Complex64>(t)
                .map(Tensor::C64),
            (Tensor::C64(t), crate::DType::F32) => self.convert_c64_to_f32(t).map(Tensor::F32),
            (Tensor::C64(t), crate::DType::F64) => self.convert_c64_to_f64(t).map(Tensor::F64),
            (Tensor::C64(_), crate::DType::I64) => Err(unsupported_dtype("convert", to)),
            (Tensor::C64(t), crate::DType::C32) => self
                .convert_complex_to_complex::<Complex64, Complex32>(t)
                .map(Tensor::C32),
            (Tensor::C64(t), crate::DType::C64) => Ok(Tensor::C64(t.clone())),
        }
    }

    fn extract_diagonal(
        &mut self,
        input: &Tensor,
        axis_a: usize,
        axis_b: usize,
    ) -> crate::Result<Tensor> {
        match input {
            Tensor::F32(t) => self
                .extract_diagonal_typed(t, axis_a, axis_b)
                .map(Tensor::F32),
            Tensor::F64(t) => self
                .extract_diagonal_typed(t, axis_a, axis_b)
                .map(Tensor::F64),
            Tensor::I64(t) => self
                .extract_diagonal_typed(t, axis_a, axis_b)
                .map(Tensor::I64),
            Tensor::C32(t) => self
                .extract_diagonal_typed(t, axis_a, axis_b)
                .map(Tensor::C32),
            Tensor::C64(t) => self
                .extract_diagonal_typed(t, axis_a, axis_b)
                .map(Tensor::C64),
        }
    }

    fn embed_diagonal(
        &mut self,
        input: &Tensor,
        axis_a: usize,
        axis_b: usize,
    ) -> crate::Result<Tensor> {
        match input {
            Tensor::F32(t) => self
                .embed_diagonal_typed(t, axis_a, axis_b)
                .map(Tensor::F32),
            Tensor::F64(t) => self
                .embed_diagonal_typed(t, axis_a, axis_b)
                .map(Tensor::F64),
            Tensor::I64(t) => self
                .embed_diagonal_typed(t, axis_a, axis_b)
                .map(Tensor::I64),
            Tensor::C32(t) => self
                .embed_diagonal_typed(t, axis_a, axis_b)
                .map(Tensor::C32),
            Tensor::C64(t) => self
                .embed_diagonal_typed(t, axis_a, axis_b)
                .map(Tensor::C64),
        }
    }

    fn tril(&mut self, input: &Tensor, k: i64) -> crate::Result<Tensor> {
        match input {
            Tensor::F32(t) => self.tril_typed(t, k).map(Tensor::F32),
            Tensor::F64(t) => self.tril_typed(t, k).map(Tensor::F64),
            Tensor::I64(t) => self.tril_typed(t, k).map(Tensor::I64),
            Tensor::C32(t) => self.tril_typed(t, k).map(Tensor::C32),
            Tensor::C64(t) => self.tril_typed(t, k).map(Tensor::C64),
        }
    }

    fn triu(&mut self, input: &Tensor, k: i64) -> crate::Result<Tensor> {
        match input {
            Tensor::F32(t) => self.triu_typed(t, k).map(Tensor::F32),
            Tensor::F64(t) => self.triu_typed(t, k).map(Tensor::F64),
            Tensor::I64(t) => self.triu_typed(t, k).map(Tensor::I64),
            Tensor::C32(t) => self.triu_typed(t, k).map(Tensor::C32),
            Tensor::C64(t) => self.triu_typed(t, k).map(Tensor::C64),
        }
    }

    fn reduce_sum(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
        match input {
            Tensor::F32(t) => self.reduce_sum_float_typed(t, axes).map(Tensor::F32),
            Tensor::F64(t) => self.reduce_sum_float_typed(t, axes).map(Tensor::F64),
            Tensor::I64(t) => self.reduce_sum_int_typed(t, axes).map(Tensor::I64),
            Tensor::C32(t) => self.reduce_sum_complex_typed(t, axes).map(Tensor::C32),
            Tensor::C64(t) => self.reduce_sum_complex_typed(t, axes).map(Tensor::C64),
        }
    }

    fn reduce_prod(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
        match input {
            Tensor::F32(t) => self.reduce_prod_float_typed(t, axes).map(Tensor::F32),
            Tensor::F64(t) => self.reduce_prod_float_typed(t, axes).map(Tensor::F64),
            Tensor::I64(t) => self.reduce_prod_int_typed(t, axes).map(Tensor::I64),
            Tensor::C32(t) => self.reduce_prod_complex_typed(t, axes).map(Tensor::C32),
            Tensor::C64(t) => self.reduce_prod_complex_typed(t, axes).map(Tensor::C64),
        }
    }

    fn reduce_max(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
        match input {
            Tensor::F32(t) => self.reduce_max_typed(t, axes).map(Tensor::F32),
            Tensor::F64(t) => self.reduce_max_typed(t, axes).map(Tensor::F64),
            Tensor::I64(_) | Tensor::C32(_) | Tensor::C64(_) => Err(crate::Error::BackendFailure {
                op: "reduce_max",
                message: format!("unsupported dtype {:?}", input.dtype()),
            }),
        }
    }

    fn reduce_min(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
        match input {
            Tensor::F32(t) => self.reduce_min_typed(t, axes).map(Tensor::F32),
            Tensor::F64(t) => self.reduce_min_typed(t, axes).map(Tensor::F64),
            Tensor::I64(_) | Tensor::C32(_) | Tensor::C64(_) => Err(crate::Error::BackendFailure {
                op: "reduce_min",
                message: format!("unsupported dtype {:?}", input.dtype()),
            }),
        }
    }

    fn dot_general(
        &mut self,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
    ) -> crate::Result<Tensor> {
        gemm::dot_general(self, lhs, rhs, config)
    }

    fn gather(
        &mut self,
        operand: &Tensor,
        start_indices: &Tensor,
        config: &GatherConfig,
    ) -> crate::Result<Tensor> {
        match (operand, start_indices) {
            (Tensor::F32(operand), Tensor::F32(indices)) => {
                self.gather_typed(operand, indices, config).map(Tensor::F32)
            }
            (Tensor::F64(operand), Tensor::F32(indices)) => {
                self.gather_typed(operand, indices, config).map(Tensor::F64)
            }
            (Tensor::C32(operand), Tensor::F32(indices)) => {
                self.gather_typed(operand, indices, config).map(Tensor::C32)
            }
            (Tensor::C64(operand), Tensor::F32(indices)) => {
                self.gather_typed(operand, indices, config).map(Tensor::C64)
            }
            (Tensor::F32(operand), Tensor::F64(indices)) => {
                self.gather_typed(operand, indices, config).map(Tensor::F32)
            }
            (Tensor::F64(operand), Tensor::F64(indices)) => {
                self.gather_typed(operand, indices, config).map(Tensor::F64)
            }
            (Tensor::C32(operand), Tensor::F64(indices)) => {
                self.gather_typed(operand, indices, config).map(Tensor::C32)
            }
            (Tensor::C64(operand), Tensor::F64(indices)) => {
                self.gather_typed(operand, indices, config).map(Tensor::C64)
            }
            (Tensor::F32(operand), Tensor::I64(indices)) => {
                let indices = self.i64_indices_as_f64(indices)?;
                self.gather_typed(operand, &indices, config)
                    .map(Tensor::F32)
            }
            (Tensor::F64(operand), Tensor::I64(indices)) => {
                let indices = self.i64_indices_as_f64(indices)?;
                self.gather_typed(operand, &indices, config)
                    .map(Tensor::F64)
            }
            (Tensor::C32(operand), Tensor::I64(indices)) => {
                let indices = self.i64_indices_as_f64(indices)?;
                self.gather_typed(operand, &indices, config)
                    .map(Tensor::C32)
            }
            (Tensor::C64(operand), Tensor::I64(indices)) => {
                let indices = self.i64_indices_as_f64(indices)?;
                self.gather_typed(operand, &indices, config)
                    .map(Tensor::C64)
            }
            (_, Tensor::C32(_) | Tensor::C64(_)) => Err(crate::Error::BackendFailure {
                op: "gather",
                message: "complex index tensors are not supported".into(),
            }),
            (Tensor::I64(_), _) => Err(unsupported_dtype("gather", operand.dtype())),
        }
    }

    fn scatter(
        &mut self,
        operand: &Tensor,
        scatter_indices: &Tensor,
        updates: &Tensor,
        config: &ScatterConfig,
    ) -> crate::Result<Tensor> {
        match (operand, scatter_indices, updates) {
            (Tensor::F32(operand), Tensor::F32(indices), Tensor::F32(updates)) => self
                .scatter_float_typed(operand, indices, updates, config)
                .map(Tensor::F32),
            (Tensor::F64(operand), Tensor::F32(indices), Tensor::F64(updates)) => self
                .scatter_float_typed(operand, indices, updates, config)
                .map(Tensor::F64),
            (Tensor::C32(operand), Tensor::F32(indices), Tensor::C32(updates)) => self
                .scatter_complex_typed::<_, f32, _>(operand, indices, updates, config)
                .map(Tensor::C32),
            (Tensor::C64(operand), Tensor::F32(indices), Tensor::C64(updates)) => self
                .scatter_complex_typed::<_, f64, _>(operand, indices, updates, config)
                .map(Tensor::C64),
            (Tensor::F32(operand), Tensor::F64(indices), Tensor::F32(updates)) => self
                .scatter_float_typed(operand, indices, updates, config)
                .map(Tensor::F32),
            (Tensor::F64(operand), Tensor::F64(indices), Tensor::F64(updates)) => self
                .scatter_float_typed(operand, indices, updates, config)
                .map(Tensor::F64),
            (Tensor::C32(operand), Tensor::F64(indices), Tensor::C32(updates)) => self
                .scatter_complex_typed::<_, f32, _>(operand, indices, updates, config)
                .map(Tensor::C32),
            (Tensor::C64(operand), Tensor::F64(indices), Tensor::C64(updates)) => self
                .scatter_complex_typed::<_, f64, _>(operand, indices, updates, config)
                .map(Tensor::C64),
            (Tensor::F32(operand), Tensor::I64(indices), Tensor::F32(updates)) => {
                let indices = self.i64_indices_as_f64(indices)?;
                self.scatter_float_typed(operand, &indices, updates, config)
                    .map(Tensor::F32)
            }
            (Tensor::F64(operand), Tensor::I64(indices), Tensor::F64(updates)) => {
                let indices = self.i64_indices_as_f64(indices)?;
                self.scatter_float_typed(operand, &indices, updates, config)
                    .map(Tensor::F64)
            }
            (Tensor::C32(operand), Tensor::I64(indices), Tensor::C32(updates)) => {
                let indices = self.i64_indices_as_f64(indices)?;
                self.scatter_complex_typed::<_, f32, _>(operand, &indices, updates, config)
                    .map(Tensor::C32)
            }
            (Tensor::C64(operand), Tensor::I64(indices), Tensor::C64(updates)) => {
                let indices = self.i64_indices_as_f64(indices)?;
                self.scatter_complex_typed::<_, f64, _>(operand, &indices, updates, config)
                    .map(Tensor::C64)
            }
            (_, Tensor::C32(_) | Tensor::C64(_), _) => Err(crate::Error::BackendFailure {
                op: "scatter",
                message: "complex index tensors are not supported".into(),
            }),
            (_, _, _) => Err(ternary_dtype_mismatch(
                "scatter",
                operand,
                scatter_indices,
                updates,
            )),
        }
    }

    fn slice(&mut self, input: &Tensor, config: &SliceConfig) -> crate::Result<Tensor> {
        match input {
            Tensor::F32(t) => self.slice_typed(t, config).map(Tensor::F32),
            Tensor::F64(t) => self.slice_typed(t, config).map(Tensor::F64),
            Tensor::I64(t) => self.slice_typed(t, config).map(Tensor::I64),
            Tensor::C32(t) => self.slice_typed(t, config).map(Tensor::C32),
            Tensor::C64(t) => self.slice_typed(t, config).map(Tensor::C64),
        }
    }

    fn dynamic_slice(
        &mut self,
        input: &Tensor,
        starts: &Tensor,
        slice_sizes: &[usize],
    ) -> crate::Result<Tensor> {
        match (input, starts) {
            (Tensor::F32(input), Tensor::F32(starts)) => self
                .dynamic_slice_typed(input, starts, slice_sizes)
                .map(Tensor::F32),
            (Tensor::F64(input), Tensor::F32(starts)) => self
                .dynamic_slice_typed(input, starts, slice_sizes)
                .map(Tensor::F64),
            (Tensor::C32(input), Tensor::F32(starts)) => self
                .dynamic_slice_typed(input, starts, slice_sizes)
                .map(Tensor::C32),
            (Tensor::C64(input), Tensor::F32(starts)) => self
                .dynamic_slice_typed(input, starts, slice_sizes)
                .map(Tensor::C64),
            (Tensor::F32(input), Tensor::F64(starts)) => self
                .dynamic_slice_typed(input, starts, slice_sizes)
                .map(Tensor::F32),
            (Tensor::F64(input), Tensor::F64(starts)) => self
                .dynamic_slice_typed(input, starts, slice_sizes)
                .map(Tensor::F64),
            (Tensor::C32(input), Tensor::F64(starts)) => self
                .dynamic_slice_typed(input, starts, slice_sizes)
                .map(Tensor::C32),
            (Tensor::C64(input), Tensor::F64(starts)) => self
                .dynamic_slice_typed(input, starts, slice_sizes)
                .map(Tensor::C64),
            (Tensor::F32(input), Tensor::I64(starts)) => {
                let starts = self.i64_indices_as_f64(starts)?;
                self.dynamic_slice_typed(input, &starts, slice_sizes)
                    .map(Tensor::F32)
            }
            (Tensor::F64(input), Tensor::I64(starts)) => {
                let starts = self.i64_indices_as_f64(starts)?;
                self.dynamic_slice_typed(input, &starts, slice_sizes)
                    .map(Tensor::F64)
            }
            (Tensor::C32(input), Tensor::I64(starts)) => {
                let starts = self.i64_indices_as_f64(starts)?;
                self.dynamic_slice_typed(input, &starts, slice_sizes)
                    .map(Tensor::C32)
            }
            (Tensor::C64(input), Tensor::I64(starts)) => {
                let starts = self.i64_indices_as_f64(starts)?;
                self.dynamic_slice_typed(input, &starts, slice_sizes)
                    .map(Tensor::C64)
            }
            (_, Tensor::C32(_) | Tensor::C64(_)) => Err(crate::Error::BackendFailure {
                op: "dynamic_slice",
                message: "complex index tensors are not supported".into(),
            }),
            (Tensor::I64(_), _) => Err(unsupported_dtype("dynamic_slice", input.dtype())),
        }
    }

    fn dynamic_update_slice(
        &mut self,
        _operand: &Tensor,
        _update: &Tensor,
        _starts: &Tensor,
    ) -> crate::Result<Tensor> {
        Err(crate::Error::BackendFailure {
            op: "dynamic_update_slice",
            message: "dynamic_update_slice is not implemented for the CubeCL backend".into(),
        })
    }

    fn pad(&mut self, input: &Tensor, config: &PadConfig) -> crate::Result<Tensor> {
        match input {
            Tensor::F32(t) => self.pad_typed(t, config).map(Tensor::F32),
            Tensor::F64(t) => self.pad_typed(t, config).map(Tensor::F64),
            Tensor::I64(t) => self.pad_typed(t, config).map(Tensor::I64),
            Tensor::C32(t) => self.pad_typed(t, config).map(Tensor::C32),
            Tensor::C64(t) => self.pad_typed(t, config).map(Tensor::C64),
        }
    }

    fn concatenate(&mut self, inputs: &[&Tensor], axis: usize) -> crate::Result<Tensor> {
        let first = inputs
            .first()
            .copied()
            .ok_or_else(|| crate::Error::InvalidConfig {
                op: "concatenate",
                message: "concatenate requires at least one input".into(),
            })?;
        match first {
            Tensor::F32(_) => {
                let typed: crate::Result<Vec<&TypedTensor<f32>>> = inputs
                    .iter()
                    .map(|tensor| match tensor {
                        Tensor::F32(t) => Ok(t),
                        _ => Err(dtype_mismatch("concatenate", first, tensor)),
                    })
                    .collect();
                self.concatenate_typed(&typed?, axis).map(Tensor::F32)
            }
            Tensor::F64(_) => {
                let typed: crate::Result<Vec<&TypedTensor<f64>>> = inputs
                    .iter()
                    .map(|tensor| match tensor {
                        Tensor::F64(t) => Ok(t),
                        _ => Err(dtype_mismatch("concatenate", first, tensor)),
                    })
                    .collect();
                self.concatenate_typed(&typed?, axis).map(Tensor::F64)
            }
            Tensor::I64(_) => {
                let typed: crate::Result<Vec<&TypedTensor<i64>>> = inputs
                    .iter()
                    .map(|tensor| match tensor {
                        Tensor::I64(t) => Ok(t),
                        _ => Err(dtype_mismatch("concatenate", first, tensor)),
                    })
                    .collect();
                self.concatenate_typed(&typed?, axis).map(Tensor::I64)
            }
            Tensor::C32(_) => {
                let typed: crate::Result<Vec<&TypedTensor<Complex32>>> = inputs
                    .iter()
                    .map(|tensor| match tensor {
                        Tensor::C32(t) => Ok(t),
                        _ => Err(dtype_mismatch("concatenate", first, tensor)),
                    })
                    .collect();
                self.concatenate_typed(&typed?, axis).map(Tensor::C32)
            }
            Tensor::C64(_) => {
                let typed: crate::Result<Vec<&TypedTensor<Complex64>>> = inputs
                    .iter()
                    .map(|tensor| match tensor {
                        Tensor::C64(t) => Ok(t),
                        _ => Err(dtype_mismatch("concatenate", first, tensor)),
                    })
                    .collect();
                self.concatenate_typed(&typed?, axis).map(Tensor::C64)
            }
        }
    }

    fn reverse(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
        match input {
            Tensor::F32(t) => self.reverse_typed(t, axes).map(Tensor::F32),
            Tensor::F64(t) => self.reverse_typed(t, axes).map(Tensor::F64),
            Tensor::I64(t) => self.reverse_typed(t, axes).map(Tensor::I64),
            Tensor::C32(t) => self.reverse_typed(t, axes).map(Tensor::C32),
            Tensor::C64(t) => self.reverse_typed(t, axes).map(Tensor::C64),
        }
    }

    fn cholesky(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        linalg::cholesky(self, input)
    }

    fn triangular_solve(
        &mut self,
        a: &Tensor,
        b: &Tensor,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    ) -> crate::Result<Tensor> {
        linalg::triangular_solve(self, a, b, left_side, lower, transpose_a, unit_diagonal)
    }

    fn lu(&mut self, input: &Tensor) -> crate::Result<Vec<Tensor>> {
        linalg::lu(self, input)
    }

    fn full_piv_lu(&mut self, input: &Tensor) -> crate::Result<Vec<Tensor>> {
        linalg::full_piv_lu(self, input)
    }

    fn full_piv_lu_solve(
        &mut self,
        a: &Tensor,
        b: &Tensor,
        transpose_a: bool,
    ) -> crate::Result<Tensor> {
        linalg::full_piv_lu_solve(self, a, b, transpose_a)
    }

    fn svd(&mut self, input: &Tensor) -> crate::Result<Vec<Tensor>> {
        linalg::svd(self, input)
    }

    fn qr(&mut self, input: &Tensor) -> crate::Result<Vec<Tensor>> {
        linalg::qr(self, input)
    }

    fn eigh(&mut self, input: &Tensor) -> crate::Result<Vec<Tensor>> {
        linalg::eigh(self, input)
    }

    fn eig(&mut self, input: &Tensor) -> crate::Result<Vec<Tensor>> {
        linalg::eig(self, input)
    }

    fn solve(&mut self, a: &Tensor, b: &Tensor) -> crate::Result<Tensor> {
        linalg::solve(self, a, b)
    }

    fn download_to_host(&mut self, tensor: &Tensor) -> crate::Result<Tensor> {
        download_tensor(self.runtime(), tensor)
    }

    fn upload_host_tensor(&mut self, tensor: &Tensor) -> crate::Result<Tensor> {
        upload_tensor(self.runtime(), tensor)
    }

    fn execute_elementwise_fusion(
        &mut self,
        inputs: &[&Tensor],
        plan: &crate::ElementwiseFusionPlan,
    ) -> crate::Result<Option<Vec<Tensor>>> {
        fusion::execute_elementwise_fusion(self, inputs, plan)
    }
}

fn validate_permutation(op: &'static str, perm: &[usize], rank: usize) -> crate::Result<()> {
    ensure_rank(op, rank, perm.len())?;
    ensure_axes_unique(op, "perm", perm, rank)
}

fn validate_broadcast_in_dim(
    input_shape: &[usize],
    shape: &[usize],
    dims: &[usize],
) -> crate::Result<()> {
    ensure_rank("broadcast_in_dim", input_shape.len(), dims.len())?;
    let mut seen = vec![false; shape.len()];
    for (src_axis, &dst_axis) in dims.iter().enumerate() {
        ensure_axis("broadcast_in_dim", dst_axis, shape.len())?;
        if seen[dst_axis] {
            return Err(crate::Error::DuplicateAxis {
                op: "broadcast_in_dim",
                axis: dst_axis,
                role: "dims",
            });
        }
        seen[dst_axis] = true;
        let src = input_shape[src_axis];
        let dst = shape[dst_axis];
        if src != dst && src != 1 {
            return Err(crate::Error::ShapeMismatch {
                op: "broadcast_in_dim",
                lhs: input_shape.to_vec(),
                rhs: shape.to_vec(),
            });
        }
    }
    Ok(())
}

fn extract_diagonal_shape(
    input_shape: &[usize],
    axis_a: usize,
    axis_b: usize,
) -> crate::Result<(Vec<usize>, usize)> {
    ensure_axis("extract_diagonal", axis_a, input_shape.len())?;
    ensure_axis("extract_diagonal", axis_b, input_shape.len())?;
    if axis_a == axis_b {
        return Err(crate::Error::DuplicateAxis {
            op: "extract_diagonal",
            axis: axis_a,
            role: "axes",
        });
    }
    let diag_output_axis = if axis_a < axis_b { axis_a } else { axis_a - 1 };
    let diag_dim = input_shape[axis_a].min(input_shape[axis_b]);
    let mut output_shape = input_shape.to_vec();
    output_shape.remove(axis_b);
    output_shape[diag_output_axis] = diag_dim;
    Ok((output_shape, diag_output_axis))
}

fn embed_diagonal_shape(
    input_shape: &[usize],
    axis_a: usize,
    axis_b: usize,
) -> crate::Result<Vec<usize>> {
    ensure_axis("embed_diagonal", axis_a, input_shape.len())?;
    if axis_b > input_shape.len() {
        return Err(crate::Error::AxisOutOfBounds {
            op: "embed_diagonal",
            axis: axis_b,
            rank: input_shape.len(),
        });
    }
    let mut output_shape = input_shape.to_vec();
    output_shape.insert(axis_b, input_shape[axis_a]);
    Ok(output_shape)
}

fn reduction_output_shape(input_shape: &[usize], axes: &[usize]) -> Vec<usize> {
    let shape: Vec<usize> = input_shape
        .iter()
        .enumerate()
        .filter_map(|(axis, &dim)| (!axes.contains(&axis)).then_some(dim))
        .collect();
    // cubecl Array::new(0) generates uint32 arr[0] which is invalid CUDA.
    // When all axes are reduced (scalar output), use shape [1] instead.
    if shape.is_empty() {
        vec![1]
    } else {
        shape
    }
}

fn reduction_keepdims_shape(input_shape: &[usize], axis: usize) -> Vec<usize> {
    let mut output_shape = input_shape.to_vec();
    output_shape[axis] = 1;
    output_shape
}

fn cubecl_reshape_metadata<T: CubeElement + Clone>(
    tensor: TypedTensor<T>,
    shape: Vec<usize>,
    op: &'static str,
) -> crate::Result<TypedTensor<T>> {
    let len = shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| crate::Error::BackendFailure {
            op,
            message: format!("shape product overflow for CubeCL reshape shape {shape:?}"),
        })?;
    let tensor_len = tensor.n_elements();
    if len != tensor_len {
        return Err(crate::Error::BackendFailure {
            op,
            message: format!(
                "cannot reshape CubeCL output metadata from {:?} ({tensor_len} elements) to {:?} ({len} elements)",
                tensor.shape, shape
            ),
        });
    }

    Ok(TypedTensor { shape, ..tensor })
}

fn validate_slice(input_shape: &[usize], config: &SliceConfig) -> crate::Result<Vec<usize>> {
    let rank = input_shape.len();
    ensure_rank("slice", rank, config.starts.len())?;
    ensure_rank("slice", rank, config.limits.len())?;
    ensure_rank("slice", rank, config.strides.len())?;
    input_shape
        .iter()
        .enumerate()
        .map(|(axis, &dim)| {
            let start = config.starts[axis];
            let limit = config.limits[axis];
            let stride = config.strides[axis];
            if start > limit {
                return Err(crate::Error::InvalidConfig {
                    op: "slice",
                    message: format!("start exceeds limit on axis {axis}"),
                });
            }
            if limit > dim {
                return Err(crate::Error::AxisOutOfBounds {
                    op: "slice",
                    axis,
                    rank,
                });
            }
            if stride == 0 {
                return Err(crate::Error::InvalidConfig {
                    op: "slice",
                    message: format!("stride must be positive on axis {axis}"),
                });
            }
            let span = limit - start;
            Ok(span.div_ceil(stride))
        })
        .collect()
}

fn pad_output_shape(input_shape: &[usize], config: &PadConfig) -> crate::Result<Vec<usize>> {
    let rank = input_shape.len();
    ensure_rank("pad", rank, config.edge_padding_low.len())?;
    ensure_rank("pad", rank, config.edge_padding_high.len())?;
    ensure_rank("pad", rank, config.interior_padding.len())?;
    let mut out_shape = Vec::with_capacity(rank);
    for axis in 0..rank {
        if config.interior_padding[axis] < 0 {
            return Err(crate::Error::InvalidConfig {
                op: "pad",
                message: format!("interior padding must be non-negative on axis {axis}"),
            });
        }
        let base = if input_shape[axis] == 0 {
            0
        } else {
            (input_shape[axis] as i64 - 1) * (config.interior_padding[axis] + 1) + 1
        };
        let dim = config.edge_padding_low[axis] + config.edge_padding_high[axis] + base;
        out_shape.push(
            usize::try_from(dim).map_err(|_| crate::Error::InvalidConfig {
                op: "pad",
                message: format!("negative output dimension on axis {axis}"),
            })?,
        );
    }
    Ok(out_shape)
}

fn index_vector_size(shape: &[usize], index_vector_dim: usize) -> usize {
    if index_vector_dim == shape.len() {
        1
    } else {
        shape[index_vector_dim]
    }
}

fn index_batch_shape(shape: &[usize], index_vector_dim: usize) -> Vec<usize> {
    if index_vector_dim == shape.len() {
        return shape.to_vec();
    }
    shape
        .iter()
        .enumerate()
        .filter_map(|(axis, &dim)| (axis != index_vector_dim).then_some(dim))
        .collect()
}

fn operand_window_dims(rank: usize, collapsed_or_inserted: &[usize]) -> Vec<usize> {
    (0..rank)
        .filter(|dim| !collapsed_or_inserted.contains(dim))
        .collect()
}

struct GatherLaunchMeta {
    output_shape: Vec<usize>,
    batch_shape: Vec<usize>,
    window_dims: Vec<usize>,
}

fn gather_launch_meta(
    operand_shape: &[usize],
    start_indices_shape: &[usize],
    config: &GatherConfig,
) -> crate::Result<GatherLaunchMeta> {
    ensure_rank("gather", operand_shape.len(), config.slice_sizes.len())?;
    if config.index_vector_dim > start_indices_shape.len() {
        return Err(crate::Error::AxisOutOfBounds {
            op: "gather",
            axis: config.index_vector_dim,
            rank: start_indices_shape.len(),
        });
    }
    let index_size = index_vector_size(start_indices_shape, config.index_vector_dim);
    if index_size != config.start_index_map.len() {
        return Err(crate::Error::InvalidConfig {
            op: "gather",
            message: "start_index_map length mismatch".into(),
        });
    }
    ensure_axes_unique(
        "gather",
        "collapsed_slice_dims",
        &config.collapsed_slice_dims,
        operand_shape.len(),
    )?;
    ensure_axes_unique(
        "gather",
        "offset_dims",
        &config.offset_dims,
        operand_shape.len(),
    )?;
    ensure_axes_unique(
        "gather",
        "start_index_map",
        &config.start_index_map,
        operand_shape.len(),
    )?;
    let window_dims = operand_window_dims(operand_shape.len(), &config.collapsed_slice_dims);
    if config.offset_dims.len() != window_dims.len() {
        return Err(crate::Error::InvalidConfig {
            op: "gather",
            message: "offset_dims length mismatch".into(),
        });
    }
    let batch_shape = index_batch_shape(start_indices_shape, config.index_vector_dim);
    let out_rank = batch_shape.len() + config.offset_dims.len();
    let mut output_shape = vec![0usize; out_rank];
    let mut out_axis_to_operand_dim = vec![None; out_rank];
    for (offset_axis, &out_axis) in config.offset_dims.iter().enumerate() {
        out_axis_to_operand_dim[out_axis] = Some(window_dims[offset_axis]);
    }
    let mut batch_axis = 0usize;
    for out_axis in 0..out_rank {
        if let Some(operand_dim) = out_axis_to_operand_dim[out_axis] {
            output_shape[out_axis] = config.slice_sizes[operand_dim];
        } else {
            output_shape[out_axis] = batch_shape[batch_axis];
            batch_axis += 1;
        }
    }
    Ok(GatherLaunchMeta {
        output_shape,
        batch_shape,
        window_dims,
    })
}

struct ScatterLaunchMeta {
    batch_shape: Vec<usize>,
    window_dims: Vec<usize>,
    window_shape_updates: Vec<usize>,
}

fn scatter_launch_meta(
    operand_shape: &[usize],
    scatter_indices_shape: &[usize],
    updates_shape: &[usize],
    config: &ScatterConfig,
) -> crate::Result<ScatterLaunchMeta> {
    if config.index_vector_dim > scatter_indices_shape.len() {
        return Err(crate::Error::AxisOutOfBounds {
            op: "scatter",
            axis: config.index_vector_dim,
            rank: scatter_indices_shape.len(),
        });
    }
    let index_size = index_vector_size(scatter_indices_shape, config.index_vector_dim);
    if index_size != config.scatter_dims_to_operand_dims.len() {
        return Err(crate::Error::InvalidConfig {
            op: "scatter",
            message: "scatter_dims_to_operand_dims length mismatch".into(),
        });
    }
    ensure_axes_unique(
        "scatter",
        "inserted_window_dims",
        &config.inserted_window_dims,
        operand_shape.len(),
    )?;
    ensure_axes_unique(
        "scatter",
        "scatter_dims_to_operand_dims",
        &config.scatter_dims_to_operand_dims,
        operand_shape.len(),
    )?;
    ensure_axes_unique(
        "scatter",
        "update_window_dims",
        &config.update_window_dims,
        updates_shape.len(),
    )?;
    let batch_shape = index_batch_shape(scatter_indices_shape, config.index_vector_dim);
    let window_dims = operand_window_dims(operand_shape.len(), &config.inserted_window_dims);
    if config.update_window_dims.len() != window_dims.len() {
        return Err(crate::Error::InvalidConfig {
            op: "scatter",
            message: "update_window_dims length mismatch".into(),
        });
    }
    if updates_shape.len() - config.update_window_dims.len() != batch_shape.len() {
        return Err(crate::Error::InvalidConfig {
            op: "scatter",
            message: "updates batch rank mismatch".into(),
        });
    }
    let window_shape_updates = config
        .update_window_dims
        .iter()
        .map(|&axis| updates_shape[axis])
        .collect();
    Ok(ScatterLaunchMeta {
        batch_shape,
        window_dims,
        window_shape_updates,
    })
}

fn concatenate_output_shape<T>(
    inputs: &[&TypedTensor<T>],
    axis: usize,
) -> crate::Result<Vec<usize>> {
    let first = inputs[0];
    let rank = first.shape.len();
    ensure_axis("concatenate", axis, rank)?;
    let mut out_shape = first.shape.clone();
    let mut axis_extent = 0usize;
    for input in inputs {
        ensure_rank("concatenate", rank, input.shape.len())?;
        for dim in 0..rank {
            if dim == axis {
                axis_extent += input.shape[dim];
            } else if input.shape[dim] != first.shape[dim] {
                return Err(crate::Error::ShapeMismatch {
                    op: "concatenate",
                    lhs: first.shape.clone(),
                    rhs: input.shape.clone(),
                });
            }
        }
    }
    out_shape[axis] = axis_extent;
    Ok(out_shape)
}

#[cfg(test)]
mod tests;
