//! CubeCL WebGPU provider runtime and backend skeleton.

use cubecl::prelude::{CubeCount, CubeDim, CubeElement, CubeType, Sequence, TensorBinding};
use cubecl_wgpu::WgpuRuntime;
use std::fmt;
use std::sync::Arc;

use crate::{
    BackendBuffer, BackendCachedDot, BackendRuntimeCache, BackendSessionHost, Buffer, CompareDir,
    DType, DeviceId, DeviceKind, DotGeneralConfig, Error, GatherConfig, GpuBackendKind, MemoryKind,
    PadConfig, Placement, ScatterConfig, SliceConfig, Tensor, TensorAnalytic, TensorBackend,
    TensorBuffer, TensorDeviceTransfer, TensorDot, TensorElementwise, TensorFusion, TensorIndexing,
    TensorReduction, TensorStructural, TypedTensor,
};

const DEFAULT_CUBE_DIM_X: u32 = 256;

mod gemm;
mod kernels;
mod memory;
mod runtime;

pub use memory::{download_webgpu_tensor, upload_webgpu_tensor};
pub use runtime::{webgpu_available, WebGpuRuntime};

/// CubeCL-managed WebGPU buffer stored behind tensor backend-buffer trait objects.
#[derive(Clone)]
pub(crate) struct WebGpuBuffer<T> {
    handle: cubecl_runtime::server::Handle,
    len: usize,
    _marker: std::marker::PhantomData<T>,
}

impl<T> std::fmt::Debug for WebGpuBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WebGpuBuffer")
            .field("len", &self.len)
            .finish()
    }
}

impl<T> WebGpuBuffer<T> {
    fn new(handle: cubecl_runtime::server::Handle, len: usize) -> Self {
        Self {
            handle,
            len,
            _marker: std::marker::PhantomData,
        }
    }

    fn handle(&self) -> &cubecl_runtime::server::Handle {
        &self.handle
    }

    fn element_len(&self) -> usize {
        self.len
    }
}

impl<T: Send + Sync + 'static> BackendBuffer<T> for WebGpuBuffer<T> {
    fn backend_family(&self) -> &'static str {
        "cubecl-webgpu"
    }

    fn len(&self) -> usize {
        self.len
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

fn webgpu_handle_from_backend<T: 'static>(
    buffer: &dyn BackendBuffer<T>,
    op: &'static str,
) -> crate::Result<cubecl_runtime::server::Handle> {
    buffer
        .as_any()
        .downcast_ref::<WebGpuBuffer<T>>()
        .map(|buffer| buffer.handle().clone())
        .ok_or_else(|| {
            crate::Error::backend_failure(
                op,
                format!(
                    "expected WebGPU backend buffer, got `{}` backend buffer",
                    buffer.backend_family()
                ),
            )
        })
}

fn checked_shape_product(op: &'static str, shape: &[usize]) -> crate::Result<usize> {
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| {
            Error::backend_failure(op, format!("shape product overflow for shape {shape:?}"))
        })
}

fn cube_count_for_len(len: usize) -> crate::Result<CubeCount> {
    let cubes = len.div_ceil(DEFAULT_CUBE_DIM_X as usize);
    let cubes = u32::try_from(cubes).map_err(|_| {
        Error::backend_failure(
            "cube_count_for_len",
            format!(
                "1D WebGPU launch for {len} elements requires {cubes} cubes, \
                 which exceeds u32::MAX"
            ),
        )
    })?;
    Ok(CubeCount::Static(cubes.max(1), 1, 1))
}

fn cube_dim_1d() -> CubeDim {
    CubeDim::new_1d(DEFAULT_CUBE_DIM_X)
}

fn comptime_sequence<T: CubeType>(values: &[T]) -> Sequence<T>
where
    T: Clone,
{
    let mut out = Sequence::new();
    for value in values {
        out.push(value.clone());
    }
    out
}

fn validate_webgpu_buffer_len<T>(
    tensor: &TypedTensor<T>,
    buffer: &WebGpuBuffer<T>,
    op: &'static str,
) -> crate::Result<()> {
    let expected_len = checked_shape_product(op, tensor.shape())?;
    let actual_len = buffer.element_len();
    if expected_len != actual_len {
        return Err(Error::backend_failure(
            op,
            format!(
                "expected shape product {expected_len} elements, actual WebGpuBuffer::len {actual_len}"
            ),
        ));
    }
    Ok(())
}

fn webgpu_buffer<'a, T: 'static>(
    tensor: &'a TypedTensor<T>,
    op: &'static str,
) -> crate::Result<&'a WebGpuBuffer<T>> {
    match tensor.buffer() {
        Buffer::Host(_) => Err(Error::backend_failure(
            op,
            "expected WebGPU tensor, got host tensor. \
             Use upload_webgpu_tensor() to transfer to WebGPU before calling WebGPU ops.",
        )),
        Buffer::Backend(buffer) => buffer
            .as_any()
            .downcast_ref::<WebGpuBuffer<T>>()
            .ok_or_else(|| {
                Error::backend_failure(
                    op,
                    format!(
                        "expected WebGPU tensor, got backend buffer family `{}`",
                        buffer.backend_family()
                    ),
                )
            }),
    }
}

fn typed_tensor_binding_with_layout<T: CubeElement + Clone>(
    tensor: &TypedTensor<T>,
    shape: &[usize],
    strides: &[usize],
    op: &'static str,
) -> crate::Result<TensorBinding<WgpuRuntime>> {
    if shape.len() != strides.len() {
        return Err(Error::backend_failure(
            op,
            format!(
                "WebGPU tensor binding layout rank mismatch: shape rank {} stride rank {}",
                shape.len(),
                strides.len()
            ),
        ));
    }
    let buffer = webgpu_buffer(tensor, op)?;
    validate_webgpu_buffer_len(tensor, buffer, op)?;
    let layout_len = checked_shape_product(op, shape)?;
    if layout_len != buffer.element_len() {
        return Err(Error::backend_failure(
            op,
            format!(
                "WebGPU tensor binding layout covers {layout_len} elements, backing buffer has {}",
                buffer.element_len()
            ),
        ));
    }

    let (shape, strides) = if shape.is_empty() {
        (vec![1], vec![1])
    } else {
        (shape.to_vec(), strides.to_vec())
    };

    // SAFETY: The tensor buffer is a validated WebGPU allocation for `tensor`.
    // The caller-provided shape/stride metadata is checked to cover exactly the
    // same element count, so CubeCL logical indexing remains inside the backing
    // allocation.
    Ok(unsafe {
        TensorBinding::from_raw_parts(buffer.handle().clone(), strides.into(), shape.into())
    })
}

pub(super) fn ensure_resident_on_runtime<T: 'static>(
    rt: &WebGpuRuntime,
    tensor: &TypedTensor<T>,
    op: &'static str,
) -> crate::Result<()> {
    webgpu_buffer(tensor, op)?;
    ensure_placement_resident_on_runtime(rt, tensor.placement(), op)
}

fn ensure_placement_resident_on_runtime(
    rt: &WebGpuRuntime,
    placement: &Placement,
    op: &'static str,
) -> crate::Result<()> {
    if !matches!(&placement.memory_kind, MemoryKind::Device) {
        return Err(Error::backend_failure(
            op,
            format!(
                "expected WebGPU tensor placement, got {:?}",
                placement.memory_kind
            ),
        ));
    }
    match &placement.device {
        Some(device)
            if device.kind == DeviceKind::Gpu(GpuBackendKind::WebGpu)
                && device.ordinal == rt.device_ordinal() =>
        {
            Ok(())
        }
        Some(device) => Err(Error::backend_failure(
            op,
            format!(
                "expected WebGPU tensor resident on webgpu:{}, got {:?}:{}",
                rt.device_ordinal(),
                device.kind,
                device.ordinal
            ),
        )),
        None => Err(Error::backend_failure(
            op,
            format!(
                "expected WebGPU tensor resident on webgpu:{}, got missing device metadata",
                rt.device_ordinal()
            ),
        )),
    }
}

fn typed_from_webgpu<T: Send + Sync + 'static>(
    shape: Vec<usize>,
    buffer: WebGpuBuffer<T>,
    device_ordinal: usize,
) -> crate::Result<TypedTensor<T>> {
    Ok(TypedTensor::from_buffer_col_major(
        shape,
        Buffer::Backend(Arc::new(buffer)),
        webgpu_placement(device_ordinal),
    )?)
}

fn alloc_output<T: CubeElement + Clone + Send + Sync + 'static>(
    rt: &WebGpuRuntime,
    shape: &[usize],
    op: &'static str,
) -> crate::Result<TypedTensor<T>> {
    let len = checked_shape_product(op, shape)?;
    let bytes = len.checked_mul(core::mem::size_of::<T>()).ok_or_else(|| {
        Error::backend_failure(
            op,
            format!("WebGPU output byte length overflow for shape {shape:?}"),
        )
    })?;
    let handle = rt.client().empty(bytes);
    typed_from_webgpu(
        shape.to_vec(),
        WebGpuBuffer::new(handle, len),
        rt.device_ordinal(),
    )
}

fn webgpu_placement(device_ordinal: usize) -> Placement {
    Placement {
        memory_kind: MemoryKind::Device,
        device: Some(DeviceId {
            kind: DeviceKind::Gpu(GpuBackendKind::WebGpu),
            ordinal: device_ordinal,
        }),
    }
}

/// CubeCL WebGPU tensor backend.
///
/// # Examples
///
/// ```
/// use tenferro_gpu::WebGpuBackend;
///
/// let _ctor: fn(usize) -> tenferro_tensor::Result<WebGpuBackend> = WebGpuBackend::new;
/// ```
pub struct WebGpuBackend {
    runtime: WebGpuRuntime,
}

impl fmt::Debug for WebGpuBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WebGpuBackend")
            .field("runtime", &self.runtime)
            .finish_non_exhaustive()
    }
}

impl WebGpuBackend {
    /// Initialize a WebGPU backend for a discrete GPU ordinal.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::WebGpuBackend;
    ///
    /// let _ctor: fn(usize) -> tenferro_tensor::Result<WebGpuBackend> = WebGpuBackend::new;
    /// ```
    pub fn new(device_ordinal: usize) -> crate::Result<Self> {
        WebGpuRuntime::new(device_ordinal).map(Self::from_runtime)
    }

    /// Initialize a WebGPU backend using CubeCL's default adapter selection.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::WebGpuBackend;
    ///
    /// let _ctor: fn() -> tenferro_tensor::Result<WebGpuBackend> = WebGpuBackend::new_default;
    /// ```
    pub fn new_default() -> crate::Result<Self> {
        WebGpuRuntime::new_default().map(Self::from_runtime)
    }

    /// Build a WebGPU backend from an initialized runtime.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::{WebGpuBackend, WebGpuRuntime};
    ///
    /// let _from_runtime: fn(WebGpuRuntime) -> WebGpuBackend = WebGpuBackend::from_runtime;
    /// ```
    pub fn from_runtime(runtime: WebGpuRuntime) -> Self {
        Self { runtime }
    }

    /// Return this backend's WebGPU runtime.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::{WebGpuBackend, WebGpuRuntime};
    ///
    /// let _runtime: fn(&WebGpuBackend) -> &WebGpuRuntime = WebGpuBackend::runtime;
    /// ```
    pub fn runtime(&self) -> &WebGpuRuntime {
        &self.runtime
    }

    /// Block until queued WebGPU work completes.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::WebGpuBackend;
    ///
    /// let _sync: fn(&WebGpuBackend) -> tenferro_tensor::Result<()> = WebGpuBackend::synchronize;
    /// ```
    pub fn synchronize(&self) -> crate::Result<()> {
        self.runtime.synchronize()
    }
}

fn unsupported_op(op: &'static str) -> crate::Error {
    crate::Error::backend_failure(
        op,
        "WebGPU backend does not support this operation yet; upload/download explicitly and use a supported backend operation",
    )
}

macro_rules! unsupported {
    ($op:literal) => {
        Err(unsupported_op($op))
    };
}

impl TensorElementwise for WebGpuBackend {
    fn add(&mut self, _lhs: &Tensor, _rhs: &Tensor) -> crate::Result<Tensor> {
        unsupported!("webgpu_add")
    }

    fn mul(&mut self, _lhs: &Tensor, _rhs: &Tensor) -> crate::Result<Tensor> {
        unsupported!("webgpu_mul")
    }

    fn neg(&mut self, _input: &Tensor) -> crate::Result<Tensor> {
        unsupported!("webgpu_neg")
    }

    fn conj(&mut self, _input: &Tensor) -> crate::Result<Tensor> {
        unsupported!("webgpu_conj")
    }

    fn div(&mut self, _lhs: &Tensor, _rhs: &Tensor) -> crate::Result<Tensor> {
        unsupported!("webgpu_div")
    }

    fn abs(&mut self, _input: &Tensor) -> crate::Result<Tensor> {
        unsupported!("webgpu_abs")
    }

    fn sign(&mut self, _input: &Tensor) -> crate::Result<Tensor> {
        unsupported!("webgpu_sign")
    }

    fn maximum(&mut self, _lhs: &Tensor, _rhs: &Tensor) -> crate::Result<Tensor> {
        unsupported!("webgpu_maximum")
    }

    fn minimum(&mut self, _lhs: &Tensor, _rhs: &Tensor) -> crate::Result<Tensor> {
        unsupported!("webgpu_minimum")
    }

    fn compare(
        &mut self,
        _lhs: &Tensor,
        _rhs: &Tensor,
        _dir: &CompareDir,
    ) -> crate::Result<Tensor> {
        unsupported!("webgpu_compare")
    }

    fn select(
        &mut self,
        _pred: &Tensor,
        _on_true: &Tensor,
        _on_false: &Tensor,
    ) -> crate::Result<Tensor> {
        unsupported!("webgpu_select")
    }

    fn clamp(
        &mut self,
        _input: &Tensor,
        _lower: &Tensor,
        _upper: &Tensor,
    ) -> crate::Result<Tensor> {
        unsupported!("webgpu_clamp")
    }
}

impl TensorAnalytic for WebGpuBackend {
    fn exp(&mut self, _input: &Tensor) -> crate::Result<Tensor> {
        unsupported!("webgpu_exp")
    }

    fn log(&mut self, _input: &Tensor) -> crate::Result<Tensor> {
        unsupported!("webgpu_log")
    }

    fn sin(&mut self, _input: &Tensor) -> crate::Result<Tensor> {
        unsupported!("webgpu_sin")
    }

    fn cos(&mut self, _input: &Tensor) -> crate::Result<Tensor> {
        unsupported!("webgpu_cos")
    }

    fn tanh(&mut self, _input: &Tensor) -> crate::Result<Tensor> {
        unsupported!("webgpu_tanh")
    }

    fn sqrt(&mut self, _input: &Tensor) -> crate::Result<Tensor> {
        unsupported!("webgpu_sqrt")
    }

    fn rsqrt(&mut self, _input: &Tensor) -> crate::Result<Tensor> {
        unsupported!("webgpu_rsqrt")
    }

    fn pow(&mut self, _lhs: &Tensor, _rhs: &Tensor) -> crate::Result<Tensor> {
        unsupported!("webgpu_pow")
    }

    fn expm1(&mut self, _input: &Tensor) -> crate::Result<Tensor> {
        unsupported!("webgpu_expm1")
    }

    fn log1p(&mut self, _input: &Tensor) -> crate::Result<Tensor> {
        unsupported!("webgpu_log1p")
    }
}

impl TensorStructural for WebGpuBackend {
    fn transpose(&mut self, _input: &Tensor, _perm: &[usize]) -> crate::Result<Tensor> {
        unsupported!("webgpu_transpose")
    }

    fn reshape(&mut self, _input: &Tensor, _shape: &[usize]) -> crate::Result<Tensor> {
        unsupported!("webgpu_reshape")
    }

    fn broadcast_in_dim(
        &mut self,
        _input: &Tensor,
        _shape: &[usize],
        _dims: &[usize],
    ) -> crate::Result<Tensor> {
        unsupported!("webgpu_broadcast_in_dim")
    }

    fn cast(&mut self, _input: &Tensor, _to: DType) -> crate::Result<Tensor> {
        unsupported!("webgpu_cast")
    }

    fn extract_diagonal(
        &mut self,
        _input: &Tensor,
        _axis_a: usize,
        _axis_b: usize,
    ) -> crate::Result<Tensor> {
        unsupported!("webgpu_extract_diagonal")
    }

    fn embed_diagonal(
        &mut self,
        _input: &Tensor,
        _axis_a: usize,
        _axis_b: usize,
    ) -> crate::Result<Tensor> {
        unsupported!("webgpu_embed_diagonal")
    }

    fn tril(&mut self, _input: &Tensor, _k: i64) -> crate::Result<Tensor> {
        unsupported!("webgpu_tril")
    }

    fn triu(&mut self, _input: &Tensor, _k: i64) -> crate::Result<Tensor> {
        unsupported!("webgpu_triu")
    }
}

impl TensorReduction for WebGpuBackend {
    fn reduce_sum(&mut self, _input: &Tensor, _axes: &[usize]) -> crate::Result<Tensor> {
        unsupported!("webgpu_reduce_sum")
    }

    fn reduce_prod(&mut self, _input: &Tensor, _axes: &[usize]) -> crate::Result<Tensor> {
        unsupported!("webgpu_reduce_prod")
    }

    fn reduce_max(&mut self, _input: &Tensor, _axes: &[usize]) -> crate::Result<Tensor> {
        unsupported!("webgpu_reduce_max")
    }

    fn reduce_min(&mut self, _input: &Tensor, _axes: &[usize]) -> crate::Result<Tensor> {
        unsupported!("webgpu_reduce_min")
    }
}

impl TensorDot for WebGpuBackend {
    fn dot_general(
        &mut self,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
    ) -> crate::Result<Tensor> {
        gemm::dot_general(self, lhs, rhs, config)
    }

    fn dot_general_with_conj(
        &mut self,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
        lhs_conj: bool,
        rhs_conj: bool,
    ) -> crate::Result<Tensor> {
        gemm::dot_general_with_conj(self, lhs, rhs, config, lhs_conj, rhs_conj)
    }
}

impl TensorIndexing for WebGpuBackend {
    fn gather(
        &mut self,
        _operand: &Tensor,
        _start_indices: &Tensor,
        _config: &GatherConfig,
    ) -> crate::Result<Tensor> {
        unsupported!("webgpu_gather")
    }

    fn scatter(
        &mut self,
        _operand: &Tensor,
        _scatter_indices: &Tensor,
        _updates: &Tensor,
        _config: &ScatterConfig,
    ) -> crate::Result<Tensor> {
        unsupported!("webgpu_scatter")
    }

    fn slice(&mut self, _input: &Tensor, _config: &SliceConfig) -> crate::Result<Tensor> {
        unsupported!("webgpu_slice")
    }

    fn dynamic_slice(
        &mut self,
        _input: &Tensor,
        _starts: &Tensor,
        _slice_sizes: &[usize],
    ) -> crate::Result<Tensor> {
        unsupported!("webgpu_dynamic_slice")
    }

    fn dynamic_update_slice(
        &mut self,
        _operand: &Tensor,
        _update: &Tensor,
        _starts: &Tensor,
    ) -> crate::Result<Tensor> {
        unsupported!("webgpu_dynamic_update_slice")
    }

    fn pad(&mut self, _input: &Tensor, _config: &PadConfig) -> crate::Result<Tensor> {
        unsupported!("webgpu_pad")
    }

    fn concatenate(&mut self, _inputs: &[&Tensor], _axis: usize) -> crate::Result<Tensor> {
        unsupported!("webgpu_concatenate")
    }

    fn reverse(&mut self, _input: &Tensor, _axes: &[usize]) -> crate::Result<Tensor> {
        unsupported!("webgpu_reverse")
    }
}

impl TensorFusion for WebGpuBackend {}

impl TensorBuffer for WebGpuBackend {}

impl TensorDeviceTransfer for WebGpuBackend {
    fn download_to_host(&mut self, tensor: &Tensor) -> crate::Result<Tensor> {
        download_webgpu_tensor(self.runtime(), tensor)
    }

    fn upload_host_tensor(&mut self, tensor: &Tensor) -> crate::Result<Tensor> {
        upload_webgpu_tensor(self.runtime(), tensor)
    }
}

impl BackendRuntimeCache for WebGpuBackend {
    type RuntimeCache = ();
}

impl BackendCachedDot for WebGpuBackend {}

impl BackendSessionHost for WebGpuBackend {}

impl TensorBackend for WebGpuBackend {}
