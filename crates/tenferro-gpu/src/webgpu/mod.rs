//! CubeCL WebGPU provider runtime and backend skeleton.

use cubecl::prelude::{CubeCount, CubeDim, CubeElement, CubeType, Sequence, TensorBinding};
use cubecl_wgpu::WgpuRuntime;
use std::fmt;
use std::sync::Arc;

use crate::{
    AllocationDomainId, AllocationId, BackendBuffer, BackendCachedDot, BackendRuntimeCache,
    BackendSessionHost, Buffer, CompareDir, DType, DeviceId, DeviceKind, DotGeneralConfig, Error,
    GatherConfig, GpuBackendKind, HostAccessError, HostReadGuard, HostWriteGuard, MemoryKind,
    PadConfig, Placement, ScatterConfig, SliceConfig, Tensor, TensorAnalytic, TensorBackend,
    TensorBuffer, TensorDeviceTransfer, TensorDot, TensorElementwise, TensorFusion, TensorIndexing,
    TensorRead, TensorReduction, TensorStructural, TensorWrite, TypedTensor,
};

const DEFAULT_CUBE_DIM_X: u32 = 256;

mod apple;
mod error;
mod gemm;
pub(crate) mod interop;
mod kernels;
mod memory;
mod runtime;
mod runtime_adapter;

pub use apple::{AppleContext, AppleTransferStats};
pub(crate) use error::{unsupported_dtype, unsupported_operation};
pub use memory::{download_webgpu_tensor, upload_webgpu_tensor};
pub use runtime::{webgpu_available, WebGpuRuntime};
pub use runtime_adapter::{
    webgpu_runtime_engine_id, webgpu_runtime_engine_registration, webgpu_runtime_hardware_class,
};

/// CubeCL-managed WebGPU buffer stored behind tensor backend-buffer trait objects.
#[derive(Clone)]
pub(crate) struct WebGpuBuffer<T> {
    handle: cubecl_runtime::server::Handle,
    len: usize,
    managed: Option<Arc<cubecl_runtime::storage::ManagedResource<cubecl_wgpu::WgpuResource>>>,
    domain: Option<Arc<apple::AppleDomainState>>,
    _marker: std::marker::PhantomData<T>,
}

impl<T> std::fmt::Debug for WebGpuBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WebGpuBuffer")
            .field("len", &self.len)
            .field(
                "allocation_domain",
                &self.domain.as_ref().map(|domain| domain.id),
            )
            .finish()
    }
}

impl<T> WebGpuBuffer<T> {
    fn new(handle: cubecl_runtime::server::Handle, len: usize) -> Self {
        Self {
            handle,
            len,
            managed: None,
            domain: None,
            _marker: std::marker::PhantomData,
        }
    }

    fn new_for_runtime(
        rt: &WebGpuRuntime,
        handle: cubecl_runtime::server::Handle,
        len: usize,
        op: &'static str,
    ) -> crate::Result<Self> {
        let Some(domain) = rt.allocation_domain() else {
            return Ok(Self::new(handle, len));
        };
        let managed = rt
            .client()
            .get_resource(handle.clone())
            .map_err(|error| crate::Error::backend_source(op, error))?;
        Ok(Self {
            handle,
            len,
            managed: Some(Arc::new(managed)),
            domain: Some(Arc::clone(domain)),
            _marker: std::marker::PhantomData,
        })
    }

    fn handle(&self) -> &cubecl_runtime::server::Handle {
        &self.handle
    }

    fn element_len(&self) -> usize {
        self.len
    }
}

struct TypedMappedRead<T> {
    bytes: cubecl_wgpu::WgpuMappedReadGuard,
    marker: std::marker::PhantomData<T>,
}

impl<T> std::ops::Deref for TypedMappedRead<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        // SAFETY: WebGpuBuffer is private and is constructed only by typed allocation helpers.
        // Bool uses a separate byte representation and is rejected before this guard is created.
        unsafe {
            std::slice::from_raw_parts(
                self.bytes.as_ptr().cast::<T>(),
                self.bytes.len() / core::mem::size_of::<T>(),
            )
        }
    }
}

fn host_access_error(error: cubecl_wgpu::HostAccessError) -> HostAccessError {
    match error {
        cubecl_wgpu::HostAccessError::DeviceLocalAllocation => HostAccessError::Unsupported {
            backend: "cubecl-webgpu",
        },
        cubecl_wgpu::HostAccessError::OverlappingHostMapping => {
            HostAccessError::OverlappingHostMapping
        }
        cubecl_wgpu::HostAccessError::GpuAccessInProgress => HostAccessError::GpuAccessInProgress,
        cubecl_wgpu::HostAccessError::MappedForHost => HostAccessError::MappedForHost,
        other => HostAccessError::BackendFailure {
            message: other.to_string(),
        },
    }
}

fn supports_typed_mapping<T: 'static>() -> bool {
    let type_id = std::any::TypeId::of::<T>();
    type_id == std::any::TypeId::of::<f32>()
        || type_id == std::any::TypeId::of::<f64>()
        || type_id == std::any::TypeId::of::<i32>()
        || type_id == std::any::TypeId::of::<i64>()
        || type_id == std::any::TypeId::of::<num_complex::Complex32>()
        || type_id == std::any::TypeId::of::<num_complex::Complex64>()
}

fn validate_typed_mapping_len<T: 'static>(
    mapped_bytes: usize,
    len: usize,
) -> Result<(), HostAccessError> {
    if !supports_typed_mapping::<T>() {
        return Err(HostAccessError::Unsupported {
            backend: "cubecl-webgpu",
        });
    }
    let expected = len.checked_mul(core::mem::size_of::<T>()).ok_or_else(|| {
        HostAccessError::BackendFailure {
            message: "mapped byte length overflow".to_owned(),
        }
    })?;
    if mapped_bytes != expected {
        return Err(HostAccessError::BackendFailure {
            message: format!(
                "mapped byte length mismatch: expected {expected}, got {mapped_bytes}",
            ),
        });
    }
    Ok(())
}

fn validate_typed_read_mapping<T: 'static>(
    bytes: &[u8],
    len: usize,
) -> Result<(), HostAccessError> {
    validate_typed_mapping_len::<T>(bytes.len(), len)?;
    if bytes.as_ptr().align_offset(core::mem::align_of::<T>()) != 0 {
        return Err(HostAccessError::BackendFailure {
            message: format!(
                "mapped pointer is not aligned for {}",
                std::any::type_name::<T>()
            ),
        });
    }
    Ok(())
}

impl<T: Send + Sync + 'static> BackendBuffer<T> for WebGpuBuffer<T> {
    fn backend_family(&self) -> &'static str {
        "cubecl-webgpu"
    }

    fn len(&self) -> usize {
        self.len
    }

    fn allocation_domain(&self) -> Option<AllocationDomainId> {
        self.domain.as_ref().map(|domain| domain.id)
    }

    fn allocation_id(&self) -> Option<AllocationId> {
        self.managed
            .as_ref()
            .map(|managed| AllocationId::from_backend_id(managed.resource().allocation_id()))
    }

    fn map_read(&self) -> Result<HostReadGuard<'_, T>, HostAccessError> {
        let managed = self.managed.as_ref().ok_or(HostAccessError::Unsupported {
            backend: self.backend_family(),
        })?;
        let bytes = managed.resource().map_read().map_err(host_access_error)?;
        validate_typed_read_mapping::<T>(&bytes, self.len)?;
        Ok(HostReadGuard::new(TypedMappedRead {
            bytes,
            marker: std::marker::PhantomData,
        }))
    }

    fn map_write(&self) -> Result<HostWriteGuard<'_, T>, HostAccessError> {
        let managed = self.managed.as_ref().ok_or(HostAccessError::Unsupported {
            backend: self.backend_family(),
        })?;
        let mut bytes = managed.resource().map_write().map_err(host_access_error)?;
        validate_typed_mapping_len::<T>(bytes.len(), self.len)?;
        Ok(HostWriteGuard::new(self.len, move |source: &[T]| {
            let byte_len = source
                .len()
                .checked_mul(core::mem::size_of::<T>())
                .ok_or_else(|| HostAccessError::BackendFailure {
                    message: "mapped write byte length overflow".to_owned(),
                })?;
            // SAFETY: the source is borrowed for this call and byte slices may inspect any
            // initialized Rust value. Private constructors restrict mapped tensors to supported
            // scalar representations; bool was rejected above.
            let source_bytes =
                unsafe { std::slice::from_raw_parts(source.as_ptr().cast::<u8>(), byte_len) };
            bytes.copy_from_slice(source_bytes);
            Ok(())
        }))
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
            crate::Error::runtime_state(
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
            Error::invalid_argument(
                op,
                "shape",
                format!("shape product overflow for shape {shape:?}"),
            )
        })
}

fn cube_count_for_len(len: usize) -> crate::Result<CubeCount> {
    let cubes = len.div_ceil(DEFAULT_CUBE_DIM_X as usize);
    let cubes = u32::try_from(cubes).map_err(|_| {
        Error::invalid_argument(
            "cube_count_for_len",
            "length",
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

fn comptime_sequence<T: CubeType + Clone>(values: &[T]) -> Sequence<T> {
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
        return Err(Error::runtime_state(
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
        Buffer::Host(_) => Err(Error::runtime_state(
            op,
            "expected WebGPU tensor, got host tensor. \
             Use upload_webgpu_tensor() to transfer to WebGPU before calling WebGPU ops.",
        )),
        Buffer::Backend(buffer) => buffer
            .as_any()
            .downcast_ref::<WebGpuBuffer<T>>()
            .ok_or_else(|| {
                Error::runtime_state(
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
        return Err(Error::rank_mismatch(op, shape.len(), strides.len()));
    }
    let buffer = webgpu_buffer(tensor, op)?;
    validate_webgpu_buffer_len(tensor, buffer, op)?;
    let layout_len = checked_shape_product(op, shape)?;
    if layout_len != buffer.element_len() {
        return Err(Error::runtime_state(
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
    let buffer = webgpu_buffer(tensor, op)?;
    if let Some(expected) = rt.allocation_domain() {
        match buffer.domain.as_ref().map(|domain| domain.id) {
            Some(actual) if actual == expected.id => {}
            Some(actual) => {
                return Err(Error::host_access(
                    op,
                    HostAccessError::ForeignDomain {
                        expected: expected.id,
                        actual,
                    },
                ));
            }
            None => {
                return Err(Error::runtime_state(
                    op,
                    "Apple runtime requires a managed allocation from its domain",
                ));
            }
        }
    }
    ensure_placement_resident_on_runtime(rt, tensor.placement(), op)
}

fn ensure_placement_resident_on_runtime(
    rt: &WebGpuRuntime,
    placement: &Placement,
    op: &'static str,
) -> crate::Result<()> {
    let expected_memory = if rt.allocation_domain().is_some() {
        MemoryKind::Managed
    } else {
        MemoryKind::Device
    };
    if placement.memory_kind != expected_memory {
        return Err(Error::runtime_state(
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
        Some(device) => Err(Error::runtime_state(
            op,
            format!(
                "expected WebGPU tensor resident on webgpu:{}, got {:?}:{}",
                rt.device_ordinal(),
                device.kind,
                device.ordinal
            ),
        )),
        None => Err(Error::runtime_state(
            op,
            format!(
                "expected WebGPU tensor resident on webgpu:{}, got missing device metadata",
                rt.device_ordinal()
            ),
        )),
    }
}

pub(super) fn typed_from_webgpu<T: Send + Sync + 'static>(
    shape: Vec<usize>,
    buffer: WebGpuBuffer<T>,
    rt: &WebGpuRuntime,
) -> crate::Result<TypedTensor<T>> {
    TypedTensor::from_buffer_col_major(
        shape,
        Buffer::Backend(Arc::new(buffer)),
        webgpu_placement(rt),
    )
}

fn alloc_output<T: CubeElement + Clone + Send + Sync + 'static>(
    rt: &WebGpuRuntime,
    shape: &[usize],
    op: &'static str,
) -> crate::Result<TypedTensor<T>> {
    let len = checked_shape_product(op, shape)?;
    let bytes = len.checked_mul(core::mem::size_of::<T>()).ok_or_else(|| {
        Error::invalid_argument(
            op,
            "shape",
            format!("WebGPU output byte length overflow for shape {shape:?}"),
        )
    })?;
    let handle = rt.client().empty(bytes);
    let buffer = WebGpuBuffer::new_for_runtime(rt, handle, len, op)?;
    typed_from_webgpu(shape.to_vec(), buffer, rt)
}

pub(super) fn alloc_tensor_in_runtime(
    rt: &WebGpuRuntime,
    dtype: DType,
    shape: &[usize],
) -> crate::Result<Tensor> {
    match dtype {
        DType::F32 => alloc_output::<f32>(rt, shape, "apple_alloc").map(Tensor::F32),
        DType::F64 => alloc_output::<f64>(rt, shape, "apple_alloc").map(Tensor::F64),
        DType::I32 => alloc_output::<i32>(rt, shape, "apple_alloc").map(Tensor::I32),
        DType::I64 => alloc_output::<i64>(rt, shape, "apple_alloc").map(Tensor::I64),
        DType::C32 => {
            alloc_output::<num_complex::Complex32>(rt, shape, "apple_alloc").map(Tensor::C32)
        }
        DType::C64 => {
            alloc_output::<num_complex::Complex64>(rt, shape, "apple_alloc").map(Tensor::C64)
        }
        DType::Bool => {
            let len = checked_shape_product("apple_alloc", shape)?;
            let handle = rt.client().empty(len);
            let buffer = WebGpuBuffer::new_for_runtime(rt, handle, len, "apple_alloc")?;
            Ok(Tensor::Bool(TypedTensor::from_buffer_col_major(
                shape.to_vec(),
                Buffer::Backend(Arc::new(buffer)),
                webgpu_placement(rt),
            )?))
        }
    }
}

fn webgpu_placement(rt: &WebGpuRuntime) -> Placement {
    Placement {
        memory_kind: if rt.allocation_domain().is_some() {
            MemoryKind::Managed
        } else {
            MemoryKind::Device
        },
        device: Some(DeviceId {
            kind: DeviceKind::Gpu(GpuBackendKind::WebGpu),
            ordinal: rt.device_ordinal(),
        }),
        cpu_affinity: None,
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
#[derive(Clone)]
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
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] when no adapter/device is
    /// available, or [`crate::Error::BackendSource`] when CubeCL initialization
    /// fails.
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
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] when default adapter selection
    /// is unavailable, or [`crate::Error::BackendSource`] when initialization
    /// fails.
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
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::BackendSource`] when queue flush or
    /// synchronization fails, or [`crate::Error::RuntimeState`] when the
    /// runtime has lost its device state.
    pub fn synchronize(&self) -> crate::Result<()> {
        self.runtime.synchronize()
    }
}

fn unsupported_op(op: &'static str) -> crate::Error {
    crate::Error::unsupported(
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

    fn sub(&mut self, _lhs: &Tensor, _rhs: &Tensor) -> crate::Result<Tensor> {
        unsupported!("webgpu_sub")
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
    fn to_contiguous_read(&mut self, _input: TensorRead<'_>) -> crate::Result<Tensor> {
        unsupported!("WebGpuBackend::to_contiguous_read")
    }

    fn copy_read_into(&mut self, _src: TensorRead<'_>, _dst: TensorWrite<'_>) -> crate::Result<()> {
        unsupported!("WebGpuBackend::copy_read_into")
    }

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
