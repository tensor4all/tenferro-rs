#![doc(hidden)]

//! Internal CPU kernel and scratch-pool implementation crate.

pub type Result<T> = tenferro_tensor::Result<T>;
pub use tenferro_tensor::{CacheStats, Error, ErrorKind};

pub mod buffer_pool;
pub mod elementwise;

use num_complex::{Complex32, Complex64};
use strided_kernel::{col_major_strides as kernel_col_major_strides, StridedArray, StridedView};
#[cfg(test)]
use strided_kernel::{copy_into, Identity};
use tenferro_tensor::{
    validate::checked_shape_product, Buffer, DType, TensorRank, TypedTensor, TypedTensorView,
};
#[cfg(test)]
use tenferro_tensor::{Tensor, TensorRead, TensorView};

use crate::buffer_pool::{BufferPool, PoolScalar};

pub(crate) fn cpu_backend_buffer_error(op: &'static str) -> Error {
    Error::runtime_state(
        op,
        "CPU backend received backend buffer; download to host before CPU execution",
    )
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum CpuNumericalError {
    #[error("{op} detected division by zero for dtype {dtype:?}")]
    DivisionByZero { op: &'static str, dtype: DType },
}

pub(crate) fn cpu_division_by_zero(op: &'static str, dtype: DType) -> Error {
    Error::extension(
        op,
        "cpu",
        ErrorKind::NumericalFailure,
        CpuNumericalError::DivisionByZero { op, dtype },
    )
}

#[doc(hidden)]
pub trait ConjElem {
    fn conj_elem(self) -> Self;
}

impl ConjElem for f32 {
    fn conj_elem(self) -> Self {
        self
    }
}

impl ConjElem for f64 {
    fn conj_elem(self) -> Self {
        self
    }
}

impl ConjElem for Complex32 {
    fn conj_elem(self) -> Self {
        self.conj()
    }
}

impl ConjElem for Complex64 {
    fn conj_elem(self) -> Self {
        self.conj()
    }
}

pub(crate) fn typed_host_data<'a, T>(
    op: &'static str,
    tensor: &'a TypedTensor<T>,
) -> Result<&'a [T]> {
    match tensor.buffer() {
        Buffer::Host(data) => Ok(data.as_slice()),
        Buffer::Backend(_) => Err(cpu_backend_buffer_error(op)),
    }
}

pub(crate) fn typed_view<'a, T: Copy>(
    op: &'static str,
    tensor: &'a TypedTensor<T>,
) -> Result<StridedView<'a, T>> {
    match tensor.buffer() {
        Buffer::Host(data) => {
            let strides = kernel_col_major_strides(tensor.shape());
            StridedView::new(data.as_slice(), tensor.shape(), &strides, 0)
                .map_err(|err| Error::backend_source(op, err))
        }
        Buffer::Backend(_) => Err(cpu_backend_buffer_error(op)),
    }
}

pub(crate) fn typed_view_from_view<'a, T: Copy + 'static, R: TensorRank>(
    op: &'static str,
    view: &TypedTensorView<'a, T, R>,
) -> Result<StridedView<'a, T>> {
    if view.backend_buffer().is_some() {
        return Err(cpu_backend_buffer_error(op));
    }
    StridedView::new(
        view.host_storage()?,
        view.shape(),
        view.strides(),
        view.offset(),
    )
    .map_err(|err| Error::backend_source(op, err))
}

/// Create an output array from the CPU buffer pool WITHOUT initializing values.
///
/// # Safety
/// Caller must write every element before reading. The returned array contains
/// uninitialized or stale data acquired from `buffers`.
pub(crate) unsafe fn typed_array_uninit_from_pool<T>(
    buffers: &mut BufferPool,
    shape: &[usize],
) -> Result<StridedArray<T>>
where
    T: PoolScalar,
{
    let total = checked_shape_product("typed_array_uninit_from_pool", "shape", shape)?;
    let strides = kernel_col_major_strides(shape);
    // SAFETY: callers use this only for operation outputs that fully overwrite every element.
    let data = unsafe { T::pool_acquire(buffers, total) };
    // Invariant: callers pass validated tensor-derived or prechecked output
    // shapes, and `strides` is their compact column-major layout.
    StridedArray::from_parts(data, shape, &strides, 0)
        .map_err(|err| Error::backend_source("typed_array_uninit_from_pool", err))
}

pub(crate) fn tensor_from_array<T: Clone>(array: StridedArray<T>) -> TypedTensor<T> {
    // Invariant: `StridedArray` owns data whose length matches its validated dimensions.
    TypedTensor::from_vec_col_major(array.dims().to_vec(), array.into_data())
        .expect("strided array dimensions match owned data length")
}

#[cfg(test)]
pub(crate) fn materialize_tensor_read(
    buffers: &mut BufferPool,
    op: &'static str,
    input: TensorRead<'_>,
) -> Result<Tensor> {
    match input {
        TensorRead::Tensor(tensor) => clone_host_tensor_read(op, tensor),
        TensorRead::View(view) => materialize_tensor_view(buffers, op, view),
    }
}

#[cfg(test)]
fn clone_host_tensor_read(op: &'static str, tensor: &Tensor) -> Result<Tensor> {
    macro_rules! clone_host {
        ($variant:ident, $tensor:expr) => {{
            typed_host_data(op, $tensor)?;
            Ok(Tensor::$variant($tensor.clone()))
        }};
    }

    match tensor {
        Tensor::F32(tensor) => clone_host!(F32, tensor),
        Tensor::F64(tensor) => clone_host!(F64, tensor),
        Tensor::I32(tensor) => clone_host!(I32, tensor),
        Tensor::I64(tensor) => clone_host!(I64, tensor),
        Tensor::Bool(tensor) => clone_host!(Bool, tensor),
        Tensor::C32(tensor) => clone_host!(C32, tensor),
        Tensor::C64(tensor) => clone_host!(C64, tensor),
    }
}

#[cfg(test)]
fn materialize_tensor_view(
    buffers: &mut BufferPool,
    op: &'static str,
    view: TensorView<'_>,
) -> Result<Tensor> {
    macro_rules! materialize {
        ($variant:ident, $view:expr) => {{
            Ok(Tensor::$variant(typed_materialize_view_for_tests(
                buffers, &$view, op,
            )?))
        }};
    }

    match view {
        TensorView::F32(view) => materialize!(F32, view),
        TensorView::F64(view) => materialize!(F64, view),
        TensorView::I32(view) => materialize!(I32, view),
        TensorView::I64(view) => materialize!(I64, view),
        TensorView::Bool(view) => materialize!(Bool, view),
        TensorView::C32(view) => materialize!(C32, view),
        TensorView::C64(view) => materialize!(C64, view),
    }
}

#[cfg(test)]
fn typed_materialize_view_for_tests<T, R>(
    buffers: &mut BufferPool,
    view: &TypedTensorView<'_, T, R>,
    op: &'static str,
) -> Result<TypedTensor<T, R>>
where
    T: Copy + Clone + PoolScalar + 'static,
    R: TensorRank,
{
    if view.backend_buffer().is_some() {
        return Err(cpu_backend_buffer_error(op));
    }
    let src: StridedView<'_, T, Identity> = StridedView::new(
        view.host_storage()?,
        view.shape(),
        view.strides(),
        view.offset(),
    )
    .map_err(|err| Error::backend_source(op, err))?;
    // SAFETY: copy_into overwrites every logical output element.
    let mut out = unsafe { typed_array_uninit_from_pool(buffers, view.shape()) }?;
    copy_into(&mut out.view_mut(), &src).map_err(|err| Error::backend_source(op, err))?;
    let shape = R::shape_from_vec(view.shape().to_vec().into())
        .map_err(|err| Error::backend_source(op, err))?;
    TypedTensor::from_buffer_col_major(
        shape,
        Buffer::Host(out.into_data()),
        view.placement().clone(),
    )
}
