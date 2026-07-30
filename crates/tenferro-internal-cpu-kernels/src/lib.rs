#![doc(hidden)]

//! Internal CPU kernel and scratch-pool implementation crate.

#[cfg(test)]
use std::mem::MaybeUninit;

pub type Result<T> = tenferro_tensor::Result<T>;
pub use tenferro_tensor::{CacheStats, Error, ErrorKind};

pub mod buffer_pool;
mod pooled_uninit_output;
/// Canonical pooled full-overwrite owner shared with the sibling `tenferro-cpu`
/// crate. This is the minimum cross-crate surface required for one ownership
/// contract; `pub(crate)` cannot cross that crate boundary. Construction only
/// exposes `MaybeUninit` storage, with initialization completed by an explicit
/// unsafe handoff.
pub use pooled_uninit_output::PooledUninitOutput;
pub mod elementwise;

use num_complex::{Complex32, Complex64};
use strided_kernel::{col_major_strides as kernel_col_major_strides, StridedView};
#[cfg(test)]
use strided_kernel::{map_into, Identity};
use tenferro_tensor::{Buffer, DType, TensorRank, TypedTensor, TypedTensorView};
#[cfg(test)]
use tenferro_tensor::{Tensor, TensorRead, TensorView};

#[cfg(test)]
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
    let mut out = PooledUninitOutput::<T>::new(buffers, view.shape().to_vec())?;
    map_into(&mut out.as_uninit_view_mut()?, &src, |x| {
        MaybeUninit::new(x)
    })
    .map_err(|err| Error::backend_source(op, err))?;
    // SAFETY: the successful materialize map replay writes every logical destination element and retains no destination view.
    let out = unsafe { out.assume_init_as::<R>()? };
    let shape = R::shape_from_vec(view.shape().to_vec().into())
        .map_err(|err| Error::backend_source(op, err))?;
    TypedTensor::from_buffer_col_major(
        shape,
        Buffer::Host(out.into_vec_col_major()?.1),
        view.placement().clone(),
    )
}
