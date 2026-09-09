#![doc(hidden)]

//! Internal ordinary CPU kernel implementations.
//!
//! Shared resource ownership is implemented by `tenferro-cpu-basic`; this crate
//! owns the ordinary dtype-dispatch kernel family.

pub type Result<T> = tenferro_tensor::Result<T>;
pub use tenferro_cpu_basic::{
    cpu_backend_buffer_error, cpu_division_by_zero, typed_host_data, typed_view,
    typed_view_from_view, ConjElem, CpuNumericalError,
};
pub use tenferro_tensor::{CacheStats, DType, Error, ErrorKind};

pub mod elementwise;
pub mod read_into;
pub use read_into::elementwise_read_into_with_context;

#[cfg(test)]
use std::mem::MaybeUninit;
#[cfg(test)]
use strided_kernel::{map_into, Identity, StridedView};
#[cfg(test)]
use tenferro_cpu_basic::{BufferPool, PoolScalar, PooledUninitOutput};
#[cfg(test)]
use tenferro_tensor::{Tensor, TensorRank, TensorRead, TensorView, TypedTensor, TypedTensorView};

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
            Ok(Tensor::$variant($tensor.duplicate()?))
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
    // SAFETY: the successful map replay writes every logical destination element.
    let out = unsafe { out.assume_init_as::<R>()? };
    let shape = R::shape_from_vec(view.shape().to_vec().into())
        .map_err(|err| Error::backend_source(op, err))?;
    let mut tensor = TypedTensor::from_vec_col_major(shape, out.into_vec_col_major()?.1)?;
    tensor.set_placement(view.placement().clone());
    Ok(tensor)
}
