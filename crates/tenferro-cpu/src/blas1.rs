use std::ops::{Add, Mul};
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};

use num_complex::{Complex32, Complex64};
use rayon::prelude::*;
use tenferro_tensor::{
    ContractionScalar, DType, MemoryKind, Tensor, TensorRead, TensorViewMut, TensorWrite,
};

use crate::buffer_pool::BufferPool;
use crate::provider::CpuExecutionContext;

#[cfg(test)]
static MATERIALIZATIONS: AtomicUsize = AtomicUsize::new(0);

#[cfg(test)]
pub(crate) fn reset_materializations_for_test() {
    MATERIALIZATIONS.store(0, Ordering::SeqCst);
}

#[cfg(test)]
pub(crate) fn materializations_for_test() -> usize {
    MATERIALIZATIONS.load(Ordering::SeqCst)
}

pub(crate) fn validate_cpu_read(op: &'static str, input: &TensorRead<'_>) -> crate::Result<()> {
    if input.backend_family().is_some() || input.placement().memory_kind == MemoryKind::Device {
        Err(crate::cpu_backend_buffer_error(op))
    } else {
        Ok(())
    }
}

pub(crate) fn axpby_read_into_accum(
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    alpha: ContractionScalar,
    x: TensorRead<'_>,
    beta: ContractionScalar,
    y: TensorWrite<'_>,
) -> crate::Result<()> {
    validate_cpu_read("BackendSession::axpby_read_into_accum", &x)?;
    validate_cpu_read("BackendSession::axpby_read_into_accum", &y.as_read())?;
    let mut materialized_x = None;
    if !x.is_col_major_contiguous()? {
        #[cfg(test)]
        MATERIALIZATIONS.fetch_add(1, Ordering::SeqCst);
        materialized_x = Some(crate::materialize_tensor_read(
            buffers,
            "BackendSession::axpby_read_into_accum",
            x.clone(),
        )?);
    }
    let x = materialized_x
        .as_ref()
        .map(TensorRead::from_tensor)
        .unwrap_or(x);

    match (x.dtype(), alpha, beta) {
        (DType::F32, ContractionScalar::F32(alpha), ContractionScalar::F32(beta)) => {
            let x_data = x.as_slice::<f32>()?;
            axpby_write_f32(context, x_data, y, alpha, beta)
        }
        (DType::F64, ContractionScalar::F64(alpha), ContractionScalar::F64(beta)) => {
            let x_data = x.as_slice::<f64>()?;
            axpby_write_f64(context, x_data, y, alpha, beta)
        }
        (DType::C32, ContractionScalar::C32(alpha), ContractionScalar::C32(beta)) => {
            let x_data = x.as_slice::<Complex32>()?;
            axpby_write_c32(context, x_data, y, alpha, beta)
        }
        (DType::C64, ContractionScalar::C64(alpha), ContractionScalar::C64(beta)) => {
            let x_data = x.as_slice::<Complex64>()?;
            axpby_write_c64(context, x_data, y, alpha, beta)
        }
        // INVARIANT: shared validation proves x, y, alpha, and beta have one
        // identical supported dtype before execution dispatch.
        _ => Err(crate::Error::Internal(
            "validated AXPBY dtype changed before execution".into(),
        )),
    }
}

fn axpby_typed<T>(
    context: &CpuExecutionContext<'_>,
    x_data: &[T],
    y_data: &mut [T],
    alpha: T,
    beta: T,
) -> crate::Result<()>
where
    T: Copy + Send + Sync + Add<Output = T> + Mul<Output = T>,
{
    debug_assert_eq!(x_data.len(), y_data.len());
    let len = y_data.len();

    // INVARIANT: shared validation proves compact, injective y and conservative
    // x/y non-overlap before this single read-modify-write pass begins. The
    // accepted tenferro-owned exception is necessary because strided-rs has no
    // primitive for reading and updating one caller-owned destination.
    if context.thread_budget().get() > 1 && len > 1 {
        let chunks = context.thread_budget().get();
        let chunk_len = len.div_ceil(chunks).max(1);
        y_data
            .par_chunks_mut(chunk_len)
            .zip(x_data.par_chunks(chunk_len))
            .for_each(|(ys, xs)| {
                for (dst, src) in ys.iter_mut().zip(xs.iter()) {
                    *dst = alpha * *src + beta * *dst;
                }
            });
    } else {
        for (dst, src) in y_data.iter_mut().zip(x_data.iter()) {
            *dst = alpha * *src + beta * *dst;
        }
    }
    Ok(())
}

macro_rules! axpby_write {
    ($name:ident, $variant:ident, $ty:ty) => {
        fn $name(
            context: &CpuExecutionContext<'_>,
            x_data: &[$ty],
            y: TensorWrite<'_>,
            alpha: $ty,
            beta: $ty,
        ) -> crate::Result<()> {
            match y {
                TensorWrite::Tensor(Tensor::$variant(tensor)) => {
                    axpby_typed(context, x_data, tensor.host_data_mut()?, alpha, beta)
                }
                TensorWrite::View(TensorViewMut::$variant(mut view)) => {
                    let len = x_data.len();
                    if len == 0 {
                        return axpby_typed(context, x_data, &mut [], alpha, beta);
                    }
                    // INVARIANT: TensorViewMut construction proves reachable
                    // bounds and injectivity; shared AXPBY validation proves a
                    // compact layout with the same shape as x. A non-empty
                    // compact view therefore has a nonnegative offset and one
                    // contiguous in-bounds region of exactly `len` elements.
                    let offset = view.offset() as usize;
                    let storage = view.host_storage_mut()?;
                    // SAFETY: the invariant above proves `offset..offset+len`
                    // is within `storage` and uniquely writable.
                    let y_data = unsafe {
                        std::slice::from_raw_parts_mut(storage.as_mut_ptr().add(offset), len)
                    };
                    axpby_typed(context, x_data, y_data, alpha, beta)
                }
                // INVARIANT: shared validation proves y has the dtype selected
                // by the outer dispatch before this helper is called.
                _ => Err(crate::Error::Internal(
                    "validated AXPBY output dtype changed before execution".into(),
                )),
            }
        }
    };
}

axpby_write!(axpby_write_f32, F32, f32);
axpby_write!(axpby_write_f64, F64, f64);
axpby_write!(axpby_write_c32, C32, Complex32);
axpby_write!(axpby_write_c64, C64, Complex64);
