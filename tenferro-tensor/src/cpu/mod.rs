use std::sync::Arc;

pub mod analytic;
pub mod backend;
pub mod elementwise;
pub mod gemm;
pub mod indexing;
pub mod linalg;
pub mod reduction;
pub mod structural;

use strided_kernel::{col_major_strides, StridedArray, StridedView};

use crate::{Buffer, TypedTensor};

pub use backend::CpuBackend;
pub use elementwise::{
    abs, add, clamp, compare, conj, div, maximum, minimum, mul, neg, select, sign,
};
pub use indexing::{dynamic_slice, gather, pad, scatter};
pub use reduction::{reduce_max, reduce_min, reduce_prod, reduce_sum};
pub use structural::{
    broadcast_in_dim, convert, embed_diagonal, extract_diagonal, reshape, transpose, tril, triu,
};

pub(crate) fn typed_view<T: Copy>(tensor: &TypedTensor<T>) -> StridedView<'_, T> {
    match &tensor.buffer {
        Buffer::Host(data) => StridedView::new(data, &tensor.shape, &tensor.strides, tensor.offset)
            .expect("valid host tensor view"),
        Buffer::Backend(_) => todo!("typed_view for backend buffers"),
    }
}

pub(crate) fn typed_array<T: Clone>(shape: &[usize], fill: T) -> StridedArray<T> {
    let total: usize = shape.iter().product();
    let strides = col_major_strides(shape);
    StridedArray::from_parts(vec![fill; total], shape, &strides, 0)
        .expect("column-major output array")
}

pub(crate) fn tensor_from_array<T: Clone>(array: StridedArray<T>) -> TypedTensor<T> {
    TypedTensor {
        buffer: Buffer::Host(Arc::new(array.data().to_vec())),
        shape: array.dims().to_vec(),
        strides: array.strides().to_vec(),
        offset: array.view().offset(),
        placement: crate::types::default_placement(),
    }
}
