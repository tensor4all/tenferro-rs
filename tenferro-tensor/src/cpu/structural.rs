use strided_kernel::{copy_into, StridedView};

use num_traits::Zero;

use crate::types::{dispatch_tensor, flat_to_multi, Tensor, TypedTensor};

use super::{tensor_from_array, typed_array, typed_view};

pub fn transpose(input: &Tensor, perm: &[usize]) -> Tensor {
    dispatch_tensor!(input, t => typed_transpose(t, perm))
}

pub fn reshape(input: &Tensor, shape: &[usize]) -> Tensor {
    dispatch_tensor!(input, t => typed_reshape(t, shape))
}

pub fn broadcast_in_dim(input: &Tensor, shape: &[usize], dims: &[usize]) -> Tensor {
    dispatch_tensor!(input, t => typed_broadcast_in_dim(t, shape, dims))
}

pub fn extract_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> Tensor {
    dispatch_tensor!(input, t => typed_extract_diagonal(t, axis_a, axis_b))
}

pub fn embed_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> Tensor {
    dispatch_tensor!(input, t => typed_embed_diagonal(t, axis_a, axis_b))
}

pub fn tril(input: &Tensor, k: i64) -> Tensor {
    dispatch_tensor!(input, t => typed_tril(t, k))
}

pub fn triu(input: &Tensor, k: i64) -> Tensor {
    dispatch_tensor!(input, t => typed_triu(t, k))
}

pub fn typed_transpose<T: Copy + Zero + Clone>(
    tensor: &TypedTensor<T>,
    perm: &[usize],
) -> TypedTensor<T> {
    let src = typed_view(tensor);
    let permuted = src.permute(perm).expect("transpose perm");
    let mut out = typed_array(permuted.dims(), T::zero());
    copy_into(&mut out.view_mut(), &permuted).expect("transpose copy");
    tensor_from_array(out)
}

pub fn typed_reshape<T: Clone>(tensor: &TypedTensor<T>, shape: &[usize]) -> TypedTensor<T> {
    let old_n: usize = tensor.shape.iter().product();
    let new_n: usize = shape.iter().product();
    assert_eq!(old_n, new_n, "reshape: element count mismatch");
    TypedTensor {
        buffer: tensor.buffer.clone(),
        shape: shape.to_vec(),
        placement: tensor.placement.clone(),
    }
}

pub fn typed_broadcast_in_dim<T: Copy + Zero + Clone>(
    tensor: &TypedTensor<T>,
    shape: &[usize],
    dims: &[usize],
) -> TypedTensor<T> {
    assert_eq!(
        dims.len(),
        tensor.shape.len(),
        "broadcast_in_dim rank mismatch"
    );
    let mut base_dims = vec![1usize; shape.len()];
    let mut base_strides = vec![0isize; shape.len()];
    let source_strides = crate::col_major_strides(&tensor.shape);
    for (src_axis, &dst_axis) in dims.iter().enumerate() {
        base_dims[dst_axis] = tensor.shape[src_axis];
        base_strides[dst_axis] = source_strides[src_axis];
    }
    let base: StridedView<'_, T> = match &tensor.buffer {
        crate::Buffer::Host(data) => {
            StridedView::new(data, &base_dims, &base_strides, 0).expect("broadcast base view")
        }
        crate::Buffer::Backend(_) => todo!("broadcast_in_dim for backend buffers"),
    };
    let broadcast = base.broadcast(shape).expect("broadcast shape");
    let mut out = typed_array(shape, T::zero());
    copy_into(&mut out.view_mut(), &broadcast).expect("broadcast copy");
    tensor_from_array(out)
}

pub fn typed_extract_diagonal<T: Copy + Zero + Clone>(
    tensor: &TypedTensor<T>,
    axis_a: usize,
    axis_b: usize,
) -> TypedTensor<T> {
    let diag = typed_view(tensor)
        .diagonal_view(&[(axis_a, axis_b)])
        .expect("diagonal_view");
    let mut out = typed_array(diag.dims(), T::zero());
    copy_into(&mut out.view_mut(), &diag).expect("extract_diagonal copy");
    tensor_from_array(out)
}

pub fn typed_embed_diagonal<T: Copy + Zero + Clone>(
    tensor: &TypedTensor<T>,
    axis_a: usize,
    axis_b: usize,
) -> TypedTensor<T> {
    assert!(
        axis_a < tensor.shape.len(),
        "embed_diagonal: axis_a out of range"
    );
    assert!(
        axis_b <= tensor.shape.len(),
        "embed_diagonal: axis_b must be <= rank"
    );

    let n = tensor.shape[axis_a];
    let mut out_shape = tensor.shape.clone();
    out_shape.insert(axis_b, n);
    let mut out = TypedTensor::zeros(out_shape);

    let in_rank = tensor.shape.len();
    let out_rank = out.shape.len();
    let mut in_idx = vec![0usize; in_rank];
    let mut out_idx = vec![0usize; out_rank];

    for flat in 0..tensor.n_elements() {
        flat_to_multi(flat, &tensor.shape, &mut in_idx);
        let diag_val = in_idx[axis_a];
        let mut src_axis = 0usize;
        for out_axis in 0..out_rank {
            if out_axis == axis_b {
                out_idx[out_axis] = diag_val;
            } else {
                out_idx[out_axis] = in_idx[src_axis];
                src_axis += 1;
            }
        }
        *out.get_mut(&out_idx) = *tensor.get(&in_idx);
    }
    out
}

pub fn typed_tril<T: Copy + Zero + Clone>(tensor: &TypedTensor<T>, k: i64) -> TypedTensor<T> {
    typed_triangular_mask(tensor, k, false)
}

pub fn typed_triu<T: Copy + Zero + Clone>(tensor: &TypedTensor<T>, k: i64) -> TypedTensor<T> {
    typed_triangular_mask(tensor, k, true)
}

fn typed_triangular_mask<T: Copy + Zero + Clone>(
    tensor: &TypedTensor<T>,
    k: i64,
    upper: bool,
) -> TypedTensor<T> {
    assert!(
        tensor.shape.len() >= 2,
        "triangular mask: expected rank >= 2, got {}",
        tensor.shape.len()
    );

    let rows = tensor.shape[0];
    let cols = tensor.shape[1];
    if tensor.shape.contains(&0) {
        return tensor.clone();
    }

    let batch_count: usize = tensor.shape[2..].iter().product();
    let block_size = rows * cols;
    let mut out = tensor.clone();
    let data = out.host_data_mut();

    for batch_idx in 0..batch_count {
        let base = batch_idx * block_size;
        for col in 0..cols {
            let boundary = col as i64 - k;
            for row in 0..rows {
                let keep = if upper {
                    (row as i64) <= boundary
                } else {
                    (row as i64) >= boundary
                };
                if !keep {
                    data[base + row + col * rows] = T::zero();
                }
            }
        }
    }

    out
}
