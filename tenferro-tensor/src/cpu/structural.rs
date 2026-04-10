use num_complex::{Complex32, Complex64};
use num_traits::Zero;
use strided_kernel::{copy_into, map_into, Identity, StridedView};

use crate::{
    types::{
        col_major_strides, flat_to_multi, row_major_strides, LayoutOrder, Tensor, TypedTensor,
    },
    DType,
};

use super::{tensor_from_array, typed_array, typed_view};

fn backend_failure(op: &'static str, err: impl ToString) -> crate::Error {
    crate::Error::BackendFailure {
        op,
        message: err.to_string(),
    }
}

fn validate_rank(op: &'static str, expected: usize, actual: usize) -> crate::Result<()> {
    if expected != actual {
        return Err(crate::Error::RankMismatch {
            op,
            expected,
            actual,
        });
    }
    Ok(())
}

fn validate_axis(op: &'static str, axis: usize, rank: usize) -> crate::Result<()> {
    if axis >= rank {
        return Err(crate::Error::AxisOutOfBounds { op, axis, rank });
    }
    Ok(())
}

fn validate_axes_distinct(op: &'static str, axis_a: usize, axis_b: usize) -> crate::Result<()> {
    if axis_a == axis_b {
        return Err(crate::Error::DuplicateAxis {
            op,
            axis: axis_a,
            role: "axes",
        });
    }
    Ok(())
}

fn validate_permutation(op: &'static str, perm: &[usize], rank: usize) -> crate::Result<()> {
    validate_rank(op, rank, perm.len())?;
    let mut seen = vec![false; rank];
    for &axis in perm {
        validate_axis(op, axis, rank)?;
        if seen[axis] {
            return Err(crate::Error::DuplicateAxis {
                op,
                axis,
                role: "perm",
            });
        }
        seen[axis] = true;
    }
    Ok(())
}

fn host_view<T: Copy>(tensor: &TypedTensor<T>) -> crate::Result<StridedView<'_, T, Identity>> {
    match &tensor.buffer {
        crate::Buffer::Host(data) => {
            StridedView::new(data, &tensor.shape, &tensor.strides, tensor.offset)
                .map_err(|err| backend_failure("structural", err))
        }
        crate::Buffer::Backend(_) => Err(crate::Error::BackendFailure {
            op: "structural",
            message: "backend buffers are not supported for structural CPU helpers".into(),
        }),
    }
}

pub fn transpose(input: &Tensor, perm: &[usize]) -> crate::Result<Tensor> {
    match input {
        Tensor::F32(t) => Ok(Tensor::F32(typed_transpose(t, perm)?)),
        Tensor::F64(t) => Ok(Tensor::F64(typed_transpose(t, perm)?)),
        Tensor::C32(t) => Ok(Tensor::C32(typed_transpose(t, perm)?)),
        Tensor::C64(t) => Ok(Tensor::C64(typed_transpose(t, perm)?)),
    }
}

pub fn reshape(input: &Tensor, shape: &[usize]) -> crate::Result<Tensor> {
    match input {
        Tensor::F32(t) => Ok(Tensor::F32(typed_reshape(t, shape)?)),
        Tensor::F64(t) => Ok(Tensor::F64(typed_reshape(t, shape)?)),
        Tensor::C32(t) => Ok(Tensor::C32(typed_reshape(t, shape)?)),
        Tensor::C64(t) => Ok(Tensor::C64(typed_reshape(t, shape)?)),
    }
}

pub fn broadcast_in_dim(input: &Tensor, shape: &[usize], dims: &[usize]) -> crate::Result<Tensor> {
    match input {
        Tensor::F32(t) => Ok(Tensor::F32(typed_broadcast_in_dim(t, shape, dims)?)),
        Tensor::F64(t) => Ok(Tensor::F64(typed_broadcast_in_dim(t, shape, dims)?)),
        Tensor::C32(t) => Ok(Tensor::C32(typed_broadcast_in_dim(t, shape, dims)?)),
        Tensor::C64(t) => Ok(Tensor::C64(typed_broadcast_in_dim(t, shape, dims)?)),
    }
}

pub fn convert(input: &Tensor, to: DType) -> Tensor {
    match (input, to) {
        (Tensor::F32(t), DType::F32) => Tensor::F32(t.clone()),
        (Tensor::F32(t), DType::F64) => Tensor::F64(typed_convert(t, |x| x as f64)),
        (Tensor::F32(t), DType::C32) => Tensor::C32(typed_convert(t, |x| Complex32::new(x, 0.0))),
        (Tensor::F32(t), DType::C64) => {
            Tensor::C64(typed_convert(t, |x| Complex64::new(x as f64, 0.0)))
        }
        (Tensor::F64(t), DType::F32) => Tensor::F32(typed_convert(t, |x| x as f32)),
        (Tensor::F64(t), DType::F64) => Tensor::F64(t.clone()),
        (Tensor::F64(t), DType::C32) => {
            Tensor::C32(typed_convert(t, |x| Complex32::new(x as f32, 0.0)))
        }
        (Tensor::F64(t), DType::C64) => Tensor::C64(typed_convert(t, |x| Complex64::new(x, 0.0))),
        (Tensor::C32(t), DType::F32) => Tensor::F32(typed_convert(t, |z| z.re)),
        (Tensor::C32(t), DType::F64) => Tensor::F64(typed_convert(t, |z| z.re as f64)),
        (Tensor::C32(t), DType::C32) => Tensor::C32(t.clone()),
        (Tensor::C32(t), DType::C64) => Tensor::C64(typed_convert(t, |z| {
            Complex64::new(z.re as f64, z.im as f64)
        })),
        (Tensor::C64(t), DType::F32) => Tensor::F32(typed_convert(t, |z| z.re as f32)),
        (Tensor::C64(t), DType::F64) => Tensor::F64(typed_convert(t, |z| z.re)),
        (Tensor::C64(t), DType::C32) => Tensor::C32(typed_convert(t, |z| {
            Complex32::new(z.re as f32, z.im as f32)
        })),
        (Tensor::C64(t), DType::C64) => Tensor::C64(t.clone()),
    }
}

pub fn extract_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> crate::Result<Tensor> {
    match input {
        Tensor::F32(t) => Ok(Tensor::F32(typed_extract_diagonal(t, axis_a, axis_b)?)),
        Tensor::F64(t) => Ok(Tensor::F64(typed_extract_diagonal(t, axis_a, axis_b)?)),
        Tensor::C32(t) => Ok(Tensor::C32(typed_extract_diagonal(t, axis_a, axis_b)?)),
        Tensor::C64(t) => Ok(Tensor::C64(typed_extract_diagonal(t, axis_a, axis_b)?)),
    }
}

pub fn embed_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> crate::Result<Tensor> {
    match input {
        Tensor::F32(t) => Ok(Tensor::F32(typed_embed_diagonal(t, axis_a, axis_b)?)),
        Tensor::F64(t) => Ok(Tensor::F64(typed_embed_diagonal(t, axis_a, axis_b)?)),
        Tensor::C32(t) => Ok(Tensor::C32(typed_embed_diagonal(t, axis_a, axis_b)?)),
        Tensor::C64(t) => Ok(Tensor::C64(typed_embed_diagonal(t, axis_a, axis_b)?)),
    }
}

pub fn tril(input: &Tensor, k: i64) -> crate::Result<Tensor> {
    match input {
        Tensor::F32(t) => Ok(Tensor::F32(typed_tril(t, k)?)),
        Tensor::F64(t) => Ok(Tensor::F64(typed_tril(t, k)?)),
        Tensor::C32(t) => Ok(Tensor::C32(typed_tril(t, k)?)),
        Tensor::C64(t) => Ok(Tensor::C64(typed_tril(t, k)?)),
    }
}

pub fn triu(input: &Tensor, k: i64) -> crate::Result<Tensor> {
    match input {
        Tensor::F32(t) => Ok(Tensor::F32(typed_triu(t, k)?)),
        Tensor::F64(t) => Ok(Tensor::F64(typed_triu(t, k)?)),
        Tensor::C32(t) => Ok(Tensor::C32(typed_triu(t, k)?)),
        Tensor::C64(t) => Ok(Tensor::C64(typed_triu(t, k)?)),
    }
}

fn permute_axes<T: Clone>(tensor: &TypedTensor<T>, perm: &[usize]) -> TypedTensor<T> {
    let shape = perm.iter().map(|&axis| tensor.shape[axis]).collect();
    let strides = perm.iter().map(|&axis| tensor.strides[axis]).collect();
    TypedTensor {
        buffer: tensor.buffer.clone(),
        shape,
        strides,
        offset: tensor.offset,
        placement: tensor.placement.clone(),
    }
}

fn reshape_strides(tensor: &TypedTensor<impl Clone>, shape: &[usize]) -> crate::Result<Vec<isize>> {
    if tensor.strides.iter().all(|&stride| stride == 0) {
        return Ok(vec![0; shape.len()]);
    }
    if tensor.strides == col_major_strides(&tensor.shape) {
        return Ok(col_major_strides(shape));
    }
    if tensor.strides == row_major_strides(&tensor.shape) {
        return Ok(row_major_strides(shape));
    }
    if let Some(strides) = reshape_singleton_only_strides(tensor, shape) {
        return Ok(strides);
    }
    Err(crate::Error::BackendFailure {
        op: "reshape",
        message: "reshape requires contiguous tensor".into(),
    })
}

fn reshape_singleton_only_strides(
    tensor: &TypedTensor<impl Clone>,
    new_shape: &[usize],
) -> Option<Vec<isize>> {
    let old_non_singleton = tensor
        .shape
        .iter()
        .copied()
        .zip(tensor.strides.iter().copied())
        .filter(|(dim, _)| *dim != 1)
        .collect::<Vec<_>>();
    let new_non_singleton = new_shape
        .iter()
        .enumerate()
        .filter_map(|(idx, &dim)| (dim != 1).then_some((idx, dim)))
        .collect::<Vec<_>>();
    if old_non_singleton.len() != new_non_singleton.len() {
        return None;
    }
    if old_non_singleton
        .iter()
        .map(|(dim, _)| *dim)
        .ne(new_non_singleton.iter().map(|(_, dim)| *dim))
    {
        return None;
    }

    let mut new_strides = vec![0; new_shape.len()];
    for ((_, stride), (new_idx, _)) in old_non_singleton.iter().zip(new_non_singleton.iter()) {
        new_strides[*new_idx] = *stride;
    }
    Some(new_strides)
}

pub fn typed_transpose<T: Clone>(
    tensor: &TypedTensor<T>,
    perm: &[usize],
) -> crate::Result<TypedTensor<T>> {
    validate_permutation("transpose", perm, tensor.shape.len())?;
    Ok(permute_axes(tensor, perm))
}

pub fn typed_reshape<T: Clone>(
    tensor: &TypedTensor<T>,
    shape: &[usize],
) -> crate::Result<TypedTensor<T>> {
    let old_n: usize = tensor.shape.iter().product();
    let new_n: usize = shape.iter().product();
    if old_n != new_n {
        return Err(crate::Error::ShapeMismatch {
            op: "reshape",
            lhs: tensor.shape.clone(),
            rhs: shape.to_vec(),
        });
    }
    if let Ok(strides) = reshape_strides(tensor, shape) {
        return Ok(TypedTensor {
            buffer: tensor.buffer.clone(),
            shape: shape.to_vec(),
            strides,
            offset: tensor.offset,
            placement: tensor.placement.clone(),
        });
    }

    let mut data = Vec::with_capacity(old_n);
    let mut idx = vec![0usize; tensor.shape.len()];
    for flat in 0..old_n {
        flat_to_multi(flat, &tensor.shape, &mut idx);
        data.push(tensor.get(&idx).clone());
    }

    let mut dense = TypedTensor::from_vec(shape.to_vec(), data);
    dense.placement = tensor.placement.clone();
    Ok(dense)
}

pub fn typed_broadcast_in_dim<T: Clone>(
    tensor: &TypedTensor<T>,
    shape: &[usize],
    dims: &[usize],
) -> crate::Result<TypedTensor<T>> {
    validate_rank("broadcast_in_dim", tensor.shape.len(), dims.len())?;
    let mut seen = vec![false; shape.len()];
    let mut out_strides = vec![0isize; shape.len()];
    for (src_axis, &dst_axis) in dims.iter().enumerate() {
        validate_axis("broadcast_in_dim", dst_axis, shape.len())?;
        if seen[dst_axis] {
            return Err(crate::Error::DuplicateAxis {
                op: "broadcast_in_dim",
                axis: dst_axis,
                role: "dims",
            });
        }
        seen[dst_axis] = true;
        let source_dim = tensor.shape[src_axis];
        let target_dim = shape[dst_axis];
        if source_dim != target_dim && source_dim != 1 {
            return Err(crate::Error::ShapeMismatch {
                op: "broadcast_in_dim",
                lhs: tensor.shape.clone(),
                rhs: shape.to_vec(),
            });
        }
        out_strides[dst_axis] = if source_dim == 1 && target_dim > 1 {
            0
        } else {
            tensor.strides[src_axis]
        };
    }
    Ok(TypedTensor {
        buffer: tensor.buffer.clone(),
        shape: shape.to_vec(),
        strides: out_strides,
        offset: tensor.offset,
        placement: tensor.placement.clone(),
    })
}

fn typed_convert<S, T>(tensor: &TypedTensor<S>, f: impl Fn(S) -> T) -> TypedTensor<T>
where
    S: Copy,
    T: Copy + Clone + Zero,
{
    let mut out = typed_array(&tensor.shape, T::zero());
    map_into(&mut out.view_mut(), &typed_view(tensor), f).expect("typed_convert");
    tensor_from_array(out)
}

pub fn typed_extract_diagonal<T: Copy + Zero + Clone>(
    tensor: &TypedTensor<T>,
    axis_a: usize,
    axis_b: usize,
) -> crate::Result<TypedTensor<T>> {
    validate_axis("extract_diagonal", axis_a, tensor.shape.len())?;
    validate_axis("extract_diagonal", axis_b, tensor.shape.len())?;
    validate_axes_distinct("extract_diagonal", axis_a, axis_b)?;

    let diag = host_view(tensor)?
        .diagonal_view(&[(axis_a, axis_b)])
        .map_err(|err| backend_failure("extract_diagonal", err))?;
    let mut out = typed_array(diag.dims(), T::zero());
    copy_into(&mut out.view_mut(), &diag)
        .map_err(|err| backend_failure("extract_diagonal", err))?;
    Ok(tensor_from_array(out))
}

pub fn typed_embed_diagonal<T: Copy + Zero + Clone>(
    tensor: &TypedTensor<T>,
    axis_a: usize,
    axis_b: usize,
) -> crate::Result<TypedTensor<T>> {
    validate_axis("embed_diagonal", axis_a, tensor.shape.len())?;
    if axis_b > tensor.shape.len() {
        return Err(crate::Error::AxisOutOfBounds {
            op: "embed_diagonal",
            axis: axis_b,
            rank: tensor.shape.len(),
        });
    }

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
    Ok(out)
}

pub fn typed_tril<T: Copy + Zero + Clone + Default>(
    tensor: &TypedTensor<T>,
    k: i64,
) -> crate::Result<TypedTensor<T>> {
    typed_triangular_mask(tensor, k, false)
}

pub fn typed_triu<T: Copy + Zero + Clone + Default>(
    tensor: &TypedTensor<T>,
    k: i64,
) -> crate::Result<TypedTensor<T>> {
    typed_triangular_mask(tensor, k, true)
}

fn typed_triangular_mask<T: Copy + Zero + Clone + Default>(
    tensor: &TypedTensor<T>,
    k: i64,
    upper: bool,
) -> crate::Result<TypedTensor<T>> {
    if tensor.shape.len() < 2 {
        return Err(crate::Error::RankMismatch {
            op: if upper { "triu" } else { "tril" },
            expected: 2,
            actual: tensor.shape.len(),
        });
    }

    let rows = tensor.shape[0];
    let cols = tensor.shape[1];
    if tensor.shape.contains(&0) {
        return Ok(tensor.clone());
    }

    let batch_count: usize = tensor.shape[2..].iter().product();
    let block_size = rows * cols;
    let mut out = tensor.to_contiguous(LayoutOrder::ColumnMajor)?;
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

    Ok(out)
}
