use num_complex::{Complex32, Complex64};
use num_traits::Zero;
use strided_kernel::{copy_into, map_into, Identity, StridedView};

use crate::{
    types::{flat_to_multi, Tensor, TypedTensor},
    DType,
};

use super::{tensor_from_array, typed_array_uninit, typed_view};

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
            let strides = crate::col_major_strides(&tensor.shape);
            StridedView::new(data, &tensor.shape, &strides, 0)
                .map_err(|err| backend_failure("structural", err))
        }
        crate::Buffer::Backend(_) => Err(crate::Error::BackendFailure {
            op: "structural",
            message: "backend buffers are not supported for structural CPU helpers".into(),
        }),
        #[cfg(feature = "cubecl")]
        crate::Buffer::Cubecl(_) => panic!("GPU tensor reached CPU kernel path"),
    }
}

fn copy_view_to_array<T: Copy + Clone>(
    op: &'static str,
    mut out: strided_kernel::StridedArray<T>,
    src: &StridedView<'_, T>,
) -> crate::Result<TypedTensor<T>> {
    copy_into(&mut out.view_mut(), src).map_err(|err| backend_failure(op, err))?;
    Ok(tensor_from_array(out))
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

pub fn typed_transpose<T: Copy + Zero + Clone>(
    tensor: &TypedTensor<T>,
    perm: &[usize],
) -> crate::Result<TypedTensor<T>> {
    validate_permutation("transpose", perm, tensor.shape.len())?;
    let src = host_view(tensor)?;
    let permuted = src
        .permute(perm)
        .map_err(|err| backend_failure("transpose", err))?;
    // SAFETY: copy_into overwrites every output element.
    let out = unsafe { typed_array_uninit(permuted.dims()) };
    copy_view_to_array("transpose", out, &permuted)
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
    Ok(TypedTensor {
        buffer: tensor.buffer.clone(),
        shape: shape.to_vec(),
        placement: tensor.placement.clone(),
    })
}

pub fn typed_broadcast_in_dim<T: Copy + Zero + Clone>(
    tensor: &TypedTensor<T>,
    shape: &[usize],
    dims: &[usize],
) -> crate::Result<TypedTensor<T>> {
    validate_rank("broadcast_in_dim", tensor.shape.len(), dims.len())?;
    let mut seen = vec![false; shape.len()];
    let mut base_dims = vec![1usize; shape.len()];
    let mut base_strides = vec![0isize; shape.len()];
    let source_strides = crate::col_major_strides(&tensor.shape);
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
        base_dims[dst_axis] = source_dim;
        base_strides[dst_axis] = source_strides[src_axis];
    }
    let base: StridedView<'_, T, Identity> = match &tensor.buffer {
        crate::Buffer::Host(data) => StridedView::new(data, &base_dims, &base_strides, 0)
            .map_err(|err| backend_failure("broadcast_in_dim", err))?,
        crate::Buffer::Backend(_) => {
            return Err(crate::Error::BackendFailure {
                op: "broadcast_in_dim",
                message: "backend buffers are not supported for structural CPU helpers".into(),
            })
        }
        #[cfg(feature = "cubecl")]
        crate::Buffer::Cubecl(_) => panic!("GPU tensor reached CPU kernel path"),
    };
    let broadcast: StridedView<'_, T, Identity> = base
        .broadcast(shape)
        .map_err(|err| backend_failure("broadcast_in_dim", err))?;
    // SAFETY: copy_into overwrites every output element.
    let mut out = unsafe { typed_array_uninit(shape) };
    copy_into(&mut out.view_mut(), &broadcast)
        .map_err(|err| backend_failure("broadcast_in_dim", err))?;
    Ok(tensor_from_array(out))
}

fn typed_convert<S, T>(tensor: &TypedTensor<S>, f: impl Fn(S) -> T) -> TypedTensor<T>
where
    S: Copy,
    T: Copy + Clone + Zero,
{
    // SAFETY: map_into overwrites every output element.
    let mut out = unsafe { typed_array_uninit(&tensor.shape) };
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
    // SAFETY: copy_into overwrites every output element.
    let mut out = unsafe { typed_array_uninit(diag.dims()) };
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

    let input_data = match &tensor.buffer {
        crate::Buffer::Host(data) => data,
        crate::Buffer::Backend(_) => {
            return Err(crate::Error::BackendFailure {
                op: "embed_diagonal",
                message: "backend buffers are not supported for structural CPU helpers".into(),
            })
        }
        #[cfg(feature = "cubecl")]
        crate::Buffer::Cubecl(_) => panic!("GPU tensor reached CPU kernel path"),
    };

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
        *out.get_mut(&out_idx) = input_data[flat];
    }
    Ok(out)
}

pub fn typed_tril<T: Copy + Zero + Clone>(
    tensor: &TypedTensor<T>,
    k: i64,
) -> crate::Result<TypedTensor<T>> {
    typed_triangular_mask(tensor, k, false)
}

pub fn typed_triu<T: Copy + Zero + Clone>(
    tensor: &TypedTensor<T>,
    k: i64,
) -> crate::Result<TypedTensor<T>> {
    typed_triangular_mask(tensor, k, true)
}

fn typed_triangular_mask<T: Copy + Zero + Clone>(
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
    let mut out = tensor.clone();
    let data = match &mut out.buffer {
        crate::Buffer::Host(data) => data,
        crate::Buffer::Backend(_) => {
            return Err(crate::Error::BackendFailure {
                op: if upper { "triu" } else { "tril" },
                message: "backend buffers are not supported for structural CPU helpers".into(),
            })
        }
        #[cfg(feature = "cubecl")]
        crate::Buffer::Cubecl(_) => panic!("GPU tensor reached CPU kernel path"),
    };

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
