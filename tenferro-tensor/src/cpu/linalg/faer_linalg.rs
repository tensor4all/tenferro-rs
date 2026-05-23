use faer::dyn_stack::{MemBuffer, MemStack};
use faer::{
    diag::{Diag, DiagRef},
    Conj, Mat, MatMut, MatRef,
};
use num_complex::{Complex32, Complex64};
use std::ops::Range;

use crate::buffer_pool::{BufferPool, PoolScalar};
use crate::cpu::CpuContext;
use crate::{Buffer, Tensor, TypedTensor};

pub(crate) trait FaerLinalg: Copy + Clone + PoolScalar {
    fn parity_one() -> Self;
    fn cholesky_2d(
        ctx: &CpuContext,
        buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
    ) -> crate::Result<TypedTensor<Self>>;
    fn lu_2d(
        ctx: &CpuContext,
        buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
    ) -> crate::Result<Vec<TypedTensor<Self>>>;
    fn full_piv_lu_2d(
        ctx: &CpuContext,
        buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
    ) -> crate::Result<Vec<TypedTensor<Self>>>;
    fn full_piv_lu_solve_2d(
        ctx: &CpuContext,
        buffers: &mut BufferPool,
        a: &TypedTensor<Self>,
        b: &TypedTensor<Self>,
        transpose_a: bool,
    ) -> crate::Result<TypedTensor<Self>>;
    fn solve_2d(
        ctx: &CpuContext,
        buffers: &mut BufferPool,
        a: &TypedTensor<Self>,
        b: &TypedTensor<Self>,
        transpose_a: bool,
    ) -> crate::Result<TypedTensor<Self>>;
    fn triangular_solve_2d(
        ctx: &CpuContext,
        buffers: &mut BufferPool,
        a: &TypedTensor<Self>,
        b: &TypedTensor<Self>,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    ) -> crate::Result<TypedTensor<Self>>;
    fn svd_2d(
        ctx: &CpuContext,
        buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
    ) -> crate::Result<Vec<TypedTensor<Self>>>;
    fn qr_2d(
        ctx: &CpuContext,
        buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
    ) -> crate::Result<Vec<TypedTensor<Self>>>;
    fn eigh_2d(
        ctx: &CpuContext,
        buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
    ) -> crate::Result<Vec<TypedTensor<Self>>>;
}

fn matrix_dims<T>(input: &TypedTensor<T>, op: &'static str) -> crate::Result<(usize, usize)> {
    if input.shape.len() != 2 {
        return Err(crate::Error::RankMismatch {
            op,
            expected: 2,
            actual: input.shape.len(),
        });
    }
    Ok((input.shape[0], input.shape[1]))
}

fn square_matrix_dim<T>(input: &TypedTensor<T>, op: &'static str) -> crate::Result<usize> {
    let (rows, cols) = matrix_dims(input, op)?;
    if rows != cols {
        return Err(crate::Error::ShapeMismatch {
            op,
            lhs: vec![rows],
            rhs: vec![cols],
        });
    }
    Ok(rows)
}

fn tensor_from_vec_with_template<T: Clone, U>(
    shape: Vec<usize>,
    data: Vec<T>,
    template: &TypedTensor<U>,
) -> TypedTensor<T> {
    TypedTensor {
        buffer: Buffer::Host(data),
        shape,
        placement: template.placement.clone(),
    }
}

fn col_major_vec_from_mat<T: Copy + PoolScalar>(
    buffers: &mut BufferPool,
    mat: MatRef<'_, T>,
) -> Vec<T> {
    let (rows, cols) = mat.shape();
    let mut data = buffers.acquire_with_capacity::<T>(rows * cols);
    for j in 0..cols {
        for i in 0..rows {
            data.push(mat[(i, j)]);
        }
    }
    data
}

fn vec_from_diag<T: Copy + PoolScalar>(buffers: &mut BufferPool, diag: DiagRef<'_, T>) -> Vec<T> {
    let col = diag.column_vector();
    let mut data = buffers.acquire_with_capacity::<T>(col.nrows());
    for i in 0..col.nrows() {
        data.push(col[i]);
    }
    data
}

macro_rules! impl_complex_faer_casts {
    ($to_faer_slice:ident, $to_faer_slice_mut:ident, $complex:ty, $faer_complex:ty) => {
        fn $to_faer_slice(data: &[$complex]) -> &[$faer_complex] {
            assert_eq!(
                std::mem::size_of::<$complex>(),
                std::mem::size_of::<$faer_complex>()
            );
            assert_eq!(
                std::mem::align_of::<$complex>(),
                std::mem::align_of::<$faer_complex>()
            );

            // SAFETY: size and alignment are checked above in release builds,
            // and both types represent one complex scalar over the same real type.
            unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<$faer_complex>(), data.len()) }
        }

        fn $to_faer_slice_mut(data: &mut [$complex]) -> &mut [$faer_complex] {
            assert_eq!(
                std::mem::size_of::<$complex>(),
                std::mem::size_of::<$faer_complex>()
            );
            assert_eq!(
                std::mem::align_of::<$complex>(),
                std::mem::align_of::<$faer_complex>()
            );

            // SAFETY: size and alignment are checked above in release builds,
            // and both types represent one complex scalar over the same real type.
            unsafe {
                std::slice::from_raw_parts_mut(
                    data.as_mut_ptr().cast::<$faer_complex>(),
                    data.len(),
                )
            }
        }
    };
}

impl_complex_faer_casts!(
    complex32_to_faer_slice,
    complex32_to_faer_slice_mut,
    Complex32,
    faer::c32
);
impl_complex_faer_casts!(
    complex64_to_faer_slice,
    complex64_to_faer_slice_mut,
    Complex64,
    faer::c64
);

fn decomposition_failed(op: &'static str) -> crate::Error {
    crate::Error::BackendFailure {
        op,
        message: "decomposition failed".into(),
    }
}

fn invalid_config(op: &'static str, message: impl Into<String>) -> crate::Error {
    crate::Error::InvalidConfig {
        op,
        message: message.into(),
    }
}

fn checked_product(op: &'static str, role: &'static str, shape: &[usize]) -> crate::Result<usize> {
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| invalid_config(op, format!("{role} element count overflows usize")))
}

fn checked_repeated_len(
    op: &'static str,
    role: &'static str,
    per_batch: usize,
    batch_count: usize,
) -> crate::Result<usize> {
    per_batch
        .checked_mul(batch_count)
        .ok_or_else(|| invalid_config(op, format!("{role} repeated batch length overflows usize")))
}

fn checked_slice_range(
    op: &'static str,
    batch_idx: usize,
    slice_size: usize,
) -> crate::Result<Range<usize>> {
    let start = batch_idx
        .checked_mul(slice_size)
        .ok_or_else(|| invalid_config(op, "batch slice start overflows usize"))?;
    let end = start
        .checked_add(slice_size)
        .ok_or_else(|| invalid_config(op, "batch slice end overflows usize"))?;
    Ok(start..end)
}

macro_rules! impl_complex_vec_helpers {
    (
        $vec_from_real_diag:ident,
        $vec_from_diag:ident,
        $vec_from_mat:ident,
        $matrix_from_predicate:ident,
        $complex:ty,
        $faer_complex:ty
    ) => {
        fn $vec_from_real_diag(
            buffers: &mut BufferPool,
            diag: DiagRef<'_, $faer_complex>,
        ) -> Vec<$complex> {
            let col = diag.column_vector();
            let mut data = buffers.acquire_with_capacity::<$complex>(col.nrows());
            for i in 0..col.nrows() {
                data.push(<$complex>::new(col[i].re, 0.0));
            }
            data
        }

        fn $vec_from_diag(
            buffers: &mut BufferPool,
            diag: DiagRef<'_, $faer_complex>,
        ) -> Vec<$complex> {
            let col = diag.column_vector();
            let mut data = buffers.acquire_with_capacity::<$complex>(col.nrows());
            for i in 0..col.nrows() {
                data.push(<$complex>::new(col[i].re, col[i].im));
            }
            data
        }

        fn $vec_from_mat(
            buffers: &mut BufferPool,
            mat: MatRef<'_, $faer_complex>,
        ) -> Vec<$complex> {
            let (rows, cols) = mat.shape();
            let mut data = buffers.acquire_with_capacity::<$complex>(rows * cols);
            for j in 0..cols {
                for i in 0..rows {
                    let value = mat[(i, j)];
                    data.push(<$complex>::new(value.re, value.im));
                }
            }
            data
        }

        fn $matrix_from_predicate(
            mat: MatRef<'_, $faer_complex>,
            rows: usize,
            cols: usize,
            predicate: impl Fn(usize, usize) -> bool,
        ) -> Vec<$complex> {
            let mut data = vec![<$complex>::new(0.0, 0.0); rows * cols];
            for j in 0..cols {
                for i in 0..rows {
                    if predicate(i, j) {
                        let value = mat[(i, j)];
                        data[i + j * rows] = <$complex>::new(value.re, value.im);
                    }
                }
            }
            data
        }
    };
}

fn matrix_from_predicate<T: Copy + Default>(
    mat: MatRef<'_, T>,
    rows: usize,
    cols: usize,
    predicate: impl Fn(usize, usize) -> bool,
) -> Vec<T> {
    let mut data = vec![T::default(); rows * cols];
    for j in 0..cols {
        for i in 0..rows {
            if predicate(i, j) {
                data[i + j * rows] = mat[(i, j)];
            }
        }
    }
    data
}

fn lower_triangle_vec_from_mat<T: Copy + Default>(mat: MatRef<'_, T>) -> Vec<T> {
    let (rows, cols) = mat.shape();
    matrix_from_predicate(mat, rows, cols, |row, col| row >= col)
}

fn upper_triangle_vec_from_mat<T: Copy + Default>(mat: MatRef<'_, T>) -> Vec<T> {
    let (rows, cols) = mat.shape();
    matrix_from_predicate(mat, rows, cols, |row, col| row <= col)
}

fn permutation_matrix<T: Copy + Default>(perm: &[usize], one: T) -> Vec<T> {
    let n = perm.len();
    let mut data = vec![T::default(); n * n];
    for (row, &source) in perm.iter().enumerate() {
        data[row + source * n] = one;
    }
    data
}

impl_complex_vec_helpers!(
    complex32_vec_from_real_diag,
    complex32_vec_from_diag,
    complex32_vec_from_mat,
    complex32_matrix_from_predicate,
    Complex32,
    faer::c32
);
impl_complex_vec_helpers!(
    complex64_vec_from_real_diag,
    complex64_vec_from_diag,
    complex64_vec_from_mat,
    complex64_matrix_from_predicate,
    Complex64,
    faer::c64
);

macro_rules! impl_real_eig_to_complex_outputs {
    ($name:ident, $real:ty, $complex:ty) => {
        fn $name(
            buffers: &mut BufferPool,
            u_real: MatRef<'_, $real>,
            s_re: DiagRef<'_, $real>,
            s_im: DiagRef<'_, $real>,
        ) -> (Vec<$complex>, Vec<$complex>) {
            let n = u_real.nrows();
            // SAFETY: the loop below writes every element of `u` and `s` before any read.
            let mut u = unsafe { <$complex as PoolScalar>::pool_acquire(buffers, n * n) };
            // SAFETY: the loop below writes every element of `u` and `s` before any read.
            let mut s = unsafe { <$complex as PoolScalar>::pool_acquire(buffers, n) };
            let mut j = 0;
            while j < n {
                if s_im[j] == 0.0 {
                    s[j] = <$complex>::new(s_re[j], 0.0);
                    for i in 0..n {
                        u[i + j * n] = <$complex>::new(u_real[(i, j)], 0.0);
                    }
                    j += 1;
                } else {
                    s[j] = <$complex>::new(s_re[j], s_im[j]);
                    s[j + 1] = <$complex>::new(s_re[j], -s_im[j]);
                    for i in 0..n {
                        u[i + j * n] = <$complex>::new(u_real[(i, j)], u_real[(i, j + 1)]);
                        u[i + (j + 1) * n] = <$complex>::new(u_real[(i, j)], -u_real[(i, j + 1)]);
                    }
                    j += 2;
                }
            }
            (u, s)
        }
    };
}

impl_real_eig_to_complex_outputs!(real32_eig_to_complex_outputs, f32, Complex32);
impl_real_eig_to_complex_outputs!(real64_eig_to_complex_outputs, f64, Complex64);

fn split_shape_core_and_batch<'a>(
    shape: &'a [usize],
    core_rank: usize,
    op: &'static str,
) -> crate::Result<(&'a [usize], &'a [usize])> {
    if shape.len() < core_rank {
        return Err(crate::Error::RankMismatch {
            op,
            expected: core_rank,
            actual: shape.len(),
        });
    }
    Ok(shape.split_at(core_rank))
}

fn split_core_and_batch<'a, T>(
    input: &'a TypedTensor<T>,
    core_rank: usize,
    op: &'static str,
) -> crate::Result<(&'a [usize], &'a [usize])> {
    split_shape_core_and_batch(&input.shape, core_rank, op)
}

fn matrix_core_and_batch<'a, T>(
    input: &'a TypedTensor<T>,
    op: &'static str,
) -> crate::Result<(usize, usize, &'a [usize])> {
    let (matrix_shape, batch_shape) = split_core_and_batch(input, 2, op)?;
    Ok((matrix_shape[0], matrix_shape[1], batch_shape))
}

fn square_core_and_batch<'a, T>(
    input: &'a TypedTensor<T>,
    op: &'static str,
) -> crate::Result<(usize, &'a [usize])> {
    let (rows, cols, batch_shape) = matrix_core_and_batch(input, op)?;
    if rows != cols {
        return Err(crate::Error::ShapeMismatch {
            op,
            lhs: vec![rows],
            rhs: vec![cols],
        });
    }
    Ok((rows, batch_shape))
}

fn transpose_col_major_data<T: Copy + PoolScalar>(
    buffers: &mut BufferPool,
    data: &[T],
    rows: usize,
    cols: usize,
) -> Vec<T> {
    let mut transposed = buffers.acquire_with_capacity::<T>(data.len());
    for j in 0..rows {
        for i in 0..cols {
            transposed.push(data[j + i * rows]);
        }
    }
    transposed
}

fn batched_single<T, F>(
    op_name: &'static str,
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
    core_rank: usize,
    op: F,
) -> crate::Result<TypedTensor<T>>
where
    T: Clone + PoolScalar,
    F: Fn(&mut BufferPool, &TypedTensor<T>) -> crate::Result<TypedTensor<T>>,
{
    let (core_shape, batch_shape) = split_core_and_batch(input, core_rank, op_name)?;
    if batch_shape.is_empty() {
        return op(buffers, input);
    }

    let slice_size = checked_product(op_name, "core shape", core_shape)?;
    let batch_count = checked_product(op_name, "batch shape", batch_shape)?;
    if batch_count == 0 {
        return Err(invalid_config(
            op_name,
            "zero-sized batch dims must be handled by the caller",
        ));
    }

    let mut out_core_shape: Option<Vec<usize>> = None;
    let mut out_data: Option<Vec<T>> = None;

    for batch_idx in 0..batch_count {
        let range = checked_slice_range(op_name, batch_idx, slice_size)?;
        let batch_input = tensor_from_vec_with_template(
            core_shape.to_vec(),
            input.host_data()[range].to_vec(),
            input,
        );
        let batch_output = op(buffers, &batch_input)?;

        if let Some(expected_shape) = &out_core_shape {
            if batch_output.shape.as_slice() != expected_shape.as_slice() {
                return Err(crate::Error::ShapeMismatch {
                    op: op_name,
                    lhs: batch_output.shape.clone(),
                    rhs: expected_shape.clone(),
                });
            }
        } else {
            let output_elements =
                checked_product(op_name, "output core shape", &batch_output.shape)?;
            let capacity =
                checked_repeated_len(op_name, "output buffer", output_elements, batch_count)?;
            out_data = Some(buffers.acquire_with_capacity::<T>(capacity));
            out_core_shape = Some(batch_output.shape.clone());
        }

        match &mut out_data {
            Some(data) => data.extend_from_slice(batch_output.host_data()),
            None => {
                return Err(invalid_config(
                    op_name,
                    "missing output buffer after first batch",
                ));
            }
        }
    }

    let mut out_shape =
        out_core_shape.ok_or_else(|| invalid_config(op_name, "missing output shape"))?;
    out_shape.extend_from_slice(batch_shape);
    let out_data = out_data.ok_or_else(|| invalid_config(op_name, "missing output data"))?;
    Ok(tensor_from_vec_with_template(out_shape, out_data, input))
}

fn batched_multi_result<T, F>(
    op_name: &'static str,
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
    core_rank: usize,
    op: F,
) -> crate::Result<Vec<TypedTensor<T>>>
where
    T: Clone + PoolScalar,
    F: Fn(&mut BufferPool, &TypedTensor<T>) -> crate::Result<Vec<TypedTensor<T>>>,
{
    let (core_shape, batch_shape) = split_core_and_batch(input, core_rank, op_name)?;
    if batch_shape.is_empty() {
        return op(buffers, input);
    }

    let slice_size = checked_product(op_name, "core shape", core_shape)?;
    let batch_count = checked_product(op_name, "batch shape", batch_shape)?;
    if batch_count == 0 {
        return Err(invalid_config(
            op_name,
            "zero-sized batch dims must be handled by the caller",
        ));
    }

    let mut out_shapes: Vec<Vec<usize>> = Vec::new();
    let mut out_data: Vec<Vec<T>> = Vec::new();

    for batch_idx in 0..batch_count {
        let range = checked_slice_range(op_name, batch_idx, slice_size)?;
        let batch_input = tensor_from_vec_with_template(
            core_shape.to_vec(),
            input.host_data()[range].to_vec(),
            input,
        );
        let batch_outputs = op(buffers, &batch_input)?;

        if out_shapes.is_empty() {
            out_shapes = batch_outputs
                .iter()
                .map(|tensor| tensor.shape.clone())
                .collect();
            let mut pooled_outputs = Vec::with_capacity(batch_outputs.len());
            for tensor in &batch_outputs {
                let output_elements = checked_product(op_name, "output core shape", &tensor.shape)?;
                let capacity =
                    checked_repeated_len(op_name, "output buffer", output_elements, batch_count)?;
                pooled_outputs.push(buffers.acquire_with_capacity::<T>(capacity));
            }
            out_data = pooled_outputs;
        } else {
            if batch_outputs.len() != out_shapes.len() {
                return Err(crate::Error::ShapeMismatch {
                    op: op_name,
                    lhs: vec![batch_outputs.len()],
                    rhs: vec![out_shapes.len()],
                });
            }
        }

        for (idx, batch_output) in batch_outputs.iter().enumerate() {
            if batch_output.shape.as_slice() != out_shapes[idx].as_slice() {
                return Err(crate::Error::ShapeMismatch {
                    op: op_name,
                    lhs: batch_output.shape.clone(),
                    rhs: out_shapes[idx].clone(),
                });
            }
            out_data[idx].extend_from_slice(batch_output.host_data());
        }
    }

    Ok(out_shapes
        .into_iter()
        .zip(out_data)
        .map(|(mut out_shape, out_data)| {
            out_shape.extend_from_slice(batch_shape);
            tensor_from_vec_with_template(out_shape, out_data, input)
        })
        .collect())
}

fn batched_multi_convert_result<InT, OutT, F>(
    op_name: &'static str,
    buffers: &mut BufferPool,
    input: &TypedTensor<InT>,
    core_rank: usize,
    op: F,
) -> crate::Result<Vec<TypedTensor<OutT>>>
where
    InT: Clone,
    OutT: Clone + PoolScalar,
    F: Fn(&mut BufferPool, &TypedTensor<InT>) -> crate::Result<Vec<TypedTensor<OutT>>>,
{
    let (core_shape, batch_shape) = split_core_and_batch(input, core_rank, op_name)?;
    if batch_shape.is_empty() {
        return op(buffers, input);
    }

    let slice_size = checked_product(op_name, "core shape", core_shape)?;
    let batch_count = checked_product(op_name, "batch shape", batch_shape)?;
    if batch_count == 0 {
        return Err(invalid_config(
            op_name,
            "zero-sized batch dims must be handled by the caller",
        ));
    }

    let mut out_shapes: Vec<Vec<usize>> = Vec::new();
    let mut out_data: Vec<Vec<OutT>> = Vec::new();

    for batch_idx in 0..batch_count {
        let range = checked_slice_range(op_name, batch_idx, slice_size)?;
        let batch_input = tensor_from_vec_with_template(
            core_shape.to_vec(),
            input.host_data()[range].to_vec(),
            input,
        );
        let batch_outputs = op(buffers, &batch_input)?;

        if out_shapes.is_empty() {
            out_shapes = batch_outputs
                .iter()
                .map(|tensor| tensor.shape.clone())
                .collect();
            let mut pooled_outputs = Vec::with_capacity(batch_outputs.len());
            for tensor in &batch_outputs {
                let output_elements = checked_product(op_name, "output core shape", &tensor.shape)?;
                let capacity =
                    checked_repeated_len(op_name, "output buffer", output_elements, batch_count)?;
                pooled_outputs.push(buffers.acquire_with_capacity::<OutT>(capacity));
            }
            out_data = pooled_outputs;
        } else {
            if batch_outputs.len() != out_shapes.len() {
                return Err(crate::Error::ShapeMismatch {
                    op: op_name,
                    lhs: vec![batch_outputs.len()],
                    rhs: vec![out_shapes.len()],
                });
            }
        }

        for (idx, batch_output) in batch_outputs.iter().enumerate() {
            if batch_output.shape.as_slice() != out_shapes[idx].as_slice() {
                return Err(crate::Error::ShapeMismatch {
                    op: op_name,
                    lhs: batch_output.shape.clone(),
                    rhs: out_shapes[idx].clone(),
                });
            }
            out_data[idx].extend_from_slice(batch_output.host_data());
        }
    }

    Ok(out_shapes
        .into_iter()
        .zip(out_data)
        .map(|(mut out_shape, out_data)| {
            out_shape.extend_from_slice(batch_shape);
            tensor_from_vec_with_template(out_shape, out_data, input)
        })
        .collect())
}

fn batched_binary_result<T, F>(
    op_name: &'static str,
    buffers: &mut BufferPool,
    a: &TypedTensor<T>,
    b: &TypedTensor<T>,
    core_rank_a: usize,
    core_rank_b: usize,
    op: F,
) -> crate::Result<TypedTensor<T>>
where
    T: Clone + PoolScalar,
    F: Fn(&mut BufferPool, &TypedTensor<T>, &TypedTensor<T>) -> crate::Result<TypedTensor<T>>,
{
    let (a_core_shape, a_batch_shape) = split_core_and_batch(a, core_rank_a, op_name)?;
    let (b_core_shape, b_batch_shape) = split_core_and_batch(b, core_rank_b, op_name)?;
    if a_batch_shape != b_batch_shape {
        return Err(crate::Error::ShapeMismatch {
            op: op_name,
            lhs: a_batch_shape.to_vec(),
            rhs: b_batch_shape.to_vec(),
        });
    }

    if a_batch_shape.is_empty() {
        return op(buffers, a, b);
    }

    let a_slice_size = checked_product(op_name, "lhs core shape", a_core_shape)?;
    let b_slice_size = checked_product(op_name, "rhs core shape", b_core_shape)?;
    let batch_count = checked_product(op_name, "batch shape", a_batch_shape)?;
    if batch_count == 0 {
        return Err(invalid_config(
            op_name,
            "zero-sized batch dims must be handled by the caller",
        ));
    }

    let mut out_core_shape: Option<Vec<usize>> = None;
    let mut out_data: Option<Vec<T>> = None;

    for batch_idx in 0..batch_count {
        let a_range = checked_slice_range(op_name, batch_idx, a_slice_size)?;
        let b_range = checked_slice_range(op_name, batch_idx, b_slice_size)?;

        let batch_a = tensor_from_vec_with_template(
            a_core_shape.to_vec(),
            a.host_data()[a_range].to_vec(),
            a,
        );
        let batch_b = tensor_from_vec_with_template(
            b_core_shape.to_vec(),
            b.host_data()[b_range].to_vec(),
            b,
        );
        let batch_output = op(buffers, &batch_a, &batch_b)?;

        if let Some(expected_shape) = &out_core_shape {
            if batch_output.shape.as_slice() != expected_shape.as_slice() {
                return Err(crate::Error::ShapeMismatch {
                    op: op_name,
                    lhs: batch_output.shape.clone(),
                    rhs: expected_shape.clone(),
                });
            }
        } else {
            let output_elements =
                checked_product(op_name, "output core shape", &batch_output.shape)?;
            let capacity =
                checked_repeated_len(op_name, "output buffer", output_elements, batch_count)?;
            out_data = Some(buffers.acquire_with_capacity::<T>(capacity));
            out_core_shape = Some(batch_output.shape.clone());
        }

        match &mut out_data {
            Some(data) => data.extend_from_slice(batch_output.host_data()),
            None => {
                return Err(invalid_config(
                    op_name,
                    "missing output buffer after first batch",
                ));
            }
        }
    }

    let mut out_shape =
        out_core_shape.ok_or_else(|| invalid_config(op_name, "missing output shape"))?;
    out_shape.extend_from_slice(a_batch_shape);
    let out_data = out_data.ok_or_else(|| invalid_config(op_name, "missing output data"))?;
    Ok(tensor_from_vec_with_template(out_shape, out_data, b))
}

macro_rules! impl_faer_linalg_for_real {
    ($scalar:ty) => {
        impl FaerLinalg for $scalar {
    fn parity_one() -> Self {
        1.0
    }

    fn cholesky_2d(
        ctx: &CpuContext,
        _buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
    ) -> crate::Result<TypedTensor<Self>> {
        let n = square_matrix_dim(input, "cholesky")?;
        let mut l = Mat::zeros(n, n);
        l.copy_from(MatRef::from_column_major_slice(input.host_data(), n, n));
        let mut mem = MemBuffer::new(
            faer::linalg::cholesky::llt::factor::cholesky_in_place_scratch::<Self>(
                n,
                ctx.faer_par(),
                Default::default(),
            ),
        );
        let stack = MemStack::new(&mut mem);
        faer::linalg::cholesky::llt::factor::cholesky_in_place(
            l.as_mut(),
            Default::default(),
            ctx.faer_par(),
            stack,
            Default::default(),
        )
        .map_err(|_| crate::Error::BackendFailure {
            op: "cholesky",
            message: "matrix is not positive definite".into(),
        })?;
        Ok(tensor_from_vec_with_template(
            vec![n, n],
            lower_triangle_vec_from_mat(l.as_ref()),
            input,
        ))
    }

    fn lu_2d(
        ctx: &CpuContext,
        _buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
    ) -> crate::Result<Vec<TypedTensor<Self>>> {
        let (m, n) = matrix_dims(input, "lu")?;
        let k = m.min(n);
        let mut lu = Mat::zeros(m, n);
        lu.copy_from(MatRef::from_column_major_slice(input.host_data(), m, n));
        let mut perm = vec![0usize; m];
        let mut perm_inv = vec![0usize; m];
        let mut mem = MemBuffer::new(
            faer::linalg::lu::partial_pivoting::factor::lu_in_place_scratch::<usize, Self>(
                m,
                n,
                ctx.faer_par(),
                Default::default(),
            ),
        );
        let stack = MemStack::new(&mut mem);
        let info = faer::linalg::lu::partial_pivoting::factor::lu_in_place(
            lu.as_mut(),
            &mut perm,
            &mut perm_inv,
            ctx.faer_par(),
            stack,
            Default::default(),
        )
        .0;

        let mut p_data = vec![0.0; m * m];
        for (row, &col) in perm.iter().enumerate() {
            p_data[row + col * m] = 1.0;
        }
        let parity = if info.transposition_count % 2 == 0 {
            1.0
        } else {
            -1.0
        };

        let mut l_data = matrix_from_predicate(lu.as_ref(), m, k, |row, col| row >= col);
        for i in 0..k {
            l_data[i + i * m] = 1.0;
        }
        let u_data = upper_triangle_vec_from_mat(lu.as_ref().get(..k, ..));

        Ok(vec![
            tensor_from_vec_with_template(vec![m, m], p_data, input),
            tensor_from_vec_with_template(vec![m, k], l_data, input),
            tensor_from_vec_with_template(vec![k, n], u_data, input),
            tensor_from_vec_with_template(vec![], vec![parity], input),
        ])
    }

    fn full_piv_lu_2d(
        ctx: &CpuContext,
        _buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
    ) -> crate::Result<Vec<TypedTensor<Self>>> {
        let n = square_matrix_dim(input, "full_piv_lu")?;
        let mut lu = Mat::zeros(n, n);
        lu.copy_from(MatRef::from_column_major_slice(input.host_data(), n, n));
        let mut row_perm = vec![0usize; n];
        let mut row_perm_inv = vec![0usize; n];
        let mut col_perm = vec![0usize; n];
        let mut col_perm_inv = vec![0usize; n];
        let mut mem = MemBuffer::new(
            faer::linalg::lu::full_pivoting::factor::lu_in_place_scratch::<usize, Self>(
                n,
                n,
                ctx.faer_par(),
                Default::default(),
            ),
        );
        let stack = MemStack::new(&mut mem);
        let info = faer::linalg::lu::full_pivoting::factor::lu_in_place(
            lu.as_mut(),
            &mut row_perm,
            &mut row_perm_inv,
            &mut col_perm,
            &mut col_perm_inv,
            ctx.faer_par(),
            stack,
            Default::default(),
        )
        .0;

        let mut l_data = lower_triangle_vec_from_mat(lu.as_ref());
        for i in 0..n {
            l_data[i + i * n] = 1.0;
        }
        let u_data = upper_triangle_vec_from_mat(lu.as_ref());
        let parity = if info.transposition_count % 2 == 0 {
            1.0
        } else {
            -1.0
        };

        Ok(vec![
            tensor_from_vec_with_template(vec![n, n], permutation_matrix(&row_perm, 1.0), input),
            tensor_from_vec_with_template(vec![n, n], l_data, input),
            tensor_from_vec_with_template(vec![n, n], u_data, input),
            tensor_from_vec_with_template(vec![n, n], permutation_matrix(&col_perm, 1.0), input),
            tensor_from_vec_with_template(vec![], vec![parity], input),
        ])
    }

    fn full_piv_lu_solve_2d(
        ctx: &CpuContext,
        buffers: &mut BufferPool,
        a: &TypedTensor<Self>,
        b: &TypedTensor<Self>,
        transpose_a: bool,
    ) -> crate::Result<TypedTensor<Self>> {
        let n = square_matrix_dim(a, "full_piv_lu_solve")?;
        let (b_rows, b_cols) = matrix_dims(b, "full_piv_lu_solve")?;
        if b_rows != n {
            return Err(crate::Error::ShapeMismatch {
                op: "full_piv_lu_solve",
                lhs: vec![n],
                rhs: vec![b_rows],
            });
        }

        let mut lu = Mat::zeros(n, n);
        lu.copy_from(MatRef::from_column_major_slice(a.host_data(), n, n));
        let mut row_perm = vec![0usize; n];
        let mut row_perm_inv = vec![0usize; n];
        let mut col_perm = vec![0usize; n];
        let mut col_perm_inv = vec![0usize; n];
        let mut mem = MemBuffer::new(
            faer::linalg::lu::full_pivoting::factor::lu_in_place_scratch::<usize, Self>(
                n,
                n,
                ctx.faer_par(),
                Default::default(),
            ),
        );
        let stack = MemStack::new(&mut mem);
        let (_, row_perm_ref, col_perm_ref) = faer::linalg::lu::full_pivoting::factor::lu_in_place(
            lu.as_mut(),
            &mut row_perm,
            &mut row_perm_inv,
            &mut col_perm,
            &mut col_perm_inv,
            ctx.faer_par(),
            stack,
            Default::default(),
        );
        for i in 0..n {
            if lu[(i, i)] == 0.0 {
                return Err(crate::Error::BackendFailure {
                    op: "full_piv_lu_solve",
                    message: "matrix is singular".into(),
                });
            }
        }

        let mut rhs_data = buffers.acquire_with_capacity::<Self>(b.host_data().len());
        rhs_data.extend_from_slice(b.host_data());
        let rhs = MatMut::from_column_major_slice_mut(&mut rhs_data, n, b_cols);
        let mut mem = MemBuffer::new(
            faer::linalg::lu::full_pivoting::solve::solve_in_place_scratch::<usize, Self>(
                n,
                b_cols,
                ctx.faer_par(),
            ),
        );
        let stack = MemStack::new(&mut mem);
        if transpose_a {
            faer::linalg::lu::full_pivoting::solve::solve_transpose_in_place(
                lu.as_ref(),
                lu.as_ref(),
                row_perm_ref,
                col_perm_ref,
                rhs,
                ctx.faer_par(),
                stack,
            );
        } else {
            faer::linalg::lu::full_pivoting::solve::solve_in_place(
                lu.as_ref(),
                lu.as_ref(),
                row_perm_ref,
                col_perm_ref,
                rhs,
                ctx.faer_par(),
                stack,
            );
        }
        Ok(tensor_from_vec_with_template(vec![n, b_cols], rhs_data, b))
    }

    fn solve_2d(
        ctx: &CpuContext,
        buffers: &mut BufferPool,
        a: &TypedTensor<Self>,
        b: &TypedTensor<Self>,
        transpose_a: bool,
    ) -> crate::Result<TypedTensor<Self>> {
        let n = square_matrix_dim(a, "solve")?;
        let (b_rows, b_cols) = matrix_dims(b, "solve")?;
        if b_rows != n {
            return Err(crate::Error::ShapeMismatch {
                op: "solve",
                lhs: vec![n],
                rhs: vec![b_rows],
            });
        }

        let mut lu = Mat::zeros(n, n);
        lu.copy_from(MatRef::from_column_major_slice(a.host_data(), n, n));
        let mut row_perm = vec![0usize; n];
        let mut row_perm_inv = vec![0usize; n];
        let mut mem = MemBuffer::new(
            faer::linalg::lu::partial_pivoting::factor::lu_in_place_scratch::<usize, Self>(
                n,
                n,
                ctx.faer_par(),
                Default::default(),
            ),
        );
        let stack = MemStack::new(&mut mem);
        let (_, row_perm_ref) = faer::linalg::lu::partial_pivoting::factor::lu_in_place(
            lu.as_mut(),
            &mut row_perm,
            &mut row_perm_inv,
            ctx.faer_par(),
            stack,
            Default::default(),
        );
        for i in 0..n {
            if lu[(i, i)] == 0.0 {
                return Err(crate::Error::BackendFailure {
                    op: "solve",
                    message: "matrix is singular".into(),
                });
            }
        }

        let mut rhs_data = buffers.acquire_with_capacity::<Self>(b.host_data().len());
        rhs_data.extend_from_slice(b.host_data());
        let rhs = MatMut::from_column_major_slice_mut(&mut rhs_data, n, b_cols);
        let mut mem = MemBuffer::new(if transpose_a {
            faer::linalg::lu::partial_pivoting::solve::solve_transpose_in_place_scratch::<
                usize,
                Self,
            >(n, b_cols, ctx.faer_par())
        } else {
            faer::linalg::lu::partial_pivoting::solve::solve_in_place_scratch::<usize, Self>(
                n,
                b_cols,
                ctx.faer_par(),
            )
        });
        let stack = MemStack::new(&mut mem);
        if transpose_a {
            faer::linalg::lu::partial_pivoting::solve::solve_transpose_in_place(
                lu.as_ref(),
                lu.as_ref(),
                row_perm_ref,
                rhs,
                ctx.faer_par(),
                stack,
            );
        } else {
            faer::linalg::lu::partial_pivoting::solve::solve_in_place(
                lu.as_ref(),
                lu.as_ref(),
                row_perm_ref,
                rhs,
                ctx.faer_par(),
                stack,
            );
        }
        Ok(tensor_from_vec_with_template(vec![n, b_cols], rhs_data, b))
    }

    fn triangular_solve_2d(
        ctx: &CpuContext,
        buffers: &mut BufferPool,
        a: &TypedTensor<Self>,
        b: &TypedTensor<Self>,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    ) -> crate::Result<TypedTensor<Self>> {
        let n = square_matrix_dim(a, "triangular_solve")?;
        let (b_rows, b_cols) = matrix_dims(b, "triangular_solve")?;
        let a_mat = MatRef::from_column_major_slice(a.host_data(), n, n);

        if left_side {
            if b_rows != n {
                return Err(crate::Error::ShapeMismatch {
                    op: "triangular_solve",
                    lhs: vec![n],
                    rhs: vec![b_rows],
                });
            }
            let mut rhs_data = buffers.acquire_with_capacity::<Self>(b.host_data().len());
            rhs_data.extend_from_slice(b.host_data());
            let rhs = MatMut::from_column_major_slice_mut(&mut rhs_data, n, b_cols);
            match (transpose_a, lower, unit_diagonal) {
                (false, true, false) => {
                    faer::linalg::triangular_solve::solve_lower_triangular_in_place(
                        a_mat,
                        rhs,
                        ctx.faer_par(),
                    );
                }
                (false, true, true) => {
                    faer::linalg::triangular_solve::solve_unit_lower_triangular_in_place(
                        a_mat,
                        rhs,
                        ctx.faer_par(),
                    );
                }
                (false, false, false) => {
                    faer::linalg::triangular_solve::solve_upper_triangular_in_place(
                        a_mat,
                        rhs,
                        ctx.faer_par(),
                    );
                }
                (false, false, true) => {
                    faer::linalg::triangular_solve::solve_unit_upper_triangular_in_place(
                        a_mat,
                        rhs,
                        ctx.faer_par(),
                    );
                }
                (true, true, false) => {
                    faer::linalg::triangular_solve::solve_upper_triangular_in_place(
                        a_mat.transpose(),
                        rhs,
                        ctx.faer_par(),
                    );
                }
                (true, true, true) => {
                    faer::linalg::triangular_solve::solve_unit_upper_triangular_in_place(
                        a_mat.transpose(),
                        rhs,
                        ctx.faer_par(),
                    );
                }
                (true, false, false) => {
                    faer::linalg::triangular_solve::solve_lower_triangular_in_place(
                        a_mat.transpose(),
                        rhs,
                        ctx.faer_par(),
                    );
                }
                (true, false, true) => {
                    faer::linalg::triangular_solve::solve_unit_lower_triangular_in_place(
                        a_mat.transpose(),
                        rhs,
                        ctx.faer_par(),
                    );
                }
            }
            Ok(tensor_from_vec_with_template(vec![n, b_cols], rhs_data, b))
        } else {
            if b_cols != n {
                return Err(crate::Error::ShapeMismatch {
                    op: "triangular_solve",
                    lhs: vec![n],
                    rhs: vec![b_cols],
                });
            }
            let nrhs = b_rows;
            let mut rhs_transposed = transpose_col_major_data(buffers, b.host_data(), nrhs, n);
            let rhs = MatMut::from_column_major_slice_mut(&mut rhs_transposed, n, nrhs);
            match (transpose_a, lower, unit_diagonal) {
                (false, true, false) => {
                    faer::linalg::triangular_solve::solve_upper_triangular_in_place(
                        a_mat.transpose(),
                        rhs,
                        ctx.faer_par(),
                    );
                }
                (false, true, true) => {
                    faer::linalg::triangular_solve::solve_unit_upper_triangular_in_place(
                        a_mat.transpose(),
                        rhs,
                        ctx.faer_par(),
                    );
                }
                (false, false, false) => {
                    faer::linalg::triangular_solve::solve_lower_triangular_in_place(
                        a_mat.transpose(),
                        rhs,
                        ctx.faer_par(),
                    );
                }
                (false, false, true) => {
                    faer::linalg::triangular_solve::solve_unit_lower_triangular_in_place(
                        a_mat.transpose(),
                        rhs,
                        ctx.faer_par(),
                    );
                }
                (true, true, false) => {
                    faer::linalg::triangular_solve::solve_lower_triangular_in_place(
                        a_mat,
                        rhs,
                        ctx.faer_par(),
                    );
                }
                (true, true, true) => {
                    faer::linalg::triangular_solve::solve_unit_lower_triangular_in_place(
                        a_mat,
                        rhs,
                        ctx.faer_par(),
                    );
                }
                (true, false, false) => {
                    faer::linalg::triangular_solve::solve_upper_triangular_in_place(
                        a_mat,
                        rhs,
                        ctx.faer_par(),
                    );
                }
                (true, false, true) => {
                    faer::linalg::triangular_solve::solve_unit_upper_triangular_in_place(
                        a_mat,
                        rhs,
                        ctx.faer_par(),
                    );
                }
            }
            let result = transpose_col_major_data(buffers, &rhs_transposed, n, nrhs);
            <Self as PoolScalar>::pool_release(buffers, rhs_transposed);
            Ok(tensor_from_vec_with_template(vec![nrhs, n], result, b))
        }
    }

    fn svd_2d(
        ctx: &CpuContext,
        buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
    ) -> crate::Result<Vec<TypedTensor<Self>>> {
        let (m, n) = matrix_dims(input, "svd")?;
        let k = m.min(n);
        let mat = MatRef::from_column_major_slice(input.host_data(), m, n);
        let mut u = Mat::zeros(m, k);
        let mut v = Mat::zeros(n, k);
        let mut s = Diag::zeros(k);
        let mut mem = MemBuffer::new(faer::linalg::svd::svd_scratch::<Self>(
            m,
            n,
            faer::linalg::svd::ComputeSvdVectors::Thin,
            faer::linalg::svd::ComputeSvdVectors::Thin,
            ctx.faer_par(),
            Default::default(),
        ));
        let stack = MemStack::new(&mut mem);
        faer::linalg::svd::svd(
            mat,
            s.as_mut(),
            Some(u.as_mut()),
            Some(v.as_mut()),
            ctx.faer_par(),
            stack,
            Default::default(),
        )
        .map_err(|_| decomposition_failed("svd"))?;

        let u = tensor_from_vec_with_template(
            vec![m, k],
            col_major_vec_from_mat(buffers, u.as_ref()),
            input,
        );
        let s = tensor_from_vec_with_template(vec![k], vec_from_diag(buffers, s.as_ref()), input);
        let mut vt_data = buffers.acquire_with_capacity::<Self>(k * n);
        for j in 0..n {
            for i in 0..k {
                vt_data.push(v[(j, i)]);
            }
        }
        let vt = tensor_from_vec_with_template(vec![k, n], vt_data, input);

        Ok(vec![u, s, vt])
    }

    fn qr_2d(
        ctx: &CpuContext,
        buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
    ) -> crate::Result<Vec<TypedTensor<Self>>> {
        let (m, n) = matrix_dims(input, "qr")?;
        let k = m.min(n);
        let mat = MatRef::from_column_major_slice(input.host_data(), m, n);
        let block_size =
            faer::linalg::qr::no_pivoting::factor::recommended_block_size::<Self>(m, n);
        let mut qr = Mat::zeros(m, n);
        qr.copy_from(mat);
        let mut coeff = Mat::zeros(block_size, k);
        let mut mem = MemBuffer::new(
            faer::linalg::qr::no_pivoting::factor::qr_in_place_scratch::<Self>(
                m,
                n,
                block_size,
                ctx.faer_par(),
                Default::default(),
            ),
        );
        let stack = MemStack::new(&mut mem);
        faer::linalg::qr::no_pivoting::factor::qr_in_place(
            qr.as_mut(),
            coeff.as_mut(),
            ctx.faer_par(),
            stack,
            Default::default(),
        );
        let mut q = Mat::identity(m, k);
        let mut mem = MemBuffer::new(
            faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<Self>(
                m,
                block_size,
                k,
            ),
        );
        let stack = MemStack::new(&mut mem);
        faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
            qr.as_ref().subcols(0, k),
            coeff.as_ref(),
            Conj::No,
            q.as_mut(),
            ctx.faer_par(),
            stack,
        );
        let q = tensor_from_vec_with_template(
            vec![m, k],
            col_major_vec_from_mat(buffers, q.as_ref()),
            input,
        );
        let r = tensor_from_vec_with_template(
            vec![k, n],
            upper_triangle_vec_from_mat(qr.as_ref().get(..k, ..)),
            input,
        );

        Ok(vec![q, r])
    }

    fn eigh_2d(
        ctx: &CpuContext,
        buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
    ) -> crate::Result<Vec<TypedTensor<Self>>> {
        let n = square_matrix_dim(input, "eigh")?;
        let mat = MatRef::from_column_major_slice(input.host_data(), n, n);
        let mut values = Diag::zeros(n);
        let mut vectors = Mat::zeros(n, n);
        let mut mem = MemBuffer::new(faer::linalg::evd::self_adjoint_evd_scratch::<Self>(
            n,
            faer::linalg::evd::ComputeEigenvectors::Yes,
            ctx.faer_par(),
            Default::default(),
        ));
        let stack = MemStack::new(&mut mem);
        faer::linalg::evd::self_adjoint_evd(
            mat,
            values.as_mut(),
            Some(vectors.as_mut()),
            ctx.faer_par(),
            stack,
            Default::default(),
        )
        .map_err(|_| decomposition_failed("eigh"))?;

        let values =
            tensor_from_vec_with_template(vec![n], vec_from_diag(buffers, values.as_ref()), input);
        let vectors = tensor_from_vec_with_template(
            vec![n, n],
            col_major_vec_from_mat(buffers, vectors.as_ref()),
            input,
        );

        Ok(vec![values, vectors])
    }
        }
    };
}

impl_faer_linalg_for_real!(f32);
impl_faer_linalg_for_real!(f64);

macro_rules! impl_faer_linalg_for_complex {
    (
        $complex:ty,
        $faer_complex:ty,
        $to_faer_slice:ident,
        $to_faer_slice_mut:ident,
        $vec_from_real_diag:ident,
        $vec_from_diag:ident,
        $vec_from_mat:ident,
        $matrix_from_predicate:ident
    ) => {
        impl FaerLinalg for $complex {
    fn parity_one() -> Self {
        <$complex>::new(1.0, 0.0)
    }

    fn cholesky_2d(
        ctx: &CpuContext,
        _buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
    ) -> crate::Result<TypedTensor<Self>> {
        let n = square_matrix_dim(input, "cholesky")?;
        let mut l = Mat::zeros(n, n);
        l.copy_from(MatRef::from_column_major_slice(
            $to_faer_slice(input.host_data()),
            n,
            n,
        ));
        let mut mem = MemBuffer::new(
            faer::linalg::cholesky::llt::factor::cholesky_in_place_scratch::<$faer_complex>(
                n,
                ctx.faer_par(),
                Default::default(),
            ),
        );
        let stack = MemStack::new(&mut mem);
        faer::linalg::cholesky::llt::factor::cholesky_in_place(
            l.as_mut(),
            Default::default(),
            ctx.faer_par(),
            stack,
            Default::default(),
        )
        .map_err(|_| crate::Error::BackendFailure {
            op: "cholesky",
            message: "matrix is not positive definite".into(),
        })?;
        Ok(tensor_from_vec_with_template(
            vec![n, n],
            $matrix_from_predicate(l.as_ref(), n, n, |row, col| row >= col),
            input,
        ))
    }

    fn lu_2d(
        ctx: &CpuContext,
        _buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
    ) -> crate::Result<Vec<TypedTensor<Self>>> {
        let (m, n) = matrix_dims(input, "lu")?;
        let k = m.min(n);
        let mut lu = Mat::zeros(m, n);
        lu.copy_from(MatRef::from_column_major_slice(
            $to_faer_slice(input.host_data()),
            m,
            n,
        ));
        let mut perm = vec![0usize; m];
        let mut perm_inv = vec![0usize; m];
        let mut mem = MemBuffer::new(
            faer::linalg::lu::partial_pivoting::factor::lu_in_place_scratch::<usize, $faer_complex>(
                m,
                n,
                ctx.faer_par(),
                Default::default(),
            ),
        );
        let stack = MemStack::new(&mut mem);
        let info = faer::linalg::lu::partial_pivoting::factor::lu_in_place(
            lu.as_mut(),
            &mut perm,
            &mut perm_inv,
            ctx.faer_par(),
            stack,
            Default::default(),
        )
        .0;

        let mut p_data = vec![<$complex>::new(0.0, 0.0); m * m];
        for (row, &col) in perm.iter().enumerate() {
            p_data[row + col * m] = <$complex>::new(1.0, 0.0);
        }
        let parity = if info.transposition_count % 2 == 0 {
            <$complex>::new(1.0, 0.0)
        } else {
            <$complex>::new(-1.0, 0.0)
        };
        let mut l_data = $matrix_from_predicate(lu.as_ref(), m, k, |row, col| row >= col);
        for i in 0..k {
            l_data[i + i * m] = <$complex>::new(1.0, 0.0);
        }
        let u_data = $matrix_from_predicate(lu.as_ref(), k, n, |row, col| row <= col);

        Ok(vec![
            tensor_from_vec_with_template(vec![m, m], p_data, input),
            tensor_from_vec_with_template(vec![m, k], l_data, input),
            tensor_from_vec_with_template(vec![k, n], u_data, input),
            tensor_from_vec_with_template(vec![], vec![parity], input),
        ])
    }

    fn full_piv_lu_2d(
        ctx: &CpuContext,
        _buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
    ) -> crate::Result<Vec<TypedTensor<Self>>> {
        let n = square_matrix_dim(input, "full_piv_lu")?;
        let mut lu = Mat::zeros(n, n);
        lu.copy_from(MatRef::from_column_major_slice(
            $to_faer_slice(input.host_data()),
            n,
            n,
        ));
        let mut row_perm = vec![0usize; n];
        let mut row_perm_inv = vec![0usize; n];
        let mut col_perm = vec![0usize; n];
        let mut col_perm_inv = vec![0usize; n];
        let mut mem = MemBuffer::new(
            faer::linalg::lu::full_pivoting::factor::lu_in_place_scratch::<usize, $faer_complex>(
                n,
                n,
                ctx.faer_par(),
                Default::default(),
            ),
        );
        let stack = MemStack::new(&mut mem);
        let info = faer::linalg::lu::full_pivoting::factor::lu_in_place(
            lu.as_mut(),
            &mut row_perm,
            &mut row_perm_inv,
            &mut col_perm,
            &mut col_perm_inv,
            ctx.faer_par(),
            stack,
            Default::default(),
        )
        .0;

        let mut l_data = $matrix_from_predicate(lu.as_ref(), n, n, |row, col| row >= col);
        for i in 0..n {
            l_data[i + i * n] = <$complex>::new(1.0, 0.0);
        }
        let u_data = $matrix_from_predicate(lu.as_ref(), n, n, |row, col| row <= col);
        let one = <$complex>::new(1.0, 0.0);
        let parity = if info.transposition_count % 2 == 0 {
            one
        } else {
            <$complex>::new(-1.0, 0.0)
        };

        Ok(vec![
            tensor_from_vec_with_template(vec![n, n], permutation_matrix(&row_perm, one), input),
            tensor_from_vec_with_template(vec![n, n], l_data, input),
            tensor_from_vec_with_template(vec![n, n], u_data, input),
            tensor_from_vec_with_template(vec![n, n], permutation_matrix(&col_perm, one), input),
            tensor_from_vec_with_template(vec![], vec![parity], input),
        ])
    }

    fn full_piv_lu_solve_2d(
        ctx: &CpuContext,
        buffers: &mut BufferPool,
        a: &TypedTensor<Self>,
        b: &TypedTensor<Self>,
        transpose_a: bool,
    ) -> crate::Result<TypedTensor<Self>> {
        let n = square_matrix_dim(a, "full_piv_lu_solve")?;
        let (b_rows, b_cols) = matrix_dims(b, "full_piv_lu_solve")?;
        if b_rows != n {
            return Err(crate::Error::ShapeMismatch {
                op: "full_piv_lu_solve",
                lhs: vec![n],
                rhs: vec![b_rows],
            });
        }

        let mut lu = Mat::zeros(n, n);
        lu.copy_from(MatRef::from_column_major_slice(
            $to_faer_slice(a.host_data()),
            n,
            n,
        ));
        let mut row_perm = vec![0usize; n];
        let mut row_perm_inv = vec![0usize; n];
        let mut col_perm = vec![0usize; n];
        let mut col_perm_inv = vec![0usize; n];
        let mut mem = MemBuffer::new(
            faer::linalg::lu::full_pivoting::factor::lu_in_place_scratch::<usize, $faer_complex>(
                n,
                n,
                ctx.faer_par(),
                Default::default(),
            ),
        );
        let stack = MemStack::new(&mut mem);
        let (_, row_perm_ref, col_perm_ref) = faer::linalg::lu::full_pivoting::factor::lu_in_place(
            lu.as_mut(),
            &mut row_perm,
            &mut row_perm_inv,
            &mut col_perm,
            &mut col_perm_inv,
            ctx.faer_par(),
            stack,
            Default::default(),
        );
        for i in 0..n {
            let value = lu[(i, i)];
            if value.re == 0.0 && value.im == 0.0 {
                return Err(crate::Error::BackendFailure {
                    op: "full_piv_lu_solve",
                    message: "matrix is singular".into(),
                });
            }
        }

        let mut rhs_data = buffers.acquire_with_capacity::<Self>(b.host_data().len());
        rhs_data.extend_from_slice(b.host_data());
        let rhs = MatMut::from_column_major_slice_mut(
            $to_faer_slice_mut(&mut rhs_data),
            n,
            b_cols,
        );
        let mut mem = MemBuffer::new(
            faer::linalg::lu::full_pivoting::solve::solve_in_place_scratch::<usize, $faer_complex>(
                n,
                b_cols,
                ctx.faer_par(),
            ),
        );
        let stack = MemStack::new(&mut mem);
        if transpose_a {
            faer::linalg::lu::full_pivoting::solve::solve_transpose_in_place(
                lu.as_ref(),
                lu.as_ref(),
                row_perm_ref,
                col_perm_ref,
                rhs,
                ctx.faer_par(),
                stack,
            );
        } else {
            faer::linalg::lu::full_pivoting::solve::solve_in_place(
                lu.as_ref(),
                lu.as_ref(),
                row_perm_ref,
                col_perm_ref,
                rhs,
                ctx.faer_par(),
                stack,
            );
        }
        Ok(tensor_from_vec_with_template(vec![n, b_cols], rhs_data, b))
    }

    fn solve_2d(
        ctx: &CpuContext,
        buffers: &mut BufferPool,
        a: &TypedTensor<Self>,
        b: &TypedTensor<Self>,
        transpose_a: bool,
    ) -> crate::Result<TypedTensor<Self>> {
        let n = square_matrix_dim(a, "solve")?;
        let (b_rows, b_cols) = matrix_dims(b, "solve")?;
        if b_rows != n {
            return Err(crate::Error::ShapeMismatch {
                op: "solve",
                lhs: vec![n],
                rhs: vec![b_rows],
            });
        }

        let mut lu = Mat::zeros(n, n);
        lu.copy_from(MatRef::from_column_major_slice(
            $to_faer_slice(a.host_data()),
            n,
            n,
        ));
        let mut row_perm = vec![0usize; n];
        let mut row_perm_inv = vec![0usize; n];
        let mut mem = MemBuffer::new(
            faer::linalg::lu::partial_pivoting::factor::lu_in_place_scratch::<usize, $faer_complex>(
                n,
                n,
                ctx.faer_par(),
                Default::default(),
            ),
        );
        let stack = MemStack::new(&mut mem);
        let (_, row_perm_ref) = faer::linalg::lu::partial_pivoting::factor::lu_in_place(
            lu.as_mut(),
            &mut row_perm,
            &mut row_perm_inv,
            ctx.faer_par(),
            stack,
            Default::default(),
        );
        for i in 0..n {
            let value = lu[(i, i)];
            if value.re == 0.0 && value.im == 0.0 {
                return Err(crate::Error::BackendFailure {
                    op: "solve",
                    message: "matrix is singular".into(),
                });
            }
        }

        let mut rhs_data = buffers.acquire_with_capacity::<Self>(b.host_data().len());
        rhs_data.extend_from_slice(b.host_data());
        let rhs =
            MatMut::from_column_major_slice_mut($to_faer_slice_mut(&mut rhs_data), n, b_cols);
        let mut mem = MemBuffer::new(if transpose_a {
            faer::linalg::lu::partial_pivoting::solve::solve_transpose_in_place_scratch::<
                usize,
                $faer_complex,
            >(n, b_cols, ctx.faer_par())
        } else {
            faer::linalg::lu::partial_pivoting::solve::solve_in_place_scratch::<
                usize,
                $faer_complex,
            >(n, b_cols, ctx.faer_par())
        });
        let stack = MemStack::new(&mut mem);
        if transpose_a {
            faer::linalg::lu::partial_pivoting::solve::solve_transpose_in_place(
                lu.as_ref(),
                lu.as_ref(),
                row_perm_ref,
                rhs,
                ctx.faer_par(),
                stack,
            );
        } else {
            faer::linalg::lu::partial_pivoting::solve::solve_in_place(
                lu.as_ref(),
                lu.as_ref(),
                row_perm_ref,
                rhs,
                ctx.faer_par(),
                stack,
            );
        }
        Ok(tensor_from_vec_with_template(vec![n, b_cols], rhs_data, b))
    }

    fn triangular_solve_2d(
        ctx: &CpuContext,
        buffers: &mut BufferPool,
        a: &TypedTensor<Self>,
        b: &TypedTensor<Self>,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    ) -> crate::Result<TypedTensor<Self>> {
        let n = square_matrix_dim(a, "triangular_solve")?;
        let (b_rows, b_cols) = matrix_dims(b, "triangular_solve")?;
        let a_mat = MatRef::from_column_major_slice($to_faer_slice(a.host_data()), n, n);

        if left_side {
            if b_rows != n {
                return Err(crate::Error::ShapeMismatch {
                    op: "triangular_solve",
                    lhs: vec![n],
                    rhs: vec![b_rows],
                });
            }
            let mut rhs_data = buffers.acquire_with_capacity::<Self>(b.host_data().len());
            rhs_data.extend_from_slice(b.host_data());
            let rhs = MatMut::from_column_major_slice_mut(
                $to_faer_slice_mut(&mut rhs_data),
                n,
                b_cols,
            );
            match (transpose_a, lower, unit_diagonal) {
                (false, true, false) => {
                    faer::linalg::triangular_solve::solve_lower_triangular_in_place(
                        a_mat,
                        rhs,
                        ctx.faer_par(),
                    );
                }
                (false, true, true) => {
                    faer::linalg::triangular_solve::solve_unit_lower_triangular_in_place(
                        a_mat,
                        rhs,
                        ctx.faer_par(),
                    );
                }
                (false, false, false) => {
                    faer::linalg::triangular_solve::solve_upper_triangular_in_place(
                        a_mat,
                        rhs,
                        ctx.faer_par(),
                    );
                }
                (false, false, true) => {
                    faer::linalg::triangular_solve::solve_unit_upper_triangular_in_place(
                        a_mat,
                        rhs,
                        ctx.faer_par(),
                    );
                }
                (true, true, false) => {
                    faer::linalg::triangular_solve::solve_upper_triangular_in_place(
                        a_mat.transpose(),
                        rhs,
                        ctx.faer_par(),
                    );
                }
                (true, true, true) => {
                    faer::linalg::triangular_solve::solve_unit_upper_triangular_in_place(
                        a_mat.transpose(),
                        rhs,
                        ctx.faer_par(),
                    );
                }
                (true, false, false) => {
                    faer::linalg::triangular_solve::solve_lower_triangular_in_place(
                        a_mat.transpose(),
                        rhs,
                        ctx.faer_par(),
                    );
                }
                (true, false, true) => {
                    faer::linalg::triangular_solve::solve_unit_lower_triangular_in_place(
                        a_mat.transpose(),
                        rhs,
                        ctx.faer_par(),
                    );
                }
            }
            Ok(tensor_from_vec_with_template(vec![n, b_cols], rhs_data, b))
        } else {
            if b_cols != n {
                return Err(crate::Error::ShapeMismatch {
                    op: "triangular_solve",
                    lhs: vec![n],
                    rhs: vec![b_cols],
                });
            }
            let nrhs = b_rows;
            let mut rhs_transposed = transpose_col_major_data(buffers, b.host_data(), nrhs, n);
            let rhs = MatMut::from_column_major_slice_mut(
                $to_faer_slice_mut(&mut rhs_transposed),
                n,
                nrhs,
            );
            match (transpose_a, lower, unit_diagonal) {
                (false, true, false) => {
                    faer::linalg::triangular_solve::solve_upper_triangular_in_place(
                        a_mat.transpose(),
                        rhs,
                        ctx.faer_par(),
                    );
                }
                (false, true, true) => {
                    faer::linalg::triangular_solve::solve_unit_upper_triangular_in_place(
                        a_mat.transpose(),
                        rhs,
                        ctx.faer_par(),
                    );
                }
                (false, false, false) => {
                    faer::linalg::triangular_solve::solve_lower_triangular_in_place(
                        a_mat.transpose(),
                        rhs,
                        ctx.faer_par(),
                    );
                }
                (false, false, true) => {
                    faer::linalg::triangular_solve::solve_unit_lower_triangular_in_place(
                        a_mat.transpose(),
                        rhs,
                        ctx.faer_par(),
                    );
                }
                (true, true, false) => {
                    faer::linalg::triangular_solve::solve_lower_triangular_in_place(
                        a_mat,
                        rhs,
                        ctx.faer_par(),
                    );
                }
                (true, true, true) => {
                    faer::linalg::triangular_solve::solve_unit_lower_triangular_in_place(
                        a_mat,
                        rhs,
                        ctx.faer_par(),
                    );
                }
                (true, false, false) => {
                    faer::linalg::triangular_solve::solve_upper_triangular_in_place(
                        a_mat,
                        rhs,
                        ctx.faer_par(),
                    );
                }
                (true, false, true) => {
                    faer::linalg::triangular_solve::solve_unit_upper_triangular_in_place(
                        a_mat,
                        rhs,
                        ctx.faer_par(),
                    );
                }
            }
            let result = transpose_col_major_data(buffers, &rhs_transposed, n, nrhs);
            <Self as PoolScalar>::pool_release(buffers, rhs_transposed);
            Ok(tensor_from_vec_with_template(vec![nrhs, n], result, b))
        }
    }

    fn svd_2d(
        ctx: &CpuContext,
        buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
    ) -> crate::Result<Vec<TypedTensor<Self>>> {
        let (m, n) = matrix_dims(input, "svd")?;
        let k = m.min(n);
        let mat = MatRef::from_column_major_slice($to_faer_slice(input.host_data()), m, n);
        let mut u = Mat::zeros(m, k);
        let mut v = Mat::zeros(n, k);
        let mut s = Diag::zeros(k);
        let mut mem = MemBuffer::new(faer::linalg::svd::svd_scratch::<$faer_complex>(
            m,
            n,
            faer::linalg::svd::ComputeSvdVectors::Thin,
            faer::linalg::svd::ComputeSvdVectors::Thin,
            ctx.faer_par(),
            Default::default(),
        ));
        let stack = MemStack::new(&mut mem);
        faer::linalg::svd::svd(
            mat,
            s.as_mut(),
            Some(u.as_mut()),
            Some(v.as_mut()),
            ctx.faer_par(),
            stack,
            Default::default(),
        )
        .map_err(|_| decomposition_failed("svd"))?;

        let u = tensor_from_vec_with_template(
            vec![m, k],
            $vec_from_mat(buffers, u.as_ref()),
            input,
        );
        let s = tensor_from_vec_with_template(
            vec![k],
            $vec_from_real_diag(buffers, s.as_ref()),
            input,
        );
        let mut vt_data = buffers.acquire_with_capacity::<Self>(k * n);
        for j in 0..n {
            for i in 0..k {
                vt_data.push(v[(j, i)].conj());
            }
        }
        let vt = tensor_from_vec_with_template(vec![k, n], vt_data, input);

        Ok(vec![u, s, vt])
    }

    fn qr_2d(
        ctx: &CpuContext,
        buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
    ) -> crate::Result<Vec<TypedTensor<Self>>> {
        let (m, n) = matrix_dims(input, "qr")?;
        let k = m.min(n);
        let mat = MatRef::from_column_major_slice($to_faer_slice(input.host_data()), m, n);
        let block_size =
            faer::linalg::qr::no_pivoting::factor::recommended_block_size::<$faer_complex>(m, n);
        let mut qr = Mat::zeros(m, n);
        qr.copy_from(mat);
        let mut coeff = Mat::zeros(block_size, k);
        let mut mem = MemBuffer::new(
            faer::linalg::qr::no_pivoting::factor::qr_in_place_scratch::<$faer_complex>(
                m,
                n,
                block_size,
                ctx.faer_par(),
                Default::default(),
            ),
        );
        let stack = MemStack::new(&mut mem);
        faer::linalg::qr::no_pivoting::factor::qr_in_place(
            qr.as_mut(),
            coeff.as_mut(),
            ctx.faer_par(),
            stack,
            Default::default(),
        );
        let mut q = Mat::identity(m, k);
        let mut mem = MemBuffer::new(
            faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<$faer_complex>(
                m,
                block_size,
                k,
            ),
        );
        let stack = MemStack::new(&mut mem);
        faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
            qr.as_ref().subcols(0, k),
            coeff.as_ref(),
            Conj::No,
            q.as_mut(),
            ctx.faer_par(),
            stack,
        );
        let q = tensor_from_vec_with_template(
            vec![m, k],
            $vec_from_mat(buffers, q.as_ref()),
            input,
        );
        let r = tensor_from_vec_with_template(
            vec![k, n],
            $matrix_from_predicate(qr.as_ref(), k, n, |row, col| row <= col),
            input,
        );

        Ok(vec![q, r])
    }

    fn eigh_2d(
        ctx: &CpuContext,
        buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
    ) -> crate::Result<Vec<TypedTensor<Self>>> {
        let n = square_matrix_dim(input, "eigh")?;
        let mat = MatRef::from_column_major_slice($to_faer_slice(input.host_data()), n, n);
        let mut values = Diag::zeros(n);
        let mut vectors = Mat::zeros(n, n);
        let mut mem = MemBuffer::new(faer::linalg::evd::self_adjoint_evd_scratch::<$faer_complex>(
            n,
            faer::linalg::evd::ComputeEigenvectors::Yes,
            ctx.faer_par(),
            Default::default(),
        ));
        let stack = MemStack::new(&mut mem);
        faer::linalg::evd::self_adjoint_evd(
            mat,
            values.as_mut(),
            Some(vectors.as_mut()),
            ctx.faer_par(),
            stack,
            Default::default(),
        )
        .map_err(|_| decomposition_failed("eigh"))?;

        let values = tensor_from_vec_with_template(
            vec![n],
            $vec_from_real_diag(buffers, values.as_ref()),
            input,
        );
        let vectors = tensor_from_vec_with_template(
            vec![n, n],
            $vec_from_mat(buffers, vectors.as_ref()),
            input,
        );

        Ok(vec![values, vectors])
    }
        }
    };
}

impl_faer_linalg_for_complex!(
    Complex32,
    faer::c32,
    complex32_to_faer_slice,
    complex32_to_faer_slice_mut,
    complex32_vec_from_real_diag,
    complex32_vec_from_diag,
    complex32_vec_from_mat,
    complex32_matrix_from_predicate
);
impl_faer_linalg_for_complex!(
    Complex64,
    faer::c64,
    complex64_to_faer_slice,
    complex64_to_faer_slice_mut,
    complex64_vec_from_real_diag,
    complex64_vec_from_diag,
    complex64_vec_from_mat,
    complex64_matrix_from_predicate
);

pub(crate) fn cholesky<T: FaerLinalg>(
    ctx: &CpuContext,
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>> {
    if has_zero_dim(&input.shape) {
        let (n, batch_shape) = square_core_and_batch(input, "cholesky")?;
        return Ok(tensor_from_vec_with_template(
            matrix_with_batch_shape(n, n, batch_shape),
            Vec::new(),
            input,
        ));
    }
    batched_single("cholesky", buffers, input, 2, |buffers, batch| {
        T::cholesky_2d(ctx, buffers, batch)
    })
}

pub(crate) fn lu<T: FaerLinalg>(
    ctx: &CpuContext,
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> crate::Result<Vec<TypedTensor<T>>> {
    if has_zero_dim(&input.shape) {
        let (m, n, batch_shape) = matrix_core_and_batch(input, "lu")?;
        let k = m.min(n);
        let parity_elements = checked_product("lu", "batch shape", batch_shape)?;
        return Ok(vec![
            tensor_from_vec_with_template(
                matrix_with_batch_shape(m, m, batch_shape),
                Vec::new(),
                input,
            ),
            tensor_from_vec_with_template(
                matrix_with_batch_shape(m, k, batch_shape),
                Vec::new(),
                input,
            ),
            tensor_from_vec_with_template(
                matrix_with_batch_shape(k, n, batch_shape),
                Vec::new(),
                input,
            ),
            tensor_from_vec_with_template(
                batch_shape.to_vec(),
                vec![T::parity_one(); parity_elements],
                input,
            ),
        ]);
    }
    batched_multi_result("lu", buffers, input, 2, |buffers, batch| {
        T::lu_2d(ctx, buffers, batch)
    })
}

pub(crate) fn full_piv_lu<T: FaerLinalg>(
    ctx: &CpuContext,
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> crate::Result<Vec<TypedTensor<T>>> {
    if has_zero_dim(&input.shape) {
        let (n, batch_shape) = square_core_and_batch(input, "full_piv_lu")?;
        let parity_elements = checked_product("full_piv_lu", "batch shape", batch_shape)?;
        return Ok(vec![
            tensor_from_vec_with_template(
                matrix_with_batch_shape(n, n, batch_shape),
                Vec::new(),
                input,
            ),
            tensor_from_vec_with_template(
                matrix_with_batch_shape(n, n, batch_shape),
                Vec::new(),
                input,
            ),
            tensor_from_vec_with_template(
                matrix_with_batch_shape(n, n, batch_shape),
                Vec::new(),
                input,
            ),
            tensor_from_vec_with_template(
                matrix_with_batch_shape(n, n, batch_shape),
                Vec::new(),
                input,
            ),
            tensor_from_vec_with_template(
                batch_shape.to_vec(),
                vec![T::parity_one(); parity_elements],
                input,
            ),
        ]);
    }
    batched_multi_result("full_piv_lu", buffers, input, 2, |buffers, batch| {
        T::full_piv_lu_2d(ctx, buffers, batch)
    })
}

pub(crate) fn full_piv_lu_solve<T: FaerLinalg>(
    ctx: &CpuContext,
    buffers: &mut BufferPool,
    a: &TypedTensor<T>,
    b: &TypedTensor<T>,
    transpose_a: bool,
) -> crate::Result<TypedTensor<T>> {
    if has_zero_dim(&a.shape) || has_zero_dim(&b.shape) {
        let (n, a_batch_shape) = square_core_and_batch(a, "full_piv_lu_solve")?;
        let (b_rows, _, b_batch_shape) = matrix_core_and_batch(b, "full_piv_lu_solve")?;
        if b_rows != n {
            return Err(crate::Error::ShapeMismatch {
                op: "full_piv_lu_solve",
                lhs: vec![n],
                rhs: vec![b_rows],
            });
        }
        if a_batch_shape != b_batch_shape {
            return Err(crate::Error::ShapeMismatch {
                op: "full_piv_lu_solve",
                lhs: a_batch_shape.to_vec(),
                rhs: b_batch_shape.to_vec(),
            });
        }
        return Ok(tensor_from_vec_with_template(
            b.shape.clone(),
            Vec::new(),
            b,
        ));
    }
    batched_binary_result("full_piv_lu_solve", buffers, a, b, 2, 2, |buffers, a, b| {
        T::full_piv_lu_solve_2d(ctx, buffers, a, b, transpose_a)
    })
}

pub(crate) fn solve<T: FaerLinalg>(
    ctx: &CpuContext,
    buffers: &mut BufferPool,
    a: &TypedTensor<T>,
    b: &TypedTensor<T>,
    transpose_a: bool,
) -> crate::Result<TypedTensor<T>> {
    if has_zero_dim(&a.shape) || has_zero_dim(&b.shape) {
        let (n, a_batch_shape) = square_core_and_batch(a, "solve")?;
        let (b_rows, _, b_batch_shape) = matrix_core_and_batch(b, "solve")?;
        if b_rows != n {
            return Err(crate::Error::ShapeMismatch {
                op: "solve",
                lhs: vec![n],
                rhs: vec![b_rows],
            });
        }
        if a_batch_shape != b_batch_shape {
            return Err(crate::Error::ShapeMismatch {
                op: "solve",
                lhs: a_batch_shape.to_vec(),
                rhs: b_batch_shape.to_vec(),
            });
        }
        return Ok(tensor_from_vec_with_template(
            b.shape.clone(),
            Vec::new(),
            b,
        ));
    }
    batched_binary_result("solve", buffers, a, b, 2, 2, |buffers, a, b| {
        T::solve_2d(ctx, buffers, a, b, transpose_a)
    })
}

pub(crate) fn triangular_solve<T: FaerLinalg>(
    ctx: &CpuContext,
    buffers: &mut BufferPool,
    a: &TypedTensor<T>,
    b: &TypedTensor<T>,
    left_side: bool,
    lower: bool,
    transpose_a: bool,
    unit_diagonal: bool,
) -> crate::Result<TypedTensor<T>> {
    if has_zero_dim(&a.shape) || has_zero_dim(&b.shape) {
        let (n, a_batch_shape) = square_core_and_batch(a, "triangular_solve")?;
        let (b_rows, b_cols, b_batch_shape) = matrix_core_and_batch(b, "triangular_solve")?;
        let rhs_core_dim = if left_side { b_rows } else { b_cols };
        if rhs_core_dim != n {
            return Err(crate::Error::ShapeMismatch {
                op: "triangular_solve",
                lhs: vec![n],
                rhs: vec![rhs_core_dim],
            });
        }
        if a_batch_shape != b_batch_shape {
            return Err(crate::Error::ShapeMismatch {
                op: "triangular_solve",
                lhs: a_batch_shape.to_vec(),
                rhs: b_batch_shape.to_vec(),
            });
        }
        return Ok(tensor_from_vec_with_template(
            b.shape.clone(),
            Vec::new(),
            b,
        ));
    }
    batched_binary_result("triangular_solve", buffers, a, b, 2, 2, |buffers, a, b| {
        T::triangular_solve_2d(
            ctx,
            buffers,
            a,
            b,
            left_side,
            lower,
            transpose_a,
            unit_diagonal,
        )
    })
}

pub(crate) fn svd<T: FaerLinalg>(
    ctx: &CpuContext,
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> crate::Result<Vec<TypedTensor<T>>> {
    if has_zero_dim(&input.shape) {
        let (m, n, batch_shape) = matrix_core_and_batch(input, "svd")?;
        let k = m.min(n);
        return Ok(vec![
            tensor_from_vec_with_template(
                matrix_with_batch_shape(m, k, batch_shape),
                Vec::new(),
                input,
            ),
            tensor_from_vec_with_template(
                vector_with_batch_shape(k, batch_shape),
                Vec::new(),
                input,
            ),
            tensor_from_vec_with_template(
                matrix_with_batch_shape(k, n, batch_shape),
                Vec::new(),
                input,
            ),
        ]);
    }
    batched_multi_result("svd", buffers, input, 2, |buffers, batch| {
        T::svd_2d(ctx, buffers, batch)
    })
}

pub(crate) fn qr<T: FaerLinalg>(
    ctx: &CpuContext,
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> crate::Result<Vec<TypedTensor<T>>> {
    if has_zero_dim(&input.shape) {
        let (m, n, batch_shape) = matrix_core_and_batch(input, "qr")?;
        let k = m.min(n);
        return Ok(vec![
            tensor_from_vec_with_template(
                matrix_with_batch_shape(m, k, batch_shape),
                Vec::new(),
                input,
            ),
            tensor_from_vec_with_template(
                matrix_with_batch_shape(k, n, batch_shape),
                Vec::new(),
                input,
            ),
        ]);
    }
    batched_multi_result("qr", buffers, input, 2, |buffers, batch| {
        T::qr_2d(ctx, buffers, batch)
    })
}

pub(crate) fn eigh<T: FaerLinalg>(
    ctx: &CpuContext,
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> crate::Result<Vec<TypedTensor<T>>> {
    if has_zero_dim(&input.shape) {
        let (n, batch_shape) = square_core_and_batch(input, "eigh")?;
        return Ok(vec![
            tensor_from_vec_with_template(
                vector_with_batch_shape(n, batch_shape),
                Vec::new(),
                input,
            ),
            tensor_from_vec_with_template(
                matrix_with_batch_shape(n, n, batch_shape),
                Vec::new(),
                input,
            ),
        ]);
    }
    batched_multi_result("eigh", buffers, input, 2, |buffers, batch| {
        T::eigh_2d(ctx, buffers, batch)
    })
}

macro_rules! impl_eig_real_2d {
    ($name:ident, $real:ty, $complex:ty, $real_eig_to_complex_outputs:ident) => {
        fn $name(
            ctx: &CpuContext,
            buffers: &mut BufferPool,
            input: &TypedTensor<$real>,
        ) -> crate::Result<Vec<TypedTensor<$complex>>> {
            let n = square_matrix_dim(input, "eig")?;
            let mat = MatRef::from_column_major_slice(input.host_data(), n, n);
            let mut u_real = Mat::zeros(n, n);
            let mut s_re = Diag::zeros(n);
            let mut s_im = Diag::zeros(n);
            let mut mem = MemBuffer::new(faer::linalg::evd::evd_scratch::<$real>(
                n,
                faer::linalg::evd::ComputeEigenvectors::No,
                faer::linalg::evd::ComputeEigenvectors::Yes,
                ctx.faer_par(),
                Default::default(),
            ));
            let stack = MemStack::new(&mut mem);
            faer::linalg::evd::evd_real(
                mat,
                s_re.as_mut(),
                s_im.as_mut(),
                None,
                Some(u_real.as_mut()),
                ctx.faer_par(),
                stack,
                Default::default(),
            )
            .map_err(|_| decomposition_failed("eig"))?;
            let (u, s) = $real_eig_to_complex_outputs(
                buffers,
                u_real.as_ref(),
                s_re.as_ref(),
                s_im.as_ref(),
            );

            Ok(vec![
                tensor_from_vec_with_template(vec![n], s, input),
                tensor_from_vec_with_template(vec![n, n], u, input),
            ])
        }
    };
}

macro_rules! impl_eig_complex_2d {
    (
        $name:ident,
        $complex:ty,
        $faer_complex:ty,
        $to_faer_slice:ident,
        $vec_from_diag:ident,
        $vec_from_mat:ident
    ) => {
        fn $name(
            ctx: &CpuContext,
            buffers: &mut BufferPool,
            input: &TypedTensor<$complex>,
        ) -> crate::Result<Vec<TypedTensor<$complex>>> {
            let n = square_matrix_dim(input, "eig")?;
            let mat = MatRef::from_column_major_slice($to_faer_slice(input.host_data()), n, n);
            let mut u = Mat::zeros(n, n);
            let mut s = Diag::zeros(n);
            let mut mem = MemBuffer::new(faer::linalg::evd::evd_scratch::<$faer_complex>(
                n,
                faer::linalg::evd::ComputeEigenvectors::No,
                faer::linalg::evd::ComputeEigenvectors::Yes,
                ctx.faer_par(),
                Default::default(),
            ));
            let stack = MemStack::new(&mut mem);
            faer::linalg::evd::evd_cplx(
                mat,
                s.as_mut(),
                None,
                Some(u.as_mut()),
                ctx.faer_par(),
                stack,
                Default::default(),
            )
            .map_err(|_| decomposition_failed("eig"))?;

            Ok(vec![
                tensor_from_vec_with_template(vec![n], $vec_from_diag(buffers, s.as_ref()), input),
                tensor_from_vec_with_template(
                    vec![n, n],
                    $vec_from_mat(buffers, u.as_ref()),
                    input,
                ),
            ])
        }
    };
}

impl_eig_real_2d!(eig_real32_2d, f32, Complex32, real32_eig_to_complex_outputs);
impl_eig_real_2d!(eig_real64_2d, f64, Complex64, real64_eig_to_complex_outputs);
impl_eig_complex_2d!(
    eig_complex32_2d,
    Complex32,
    faer::c32,
    complex32_to_faer_slice,
    complex32_vec_from_diag,
    complex32_vec_from_mat
);
impl_eig_complex_2d!(
    eig_complex64_2d,
    Complex64,
    faer::c64,
    complex64_to_faer_slice,
    complex64_vec_from_diag,
    complex64_vec_from_mat
);

pub(crate) fn eig(
    ctx: &CpuContext,
    buffers: &mut BufferPool,
    input: &Tensor,
) -> crate::Result<Vec<Tensor>> {
    if has_zero_dim(input.shape()) {
        let (matrix_shape, batch_shape) = split_shape_core_and_batch(input.shape(), 2, "eig")?;
        let n = matrix_shape[0];
        if matrix_shape[1] != n {
            return Err(crate::Error::ShapeMismatch {
                op: "eig",
                lhs: vec![n],
                rhs: vec![matrix_shape[1]],
            });
        }
        let value_shape = vector_with_batch_shape(n, batch_shape);
        let vector_shape = matrix_with_batch_shape(n, n, batch_shape);
        return match input {
            Tensor::F32(_) | Tensor::C32(_) => Ok(vec![
                Tensor::C32(TypedTensor::from_vec_col_major(value_shape, Vec::new())),
                Tensor::C32(TypedTensor::from_vec_col_major(vector_shape, Vec::new())),
            ]),
            Tensor::F64(_) | Tensor::C64(_) => Ok(vec![
                Tensor::C64(TypedTensor::from_vec_col_major(value_shape, Vec::new())),
                Tensor::C64(TypedTensor::from_vec_col_major(vector_shape, Vec::new())),
            ]),
            _ => Err(crate::Error::BackendFailure {
                op: "eig",
                message: format!("unsupported dtype {:?}", input.dtype()),
            }),
        };
    }

    match input {
        Tensor::F32(t) => {
            Ok(
                batched_multi_convert_result("eig", buffers, t, 2, |buffers, batch| {
                    eig_real32_2d(ctx, buffers, batch)
                })?
                .into_iter()
                .map(Tensor::C32)
                .collect(),
            )
        }
        Tensor::F64(t) => {
            Ok(
                batched_multi_convert_result("eig", buffers, t, 2, |buffers, batch| {
                    eig_real64_2d(ctx, buffers, batch)
                })?
                .into_iter()
                .map(Tensor::C64)
                .collect(),
            )
        }
        Tensor::C32(t) => {
            Ok(
                batched_multi_convert_result("eig", buffers, t, 2, |buffers, batch| {
                    eig_complex32_2d(ctx, buffers, batch)
                })?
                .into_iter()
                .map(Tensor::C32)
                .collect(),
            )
        }
        Tensor::C64(t) => {
            Ok(
                batched_multi_convert_result("eig", buffers, t, 2, |buffers, batch| {
                    eig_complex64_2d(ctx, buffers, batch)
                })?
                .into_iter()
                .map(Tensor::C64)
                .collect(),
            )
        }
        _ => Err(crate::Error::BackendFailure {
            op: "eig",
            message: format!("unsupported dtype {:?}", input.dtype()),
        }),
    }
}

fn has_zero_dim(shape: &[usize]) -> bool {
    shape.contains(&0)
}

fn matrix_with_batch_shape(rows: usize, cols: usize, batch_shape: &[usize]) -> Vec<usize> {
    let mut shape = vec![rows, cols];
    shape.extend_from_slice(batch_shape);
    shape
}

fn vector_with_batch_shape(len: usize, batch_shape: &[usize]) -> Vec<usize> {
    let mut shape = vec![len];
    shape.extend_from_slice(batch_shape);
    shape
}
