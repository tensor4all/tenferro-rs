//! Shared tensor-level CPU implementation of linalg operations.
//!
//! This module delegates batched tensor operations to the slice-level backend
//! selected in [`super::cpu`].

use num_traits::Zero;
use tenferro_algebra::Scalar;
use tenferro_device::{Error, Result};
use tenferro_tensor::Tensor;

use super::col_major_strides;
use super::tensor_api::{
    CholeskyTensorExResult, EigTensorResult, EigenTensorResult, LuTensorExResult, LuTensorResult,
    QrTensorResult, SolveTensorExResult, SvdTensorResult,
};
use super::tensor_helpers::{
    batch_count, ensure_col_major, extract_contiguous_slice, validate_matrix_shape,
    validate_solve_rhs_shape, validate_square,
};
use crate::KernelLinalgScalar;

/// Create a tensor from data in column-major layout.
fn tensor_from_data<T: Scalar>(data: Vec<T>, dims: &[usize]) -> Result<Tensor<T>> {
    let strides = col_major_strides(dims);
    Tensor::from_vec(data, dims, &strides, 0)
}

fn first_zero_pivot_from_u<T: KernelLinalgScalar>(
    u: &[T],
    diag_len: usize,
    leading_dim: usize,
) -> i32 {
    for i in 0..diag_len {
        if u[i * (leading_dim + 1)] == T::zero() {
            return (i + 1) as i32;
        }
    }
    0
}

fn extract_leading_principal_minor<T: KernelLinalgScalar>(
    a: &[T],
    n: usize,
    k: usize,
    minor: &mut [T],
) {
    for col in 0..k {
        let src = col * n;
        let dst = col * k;
        minor[dst..dst + k].copy_from_slice(&a[src..src + k]);
    }
}

fn first_failing_leading_principal_minor<T: KernelLinalgScalar>(
    a: &[T],
    n: usize,
    minor_a: &mut [T],
    minor_l: &mut [T],
) -> Result<i32> {
    for k in 1..=n {
        let minor_len = k * k;
        extract_leading_principal_minor(a, n, k, &mut minor_a[..minor_len]);
        minor_l[..minor_len].fill(T::zero());
        if super::cpu::cholesky_slices(&minor_a[..minor_len], k, &mut minor_l[..minor_len]).is_err()
        {
            return Ok(k as i32);
        }
    }

    Err(Error::DeviceError(
        "CPU cholesky_ex failed a batch but no failing leading principal minor was found".into(),
    ))
}

/// Solve `A x = b` while preserving successful batch payloads and reporting per-batch status.
pub(crate) fn solve_ex<T>(
    _ctx: &mut tenferro_prims::CpuContext,
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> Result<SolveTensorExResult<T>>
where
    T: KernelLinalgScalar,
{
    let (n, batch_dims) = validate_square(a)?;
    let rhs = validate_solve_rhs_shape(b, n, batch_dims, "solve_ex")?;
    let nrhs = rhs.nrhs;
    let bc = batch_count(batch_dims);

    let a_contig = ensure_col_major(a);
    let b_contig = ensure_col_major(b);
    let a_data = extract_contiguous_slice(&a_contig)?;
    let b_data = extract_contiguous_slice(&b_contig)?;
    let a_off = a_contig.offset() as usize;
    let b_off = b_contig.offset() as usize;

    let mat_a = n * n;
    let mat_b = n * nrhs;
    let mut solution_data = vec![T::zero(); mat_b * bc];
    let mut info = vec![0i32; bc];

    let mut perm = vec![0usize; n];
    let mut l_buf = vec![T::zero(); mat_a];
    let mut u_buf = vec![T::zero(); mat_a];

    for i in 0..bc {
        let a_slice = &a_data[a_off + i * mat_a..a_off + (i + 1) * mat_a];
        let b_slice = &b_data[b_off + i * mat_b..b_off + (i + 1) * mat_b];

        perm.fill(0);
        l_buf.fill(T::zero());
        u_buf.fill(T::zero());
        super::cpu::lu_slices(a_slice, n, n, &mut perm, &mut l_buf, &mut u_buf)?;

        let batch_info = first_zero_pivot_from_u(&u_buf, n, n);
        info[i] = batch_info;
        if batch_info == 0 {
            let solution_slice = &mut solution_data[i * mat_b..(i + 1) * mat_b];
            super::cpu::solve_slices(a_slice, b_slice, n, nrhs, solution_slice)?;
        }
    }

    Ok(SolveTensorExResult {
        solution: tensor_from_data(solution_data, &rhs.output_dims)?,
        info,
    })
}

/// Solve `A x = b` via the selected CPU slice backend.
pub(crate) fn solve<T>(
    _ctx: &mut tenferro_prims::CpuContext,
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> Result<Tensor<T>>
where
    T: KernelLinalgScalar,
{
    let (n, batch_dims) = validate_square(a)?;
    let rhs = validate_solve_rhs_shape(b, n, batch_dims, "solve")?;
    let nrhs = rhs.nrhs;
    let bc = batch_count(batch_dims);

    let a_contig = ensure_col_major(a);
    let b_contig = ensure_col_major(b);
    let a_data = extract_contiguous_slice(&a_contig)?;
    let b_data = extract_contiguous_slice(&b_contig)?;
    let a_off = a_contig.offset() as usize;
    let b_off = b_contig.offset() as usize;

    let mat_a = n * n;
    let mat_b = n * nrhs;
    let mut x_data = vec![T::zero(); mat_b * bc];

    for i in 0..bc {
        let a_slice = &a_data[a_off + i * mat_a..a_off + (i + 1) * mat_a];
        let b_slice = &b_data[b_off + i * mat_b..b_off + (i + 1) * mat_b];
        let x_slice = &mut x_data[i * mat_b..(i + 1) * mat_b];
        super::cpu::solve_slices(a_slice, b_slice, n, nrhs, x_slice)?;
    }

    tensor_from_data(x_data, &rhs.output_dims)
}

/// Solve triangular `A x = b` via the selected CPU slice backend.
pub(crate) fn solve_triangular<T>(
    _ctx: &mut tenferro_prims::CpuContext,
    a: &Tensor<T>,
    b: &Tensor<T>,
    upper: bool,
) -> Result<Tensor<T>>
where
    T: KernelLinalgScalar,
{
    let (n, batch_dims) = validate_square(a)?;
    let rhs = validate_solve_rhs_shape(b, n, batch_dims, "solve_triangular")?;
    let nrhs = rhs.nrhs;
    let bc = batch_count(batch_dims);

    let a_contig = ensure_col_major(a);
    let b_contig = ensure_col_major(b);
    let a_data = extract_contiguous_slice(&a_contig)?;
    let b_data = extract_contiguous_slice(&b_contig)?;
    let a_off = a_contig.offset() as usize;
    let b_off = b_contig.offset() as usize;

    let mat_a = n * n;
    let mat_b = n * nrhs;
    let mut x_data = vec![T::zero(); mat_b * bc];

    for i in 0..bc {
        let a_slice = &a_data[a_off + i * mat_a..a_off + (i + 1) * mat_a];
        let b_slice = &b_data[b_off + i * mat_b..b_off + (i + 1) * mat_b];
        let x_slice = &mut x_data[i * mat_b..(i + 1) * mat_b];
        super::cpu::solve_triangular_slices(a_slice, b_slice, n, nrhs, upper, x_slice)?;
    }

    tensor_from_data(x_data, &rhs.output_dims)
}

/// Thin QR decomposition via the selected CPU slice backend.
pub(crate) fn qr<T>(
    _ctx: &mut tenferro_prims::CpuContext,
    a: &Tensor<T>,
) -> Result<QrTensorResult<T>>
where
    T: KernelLinalgScalar,
{
    let (m, n, batch_dims) = validate_matrix_shape(a)?;
    let k = m.min(n);
    let bc = batch_count(batch_dims);

    let a_contig = ensure_col_major(a);
    let a_data = extract_contiguous_slice(&a_contig)?;
    let a_off = a_contig.offset() as usize;
    let mat_size = m * n;

    let mut q_data = vec![T::zero(); m * k * bc];
    let mut r_data = vec![T::zero(); k * n * bc];

    let mut q_buf = vec![T::zero(); m * k];
    let mut r_buf = vec![T::zero(); k * n];

    for i in 0..bc {
        let a_slice = &a_data[a_off + i * mat_size..a_off + (i + 1) * mat_size];
        q_buf.fill(T::zero());
        r_buf.fill(T::zero());
        super::cpu::qr_slices(a_slice, m, n, &mut q_buf, &mut r_buf)?;
        q_data[i * m * k..(i + 1) * m * k].copy_from_slice(&q_buf);
        r_data[i * k * n..(i + 1) * k * n].copy_from_slice(&r_buf);
    }

    let mut q_shape = vec![m, k];
    q_shape.extend_from_slice(batch_dims);
    let mut r_shape = vec![k, n];
    r_shape.extend_from_slice(batch_dims);

    Ok(QrTensorResult {
        q: tensor_from_data(q_data, &q_shape)?,
        r: tensor_from_data(r_data, &r_shape)?,
    })
}

/// Thin SVD via the selected CPU slice backend.
pub(crate) fn thin_svd<T>(
    _ctx: &mut tenferro_prims::CpuContext,
    a: &Tensor<T>,
) -> Result<SvdTensorResult<T>>
where
    T: KernelLinalgScalar,
{
    let (m, n, batch_dims) = validate_matrix_shape(a)?;
    let k = m.min(n);
    let bc = batch_count(batch_dims);

    let a_contig = ensure_col_major(a);
    let a_data = extract_contiguous_slice(&a_contig)?;
    let a_off = a_contig.offset() as usize;
    let mat_size = m * n;

    let mut u_data = vec![T::zero(); m * k * bc];
    let mut s_data = vec![T::Real::zero(); k * bc];
    let mut vt_data = vec![T::zero(); k * n * bc];

    let mut u_buf = vec![T::zero(); m * k];
    let mut s_buf = vec![T::Real::zero(); k];
    let mut vt_buf = vec![T::zero(); k * n];

    for i in 0..bc {
        let a_slice = &a_data[a_off + i * mat_size..a_off + (i + 1) * mat_size];
        super::cpu::thin_svd_slices(a_slice, m, n, &mut u_buf, &mut s_buf, &mut vt_buf)?;
        u_data[i * m * k..(i + 1) * m * k].copy_from_slice(&u_buf);
        s_data[i * k..(i + 1) * k].copy_from_slice(&s_buf);
        vt_data[i * k * n..(i + 1) * k * n].copy_from_slice(&vt_buf);
    }

    let mut u_shape = vec![m, k];
    u_shape.extend_from_slice(batch_dims);
    let mut s_shape = vec![k];
    s_shape.extend_from_slice(batch_dims);
    let mut vt_shape = vec![k, n];
    vt_shape.extend_from_slice(batch_dims);

    Ok(SvdTensorResult {
        u: tensor_from_data(u_data, &u_shape)?,
        s: tensor_from_data(s_data, &s_shape)?,
        vt: tensor_from_data(vt_data, &vt_shape)?,
    })
}

/// LU factorization via the selected CPU slice backend.
pub(crate) fn lu_factor<T>(
    _ctx: &mut tenferro_prims::CpuContext,
    a: &Tensor<T>,
) -> Result<LuTensorResult<T>>
where
    T: KernelLinalgScalar,
{
    let (m, n, batch_dims) = validate_matrix_shape(a)?;
    let k = m.min(n);
    let bc = batch_count(batch_dims);

    let a_contig = ensure_col_major(a);
    let a_data = extract_contiguous_slice(&a_contig)?;
    let a_off = a_contig.offset() as usize;
    let mat_size = m * n;

    let mut l_data = vec![T::zero(); m * k * bc];
    let mut u_data = vec![T::zero(); k * n * bc];
    let mut all_pivots = vec![0i32; m * bc];

    let mut perm = vec![0usize; m];
    let mut l_buf = vec![T::zero(); m * k];
    let mut u_buf = vec![T::zero(); k * n];

    for i in 0..bc {
        let a_slice = &a_data[a_off + i * mat_size..a_off + (i + 1) * mat_size];
        perm.fill(0);
        l_buf.fill(T::zero());
        u_buf.fill(T::zero());
        super::cpu::lu_slices(a_slice, m, n, &mut perm, &mut l_buf, &mut u_buf)?;
        l_data[i * m * k..(i + 1) * m * k].copy_from_slice(&l_buf);
        u_data[i * k * n..(i + 1) * k * n].copy_from_slice(&u_buf);
        for (j, &p) in perm.iter().enumerate() {
            all_pivots[i * m + j] = p as i32;
        }
    }

    let mut l_shape = vec![m, k];
    l_shape.extend_from_slice(batch_dims);
    let mut u_shape = vec![k, n];
    u_shape.extend_from_slice(batch_dims);

    Ok(LuTensorResult {
        l: tensor_from_data(l_data, &l_shape)?,
        u: tensor_from_data(u_data, &u_shape)?,
        pivots: all_pivots,
    })
}

/// LU factorization while preserving payloads and reporting per-batch status.
pub(crate) fn lu_factor_ex<T>(
    _ctx: &mut tenferro_prims::CpuContext,
    a: &Tensor<T>,
) -> Result<LuTensorExResult<T>>
where
    T: KernelLinalgScalar,
{
    let (m, n, batch_dims) = validate_matrix_shape(a)?;
    let k = m.min(n);
    let bc = batch_count(batch_dims);

    let a_contig = ensure_col_major(a);
    let a_data = extract_contiguous_slice(&a_contig)?;
    let a_off = a_contig.offset() as usize;
    let mat_size = m * n;

    let mut l_data = vec![T::zero(); m * k * bc];
    let mut u_data = vec![T::zero(); k * n * bc];
    let mut all_pivots = vec![0i32; m * bc];
    let mut info = vec![0i32; bc];

    let mut perm = vec![0usize; m];
    let mut l_buf = vec![T::zero(); m * k];
    let mut u_buf = vec![T::zero(); k * n];

    for i in 0..bc {
        let a_slice = &a_data[a_off + i * mat_size..a_off + (i + 1) * mat_size];
        perm.fill(0);
        l_buf.fill(T::zero());
        u_buf.fill(T::zero());
        super::cpu::lu_slices(a_slice, m, n, &mut perm, &mut l_buf, &mut u_buf)?;

        l_data[i * m * k..(i + 1) * m * k].copy_from_slice(&l_buf);
        u_data[i * k * n..(i + 1) * k * n].copy_from_slice(&u_buf);
        info[i] = first_zero_pivot_from_u(&u_buf, k, k);
        for (j, &p) in perm.iter().enumerate() {
            all_pivots[i * m + j] = p as i32;
        }
    }

    let mut l_shape = vec![m, k];
    l_shape.extend_from_slice(batch_dims);
    let mut u_shape = vec![k, n];
    u_shape.extend_from_slice(batch_dims);

    Ok(LuTensorExResult {
        l: tensor_from_data(l_data, &l_shape)?,
        u: tensor_from_data(u_data, &u_shape)?,
        pivots: all_pivots,
        info,
    })
}

/// Cholesky decomposition via the selected CPU slice backend.
pub(crate) fn cholesky<T>(_ctx: &mut tenferro_prims::CpuContext, a: &Tensor<T>) -> Result<Tensor<T>>
where
    T: KernelLinalgScalar,
{
    let (n, batch_dims) = validate_square(a)?;
    let bc = batch_count(batch_dims);

    let a_contig = ensure_col_major(a);
    let a_data = extract_contiguous_slice(&a_contig)?;
    let a_off = a_contig.offset() as usize;
    let mat_size = n * n;

    let mut l_data = vec![T::zero(); mat_size * bc];
    let mut l_buf = vec![T::zero(); mat_size];

    for i in 0..bc {
        let a_slice = &a_data[a_off + i * mat_size..a_off + (i + 1) * mat_size];
        l_buf.fill(T::zero());
        super::cpu::cholesky_slices(a_slice, n, &mut l_buf)?;
        l_data[i * mat_size..(i + 1) * mat_size].copy_from_slice(&l_buf);
    }

    let mut out_shape = vec![n, n];
    out_shape.extend_from_slice(batch_dims);
    tensor_from_data(l_data, &out_shape)
}

/// Cholesky decomposition while preserving payloads and reporting per-batch status.
pub(crate) fn cholesky_ex<T>(
    _ctx: &mut tenferro_prims::CpuContext,
    a: &Tensor<T>,
) -> Result<CholeskyTensorExResult<T>>
where
    T: KernelLinalgScalar,
{
    let (n, batch_dims) = validate_square(a)?;
    let bc = batch_count(batch_dims);

    let a_contig = ensure_col_major(a);
    let a_data = extract_contiguous_slice(&a_contig)?;
    let a_off = a_contig.offset() as usize;
    let mat_size = n * n;

    let mut l_data = vec![T::zero(); mat_size * bc];
    let mut info = vec![0i32; bc];

    let mut l_buf = vec![T::zero(); mat_size];
    let mut minor_a_buf = vec![T::zero(); mat_size];
    let mut minor_l_buf = vec![T::zero(); mat_size];

    for i in 0..bc {
        let a_slice = &a_data[a_off + i * mat_size..a_off + (i + 1) * mat_size];
        l_buf.fill(T::zero());

        match super::cpu::cholesky_slices(a_slice, n, &mut l_buf) {
            Ok(()) => {
                l_data[i * mat_size..(i + 1) * mat_size].copy_from_slice(&l_buf);
            }
            Err(_) => {
                info[i] = first_failing_leading_principal_minor(
                    a_slice,
                    n,
                    &mut minor_a_buf,
                    &mut minor_l_buf,
                )?;
            }
        }
    }

    let mut out_shape = vec![n, n];
    out_shape.extend_from_slice(batch_dims);
    Ok(CholeskyTensorExResult {
        l: tensor_from_data(l_data, &out_shape)?,
        info,
    })
}

/// Hermitian eigendecomposition via the selected CPU slice backend.
pub(crate) fn eigen_sym<T>(
    _ctx: &mut tenferro_prims::CpuContext,
    a: &Tensor<T>,
) -> Result<EigenTensorResult<T>>
where
    T: KernelLinalgScalar,
{
    let (n, batch_dims) = validate_square(a)?;
    let bc = batch_count(batch_dims);
    if n == 0 || bc == 0 {
        let mut val_shape = vec![n];
        val_shape.extend_from_slice(batch_dims);
        let mut vec_shape = vec![n, n];
        vec_shape.extend_from_slice(batch_dims);
        return Ok(EigenTensorResult {
            values: tensor_from_data(Vec::new(), &val_shape)?,
            vectors: tensor_from_data(Vec::new(), &vec_shape)?,
        });
    }

    let a_contig = ensure_col_major(a);
    let a_data = extract_contiguous_slice(&a_contig)?;
    let a_off = a_contig.offset() as usize;
    let mat_size = n * n;

    let mut val_data = vec![T::Real::zero(); n * bc];
    let mut vec_data = vec![T::zero(); mat_size * bc];

    let mut val_buf = vec![T::Real::zero(); n];
    let mut vec_buf = vec![T::zero(); mat_size];

    for i in 0..bc {
        let a_slice = &a_data[a_off + i * mat_size..a_off + (i + 1) * mat_size];
        super::cpu::eigen_sym_slices(a_slice, n, &mut val_buf, &mut vec_buf)?;
        val_data[i * n..(i + 1) * n].copy_from_slice(&val_buf);
        vec_data[i * mat_size..(i + 1) * mat_size].copy_from_slice(&vec_buf);
    }

    let mut val_shape = vec![n];
    val_shape.extend_from_slice(batch_dims);
    let mut vec_shape = vec![n, n];
    vec_shape.extend_from_slice(batch_dims);

    Ok(EigenTensorResult {
        values: tensor_from_data(val_data, &val_shape)?,
        vectors: tensor_from_data(vec_data, &vec_shape)?,
    })
}

/// General eigendecomposition via the selected CPU slice backend.
pub(crate) fn eig<T>(
    _ctx: &mut tenferro_prims::CpuContext,
    a: &Tensor<T>,
) -> Result<EigTensorResult<T>>
where
    T: KernelLinalgScalar,
{
    let (n, batch_dims) = validate_square(a)?;
    let bc = batch_count(batch_dims);

    let a_contig = ensure_col_major(a);
    let a_data = extract_contiguous_slice(&a_contig)?;
    let a_off = a_contig.offset() as usize;
    let mat_size = n * n;

    let (val_ri_len, vec_ri_len) = super::cpu::eig_buffer_sizes::<T>(n);

    let mut val_ri = vec![T::zero(); val_ri_len];
    let mut vec_ri = vec![T::zero(); vec_ri_len];

    let mut all_values = vec![T::Complex::zero(); n * bc];
    let mut all_vectors = vec![T::Complex::zero(); mat_size * bc];

    for i in 0..bc {
        let a_slice = &a_data[a_off + i * mat_size..a_off + (i + 1) * mat_size];
        super::cpu::eig_slices(a_slice, n, &mut val_ri, &mut vec_ri)?;

        super::cpu::eig_ri_to_complex::<T>(
            n,
            &val_ri,
            &vec_ri,
            &mut all_values[i * n..(i + 1) * n],
            &mut all_vectors[i * mat_size..(i + 1) * mat_size],
        );
    }

    let mut val_shape = vec![n];
    val_shape.extend_from_slice(batch_dims);
    let mut vec_shape = vec![n, n];
    vec_shape.extend_from_slice(batch_dims);

    Ok(EigTensorResult {
        values: tensor_from_data(all_values, &val_shape)?,
        vectors: tensor_from_data(all_vectors, &vec_shape)?,
    })
}
