use super::*;

/// Reverse-mode AD rule for matrix exponential (VJP / pullback).
///
/// Computes the gradient of the input given a cotangent for `exp(A)`.
/// Uses the auxiliary 2n x 2n matrix trick (PyTorch approach):
///
/// ```text
/// M = [[A^T, cotangent], [0, A^T]]
/// grad_A = top-right n×n block of exp(M)
/// ```
///
/// # Examples
///
/// ```
/// use tenferro_linalg::matrix_exp_rrule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 3], mem, col);
/// let cotangent = Tensor::<f64>::ones(&[3, 3], mem, col);
/// let grad_a = matrix_exp_rrule(&mut ctx, &a, &cotangent).unwrap();
/// ```
pub fn matrix_exp_rrule<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    cotangent: &Tensor<T>,
) -> AdResult<Tensor<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::MatrixExp, "matrix_exp_rrule")
        .map_err(to_ad_err)?;

    let (n, batch_dims) = validate_square(tensor).map_err(to_ad_err)?;
    let bc = batch_count(batch_dims);

    let (a_data, _) = extract_data(tensor)?;
    let (co_data, _) = extract_data(cotangent)?;

    let nn = 2 * n;
    let mut grad_data = vec![T::zero(); n * n * bc];

    for b in 0..bc {
        let a = &a_data[b * n * n..(b + 1) * n * n];
        let co = &co_data[b * n * n..(b + 1) * n * n];

        // Build 2n×2n auxiliary matrix M = [[A^T, cotangent], [0, A^T]]
        let mut m = vec![T::zero(); nn * nn];
        for j in 0..n {
            for i in 0..n {
                // A^T: transpose of A — a^T[i,j] = a[j,i] = a[j + i*n]
                let a_t_ij = a[j + i * n];
                // Top-left: A^T
                m[i + j * nn] = a_t_ij;
                // Top-right: cotangent
                m[i + (j + n) * nn] = co[i + j * n];
                // Bottom-right: A^T
                m[(i + n) + (j + n) * nn] = a_t_ij;
                // Bottom-left: already zero
            }
        }

        // Compute exp(M)
        let exp_m = matrix_exp_single(ctx, &m, nn).map_err(to_ad_err)?;

        // Extract top-right block → gradient d̄A
        for j in 0..n {
            for i in 0..n {
                grad_data[b * n * n + i + j * n] = exp_m[i + (j + n) * nn];
            }
        }
    }

    let dims = output_dims(&[n, n], batch_dims);
    tensor_from_data(grad_data, &dims).map_err(to_ad_err)
}
