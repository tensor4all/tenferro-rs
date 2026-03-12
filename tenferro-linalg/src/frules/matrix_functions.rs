use super::*;

/// Forward-mode AD rule for matrix exponential (JVP / pushforward).
///
/// Computes `exp(A)` and the Frechet derivative `d(exp(A))` in the direction `dA`.
/// Uses the auxiliary 2n x 2n matrix trick (PyTorch approach):
///
/// ```text
/// M = [[A, dA], [0, A]]
/// exp(A)    = top-left  n×n block of exp(M)
/// d(exp(A)) = top-right n×n block of exp(M)
/// ```
///
/// # Examples
///
/// ```
/// use tenferro_linalg::matrix_exp_frule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 3], mem, col);
/// let da = Tensor::<f64>::ones(&[3, 3], mem, col);
/// let (exp_a, dexp_a) = matrix_exp_frule(&mut ctx, &a, &da).unwrap();
/// ```
pub fn matrix_exp_frule<T: KernelLinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
) -> AdResult<(Tensor<T>, Tensor<T>)>
where
    T: KernelLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::MatrixExp, "matrix_exp_frule")
        .map_err(to_ad_err)?;

    let (n, batch_dims) = validate_square(tensor).map_err(to_ad_err)?;
    let bc = batch_count(batch_dims);

    let (a_data, _) = extract_data(tensor)?;
    let (da_data, _) = extract_data(tangent)?;

    let nn = 2 * n;
    let mut result_data = vec![T::zero(); n * n * bc];
    let mut tangent_data = vec![T::zero(); n * n * bc];

    for b in 0..bc {
        let a = &a_data[b * n * n..(b + 1) * n * n];
        let da = &da_data[b * n * n..(b + 1) * n * n];

        // Build 2n×2n auxiliary matrix M = [[A, dA], [0, A]]
        let mut m = vec![T::zero(); nn * nn];
        for j in 0..n {
            for i in 0..n {
                // Top-left: A
                m[i + j * nn] = a[i + j * n];
                // Top-right: dA
                m[i + (j + n) * nn] = da[i + j * n];
                // Bottom-right: A
                m[(i + n) + (j + n) * nn] = a[i + j * n];
                // Bottom-left: already zero
            }
        }

        // Compute exp(M) — call matrix_exp_single with the 2n×2n matrix
        let exp_m = matrix_exp_single(ctx, &m, nn).map_err(to_ad_err)?;

        // Extract top-left block → exp(A)
        for j in 0..n {
            for i in 0..n {
                result_data[b * n * n + i + j * n] = exp_m[i + j * nn];
            }
        }

        // Extract top-right block → d(exp(A))
        for j in 0..n {
            for i in 0..n {
                tangent_data[b * n * n + i + j * n] = exp_m[i + (j + n) * nn];
            }
        }
    }

    let dims = output_dims(&[n, n], batch_dims);
    let result = tensor_from_data(result_data, &dims).map_err(to_ad_err)?;
    let tang = tensor_from_data(tangent_data, &dims).map_err(to_ad_err)?;
    Ok((result, tang))
}
