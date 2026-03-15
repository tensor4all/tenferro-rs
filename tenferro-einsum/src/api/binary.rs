use std::collections::HashMap;

use tenferro_algebra::{Conjugate, HasAlgebra, Scalar, Semiring};
use tenferro_device::{Error, Result};
use tenferro_tensor::Tensor;

use crate::api::{einsum_with_subscripts, einsum_with_subscripts_into};
use crate::execution::backend::{BackendContext, EinsumBackend};
use crate::syntax::subscripts::Subscripts;

fn ensure_binary_subscripts(subscripts: &Subscripts) -> Result<()> {
    if subscripts.inputs.len() != 2 {
        return Err(Error::InvalidArgument(format!(
            "binary einsum requires exactly 2 inputs, got {}",
            subscripts.inputs.len()
        )));
    }
    Ok(())
}

/// Execute a binary einsum from string notation.
///
/// This is the two-input specialization of [`crate::einsum`]. It is intended as
/// a reusable primitive for building explicit contraction paths at higher layers.
///
/// # Examples
///
/// ```ignore
/// use tenferro_algebra::Standard;
/// use tenferro_einsum::einsum_binary;
/// use tenferro_tensor::{MemoryOrder, Tensor};
/// use tenferro_prims::{CpuBackend, CpuContext};
///
/// let mut ctx = CpuContext::new(1);
/// let col = MemoryOrder::ColumnMajor;
/// let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], col).unwrap();
/// let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], col).unwrap();
/// let c =
///     einsum_binary::<Standard<f64>, CpuBackend>(&mut ctx, "ij,jk->ik", &a, &b, None).unwrap();
/// ```
pub fn einsum_binary<Alg, Backend>(
    ctx: &mut BackendContext<Alg, Backend>,
    subscripts: &str,
    left: &Tensor<Alg::Scalar>,
    right: &Tensor<Alg::Scalar>,
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Tensor<Alg::Scalar>>
where
    Alg: Semiring,
    Alg::Scalar: Scalar + Conjugate + HasAlgebra<Algebra = Alg>,
    Backend: EinsumBackend<Alg>,
{
    let subs = Subscripts::parse(subscripts)?;
    einsum_binary_with_subscripts::<Alg, Backend>(ctx, &subs, left, right, size_dict)
}

/// Execute a binary einsum from pre-parsed subscripts.
///
/// # Errors
///
/// Returns an error if `subscripts` does not contain exactly two inputs.
///
/// # Examples
///
/// ```ignore
/// use tenferro_algebra::Standard;
/// use tenferro_einsum::{einsum_binary_with_subscripts, Subscripts};
/// use tenferro_tensor::{MemoryOrder, Tensor};
/// use tenferro_prims::{CpuBackend, CpuContext};
///
/// let mut ctx = CpuContext::new(1);
/// let col = MemoryOrder::ColumnMajor;
/// let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
/// let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], col).unwrap();
/// let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], col).unwrap();
/// let c =
///     einsum_binary_with_subscripts::<Standard<f64>, CpuBackend>(&mut ctx, &subs, &a, &b, None)
///         .unwrap();
/// ```
pub fn einsum_binary_with_subscripts<Alg, Backend>(
    ctx: &mut BackendContext<Alg, Backend>,
    subscripts: &Subscripts,
    left: &Tensor<Alg::Scalar>,
    right: &Tensor<Alg::Scalar>,
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Tensor<Alg::Scalar>>
where
    Alg: Semiring,
    Alg::Scalar: Scalar + Conjugate + HasAlgebra<Algebra = Alg>,
    Backend: EinsumBackend<Alg>,
{
    ensure_binary_subscripts(subscripts)?;
    einsum_with_subscripts::<Alg, Backend>(ctx, subscripts, &[left, right], size_dict)
}

/// Execute a binary einsum and accumulate into an existing output buffer.
///
/// Computes `output = alpha * einsum(left, right) + beta * output`.
///
/// # Examples
///
/// ```ignore
/// use tenferro_algebra::Standard;
/// use tenferro_einsum::einsum_binary_into;
/// use tenferro_device::LogicalMemorySpace;
/// use tenferro_tensor::{MemoryOrder, Tensor};
/// use tenferro_prims::{CpuBackend, CpuContext};
///
/// let mut ctx = CpuContext::new(1);
/// let col = MemoryOrder::ColumnMajor;
/// let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], col).unwrap();
/// let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], col).unwrap();
/// let mut c = Tensor::<f64>::zeros(&[2, 2], LogicalMemorySpace::MainMemory, col);
/// einsum_binary_into::<Standard<f64>, CpuBackend>(
///     &mut ctx, "ij,jk->ik", &a, &b, 1.0, 0.0, &mut c, None
/// )
/// .unwrap();
/// ```
pub fn einsum_binary_into<Alg, Backend>(
    ctx: &mut BackendContext<Alg, Backend>,
    subscripts: &str,
    left: &Tensor<Alg::Scalar>,
    right: &Tensor<Alg::Scalar>,
    alpha: Alg::Scalar,
    beta: Alg::Scalar,
    output: &mut Tensor<Alg::Scalar>,
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<()>
where
    Alg: Semiring,
    Alg::Scalar: Scalar + Conjugate + HasAlgebra<Algebra = Alg>,
    Backend: EinsumBackend<Alg>,
{
    let subs = Subscripts::parse(subscripts)?;
    einsum_binary_with_subscripts_into::<Alg, Backend>(
        ctx, &subs, left, right, alpha, beta, output, size_dict,
    )
}

/// Execute a binary einsum from pre-parsed subscripts, accumulating into output.
///
/// Computes `output = alpha * einsum(left, right) + beta * output`.
///
/// # Errors
///
/// Returns an error if `subscripts` does not contain exactly two inputs.
///
/// # Examples
///
/// ```ignore
/// use tenferro_algebra::Standard;
/// use tenferro_einsum::{einsum_binary_with_subscripts_into, Subscripts};
/// use tenferro_device::LogicalMemorySpace;
/// use tenferro_tensor::{MemoryOrder, Tensor};
/// use tenferro_prims::{CpuBackend, CpuContext};
///
/// let mut ctx = CpuContext::new(1);
/// let col = MemoryOrder::ColumnMajor;
/// let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
/// let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], col).unwrap();
/// let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], col).unwrap();
/// let mut c = Tensor::<f64>::zeros(&[2, 2], LogicalMemorySpace::MainMemory, col);
/// einsum_binary_with_subscripts_into::<Standard<f64>, CpuBackend>(
///     &mut ctx, &subs, &a, &b, 1.0, 0.0, &mut c, None
/// ).unwrap();
/// ```
pub fn einsum_binary_with_subscripts_into<Alg, Backend>(
    ctx: &mut BackendContext<Alg, Backend>,
    subscripts: &Subscripts,
    left: &Tensor<Alg::Scalar>,
    right: &Tensor<Alg::Scalar>,
    alpha: Alg::Scalar,
    beta: Alg::Scalar,
    output: &mut Tensor<Alg::Scalar>,
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<()>
where
    Alg: Semiring,
    Alg::Scalar: Scalar + Conjugate + HasAlgebra<Algebra = Alg>,
    Backend: EinsumBackend<Alg>,
{
    ensure_binary_subscripts(subscripts)?;
    einsum_with_subscripts_into::<Alg, Backend>(
        ctx,
        subscripts,
        &[left, right],
        alpha,
        beta,
        output,
        size_dict,
    )
}

#[cfg(test)]
mod tests;
