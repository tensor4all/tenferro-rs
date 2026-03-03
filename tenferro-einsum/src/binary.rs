use std::collections::HashMap;

use tenferro_algebra::{Algebra, HasAlgebra, Scalar};
use tenferro_device::{Error, Result};
use tenferro_prims::TensorPrims;
use tenferro_tensor::Tensor;

use crate::api::{einsum_with_subscripts, einsum_with_subscripts_into};
use crate::subscripts::Subscripts;

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
/// use tenferro_einsum::einsum_binary;
/// use tenferro_prims::{CpuBackend, CpuContext};
///
/// let mut ctx = CpuContext::new(1);
/// let c = einsum_binary::<_, CpuBackend>(&mut ctx, "ij,jk->ik", &a, &b, None).unwrap();
/// ```
pub fn einsum_binary<Alg, Backend>(
    ctx: &mut Backend::Context,
    subscripts: &str,
    left: &Tensor<Alg::Scalar>,
    right: &Tensor<Alg::Scalar>,
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Tensor<Alg::Scalar>>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
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
/// use tenferro_einsum::{einsum_binary_with_subscripts, Subscripts};
/// use tenferro_prims::{CpuBackend, CpuContext};
///
/// let mut ctx = CpuContext::new(1);
/// let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
/// let c = einsum_binary_with_subscripts::<_, CpuBackend>(&mut ctx, &subs, &a, &b, None).unwrap();
/// ```
pub fn einsum_binary_with_subscripts<Alg, Backend>(
    ctx: &mut Backend::Context,
    subscripts: &Subscripts,
    left: &Tensor<Alg::Scalar>,
    right: &Tensor<Alg::Scalar>,
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Tensor<Alg::Scalar>>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
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
/// use tenferro_einsum::einsum_binary_into;
/// use tenferro_prims::{CpuBackend, CpuContext};
///
/// let mut ctx = CpuContext::new(1);
/// einsum_binary_into::<_, CpuBackend>(&mut ctx, "ij,jk->ik", &a, &b, 1.0, 0.0, &mut c, None).unwrap();
/// ```
pub fn einsum_binary_into<Alg, Backend>(
    ctx: &mut Backend::Context,
    subscripts: &str,
    left: &Tensor<Alg::Scalar>,
    right: &Tensor<Alg::Scalar>,
    alpha: Alg::Scalar,
    beta: Alg::Scalar,
    output: &mut Tensor<Alg::Scalar>,
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<()>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
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
/// use tenferro_einsum::{einsum_binary_with_subscripts_into, Subscripts};
/// use tenferro_prims::{CpuBackend, CpuContext};
///
/// let mut ctx = CpuContext::new(1);
/// let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
/// einsum_binary_with_subscripts_into::<_, CpuBackend>(
///     &mut ctx, &subs, &a, &b, 1.0, 0.0, &mut c, None
/// ).unwrap();
/// ```
pub fn einsum_binary_with_subscripts_into<Alg, Backend>(
    ctx: &mut Backend::Context,
    subscripts: &Subscripts,
    left: &Tensor<Alg::Scalar>,
    right: &Tensor<Alg::Scalar>,
    alpha: Alg::Scalar,
    beta: Alg::Scalar,
    output: &mut Tensor<Alg::Scalar>,
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<()>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
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
mod tests {
    use tenferro_algebra::Standard;
    use tenferro_device::LogicalMemorySpace;
    use tenferro_prims::{CpuBackend, CpuContext};
    use tenferro_tensor::{MemoryOrder, Tensor};

    use super::{einsum_binary, einsum_binary_into};

    const COL: MemoryOrder = MemoryOrder::ColumnMajor;
    const MEM: LogicalMemorySpace = LogicalMemorySpace::MainMemory;

    fn mat(data: &[f64], dims: &[usize]) -> Tensor<f64> {
        Tensor::from_slice(data, dims, COL).unwrap()
    }

    #[test]
    fn binary_matmul_matches_expected_values() {
        let mut ctx = CpuContext::new(1);
        let a = mat(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = mat(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let c = einsum_binary::<Standard<f64>, CpuBackend>(&mut ctx, "ij,jk->ik", &a, &b, None)
            .unwrap();
        let data = c.buffer().as_slice().unwrap();
        assert_eq!(
            data[c.offset() as usize..c.offset() as usize + 4],
            [23.0, 34.0, 31.0, 46.0]
        );
    }

    #[test]
    fn binary_into_accumulates_with_alpha_beta() {
        let mut ctx = CpuContext::new(1);
        let a = mat(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = mat(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let mut out = Tensor::<f64>::zeros(&[2, 2], MEM, COL);
        einsum_binary_into::<Standard<f64>, CpuBackend>(
            &mut ctx,
            "ij,jk->ik",
            &a,
            &b,
            1.0,
            0.0,
            &mut out,
            None,
        )
        .unwrap();
        einsum_binary_into::<Standard<f64>, CpuBackend>(
            &mut ctx,
            "ij,jk->ik",
            &a,
            &b,
            1.0,
            1.0,
            &mut out,
            None,
        )
        .unwrap();
        let data = out.buffer().as_slice().unwrap();
        assert_eq!(
            data[out.offset() as usize..out.offset() as usize + 4],
            [46.0, 68.0, 62.0, 92.0]
        );
    }

    #[test]
    fn binary_rejects_non_binary_notation() {
        let mut ctx = CpuContext::new(1);
        let a = Tensor::<f64>::zeros(&[2, 2], MEM, COL);
        let b = Tensor::<f64>::zeros(&[2, 2], MEM, COL);
        let result =
            einsum_binary::<Standard<f64>, CpuBackend>(&mut ctx, "ij,jk,kl->il", &a, &b, None);
        match result {
            Ok(_) => panic!("expected binary notation validation error"),
            Err(err) => {
                assert!(format!("{err}").contains("binary einsum requires exactly 2 inputs"))
            }
        }
    }
}
