use std::collections::HashMap;

use tenferro_algebra::{HasAlgebra, Scalar, Semiring};
use tenferro_device::Result;
use tenferro_tensor::Tensor;

use crate::backend::{BackendContext, EinsumBackend};
use crate::subscripts::Subscripts;
use crate::tree::ContractionTree;

use super::borrowed::{einsum, einsum_with_plan, einsum_with_subscripts};

/// Execute einsum using string notation, consuming the input tensors.
///
/// # Examples
///
/// ```ignore
/// use tenferro_algebra::Standard;
/// use tenferro_einsum::einsum_owned;
/// use tenferro_prims::{CpuBackend, CpuContext};
///
/// let mut ctx = CpuContext::new(1);
/// let out = einsum_owned::<Standard<f64>, CpuBackend>(&mut ctx, "ij,jk->ik", vec![a, b], None)
///     .unwrap();
/// ```
pub fn einsum_owned<Alg, Backend>(
    ctx: &mut BackendContext<Alg, Backend>,
    subscripts: &str,
    operands: Vec<Tensor<Alg::Scalar>>,
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Tensor<Alg::Scalar>>
where
    Alg: Semiring,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: EinsumBackend<Alg>,
{
    let refs: Vec<&Tensor<Alg::Scalar>> = operands.iter().collect();
    einsum::<Alg, Backend>(ctx, subscripts, &refs, size_dict)
}

/// Execute einsum with pre-built [`Subscripts`], consuming the input tensors.
///
/// # Examples
///
/// ```ignore
/// use tenferro_algebra::Standard;
/// use tenferro_einsum::{einsum_with_subscripts_owned, Subscripts};
/// use tenferro_prims::{CpuBackend, CpuContext};
///
/// let mut ctx = CpuContext::new(1);
/// let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
/// let out =
///     einsum_with_subscripts_owned::<Standard<f64>, CpuBackend>(&mut ctx, &subs, vec![a, b], None)
///         .unwrap();
/// ```
pub fn einsum_with_subscripts_owned<Alg, Backend>(
    ctx: &mut BackendContext<Alg, Backend>,
    subscripts: &Subscripts,
    operands: Vec<Tensor<Alg::Scalar>>,
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Tensor<Alg::Scalar>>
where
    Alg: Semiring,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: EinsumBackend<Alg>,
{
    let refs: Vec<&Tensor<Alg::Scalar>> = operands.iter().collect();
    einsum_with_subscripts::<Alg, Backend>(ctx, subscripts, &refs, size_dict)
}

/// Execute einsum with a pre-optimized [`ContractionTree`], consuming the
/// input tensors.
///
/// # Examples
///
/// ```ignore
/// use tenferro_algebra::Standard;
/// use tenferro_einsum::{einsum_with_plan_owned, ContractionTree, Subscripts};
/// use tenferro_prims::{CpuBackend, CpuContext};
///
/// let mut ctx = CpuContext::new(1);
/// let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
/// let tree = ContractionTree::optimize(&subs, &[&[2, 3], &[3, 4]]).unwrap();
/// let out =
///     einsum_with_plan_owned::<Standard<f64>, CpuBackend>(&mut ctx, &tree, vec![a, b], None)
///         .unwrap();
/// ```
pub fn einsum_with_plan_owned<Alg, Backend>(
    ctx: &mut BackendContext<Alg, Backend>,
    tree: &ContractionTree,
    operands: Vec<Tensor<Alg::Scalar>>,
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Tensor<Alg::Scalar>>
where
    Alg: Semiring,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: EinsumBackend<Alg>,
{
    let refs: Vec<&Tensor<Alg::Scalar>> = operands.iter().collect();
    einsum_with_plan::<Alg, Backend>(ctx, tree, &refs, size_dict)
}
