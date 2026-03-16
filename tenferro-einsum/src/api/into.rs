use std::collections::HashMap;

use tenferro_algebra::{Conjugate, HasAlgebra, Scalar, Semiring};
use tenferro_device::Result;
use tenferro_tensor::Tensor;

use crate::execution::backend::{BackendContext, EinsumBackend};
use crate::execution::execute::{execute_nested, execute_tree};
use crate::execution::pool::BufferPool;
use crate::planning::tree::ContractionTree;
use crate::syntax::nested::NestedEinsum;
use crate::syntax::subscripts::Subscripts;

use super::canonical::canonicalize_col_major_operands;

/// Execute einsum using string notation, accumulating into an existing output.
///
/// # Examples
///
/// ```ignore
/// use tenferro_algebra::Standard;
/// use tenferro_einsum::einsum_into;
/// use tenferro_prims::{CpuBackend, CpuContext};
///
/// let mut ctx = CpuContext::new(1);
/// einsum_into::<Standard<f64>, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b], 1.0, 0.0, &mut c, None)
///     .unwrap();
/// ```
pub fn einsum_into<Alg, Backend>(
    ctx: &mut BackendContext<Alg, Backend>,
    subscripts: &str,
    operands: &[&Tensor<Alg::Scalar>],
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
    if subscripts.contains('(') {
        let nested = NestedEinsum::parse(subscripts)?;
        let result = execute_nested::<Alg, Backend>(ctx, &nested, operands, size_dict)?;
        let identity_subs = Subscripts {
            inputs: vec![subs.output.clone()],
            output: subs.output,
        };
        einsum_with_subscripts_into::<Alg, Backend>(
            ctx,
            &identity_subs,
            &[&result],
            alpha,
            beta,
            output,
            size_dict,
        )
    } else {
        einsum_with_subscripts_into::<Alg, Backend>(
            ctx, &subs, operands, alpha, beta, output, size_dict,
        )
    }
}

/// Execute einsum with pre-built [`Subscripts`], accumulating into an existing output.
///
/// # Examples
///
/// ```ignore
/// use tenferro_algebra::Standard;
/// use tenferro_einsum::{einsum_with_subscripts_into, Subscripts};
/// use tenferro_prims::{CpuBackend, CpuContext};
///
/// let mut ctx = CpuContext::new(1);
/// let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
/// einsum_with_subscripts_into::<Standard<f64>, CpuBackend>(
///     &mut ctx, &subs, &[&a, &b], 1.0, 0.0, &mut c, None,
/// ).unwrap();
/// ```
pub fn einsum_with_subscripts_into<Alg, Backend>(
    ctx: &mut BackendContext<Alg, Backend>,
    subscripts: &Subscripts,
    operands: &[&Tensor<Alg::Scalar>],
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
    let shapes: Vec<&[usize]> = operands.iter().map(|t| t.dims()).collect();
    let tree = ContractionTree::optimize(subscripts, &shapes)?;
    einsum_with_plan_into::<Alg, Backend>(ctx, &tree, operands, alpha, beta, output, size_dict)
}

/// Execute N-ary einsum with an explicit pairwise contraction path, accumulating
/// into an existing output tensor.
///
/// # Examples
///
/// ```ignore
/// use tenferro_algebra::Standard;
/// use tenferro_einsum::{einsum_with_path_into, Subscripts};
/// use tenferro_prims::{CpuBackend, CpuContext};
///
/// let mut ctx = CpuContext::new(1);
/// let subs = Subscripts::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);
/// let pairs = vec![(1, 2), (0, 3)];
/// einsum_with_path_into::<Standard<f64>, CpuBackend>(
///     &mut ctx, &subs, &pairs, &[&a, &b, &c], 1.0, 0.0, &mut out, None,
/// ).unwrap();
/// ```
pub fn einsum_with_path_into<Alg, Backend>(
    ctx: &mut BackendContext<Alg, Backend>,
    subscripts: &Subscripts,
    pairs: &[(usize, usize)],
    operands: &[&Tensor<Alg::Scalar>],
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
    let shapes: Vec<&[usize]> = operands.iter().map(|t| t.dims()).collect();
    let tree = ContractionTree::from_pairs(subscripts, &shapes, pairs)?;
    einsum_with_plan_into::<Alg, Backend>(ctx, &tree, operands, alpha, beta, output, size_dict)
}

/// Execute einsum with a pre-optimized [`ContractionTree`], accumulating
/// into an existing output.
///
/// # Examples
///
/// ```ignore
/// use tenferro_algebra::Standard;
/// use tenferro_einsum::{einsum_with_plan_into, ContractionTree, Subscripts};
/// use tenferro_prims::{CpuBackend, CpuContext};
///
/// let mut ctx = CpuContext::new(1);
/// let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
/// let tree = ContractionTree::optimize(&subs, &[&[2, 3], &[3, 4]]).unwrap();
/// einsum_with_plan_into::<Standard<f64>, CpuBackend>(
///     &mut ctx, &tree, &[&a, &b], 1.0, 0.0, &mut c, None,
/// ).unwrap();
/// ```
pub fn einsum_with_plan_into<Alg, Backend>(
    ctx: &mut BackendContext<Alg, Backend>,
    tree: &ContractionTree,
    operands: &[&Tensor<Alg::Scalar>],
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
    let _ = size_dict;
    let canonical_operands = canonicalize_col_major_operands(operands);
    let canonical_refs: Vec<&Tensor<Alg::Scalar>> = canonical_operands.iter().collect();
    let mut pool = BufferPool::new();
    execute_tree::<Alg, Backend>(
        ctx,
        tree,
        &canonical_refs,
        alpha,
        beta,
        output,
        &mut pool,
        false,
    )
}
