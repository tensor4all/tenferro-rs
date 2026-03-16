use std::collections::HashMap;

use tenferro_algebra::{Conjugate, HasAlgebra, Scalar, Semiring};
use tenferro_device::Result;
use tenferro_tensor::Tensor;

use super::canonical::canonicalize_col_major_operands_owned;
use crate::execution::backend::{BackendContext, EinsumBackend};
use crate::execution::execute::execute_nested;
use crate::execution::pool::BufferPool;
use crate::execution::util::{compute_output_shape, infer_memory_space};
use crate::planning::tree::ContractionTree;
use crate::syntax::nested::NestedEinsum;
use crate::syntax::subscripts::Subscripts;

#[cfg(test)]
mod tests;

/// Execute einsum using string notation, consuming the input tensors.
///
/// Unlike the borrowed entry point, this keeps canonicalization on an owning
/// path and only borrows after the owned operands have been normalized for
/// execution.
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
    Alg::Scalar: Scalar + Conjugate + HasAlgebra<Algebra = Alg>,
    Backend: EinsumBackend<Alg>,
{
    let subs = Subscripts::parse(subscripts)?;
    if subscripts.contains('(') {
        let nested = NestedEinsum::parse(subscripts)?;
        let canonical_operands = canonicalize_col_major_operands_owned(operands);
        let refs: Vec<&Tensor<Alg::Scalar>> = canonical_operands.iter().collect();
        return execute_nested::<Alg, Backend>(ctx, &nested, &refs, size_dict);
    }

    einsum_with_subscripts_owned::<Alg, Backend>(ctx, &subs, operands, size_dict)
}

/// Execute einsum with pre-built [`Subscripts`], consuming the input tensors.
///
/// This preserves the owned execution path through planning and operand
/// normalization.
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
    Alg::Scalar: Scalar + Conjugate + HasAlgebra<Algebra = Alg>,
    Backend: EinsumBackend<Alg>,
{
    let shapes: Vec<&[usize]> = operands.iter().map(|t| t.dims()).collect();
    let tree = ContractionTree::optimize(subscripts, &shapes)?;
    einsum_with_plan_owned::<Alg, Backend>(ctx, &tree, operands, size_dict)
}

/// Execute einsum with a pre-optimized [`ContractionTree`], consuming the
/// input tensors.
///
/// This is the lowest-level owning entry point and materializes canonical
/// owned operands before dispatching to the execution tree.
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
    Alg::Scalar: Scalar + Conjugate + HasAlgebra<Algebra = Alg>,
    Backend: EinsumBackend<Alg>,
{
    let mut sd = tree.size_dict.clone();
    if let Some(extra) = size_dict {
        for (&k, &v) in extra {
            sd.insert(k, v);
        }
    }
    let output_shape = compute_output_shape(&tree.subscripts.output, &sd)?;
    let canonical_operands = canonicalize_col_major_operands_owned(operands);
    let canonical_refs: Vec<&Tensor<Alg::Scalar>> = canonical_operands.iter().collect();
    let memory_space = infer_memory_space(&canonical_refs)?;
    let mut output = Tensor::<Alg::Scalar>::zeros(
        &output_shape,
        memory_space,
        tenferro_tensor::MemoryOrder::ColumnMajor,
    );
    let mut pool = BufferPool::new();
    crate::execution::execute::execute_tree::<Alg, Backend>(
        ctx,
        tree,
        &canonical_refs,
        Alg::one(),
        Alg::zero(),
        &mut output,
        &mut pool,
        true,
    )?;
    Ok(output)
}
