use std::collections::HashMap;

use tenferro_algebra::{Conjugate, HasAlgebra, Scalar, Semiring};
use tenferro_device::Result;
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::ad::einsum_frule_impl;
use crate::execution::backend::{BackendContext, EinsumBackend};
use crate::execution::execute::{execute_nested, execute_tree};
use crate::execution::pool::BufferPool;
use crate::execution::util::{compute_output_shape, infer_memory_space};
use crate::planning::tree::ContractionTree;
use crate::syntax::nested::NestedEinsum;
use crate::syntax::subscripts::Subscripts;

fn materialize_logical_col_major<T: Scalar + Conjugate>(tensor: &Tensor<T>) -> Tensor<T> {
    if tensor.is_col_major_contiguous() && tensor.offset() == 0 && !tensor.is_conjugated() {
        return tensor.clone();
    }

    let contiguous = tensor.contiguous(MemoryOrder::ColumnMajor);
    let offset = usize::try_from(contiguous.offset()).unwrap_or(0);
    let len = contiguous.len();
    let data = contiguous
        .buffer()
        .as_slice()
        .and_then(|values| values.get(offset..offset + len))
        .map(|values| {
            if contiguous.is_conjugated() {
                values.iter().copied().map(Conjugate::conj).collect()
            } else {
                values.to_vec()
            }
        });

    data.and_then(|values| {
        Tensor::from_slice(&values, tensor.dims(), MemoryOrder::ColumnMajor).ok()
    })
    .unwrap_or_else(|| tensor.clone().into_contiguous(MemoryOrder::ColumnMajor))
}

pub(super) fn canonicalize_col_major_operands<T: Scalar + Conjugate>(
    operands: &[&Tensor<T>],
) -> Vec<Tensor<T>> {
    operands
        .iter()
        .map(|tensor| materialize_logical_col_major(tensor))
        .collect()
}

/// Execute einsum using string notation.
///
/// Parses the subscript string, optimizes the contraction order, and
/// executes the contraction. The backend `B` and its context `ctx`
/// are passed explicitly.
///
/// Parentheses in the subscript string specify contraction order
/// explicitly (e.g., `"ij,(jk,kl)->il"` contracts B and C first).
/// Without parentheses, the contraction order is optimized automatically.
///
/// # Arguments
///
/// * `ctx` — Mutable backend context (thread pool, plan cache)
/// * `subscripts` — Einstein summation notation (e.g., `"ij,jk->ik"`)
/// * `operands` — Input tensors
/// * `size_dict` — Optional dimension sizes for output labels not in inputs
///
/// # Examples
///
/// ```ignore
/// use tenferro_algebra::Standard;
/// use tenferro_prims::{CpuBackend, CpuContext};
/// let mut ctx = CpuContext::new(4);
///
/// // Matrix multiplication
/// let c = einsum::<Standard<f64>, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b], None).unwrap();
///
/// // Trace
/// let tr = einsum::<Standard<f64>, CpuBackend>(&mut ctx, "ii->", &[&a], None).unwrap();
///
/// // Batch matrix multiplication
/// let c =
///     einsum::<Standard<f64>, CpuBackend>(&mut ctx, "bij,bjk->bik", &[&a, &b], None).unwrap();
///
/// // Explicit contraction order: contract B*C first, then A
/// let d = einsum::<Standard<f64>, CpuBackend>(&mut ctx, "ij,(jk,kl)->il", &[&a, &b, &c], None)
///     .unwrap();
/// ```
///
/// # Errors
///
/// Returns an error if the notation is invalid or tensor shapes are
/// incompatible with the subscripts.
pub fn einsum<Alg, Backend>(
    ctx: &mut BackendContext<Alg, Backend>,
    subscripts: &str,
    operands: &[&Tensor<Alg::Scalar>],
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Tensor<Alg::Scalar>>
where
    Alg: Semiring,
    Alg::Scalar: Scalar + Conjugate + HasAlgebra<Algebra = Alg>,
    Backend: EinsumBackend<Alg>,
{
    let subs = Subscripts::parse(subscripts)?;
    let nested = if subscripts.contains('(') {
        Some(NestedEinsum::parse(subscripts)?)
    } else {
        None
    };

    let mut output = if let Some(ref nested) = nested {
        execute_nested::<Alg, Backend>(ctx, nested, operands, size_dict)?
    } else {
        einsum_with_subscripts::<Alg, Backend>(ctx, &subs, operands, size_dict)?
    };

    if operands.iter().any(|t| t.has_fw_grad()) {
        let tangents: Vec<Option<&Tensor<Alg::Scalar>>> =
            operands.iter().map(|t| t.fw_grad()).collect();
        if let Ok(output_tangent) =
            einsum_frule_impl::<Alg, Backend>(ctx, &subs, nested.as_ref(), operands, &tangents)
        {
            output.set_fw_grad(output_tangent);
        }
    }

    Ok(output)
}

/// Execute einsum with pre-built [`Subscripts`].
///
/// Avoids re-parsing the subscript string on each call.
///
/// # Examples
///
/// ```ignore
/// use tenferro_algebra::Standard;
/// use tenferro_einsum::{einsum_with_subscripts, Subscripts};
/// use tenferro_prims::{CpuBackend, CpuContext};
///
/// let mut ctx = CpuContext::new(1);
/// let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
/// let out = einsum_with_subscripts::<Standard<f64>, CpuBackend>(&mut ctx, &subs, &[&a, &b], None)
///     .unwrap();
/// ```
pub fn einsum_with_subscripts<Alg, Backend>(
    ctx: &mut BackendContext<Alg, Backend>,
    subscripts: &Subscripts,
    operands: &[&Tensor<Alg::Scalar>],
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Tensor<Alg::Scalar>>
where
    Alg: Semiring,
    Alg::Scalar: Scalar + Conjugate + HasAlgebra<Algebra = Alg>,
    Backend: EinsumBackend<Alg>,
{
    let shapes: Vec<&[usize]> = operands.iter().map(|t| t.dims()).collect();
    let tree = ContractionTree::optimize(subscripts, &shapes)?;
    einsum_with_plan::<Alg, Backend>(ctx, &tree, operands, size_dict)
}

/// Execute N-ary einsum with an explicit pairwise contraction path.
///
/// # Examples
///
/// ```ignore
/// use tenferro_algebra::Standard;
/// use tenferro_einsum::{einsum_with_path, Subscripts};
/// use tenferro_prims::{CpuBackend, CpuContext};
///
/// let mut ctx = CpuContext::new(1);
/// let subs = Subscripts::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);
/// let pairs = vec![(1, 2), (0, 3)];
/// let out =
///     einsum_with_path::<Standard<f64>, CpuBackend>(&mut ctx, &subs, &pairs, &[&a, &b, &c], None)
///         .unwrap();
/// ```
pub fn einsum_with_path<Alg, Backend>(
    ctx: &mut BackendContext<Alg, Backend>,
    subscripts: &Subscripts,
    pairs: &[(usize, usize)],
    operands: &[&Tensor<Alg::Scalar>],
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Tensor<Alg::Scalar>>
where
    Alg: Semiring,
    Alg::Scalar: Scalar + Conjugate + HasAlgebra<Algebra = Alg>,
    Backend: EinsumBackend<Alg>,
{
    let shapes: Vec<&[usize]> = operands.iter().map(|t| t.dims()).collect();
    let tree = ContractionTree::from_pairs(subscripts, &shapes, pairs)?;
    einsum_with_plan::<Alg, Backend>(ctx, &tree, operands, size_dict)
}

/// Execute einsum with a pre-optimized [`ContractionTree`].
///
/// # Examples
///
/// ```ignore
/// use tenferro_algebra::Standard;
/// use tenferro_einsum::{einsum_with_plan, ContractionTree, Subscripts};
/// use tenferro_prims::{CpuBackend, CpuContext};
///
/// let mut ctx = CpuContext::new(1);
/// let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
/// let tree = ContractionTree::optimize(&subs, &[&[2, 3], &[3, 4]]).unwrap();
/// let out = einsum_with_plan::<Standard<f64>, CpuBackend>(&mut ctx, &tree, &[&a, &b], None)
///     .unwrap();
/// ```
pub fn einsum_with_plan<Alg, Backend>(
    ctx: &mut BackendContext<Alg, Backend>,
    tree: &ContractionTree,
    operands: &[&Tensor<Alg::Scalar>],
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
    let canonical_operands = canonicalize_col_major_operands(operands);
    let canonical_refs: Vec<&Tensor<Alg::Scalar>> = canonical_operands.iter().collect();
    let memory_space = infer_memory_space(&canonical_refs)?;
    let mut output =
        Tensor::<Alg::Scalar>::zeros(&output_shape, memory_space, MemoryOrder::ColumnMajor);
    let mut pool = BufferPool::new();
    execute_tree::<Alg, Backend>(
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
