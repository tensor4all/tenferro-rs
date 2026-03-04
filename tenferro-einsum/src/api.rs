use std::collections::HashMap;

use num_traits::{One, Zero};
use tenferro_algebra::{Algebra, HasAlgebra, Scalar};
use tenferro_device::Result;
use tenferro_prims::TensorPrims;
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::ad::einsum_frule_impl;
use crate::execute::{execute_nested, execute_tree};
use crate::nested::NestedEinsum;
use crate::pool::BufferPool;
use crate::subscripts::Subscripts;
use crate::tree::ContractionTree;
use crate::util::{compute_output_shape, infer_memory_space};

fn canonicalize_col_major_operands<T: Scalar>(operands: &[&Tensor<T>]) -> Vec<Tensor<T>> {
    operands
        .iter()
        .map(|t| {
            if t.is_col_major_contiguous() && t.offset() == 0 {
                (*t).clone()
            } else {
                (*t).clone().into_contiguous(MemoryOrder::ColumnMajor)
            }
        })
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
/// use tenferro_prims::{CpuBackend, CpuContext};
/// let mut ctx = CpuContext::new(4);
///
/// // Matrix multiplication
/// let c = einsum::<_, _, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b], None).unwrap();
///
/// // Trace
/// let tr = einsum::<_, _, CpuBackend>(&mut ctx, "ii->", &[&a], None).unwrap();
///
/// // Batch matrix multiplication
/// let c = einsum::<_, _, CpuBackend>(&mut ctx, "bij,bjk->bik", &[&a, &b], None).unwrap();
///
/// // Explicit contraction order: contract B*C first, then A
/// let d = einsum::<_, _, CpuBackend>(&mut ctx, "ij,(jk,kl)->il", &[&a, &b, &c], None).unwrap();
/// ```
///
/// # Errors
///
/// Returns an error if the notation is invalid or tensor shapes are
/// incompatible with the subscripts.
pub fn einsum<Alg, Backend>(
    ctx: &mut Backend::Context,
    subscripts: &str,
    operands: &[&Tensor<Alg::Scalar>],
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Tensor<Alg::Scalar>>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
{
    // Subscripts::parse strips parentheses, giving the flat form needed
    // by both the flat execution path and the frule tangent propagation.
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

    // Auto-propagate forward-mode tangents, respecting contraction order.
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
/// Avoids re-parsing the subscript string on each call. Useful when the
/// same contraction pattern is applied to tensors of varying shapes.
///
/// # Errors
///
/// Returns an error if tensor shapes are incompatible with the subscripts.
pub fn einsum_with_subscripts<Alg, Backend>(
    ctx: &mut Backend::Context,
    subscripts: &Subscripts,
    operands: &[&Tensor<Alg::Scalar>],
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Tensor<Alg::Scalar>>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
{
    let shapes: Vec<&[usize]> = operands.iter().map(|t| t.dims()).collect();
    let tree = ContractionTree::optimize(subscripts, &shapes)?;
    einsum_with_plan::<Alg, Backend>(ctx, &tree, operands, size_dict)
}

/// Execute N-ary einsum with an explicit pairwise contraction path.
///
/// This is a convenience wrapper around [`ContractionTree::from_pairs`] and
/// [`einsum_with_plan`]. It makes the "N-ary = binary composition along a path"
/// model explicit in the public API.
///
/// # Arguments
///
/// * `subscripts` — Einsum subscripts
/// * `pairs` — Ordered pairwise contraction path
/// * `operands` — Input tensors
/// * `size_dict` — Optional size overrides for output-only labels
///
/// # Errors
///
/// Returns an error if the path is invalid for the provided subscripts/shapes.
///
/// # Examples
///
/// ```ignore
/// use tenferro_einsum::{einsum_with_path, Subscripts};
/// use tenferro_prims::{CpuBackend, CpuContext};
///
/// let mut ctx = CpuContext::new(1);
/// let subs = Subscripts::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);
/// // Path: contract B*C first, then A*(BC)
/// let pairs = vec![(1, 2), (0, 3)];
/// let d = einsum_with_path::<_, CpuBackend>(&mut ctx, &subs, &pairs, &[&a, &b, &c], None).unwrap();
/// ```
pub fn einsum_with_path<Alg, Backend>(
    ctx: &mut Backend::Context,
    subscripts: &Subscripts,
    pairs: &[(usize, usize)],
    operands: &[&Tensor<Alg::Scalar>],
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Tensor<Alg::Scalar>>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
{
    let shapes: Vec<&[usize]> = operands.iter().map(|t| t.dims()).collect();
    let tree = ContractionTree::from_pairs(subscripts, &shapes, pairs)?;
    einsum_with_plan::<Alg, Backend>(ctx, &tree, operands, size_dict)
}

/// Execute einsum with a pre-optimized [`ContractionTree`].
///
/// Avoids both subscript parsing and contraction order optimization.
/// Ideal for hot loops where the same contraction is executed repeatedly
/// on tensors of the same shape.
///
/// # Errors
///
/// Returns an error if the operand shapes do not match those used to
/// build the contraction tree.
pub fn einsum_with_plan<Alg, Backend>(
    ctx: &mut Backend::Context,
    tree: &ContractionTree,
    operands: &[&Tensor<Alg::Scalar>],
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Tensor<Alg::Scalar>>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
{
    // Merge size_dict from tree and optional extra
    let mut sd = tree.size_dict.clone();
    if let Some(extra) = size_dict {
        for (&k, &v) in extra {
            sd.insert(k, v);
        }
    }
    let output_shape = compute_output_shape(&tree.subscripts.output, &sd)?;
    let canonical_operands = canonicalize_col_major_operands(operands);
    let canonical_refs: Vec<&Tensor<Alg::Scalar>> = canonical_operands.iter().collect();
    // Allocate the output tensor on the same memory space as the operands.
    let memory_space = infer_memory_space(&canonical_refs)?;
    let mut output =
        Tensor::<Alg::Scalar>::zeros(&output_shape, memory_space, MemoryOrder::ColumnMajor);
    let mut pool = BufferPool::new();
    execute_tree::<Alg, Backend>(
        ctx,
        tree,
        &canonical_refs,
        Alg::Scalar::one(),
        Alg::Scalar::zero(),
        &mut output,
        &mut pool,
        true, // lazy_final: output is internally allocated
    )?;
    Ok(output)
}

// ============================================================================
// Consuming variants (take ownership of input tensors for buffer reuse)
// ============================================================================

/// Execute einsum using string notation, consuming the input tensors.
///
/// Takes ownership of the operands, allowing the implementation to reuse
/// their buffers for intermediate results or the final output. This avoids
/// allocation when an operand buffer is already the right shape and layout.
///
/// **Note:** Buffer reuse is not yet implemented. The owned variants
/// currently delegate to the borrowed API.
///
/// # Examples
///
/// ```ignore
/// use tenferro_einsum::einsum_owned;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_prims::{CpuBackend, CpuContext};
///
/// let col = MemoryOrder::ColumnMajor;
/// let mut ctx = CpuContext::new(4);
/// let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], col).unwrap();
/// let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], col).unwrap();
///
/// // `a` and `b` are consumed; their buffers may be reused
/// let c = einsum_owned::<_, _, CpuBackend>(&mut ctx, "ij,jk->ik", vec![a, b], None).unwrap();
/// ```
///
/// # Errors
///
/// Returns an error if the notation is invalid or tensor shapes are
/// incompatible with the subscripts.
pub fn einsum_owned<Alg, Backend>(
    ctx: &mut Backend::Context,
    subscripts: &str,
    operands: Vec<Tensor<Alg::Scalar>>,
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Tensor<Alg::Scalar>>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
{
    let refs: Vec<&Tensor<Alg::Scalar>> = operands.iter().collect();
    einsum::<Alg, Backend>(ctx, subscripts, &refs, size_dict)
}

/// Execute einsum with pre-built [`Subscripts`], consuming the input tensors.
///
/// Combines the benefits of subscript caching ([`einsum_with_subscripts`])
/// with buffer reuse from owned operands.
///
/// # Errors
///
/// Returns an error if tensor shapes are incompatible with the subscripts.
pub fn einsum_with_subscripts_owned<Alg, Backend>(
    ctx: &mut Backend::Context,
    subscripts: &Subscripts,
    operands: Vec<Tensor<Alg::Scalar>>,
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Tensor<Alg::Scalar>>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
{
    let refs: Vec<&Tensor<Alg::Scalar>> = operands.iter().collect();
    einsum_with_subscripts::<Alg, Backend>(ctx, subscripts, &refs, size_dict)
}

/// Execute einsum with a pre-optimized [`ContractionTree`], consuming the
/// input tensors.
///
/// Combines the benefits of plan caching ([`einsum_with_plan`]) with
/// buffer reuse from owned operands. Ideal for hot loops where the
/// caller no longer needs the input tensors after contraction.
///
/// # Errors
///
/// Returns an error if the operand shapes do not match those used to
/// build the contraction tree.
pub fn einsum_with_plan_owned<Alg, Backend>(
    ctx: &mut Backend::Context,
    tree: &ContractionTree,
    operands: Vec<Tensor<Alg::Scalar>>,
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Tensor<Alg::Scalar>>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
{
    let refs: Vec<&Tensor<Alg::Scalar>> = operands.iter().collect();
    einsum_with_plan::<Alg, Backend>(ctx, tree, &refs, size_dict)
}

// ============================================================================
// Accumulating variants (write into pre-allocated output buffer)
// ============================================================================

/// Execute einsum using string notation, accumulating into an existing output.
///
/// Computes `output = alpha * einsum(operands) + beta * output`, writing
/// the result into the provided output tensor. This avoids allocating a new
/// output buffer on each call, which is critical for hot loops.
///
/// # Arguments
///
/// * `ctx` — Mutable backend context (thread pool, plan cache)
/// * `subscripts` — Einstein summation notation (e.g., `"ij,jk->ik"`)
/// * `operands` — Input tensors
/// * `alpha` — Scaling factor for the einsum result
/// * `beta` — Scaling factor for the existing output contents
/// * `output` — Pre-allocated output tensor (must have correct shape)
/// * `size_dict` — Optional dimension sizes for output labels not in inputs
///
/// # Examples
///
/// ```ignore
/// use tenferro_einsum::einsum_into;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
/// use tenferro_prims::{CpuBackend, CpuContext};
///
/// let col = MemoryOrder::ColumnMajor;
/// let mut ctx = CpuContext::new(4);
/// let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], col).unwrap();
/// let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], col).unwrap();
/// let mut c = Tensor::<f64>::zeros(&[2, 2], LogicalMemorySpace::MainMemory, col);
///
/// // Overwrite: C = A @ B
/// einsum_into::<_, _, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b], 1.0, 0.0, &mut c, None).unwrap();
///
/// // Accumulate: C += A @ B
/// einsum_into::<_, _, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b], 1.0, 1.0, &mut c, None).unwrap();
/// ```
///
/// # Errors
///
/// Returns an error if the notation is invalid, tensor shapes are
/// incompatible, or the output shape does not match the expected result.
pub fn einsum_into<Alg, Backend>(
    ctx: &mut Backend::Context,
    subscripts: &str,
    operands: &[&Tensor<Alg::Scalar>],
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
    if subscripts.contains('(') {
        let nested = NestedEinsum::parse(subscripts)?;
        let result = execute_nested::<Alg, Backend>(ctx, &nested, operands, size_dict)?;
        // Apply output = alpha * result + beta * output via identity einsum
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
/// Computes `output = alpha * einsum(operands) + beta * output`.
/// Avoids re-parsing the subscript string on each call.
///
/// # Examples
///
/// ```ignore
/// use tenferro_einsum::{einsum_with_subscripts_into, Subscripts};
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
/// use tenferro_prims::{CpuBackend, CpuContext};
///
/// let mut ctx = CpuContext::new(4);
/// let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
/// let mut c = Tensor::<f64>::zeros(&[3, 5], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
///
/// // C = 1.0 * (A @ B) + 0.0 * C
/// einsum_with_subscripts_into::<_, _, CpuBackend>(
///     &mut ctx, &subs, &[&a, &b], 1.0, 0.0, &mut c, None,
/// ).unwrap();
/// ```
///
/// # Errors
///
/// Returns an error if tensor shapes are incompatible with the subscripts
/// or the output shape does not match.
pub fn einsum_with_subscripts_into<Alg, Backend>(
    ctx: &mut Backend::Context,
    subscripts: &Subscripts,
    operands: &[&Tensor<Alg::Scalar>],
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
    let shapes: Vec<&[usize]> = operands.iter().map(|t| t.dims()).collect();
    let tree = ContractionTree::optimize(subscripts, &shapes)?;
    einsum_with_plan_into::<Alg, Backend>(ctx, &tree, operands, alpha, beta, output, size_dict)
}

/// Execute N-ary einsum with an explicit pairwise contraction path, accumulating
/// into an existing output tensor.
///
/// Computes `output = alpha * einsum_path(operands) + beta * output`.
///
/// # Examples
///
/// ```ignore
/// use tenferro_einsum::{einsum_with_path_into, Subscripts};
/// use tenferro_prims::{CpuBackend, CpuContext};
///
/// let mut ctx = CpuContext::new(1);
/// let subs = Subscripts::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);
/// let pairs = vec![(1, 2), (0, 3)];
/// einsum_with_path_into::<_, CpuBackend>(
///     &mut ctx, &subs, &pairs, &[&a, &b, &c], 1.0, 0.0, &mut out, None
/// ).unwrap();
/// ```
pub fn einsum_with_path_into<Alg, Backend>(
    ctx: &mut Backend::Context,
    subscripts: &Subscripts,
    pairs: &[(usize, usize)],
    operands: &[&Tensor<Alg::Scalar>],
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
    let shapes: Vec<&[usize]> = operands.iter().map(|t| t.dims()).collect();
    let tree = ContractionTree::from_pairs(subscripts, &shapes, pairs)?;
    einsum_with_plan_into::<Alg, Backend>(ctx, &tree, operands, alpha, beta, output, size_dict)
}

/// Execute einsum with a pre-optimized [`ContractionTree`], accumulating
/// into an existing output.
///
/// Computes `output = alpha * einsum(operands) + beta * output`.
/// Avoids both subscript parsing and contraction order optimization.
/// This is the fastest variant for hot loops with pre-allocated buffers.
///
/// # Examples
///
/// ```ignore
/// use tenferro_einsum::{einsum_with_plan_into, ContractionTree, Subscripts};
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
/// use tenferro_prims::{CpuBackend, CpuContext};
///
/// let col = MemoryOrder::ColumnMajor;
/// let mut ctx = CpuContext::new(4);
/// let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
/// let tree = ContractionTree::optimize(&subs, &[&[3, 4], &[4, 5]]).unwrap();
/// let mut c = Tensor::<f64>::zeros(&[3, 5], LogicalMemorySpace::MainMemory, col);
///
/// // Hot loop: reuse output buffer, no allocation per iteration
/// for _ in 0..1000 {
///     einsum_with_plan_into::<_, _, CpuBackend>(
///         &mut ctx, &tree, &[&a, &b], 1.0, 0.0, &mut c, None,
///     ).unwrap();
/// }
/// ```
///
/// # Errors
///
/// Returns an error if the operand shapes do not match those used to
/// build the contraction tree, or the output shape is incorrect.
pub fn einsum_with_plan_into<Alg, Backend>(
    ctx: &mut Backend::Context,
    tree: &ContractionTree,
    operands: &[&Tensor<Alg::Scalar>],
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
    let _ = size_dict; // size_dict already captured in tree.size_dict
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
