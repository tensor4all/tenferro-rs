use std::collections::HashMap;

use smallvec::SmallVec;
use tenferro_algebra::{Conjugate, HasAlgebra, Scalar, Semiring};
#[cfg(test)]
use tenferro_device::Error;
use tenferro_device::Result;
use tenferro_prims::TensorTempPoolContext;
use tenferro_tensor::Tensor;

use crate::execution::backend::{BackendContext, EinsumBackend};
use crate::execution::chain::execute_binary_step;
use crate::execution::pool::{with_context_buffer_pool, TensorBufferPool};
use crate::execution::strict_binary::{
    try_execute_strict_binary_plan, try_execute_strict_binary_plan_into,
};
use crate::execution::util::infer_memory_space;
use crate::planning::plan::{
    compile_binary_contraction_plan, compile_strict_binary_lowering_plan, BinaryContractionPlan,
};
#[cfg(test)]
use crate::planning::tree::ContractionTree;
use crate::syntax::subscripts::Subscripts;

use super::canonical::canonicalize_col_major_operands_borrowed;

#[cfg(test)]
fn ensure_binary_subscripts(subscripts: &Subscripts) -> Result<()> {
    if subscripts.inputs.len() != 2 {
        return Err(Error::InvalidArgument(format!(
            "binary einsum requires exactly 2 inputs, got {}",
            subscripts.inputs.len()
        )));
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn binary_contraction_tree(
    subscripts: &Subscripts,
    shapes: &[&[usize]],
) -> Result<ContractionTree> {
    ensure_binary_subscripts(subscripts)?;
    ContractionTree::from_pairs(subscripts, shapes, &[(0, 1)])
}

pub(crate) fn execute_binary_with_subscripts_impl<Alg, Backend>(
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
    BackendContext<Alg, Backend>: TensorTempPoolContext,
{
    if let Some(output) = try_execute_strict_binary_with_subscripts_impl::<Alg, Backend>(
        ctx, subscripts, left, right, size_dict,
    )? {
        return Ok(output);
    }

    execute_binary_with_subscripts_generic_impl::<Alg, Backend>(
        ctx, subscripts, left, right, size_dict,
    )
}

pub(crate) fn maybe_execute_binary_operands<Alg, Backend>(
    ctx: &mut BackendContext<Alg, Backend>,
    subscripts: &Subscripts,
    operands: &[&Tensor<Alg::Scalar>],
    size_dict: Option<&HashMap<u32, usize>>,
) -> Option<Result<Tensor<Alg::Scalar>>>
where
    Alg: Semiring,
    Alg::Scalar: Scalar + Conjugate + HasAlgebra<Algebra = Alg>,
    Backend: EinsumBackend<Alg>,
    BackendContext<Alg, Backend>: TensorTempPoolContext,
{
    (operands.len() == 2).then(|| {
        execute_binary_with_subscripts_impl::<Alg, Backend>(
            ctx,
            subscripts,
            operands[0],
            operands[1],
            size_dict,
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn maybe_execute_binary_operands_into<Alg, Backend>(
    ctx: &mut BackendContext<Alg, Backend>,
    subscripts: &Subscripts,
    operands: &[&Tensor<Alg::Scalar>],
    alpha: Alg::Scalar,
    beta: Alg::Scalar,
    output: &mut Tensor<Alg::Scalar>,
    size_dict: Option<&HashMap<u32, usize>>,
) -> Option<Result<()>>
where
    Alg: Semiring,
    Alg::Scalar: Scalar + Conjugate + HasAlgebra<Algebra = Alg>,
    Backend: EinsumBackend<Alg>,
    BackendContext<Alg, Backend>: TensorTempPoolContext,
{
    (operands.len() == 2).then(|| {
        execute_binary_with_subscripts_into_impl::<Alg, Backend>(
            ctx,
            subscripts,
            operands[0],
            operands[1],
            alpha,
            beta,
            output,
            size_dict,
        )
    })
}

fn binary_shapes<'a, T>(left: &'a Tensor<T>, right: &'a Tensor<T>) -> [&'a [usize]; 2] {
    [left.dims(), right.dims()]
}

fn compile_binary_plan_for_operands<T: Scalar>(
    subscripts: &Subscripts,
    left: &Tensor<T>,
    right: &Tensor<T>,
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<BinaryContractionPlan> {
    let shapes = binary_shapes(left, right);
    compile_binary_contraction_plan(subscripts, &shapes, size_dict)
}

#[allow(clippy::too_many_arguments)]
fn execute_binary_generic_step<Alg, Backend, P>(
    ctx: &mut BackendContext<Alg, Backend>,
    plan: &BinaryContractionPlan,
    subscripts: &Subscripts,
    left: &Tensor<Alg::Scalar>,
    right: &Tensor<Alg::Scalar>,
    alpha: Alg::Scalar,
    beta: Alg::Scalar,
    output: &mut Tensor<Alg::Scalar>,
    pool: &mut P,
    lazy_final: bool,
) -> Result<()>
where
    Alg: Semiring,
    Alg::Scalar: Scalar + Conjugate + HasAlgebra<Algebra = Alg>,
    Backend: EinsumBackend<Alg>,
    BackendContext<Alg, Backend>: TensorTempPoolContext,
    P: TensorBufferPool<Alg::Scalar> + ?Sized,
{
    let borrowed_operands: SmallVec<[&Tensor<Alg::Scalar>; 2]> = smallvec::smallvec![left, right];
    let canonical_operands = canonicalize_col_major_operands_borrowed(&borrowed_operands);
    let canonical_refs = [
        canonical_operands[0].as_tensor(),
        canonical_operands[1].as_tensor(),
    ];
    execute_binary_step::<Alg, Backend, _>(
        ctx,
        &plan.step_plan,
        &subscripts.inputs[0],
        &subscripts.inputs[1],
        &subscripts.output,
        canonical_refs[0],
        canonical_refs[1],
        alpha,
        beta,
        output,
        pool,
        lazy_final,
    )
}

pub(crate) fn execute_binary_with_subscripts_generic_impl<Alg, Backend>(
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
    BackendContext<Alg, Backend>: TensorTempPoolContext,
{
    let plan = compile_binary_plan_for_operands(subscripts, left, right, size_dict)?;
    let memory_space = infer_memory_space(&[left, right])?;
    with_context_buffer_pool(ctx, |ctx, pool| {
        let mut output = Tensor::<Alg::Scalar>::zeros(
            &plan.output_shape,
            memory_space,
            tenferro_tensor::MemoryOrder::ColumnMajor,
        )?;
        execute_binary_generic_step::<Alg, Backend, _>(
            ctx,
            &plan,
            subscripts,
            left,
            right,
            Alg::one(),
            Alg::zero(),
            &mut output,
            pool,
            true,
        )?;
        Ok(output)
    })
}

pub(crate) fn try_execute_strict_binary_with_subscripts_impl<Alg, Backend>(
    ctx: &mut BackendContext<Alg, Backend>,
    subscripts: &Subscripts,
    left: &Tensor<Alg::Scalar>,
    right: &Tensor<Alg::Scalar>,
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Option<Tensor<Alg::Scalar>>>
where
    Alg: Semiring,
    Alg::Scalar: Scalar + Conjugate + HasAlgebra<Algebra = Alg>,
    Backend: EinsumBackend<Alg>,
    BackendContext<Alg, Backend>: TensorTempPoolContext,
{
    if left.is_conjugated() || right.is_conjugated() {
        return Ok(None);
    }
    let shapes = binary_shapes(left, right);
    let Some(plan) = compile_strict_binary_lowering_plan(subscripts, &shapes, size_dict)? else {
        return Ok(None);
    };
    try_execute_strict_binary_plan::<Alg, Backend>(ctx, &plan, left, right)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_binary_with_subscripts_into_impl<Alg, Backend>(
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
    BackendContext<Alg, Backend>: TensorTempPoolContext,
{
    if try_execute_strict_binary_with_subscripts_into_impl::<Alg, Backend>(
        ctx, subscripts, left, right, alpha, beta, output, size_dict,
    )? {
        return Ok(());
    }

    execute_binary_with_subscripts_into_generic_impl::<Alg, Backend>(
        ctx, subscripts, left, right, alpha, beta, output, size_dict,
    )
}

#[allow(clippy::too_many_arguments)]
fn execute_binary_with_subscripts_into_generic_impl<Alg, Backend>(
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
    BackendContext<Alg, Backend>: TensorTempPoolContext,
{
    let plan = compile_binary_plan_for_operands(subscripts, left, right, size_dict)?;
    with_context_buffer_pool(ctx, |ctx, pool| {
        execute_binary_generic_step::<Alg, Backend, _>(
            ctx, &plan, subscripts, left, right, alpha, beta, output, pool, false,
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn try_execute_strict_binary_with_subscripts_into_impl<Alg, Backend>(
    ctx: &mut BackendContext<Alg, Backend>,
    subscripts: &Subscripts,
    left: &Tensor<Alg::Scalar>,
    right: &Tensor<Alg::Scalar>,
    alpha: Alg::Scalar,
    beta: Alg::Scalar,
    output: &mut Tensor<Alg::Scalar>,
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<bool>
where
    Alg: Semiring,
    Alg::Scalar: Scalar + Conjugate + HasAlgebra<Algebra = Alg>,
    Backend: EinsumBackend<Alg>,
    BackendContext<Alg, Backend>: TensorTempPoolContext,
{
    if left.is_conjugated() || right.is_conjugated() {
        return Ok(false);
    }
    let shapes = binary_shapes(left, right);
    let Some(plan) = compile_strict_binary_lowering_plan(subscripts, &shapes, size_dict)? else {
        return Ok(false);
    };
    try_execute_strict_binary_plan_into::<Alg, Backend>(
        ctx, &plan, left, right, alpha, beta, output,
    )
    .map(|executed| executed.is_some())
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
    BackendContext<Alg, Backend>: TensorTempPoolContext,
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
    BackendContext<Alg, Backend>: TensorTempPoolContext,
{
    execute_binary_with_subscripts_impl::<Alg, Backend>(ctx, subscripts, left, right, size_dict)
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
/// let mut c = Tensor::<f64>::zeros(&[2, 2], LogicalMemorySpace::MainMemory, col).unwrap();
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
    BackendContext<Alg, Backend>: TensorTempPoolContext,
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
/// let mut c = Tensor::<f64>::zeros(&[2, 2], LogicalMemorySpace::MainMemory, col).unwrap();
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
    BackendContext<Alg, Backend>: TensorTempPoolContext,
{
    execute_binary_with_subscripts_into_impl::<Alg, Backend>(
        ctx, subscripts, left, right, alpha, beta, output, size_dict,
    )
}

#[cfg(test)]
mod tests;
