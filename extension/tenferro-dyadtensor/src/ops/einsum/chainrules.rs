use std::sync::{Arc, Mutex};

use chainrules::{autograd, AdResult, AutogradGraph, BackwardOptions, Variable};
use tenferro_algebra::{HasAlgebra, Scalar, Standard};
use tenferro_einsum::{variable_einsum, BackendContext, EinsumBackend};
use tenferro_tensor::Tensor;

/// Creates a shared monomorphic AD context for tensor variables.
///
/// # Examples
///
/// ```
/// use tenferro_dyadtensor::chainrules_api;
///
/// let _ctx = chainrules_api::context::<f64>();
/// ```
pub fn context<T>() -> Arc<Mutex<AutogradGraph<Tensor<T>>>>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>>,
{
    AutogradGraph::new()
}

/// Creates a `Variable<Tensor<T>>` leaf attached to `ctx`.
///
/// # Examples
///
/// ```ignore
/// let ctx = tenferro_dyadtensor::chainrules_api::context::<f64>();
/// let x = tenferro_dyadtensor::chainrules_api::leaf_in(tensor, ctx, true).unwrap();
/// ```
pub fn leaf_in<T>(
    value: Tensor<T>,
    ctx: Arc<Mutex<AutogradGraph<Tensor<T>>>>,
    requires_grad: bool,
) -> AdResult<Variable<Tensor<T>>>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>>,
{
    let var = Variable::new_in(value, ctx);
    if requires_grad {
        var.requires_grad_(true)
    } else {
        Ok(var)
    }
}

/// Runs `tenferro-einsum` on `Variable<Tensor<T>>` operands.
///
/// # Examples
///
/// ```ignore
/// let out = tenferro_dyadtensor::chainrules_api::einsum(ctx, "ij,jk->ik", &[&a, &b]).unwrap();
/// ```
pub fn einsum<T, Backend>(
    runtime_ctx: Arc<Mutex<BackendContext<Standard<T>, Backend>>>,
    subscripts: &str,
    operands: &[&Variable<Tensor<T>>],
) -> AdResult<Variable<Tensor<T>>>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>> + Send + Sync + 'static,
    Backend: EinsumBackend<Standard<T>> + Send + Sync + 'static,
    BackendContext<Standard<T>, Backend>: Send,
{
    variable_einsum::<Standard<T>, Backend>(runtime_ctx, subscripts, operands)
}

/// Convenience wrapper for `loss.backward(...)`.
///
/// # Examples
///
/// ```ignore
/// tenferro_dyadtensor::chainrules_api::backward(&loss, Default::default()).unwrap();
/// ```
pub fn backward<T>(loss: &Variable<Tensor<T>>, options: BackwardOptions<Tensor<T>>) -> AdResult<()>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>>,
{
    loss.backward(options)
}

/// Convenience wrapper for `loss.backward_hvp(...)`.
///
/// # Examples
///
/// ```ignore
/// tenferro_dyadtensor::chainrules_api::backward_hvp(&loss, Default::default()).unwrap();
/// ```
pub fn backward_hvp<T>(
    loss: &Variable<Tensor<T>>,
    options: BackwardOptions<Tensor<T>>,
) -> AdResult<()>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>>,
{
    loss.backward_hvp(options)
}

/// Convenience wrapper for tangent-valued grad query.
///
/// # Examples
///
/// ```ignore
/// let grads = tenferro_dyadtensor::chainrules_api::grad_tangent(&loss, &[&x], Default::default()).unwrap();
/// ```
pub fn grad_tangent<T>(
    output: &Variable<Tensor<T>>,
    inputs: &[&Variable<Tensor<T>>],
    options: BackwardOptions<Tensor<T>>,
) -> AdResult<Vec<Tensor<T>>>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>>,
{
    autograd::grad_tangent(output, inputs, options)
}

#[cfg(test)]
mod tests;
