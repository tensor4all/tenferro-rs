use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use chainrules::{autograd, AdResult, AutogradContext, BackwardOptions, Variable};
use tenferro_algebra::{HasAlgebra, Scalar, Standard};
use tenferro_einsum::variable_einsum;
use tenferro_prims::{CpuBackend, CpuContext};
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
pub fn context<T>() -> Arc<Mutex<AutogradContext<Tensor<T>>>>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>>,
{
    AutogradContext::new()
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
    ctx: Arc<Mutex<AutogradContext<Tensor<T>>>>,
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
pub fn einsum<T>(
    runtime_ctx: Rc<RefCell<CpuContext>>,
    subscripts: &str,
    operands: &[&Variable<Tensor<T>>],
) -> AdResult<Variable<Tensor<T>>>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>> + 'static,
{
    variable_einsum::<Standard<T>, CpuBackend>(runtime_ctx, subscripts, operands)
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
mod tests {
    use super::*;
    use tenferro_device::LogicalMemorySpace;
    use tenferro_tensor::MemoryOrder;

    #[test]
    fn variable_einsum_backward_and_hvp_flow() {
        let runtime_ctx = Rc::new(RefCell::new(CpuContext::new(1)));
        let ad_ctx = context::<f64>();

        let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap();
        let b = Tensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap();

        let da = Tensor::<f64>::ones(
            &[2, 2],
            LogicalMemorySpace::MainMemory,
            MemoryOrder::ColumnMajor,
        );

        let a = leaf_in(a, Arc::clone(&ad_ctx), true)
            .unwrap()
            .with_tangent_(da)
            .unwrap();
        let b = leaf_in(b, Arc::clone(&ad_ctx), true).unwrap();

        let y = einsum(runtime_ctx.clone(), "ij,jk->ik", &[&a, &b]).unwrap();
        let loss = einsum(runtime_ctx.clone(), "ij,ij->", &[&y, &y]).unwrap();

        backward(
            &loss,
            BackwardOptions {
                retain_graph: Some(true),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(a.grad().is_some());
        assert!(b.grad().is_some());

        a.zero_grad().unwrap();
        b.zero_grad().unwrap();

        backward_hvp(&loss, BackwardOptions::default()).unwrap();
        assert!(a.grad().is_some());
        assert!(a.hvp().is_some());
    }
}
