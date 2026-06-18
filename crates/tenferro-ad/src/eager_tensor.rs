//! Eager tensor operations.
//!
//! Core eager tensor types are re-exported here. Extension crates provide
//! operation-specific helpers outside this runtime crate.

use std::sync::Arc;

use computegraph::graph::Graph;
use tenferro_ops::broadcast::{broadcast_input_plan, broadcast_shape, broadcast_shapes};
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::TensorFusion;

pub use crate::eager::{EagerRuntime, EagerTensor};
use crate::error::{Error, Result};
use crate::CompareDir;

/// Convert an eager tensor to a different dtype.
///
/// Numeric casts follow Rust primitive cast semantics. Real-to-complex
/// conversion sets the imaginary part to zero; complex-to-real or
/// complex-to-integer conversion uses the real part. Boolean conversion uses
/// nonzero testing, and `bool` converts to numeric dtypes as `0` or `1`.
///
/// # Examples
///
/// ```rust
/// # use tenferro_ad::{eager_tensor, DType, EagerRuntime, EagerTensor, Tensor};
/// # let ctx = EagerRuntime::new();
/// # let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]), ctx);
/// let y = eager_tensor::convert(&x, DType::F32).unwrap();
/// ```
pub fn convert(input: &EagerTensor, to: crate::DType) -> Result<EagerTensor> {
    input.convert(to)
}

/// Elementwise addition with NumPy-style broadcasting.
///
/// # Examples
///
/// ```rust
/// # use tenferro_ad::{eager_tensor, EagerRuntime, EagerTensor, Tensor};
/// # let ctx = EagerRuntime::new();
/// # let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]), ctx.clone());
/// # let y = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]), ctx);
/// let z = eager_tensor::add(&x, &y).unwrap();
/// ```
pub fn add(lhs: &EagerTensor, rhs: &EagerTensor) -> Result<EagerTensor> {
    let (lhs, rhs) = broadcast_binary(lhs, rhs)?;
    lhs.add(&rhs)
}

macro_rules! unary_fn {
    ($name:ident, $method:ident, $summary:literal) => {
        #[doc = $summary]
        ///
        /// # Examples
        ///
        /// ```rust
        /// # use tenferro_ad::{eager_tensor, EagerRuntime, EagerTensor, Tensor};
        /// # let ctx = EagerRuntime::new();
        /// # let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 4.0]), ctx);
        #[doc = concat!("let y = eager_tensor::", stringify!($name), "(&x).unwrap();")]
        /// ```
        pub fn $name(input: &EagerTensor) -> Result<EagerTensor> {
            input.$method()
        }
    };
}

macro_rules! binary_method_fn {
    ($name:ident, $method:ident, $summary:literal) => {
        #[doc = $summary]
        ///
        /// # Examples
        ///
        /// ```rust
        /// # use tenferro_ad::{eager_tensor, EagerRuntime, EagerTensor, Tensor};
        /// # let ctx = EagerRuntime::new();
        /// # let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]), ctx.clone());
        /// # let y = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 8.0]), ctx);
        #[doc = concat!("let z = eager_tensor::", stringify!($name), "(&x, &y).unwrap();")]
        /// ```
        pub fn $name(lhs: &EagerTensor, rhs: &EagerTensor) -> Result<EagerTensor> {
            let (lhs, rhs) = broadcast_binary(lhs, rhs)?;
            lhs.$method(&rhs)
        }
    };
}

binary_method_fn!(
    mul,
    mul,
    "Elementwise multiplication with NumPy-style broadcasting."
);
binary_method_fn!(
    div,
    div,
    "Elementwise division with NumPy-style broadcasting."
);
binary_method_fn!(pow, pow, "Elementwise power with NumPy-style broadcasting.");
binary_method_fn!(
    maximum,
    maximum,
    "Elementwise maximum with NumPy-style broadcasting."
);
binary_method_fn!(
    minimum,
    minimum,
    "Elementwise minimum with NumPy-style broadcasting."
);

unary_fn!(neg, neg, "Elementwise negation.");
unary_fn!(abs, abs, "Elementwise absolute value.");
unary_fn!(sign, sign, "Elementwise sign.");
unary_fn!(conj, conj, "Elementwise complex conjugate.");
unary_fn!(exp, exp, "Elementwise exponential.");
unary_fn!(log, log, "Elementwise natural logarithm.");
unary_fn!(sin, sin, "Elementwise sine.");
unary_fn!(cos, cos, "Elementwise cosine.");
unary_fn!(tanh, tanh, "Elementwise hyperbolic tangent.");
unary_fn!(sqrt, sqrt, "Elementwise square root.");
unary_fn!(rsqrt, rsqrt, "Elementwise reciprocal square root.");
unary_fn!(expm1, expm1, "Elementwise `exp(x) - 1`.");
unary_fn!(log1p, log1p, "Elementwise `log(1 + x)`.");

/// Elementwise subtraction with NumPy-style broadcasting.
///
/// # Examples
///
/// ```rust
/// # use tenferro_ad::{eager_tensor, EagerRuntime, EagerTensor, Tensor};
/// # let ctx = EagerRuntime::new();
/// # let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]), ctx.clone());
/// # let y = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 8.0]), ctx);
/// let z = eager_tensor::sub(&x, &y).unwrap();
/// ```
pub fn sub(lhs: &EagerTensor, rhs: &EagerTensor) -> Result<EagerTensor> {
    let (lhs, rhs) = broadcast_binary(lhs, rhs)?;
    lhs.add(&rhs.neg()?)
}

/// Elementwise comparison with NumPy-style broadcasting.
///
/// The result is a bool tensor.
///
/// # Examples
///
/// ```rust
/// # use tenferro_ad::{eager_tensor, CompareDir, EagerRuntime, EagerTensor, Tensor};
/// # let ctx = EagerRuntime::new();
/// # let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]), ctx.clone());
/// # let y = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 8.0]), ctx);
/// let z = eager_tensor::compare(&x, &y, CompareDir::Gt).unwrap();
/// assert_eq!(z.data().as_slice::<bool>().unwrap(), &[true, false]);
/// ```
pub fn compare(lhs: &EagerTensor, rhs: &EagerTensor, dir: CompareDir) -> Result<EagerTensor> {
    let (lhs, rhs) = broadcast_binary(lhs, rhs)?;
    lhs.binary_op(&rhs, StdTensorOp::Compare(dir))
}

/// Select values from `on_true` or `on_false` using a condition tensor.
///
/// This corresponds to NumPy `where(condition, x, y)`.
///
/// # Examples
///
/// ```rust
/// # use tenferro_ad::{eager_tensor, CompareDir, EagerRuntime, EagerTensor, Tensor};
/// # let ctx = EagerRuntime::new();
/// # let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]), ctx.clone());
/// # let y = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 8.0]), ctx);
/// # let condition = eager_tensor::compare(&x, &y, CompareDir::Gt).unwrap();
/// let z = eager_tensor::where_select(&condition, &x, &y).unwrap();
/// ```
pub fn where_select(
    condition: &EagerTensor,
    on_true: &EagerTensor,
    on_false: &EagerTensor,
) -> Result<EagerTensor> {
    let (condition, on_true, on_false) = broadcast_ternary(condition, on_true, on_false)?;
    EagerTensor::select(&condition, &on_true, &on_false)
}

/// Clamp values elementwise between lower and upper bounds.
///
/// # Examples
///
/// ```rust
/// # use tenferro_ad::{eager_tensor, EagerRuntime, EagerTensor, Tensor};
/// # let ctx = EagerRuntime::new();
/// # let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![-2.0_f64, 4.0]), ctx.clone());
/// # let lower = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![], vec![0.0_f64]), ctx.clone());
/// # let upper = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![], vec![3.0_f64]), ctx);
/// let z = eager_tensor::clamp(&x, &lower, &upper).unwrap();
/// ```
pub fn clamp(input: &EagerTensor, lower: &EagerTensor, upper: &EagerTensor) -> Result<EagerTensor> {
    let (input, lower, upper) = broadcast_ternary(input, lower, upper)?;
    input.clamp(&lower, &upper)
}

/// Matrix multiplication helper for rank-2 eager tensors.
///
/// This contracts the last dimension of `a` with the first dimension of `b`.
///
/// # Examples
///
/// ```rust
/// # use tenferro_ad::{eager_tensor, EagerRuntime, EagerTensor, Tensor};
/// # let ctx = EagerRuntime::new();
/// # let a = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]), ctx.clone());
/// # let b = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64; 6]), ctx);
/// let c = eager_tensor::matmul(&a, &b).unwrap();
/// ```
pub fn matmul(a: &EagerTensor, b: &EagerTensor) -> Result<EagerTensor> {
    a.matmul(b)
}

/// Apply one standard tensor op eagerly and record it for AD when needed.
///
/// This is intended for extension crates that lower a composite operation into
/// ordinary `StdTensorOp` nodes and want eager AD to see the expanded graph.
pub fn apply_standard_op(op: StdTensorOp, inputs: &[&EagerTensor]) -> Result<EagerTensor> {
    if matches!(op, StdTensorOp::Extension(_)) {
        return Err(Error::Internal(
            "eager_tensor::apply_standard_op does not accept Extension ops".into(),
        ));
    }
    EagerTensor::nary_op(inputs, op)
}

/// Apply a standard `StdTensorOp` graph eagerly and record it as one AD node.
///
/// This is a low-level extension hook for composite eager operations. The
/// callback receives graph-local input keys, must build a graph whose inputs
/// use those keys in order, and must set the graph outputs.
#[doc(hidden)]
pub fn apply_standard_graph(
    inputs: &[&EagerTensor],
    build_graph: impl FnOnce(&[TensorInputKey]) -> Result<Arc<Graph<StdTensorOp>>>,
) -> Result<Vec<EagerTensor>> {
    EagerTensor::standard_graph_op(inputs, build_graph)
}

/// Execute a backend fused broadcast-multiply for untracked eager tensors.
///
/// Returns `None` when either input participates in AD or the backend does not
/// provide a fused implementation, so callers can fall back to ordinary eager
/// `StdTensorOp` execution without changing gradients.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn backend_broadcast_multiply_untracked(
    lhs: &EagerTensor,
    lhs_shape: &[usize],
    lhs_dims: &[usize],
    rhs: &EagerTensor,
    rhs_shape: &[usize],
    rhs_dims: &[usize],
) -> Result<Option<EagerTensor>> {
    ensure_same_context(lhs, rhs)?;
    if lhs.tracks_grad() || rhs.tracks_grad() {
        return Ok(None);
    }

    let value = lhs.runtime().with_backend_mut(|backend| {
        backend.execute_broadcast_multiply_value(
            lhs.tensor_read(),
            lhs_shape,
            lhs_dims,
            rhs.tensor_read(),
            rhs_shape,
            rhs_dims,
        )
    })??;

    Ok(value.map(|value| EagerTensor::new_untracked_value_result(lhs.runtime().clone(), value)))
}

fn broadcast_binary(lhs: &EagerTensor, rhs: &EagerTensor) -> Result<(EagerTensor, EagerTensor)> {
    ensure_same_context(lhs, rhs)?;
    let shape = broadcast_shape(lhs.shape(), rhs.shape())
        .map_err(|err| Error::Internal(err.to_string()))?;
    Ok((broadcast_to(lhs, &shape)?, broadcast_to(rhs, &shape)?))
}

fn broadcast_ternary(
    first: &EagerTensor,
    second: &EagerTensor,
    third: &EagerTensor,
) -> Result<(EagerTensor, EagerTensor, EagerTensor)> {
    ensure_same_context(first, second)?;
    ensure_same_context(first, third)?;
    let shape = broadcast_shapes([first.shape(), second.shape(), third.shape()])
        .map_err(|err| Error::Internal(err.to_string()))?;
    Ok((
        broadcast_to(first, &shape)?,
        broadcast_to(second, &shape)?,
        broadcast_to(third, &shape)?,
    ))
}

fn broadcast_to(input: &EagerTensor, target_shape: &[usize]) -> Result<EagerTensor> {
    let input_shape = input.shape();
    if input_shape == target_shape {
        return Ok(input.clone());
    }

    let plan = broadcast_input_plan(input_shape, target_shape)
        .map_err(|err| Error::Internal(err.to_string()))?;
    let source = if plan.source_shape == input_shape {
        input.clone()
    } else {
        input.reshape(&plan.source_shape)?
    };
    source.broadcast_in_dim(target_shape, &plan.dims)
}

fn ensure_same_context(lhs: &EagerTensor, rhs: &EagerTensor) -> Result<()> {
    if lhs.same_context(rhs) {
        Ok(())
    } else {
        Err(Error::ContextMismatch {
            lhs: lhs.ctx_id(),
            rhs: rhs.ctx_id(),
        })
    }
}
