use tenferro_tensor::{Error, Result, Tensor, TensorBackend, TensorScalar, TypedTensor};

use crate::eager_einsum;

/// Execute eager einsum over typed tensors and return a typed result.
///
/// # Examples
///
/// ```
/// use tenferro_einsum::typed_eager_einsum;
/// use tenferro_tensor::{cpu::CpuBackend, TypedTensor};
///
/// let mut ctx = CpuBackend::new();
/// let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// let b = TypedTensor::<f64>::from_vec_col_major(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
///
/// let c = typed_eager_einsum(&mut ctx, &[&a, &b], "ij,jk->ik").unwrap();
///
/// assert_eq!(c.shape, vec![2, 2]);
/// assert_eq!(c.as_slice(), &[22.0, 28.0, 49.0, 64.0]);
/// ```
pub fn typed_eager_einsum<T: TensorScalar>(
    ctx: &mut impl TensorBackend,
    inputs: &[&TypedTensor<T>],
    subscripts: &str,
) -> Result<TypedTensor<T>> {
    let tensors: Vec<Tensor> = inputs
        .iter()
        .map(|tensor| T::into_tensor(tensor.shape.clone(), tensor.host_data().to_vec()))
        .collect();
    let refs: Vec<&Tensor> = tensors.iter().collect();
    let result = eager_einsum(ctx, &refs, subscripts)?;
    let actual = result.dtype();
    T::try_into_typed(result).ok_or_else(|| Error::DTypeMismatch {
        op: "typed_eager_einsum",
        lhs: actual,
        rhs: T::dtype(),
    })
}
