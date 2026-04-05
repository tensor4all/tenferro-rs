use std::ops::{Add, Mul, Neg};

use num_traits::Zero;
use strided_kernel::{map_into, zip_map2_into};

use crate::types::{dispatch_binary, dispatch_tensor, ConjElem, Tensor, TypedTensor};

use super::{tensor_from_array, typed_array, typed_view};

pub fn add(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    dispatch_binary!(lhs, rhs, |a, b| typed_add(a, b))
}

pub fn mul(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    dispatch_binary!(lhs, rhs, |a, b| typed_mul(a, b))
}

pub fn neg(input: &Tensor) -> Tensor {
    dispatch_tensor!(input, t => typed_neg(t))
}

pub fn conj(input: &Tensor) -> Tensor {
    dispatch_tensor!(input, t => typed_conj(t))
}

pub fn typed_add<T>(lhs: &TypedTensor<T>, rhs: &TypedTensor<T>) -> TypedTensor<T>
where
    T: Copy + Clone + Zero + Add<Output = T>,
{
    assert_eq!(lhs.shape, rhs.shape, "add: shape mismatch");
    let mut out = typed_array(&lhs.shape, T::zero());
    zip_map2_into(
        &mut out.view_mut(),
        &typed_view(lhs),
        &typed_view(rhs),
        |x, y| x + y,
    )
    .expect("typed_add");
    tensor_from_array(out)
}

pub fn typed_mul<T>(lhs: &TypedTensor<T>, rhs: &TypedTensor<T>) -> TypedTensor<T>
where
    T: Copy + Clone + Zero + Mul<Output = T>,
{
    assert_eq!(lhs.shape, rhs.shape, "mul: shape mismatch");
    let mut out = typed_array(&lhs.shape, T::zero());
    zip_map2_into(
        &mut out.view_mut(),
        &typed_view(lhs),
        &typed_view(rhs),
        |x, y| x * y,
    )
    .expect("typed_mul");
    tensor_from_array(out)
}

pub fn typed_neg<T>(input: &TypedTensor<T>) -> TypedTensor<T>
where
    T: Copy + Clone + Zero + Neg<Output = T>,
{
    let mut out = typed_array(&input.shape, T::zero());
    map_into(&mut out.view_mut(), &typed_view(input), |x| -x).expect("typed_neg");
    tensor_from_array(out)
}

pub fn typed_conj<T>(input: &TypedTensor<T>) -> TypedTensor<T>
where
    T: Copy + Clone + Zero + ConjElem,
{
    let mut out = typed_array(&input.shape, T::zero());
    map_into(&mut out.view_mut(), &typed_view(input), |x| x.conj_elem()).expect("typed_conj");
    tensor_from_array(out)
}
