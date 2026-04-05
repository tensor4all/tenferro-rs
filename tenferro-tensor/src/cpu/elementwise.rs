use std::ops::{Add, Div, Mul, Neg};

use num_complex::Complex;
use num_traits::{One, Zero};
use strided_kernel::{map_into, zip_map2_into, zip_map3_into};

use crate::{
    config::CompareDir,
    types::{dispatch_binary, dispatch_tensor, ConjElem, Tensor, TypedTensor},
};

use super::{tensor_from_array, typed_array, typed_view};

macro_rules! dispatch_ternary {
    ($a:expr, $b:expr, $c:expr, |$x:ident, $y:ident, $z:ident| $body:expr) => {
        match ($a, $b, $c) {
            (Tensor::F32($x), Tensor::F32($y), Tensor::F32($z)) => Tensor::F32($body),
            (Tensor::F64($x), Tensor::F64($y), Tensor::F64($z)) => Tensor::F64($body),
            (Tensor::C32($x), Tensor::C32($y), Tensor::C32($z)) => Tensor::C32($body),
            (Tensor::C64($x), Tensor::C64($y), Tensor::C64($z)) => Tensor::C64($body),
            _ => panic!("dtype mismatch in ternary op"),
        }
    };
}

pub(crate) trait Tier2Elem: Copy + Clone + One + Zero {
    fn abs_elem(self) -> Self;
    fn sign_elem(self) -> Self;
    fn max_elem(self, other: Self) -> Self;
    fn min_elem(self, other: Self) -> Self;
    fn compare_elem(self, other: Self, dir: &CompareDir) -> Self;
    fn is_nonzero(self) -> bool;
}

macro_rules! impl_tier2_elem_real {
    ($ty:ty) => {
        impl Tier2Elem for $ty {
            fn abs_elem(self) -> Self {
                self.abs()
            }

            fn sign_elem(self) -> Self {
                if self == Self::zero() {
                    Self::zero()
                } else {
                    self.signum()
                }
            }

            fn max_elem(self, other: Self) -> Self {
                if self >= other {
                    self
                } else {
                    other
                }
            }

            fn min_elem(self, other: Self) -> Self {
                if self <= other {
                    self
                } else {
                    other
                }
            }

            fn compare_elem(self, other: Self, dir: &CompareDir) -> Self {
                let pred = match dir {
                    CompareDir::Eq => self == other,
                    CompareDir::Lt => self < other,
                    CompareDir::Le => self <= other,
                    CompareDir::Gt => self > other,
                    CompareDir::Ge => self >= other,
                };
                if pred {
                    Self::one()
                } else {
                    Self::zero()
                }
            }

            fn is_nonzero(self) -> bool {
                self != Self::zero()
            }
        }
    };
}

macro_rules! impl_tier2_elem_complex {
    ($real:ty) => {
        impl Tier2Elem for Complex<$real> {
            fn abs_elem(self) -> Self {
                Self::new(self.norm(), <$real>::zero())
            }

            fn sign_elem(self) -> Self {
                if self.is_zero() {
                    Self::zero()
                } else {
                    self / self.abs_elem()
                }
            }

            fn max_elem(self, other: Self) -> Self {
                if self.norm_sqr() >= other.norm_sqr() {
                    self
                } else {
                    other
                }
            }

            fn min_elem(self, other: Self) -> Self {
                if self.norm_sqr() <= other.norm_sqr() {
                    self
                } else {
                    other
                }
            }

            fn compare_elem(self, other: Self, dir: &CompareDir) -> Self {
                let pred = match dir {
                    CompareDir::Eq => self == other,
                    CompareDir::Lt => self.norm_sqr() < other.norm_sqr(),
                    CompareDir::Le => self.norm_sqr() <= other.norm_sqr(),
                    CompareDir::Gt => self.norm_sqr() > other.norm_sqr(),
                    CompareDir::Ge => self.norm_sqr() >= other.norm_sqr(),
                };
                if pred {
                    Self::one()
                } else {
                    Self::zero()
                }
            }

            fn is_nonzero(self) -> bool {
                !self.is_zero()
            }
        }
    };
}

impl_tier2_elem_real!(f32);
impl_tier2_elem_real!(f64);
impl_tier2_elem_complex!(f32);
impl_tier2_elem_complex!(f64);

pub fn add(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    dispatch_binary!(lhs, rhs, |a, b| typed_add(a, b))
}

pub fn mul(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    dispatch_binary!(lhs, rhs, |a, b| typed_mul(a, b))
}

pub fn div(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    dispatch_binary!(lhs, rhs, |a, b| typed_div(a, b))
}

pub fn neg(input: &Tensor) -> Tensor {
    dispatch_tensor!(input, t => typed_neg(t))
}

pub fn conj(input: &Tensor) -> Tensor {
    dispatch_tensor!(input, t => typed_conj(t))
}

pub fn abs(input: &Tensor) -> Tensor {
    dispatch_tensor!(input, t => typed_abs(t))
}

pub fn sign(input: &Tensor) -> Tensor {
    dispatch_tensor!(input, t => typed_sign(t))
}

pub fn maximum(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    dispatch_binary!(lhs, rhs, |a, b| typed_maximum(a, b))
}

pub fn minimum(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    dispatch_binary!(lhs, rhs, |a, b| typed_minimum(a, b))
}

pub fn compare(lhs: &Tensor, rhs: &Tensor, dir: &CompareDir) -> Tensor {
    dispatch_binary!(lhs, rhs, |a, b| typed_compare(a, b, dir))
}

pub fn select(pred: &Tensor, on_true: &Tensor, on_false: &Tensor) -> Tensor {
    dispatch_ternary!(pred, on_true, on_false, |p, t, f| typed_select(p, t, f))
}

pub fn clamp(input: &Tensor, lower: &Tensor, upper: &Tensor) -> Tensor {
    dispatch_ternary!(input, lower, upper, |x, lo, hi| typed_clamp(x, lo, hi))
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

pub fn typed_div<T>(lhs: &TypedTensor<T>, rhs: &TypedTensor<T>) -> TypedTensor<T>
where
    T: Copy + Clone + Zero + Div<Output = T>,
{
    assert_eq!(lhs.shape, rhs.shape, "div: shape mismatch");
    let mut out = typed_array(&lhs.shape, T::zero());
    zip_map2_into(
        &mut out.view_mut(),
        &typed_view(lhs),
        &typed_view(rhs),
        |x, y| x / y,
    )
    .expect("typed_div");
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

pub(crate) fn typed_abs<T>(input: &TypedTensor<T>) -> TypedTensor<T>
where
    T: Tier2Elem,
{
    let mut out = typed_array(&input.shape, T::zero());
    map_into(&mut out.view_mut(), &typed_view(input), |x| x.abs_elem()).expect("typed_abs");
    tensor_from_array(out)
}

pub(crate) fn typed_sign<T>(input: &TypedTensor<T>) -> TypedTensor<T>
where
    T: Tier2Elem,
{
    let mut out = typed_array(&input.shape, T::zero());
    map_into(&mut out.view_mut(), &typed_view(input), |x| x.sign_elem()).expect("typed_sign");
    tensor_from_array(out)
}

pub(crate) fn typed_maximum<T>(lhs: &TypedTensor<T>, rhs: &TypedTensor<T>) -> TypedTensor<T>
where
    T: Tier2Elem,
{
    assert_eq!(lhs.shape, rhs.shape, "maximum: shape mismatch");
    let mut out = typed_array(&lhs.shape, T::zero());
    zip_map2_into(
        &mut out.view_mut(),
        &typed_view(lhs),
        &typed_view(rhs),
        |x, y| x.max_elem(y),
    )
    .expect("typed_maximum");
    tensor_from_array(out)
}

pub(crate) fn typed_minimum<T>(lhs: &TypedTensor<T>, rhs: &TypedTensor<T>) -> TypedTensor<T>
where
    T: Tier2Elem,
{
    assert_eq!(lhs.shape, rhs.shape, "minimum: shape mismatch");
    let mut out = typed_array(&lhs.shape, T::zero());
    zip_map2_into(
        &mut out.view_mut(),
        &typed_view(lhs),
        &typed_view(rhs),
        |x, y| x.min_elem(y),
    )
    .expect("typed_minimum");
    tensor_from_array(out)
}

pub(crate) fn typed_compare<T>(
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    dir: &CompareDir,
) -> TypedTensor<T>
where
    T: Tier2Elem,
{
    assert_eq!(lhs.shape, rhs.shape, "compare: shape mismatch");
    let mut out = typed_array(&lhs.shape, T::zero());
    zip_map2_into(
        &mut out.view_mut(),
        &typed_view(lhs),
        &typed_view(rhs),
        |x, y| x.compare_elem(y, dir),
    )
    .expect("typed_compare");
    tensor_from_array(out)
}

pub(crate) fn typed_select<T>(
    pred: &TypedTensor<T>,
    on_true: &TypedTensor<T>,
    on_false: &TypedTensor<T>,
) -> TypedTensor<T>
where
    T: Tier2Elem,
{
    assert_eq!(pred.shape, on_true.shape, "select: shape mismatch");
    assert_eq!(pred.shape, on_false.shape, "select: shape mismatch");
    let mut out = typed_array(&pred.shape, T::zero());
    zip_map3_into(
        &mut out.view_mut(),
        &typed_view(pred),
        &typed_view(on_true),
        &typed_view(on_false),
        |p, t, f| if p.is_nonzero() { t } else { f },
    )
    .expect("typed_select");
    tensor_from_array(out)
}

pub(crate) fn typed_clamp<T>(
    input: &TypedTensor<T>,
    lower: &TypedTensor<T>,
    upper: &TypedTensor<T>,
) -> TypedTensor<T>
where
    T: Tier2Elem,
{
    assert_eq!(input.shape, lower.shape, "clamp: shape mismatch");
    assert_eq!(input.shape, upper.shape, "clamp: shape mismatch");
    let mut out = typed_array(&input.shape, T::zero());
    zip_map3_into(
        &mut out.view_mut(),
        &typed_view(input),
        &typed_view(lower),
        &typed_view(upper),
        |x, lo, hi| lo.max_elem(hi.min_elem(x)),
    )
    .expect("typed_clamp");
    tensor_from_array(out)
}
