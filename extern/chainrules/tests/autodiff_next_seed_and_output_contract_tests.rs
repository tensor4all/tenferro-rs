use std::ops::{Add, Mul};

use chainrules::{autograd, AutodiffError, BackwardOptions, Differentiable, Variable};

#[derive(Clone, Copy, Debug, PartialEq)]
struct Pair {
    x: f64,
    y: f64,
}

impl Add for Pair {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl Mul for Pair {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
        }
    }
}

impl Differentiable for Pair {
    type Tangent = Self;

    fn zero_tangent(&self) -> Self::Tangent {
        Self { x: 0.0, y: 0.0 }
    }

    fn accumulate_tangent(a: Self::Tangent, b: &Self::Tangent) -> Self::Tangent {
        Self {
            x: a.x + b.x,
            y: a.y + b.y,
        }
    }

    fn num_elements(&self) -> usize {
        2
    }

    fn seed_cotangent(&self) -> Self::Tangent {
        Self { x: 1.0, y: 1.0 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct SingleSlot(f64);

impl Add for SingleSlot {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl Mul for SingleSlot {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl Differentiable for SingleSlot {
    type Tangent = Self;

    fn zero_tangent(&self) -> Self::Tangent {
        Self(0.0)
    }

    fn accumulate_tangent(a: Self::Tangent, b: &Self::Tangent) -> Self::Tangent {
        Self(a.0 + b.0)
    }

    fn num_elements(&self) -> usize {
        1
    }

    fn seed_cotangent(&self) -> Self::Tangent {
        Self(1.0)
    }
}

#[test]
fn non_scalar_backward_requires_seed_grad() {
    let x = Variable::new(Pair { x: 2.0, y: 3.0 })
        .requires_grad_(true)
        .unwrap();
    let y = autograd::square(&x).unwrap();

    let err = y.backward(BackwardOptions::default()).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn non_scalar_backward_with_seed_grad_succeeds() {
    let x = Variable::new(Pair { x: 2.0, y: 3.0 })
        .requires_grad_(true)
        .unwrap();
    let y = autograd::square(&x).unwrap();

    y.backward(BackwardOptions {
        seed_grad: Some(Pair { x: 1.0, y: 1.0 }),
        ..Default::default()
    })
    .unwrap();

    assert_eq!(x.grad(), Some(Pair { x: 4.0, y: 6.0 }));
}

#[test]
fn non_scalar_grad_queries_require_seed_grad() {
    let x = Variable::new(Pair { x: 2.0, y: 3.0 })
        .requires_grad_(true)
        .unwrap();
    let y = autograd::square(&x).unwrap();

    let err_tangent = autograd::grad_tangent(&y, &[&x], BackwardOptions::default()).unwrap_err();
    assert!(matches!(err_tangent, AutodiffError::InvalidArgument(_)));

    let err_variable = match autograd::grad_variable(&y, &[&x], BackwardOptions::default()) {
        Ok(_) => panic!("expected InvalidArgument for non-scalar grad_variable without seed"),
        Err(err) => err,
    };
    assert!(matches!(err_variable, AutodiffError::InvalidArgument(_)));
}

#[test]
fn backward_and_grad_queries_require_tracked_output() {
    let y = Variable::new(3.0_f64);

    let err_backward = y.backward(BackwardOptions::default()).unwrap_err();
    assert!(matches!(err_backward, AutodiffError::InvalidArgument(_)));

    let err_grad_tangent =
        autograd::grad_tangent(&y, &[&y], BackwardOptions::default()).unwrap_err();
    assert!(matches!(
        err_grad_tangent,
        AutodiffError::InvalidArgument(_)
    ));

    let err_grad_variable = match autograd::grad_variable(&y, &[&y], BackwardOptions::default()) {
        Ok(_) => panic!("expected InvalidArgument for grad_variable on untracked output"),
        Err(err) => err,
    };
    assert!(matches!(
        err_grad_variable,
        AutodiffError::InvalidArgument(_)
    ));
}

#[test]
fn single_element_custom_output_can_omit_seed_grad() {
    let x = Variable::new(SingleSlot(3.0)).requires_grad_(true).unwrap();
    let y = autograd::square(&x).unwrap();

    y.backward(BackwardOptions::default()).unwrap();

    assert_eq!(x.grad(), Some(SingleSlot(6.0)));
}
