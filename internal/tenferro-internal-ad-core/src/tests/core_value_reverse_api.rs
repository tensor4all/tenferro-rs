use tenferro_internal_error::Error;
use tenferro_internal_frontend_core::{DynTensor, StructuredTensor};
use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};
use tidu::expert::Tape;

use crate::AdTensor;

fn rank0_f64(value: f64) -> DenseTensor<f64> {
    DenseTensor::<f64>::from_slice(&[value], &[], MemoryOrder::ColumnMajor).unwrap()
}

fn structured_rank0_f64(value: f64) -> StructuredTensor<f64> {
    StructuredTensor::from(rank0_f64(value))
}

#[test]
fn reverse_api_transitions_between_primal_and_reverse_leaf() {
    let mut value = AdTensor::new_primal(rank0_f64(1.0));
    assert!(!value.requires_grad());
    assert!(value.grad().is_none());
    assert!(value.hvp().is_none());

    value.set_requires_grad(true).unwrap();
    assert!(value.requires_grad());

    value.set_requires_grad(false).unwrap();
    assert!(!value.requires_grad());
    assert!(value.grad().is_none());
    assert!(value.hvp().is_none());
}

#[test]
fn reverse_api_rejects_enabling_requires_grad_on_forward_tensor() {
    let mut value = AdTensor::new_forward(rank0_f64(1.0), rank0_f64(0.5)).unwrap();
    let err = value.set_requires_grad(true).unwrap_err();
    assert!(matches!(
        err,
        Error::UnsupportedAdOp {
            op: "set_requires_grad"
        }
    ));
}

#[test]
fn reverse_leaf_zero_grad_clears_accumulated_gradients_and_hvps() {
    let tape = Tape::<DynTensor>::new();
    let value = AdTensor::new_reverse_leaf(rank0_f64(1.0), &tape).unwrap();

    value
        .accumulate_leaf_grad(structured_rank0_f64(2.0))
        .unwrap();
    value
        .accumulate_leaf_grad(structured_rank0_f64(-0.5))
        .unwrap();
    value
        .accumulate_leaf_hvp(structured_rank0_f64(3.0))
        .unwrap();
    value
        .accumulate_leaf_hvp(structured_rank0_f64(1.5))
        .unwrap();

    assert_eq!(
        value.grad().unwrap().payload().buffer().as_slice().unwrap(),
        &[1.5]
    );
    assert_eq!(
        value.hvp().unwrap().payload().buffer().as_slice().unwrap(),
        &[4.5]
    );

    value.zero_grad().unwrap();
    assert!(value.grad().is_none());
    assert!(value.hvp().is_none());
}

#[test]
fn reverse_non_leaf_rejects_leaf_only_operations() {
    let tape = Tape::<DynTensor>::new();
    let value = AdTensor::new_reverse_output(rank0_f64(1.0), &tape, None).unwrap();

    let zero_grad_err = value.zero_grad().unwrap_err();
    assert!(matches!(zero_grad_err, Error::InvalidAdTensor { .. }));

    let grad_err = value
        .accumulate_leaf_grad(structured_rank0_f64(1.0))
        .unwrap_err();
    assert!(matches!(grad_err, Error::InvalidAdTensor { .. }));

    let hvp_err = value
        .accumulate_leaf_hvp(structured_rank0_f64(1.0))
        .unwrap_err();
    assert!(matches!(hvp_err, Error::InvalidAdTensor { .. }));
}

#[test]
fn reverse_non_leaf_allows_explicit_input_gradient_cache() {
    let tape = Tape::<DynTensor>::new();
    let value = AdTensor::new_reverse_output(rank0_f64(1.0), &tape, None).unwrap();

    value
        .accumulate_input_grad(structured_rank0_f64(1.0))
        .unwrap();
    value
        .accumulate_input_grad(structured_rank0_f64(2.5))
        .unwrap();

    assert_eq!(
        value.grad().unwrap().payload().buffer().as_slice().unwrap(),
        &[3.5]
    );
}
