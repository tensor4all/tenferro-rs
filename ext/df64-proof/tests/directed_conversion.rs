//! The extended-precision example and the directed conversions around it.
//!
//! This walks #1785's required example through public boundaries: two `Df64`
//! tensors holding `1` and `2^-80` are added in `Df64`, `1` is subtracted in
//! `Df64`, and the result is `2^-80`. The directed conversions in both directions
//! are then checked against their declared rounding.

use tenferro_cpu::{scalar_binary_into, AddOp, SubOp};
use tenferro_df64_proof::{conversion, Df64};
use tenferro_tensor::{DType, Tensor};
use tenferro_tensor_core::{ErasedHostTensor, HostTensor};

fn external_df64(values: &[Df64]) -> Tensor {
    Tensor::external(ErasedHostTensor::new(
        HostTensor::from_vec_col_major(vec![values.len()], values.to_vec())
            .expect("shape matches data"),
    ))
}

fn payload_of(tensor: &Tensor) -> &HostTensor<Df64> {
    match tensor.external_payload() {
        Some(payload) => payload.downcast_ref::<Df64>().expect("df64 payload"),
        None => panic!("expected an external payload"),
    }
}

fn values_of(tensor: &Tensor) -> &[Df64] {
    payload_of(tensor).as_slice()
}

/// The mutable public projection reaches the payload, so a destination is filled
/// without a copy back into the caller's tensor.
fn destination_for(tensor: &mut Tensor) -> &mut HostTensor<Df64> {
    match tensor.external_payload_mut() {
        Some(payload) => payload.downcast_mut::<Df64>().expect("df64 payload"),
        None => panic!("expected an external payload"),
    }
}

#[test]
fn the_required_extended_precision_example_runs_end_to_end() {
    let low = 2f64.powi(-80);

    // Two externally defined tensors hold 1 and 2^-80.
    let one = external_df64(&[Df64::from_f64(1.0)]);
    let tiny = external_df64(&[Df64::from_f64(low)]);

    // Add in Df64.
    let mut sum = external_df64(&[Df64::zero()]);
    scalar_binary_into::<Df64, AddOp>(
        "add",
        destination_for(&mut sum),
        payload_of(&one),
        payload_of(&tiny),
    )
    .expect("addition succeeds");
    assert_eq!(values_of(&sum), &[Df64 { hi: 1.0, lo: low }]);

    // Subtract 1 in Df64.
    let mut difference = external_df64(&[Df64::zero()]);
    scalar_binary_into::<Df64, SubOp>(
        "sub",
        destination_for(&mut difference),
        payload_of(&sum),
        payload_of(&one),
    )
    .expect("subtraction succeeds");

    // The result is 2^-80, and it survives the erased read.
    assert_eq!(values_of(&difference), &[Df64 { hi: low, lo: 0.0 }]);
    let narrowed = conversion::to_f64(&difference).expect("directed conversion");
    assert_eq!(narrowed.as_slice::<f64>().expect("f64 slice"), &[low]);
    assert_eq!(narrowed.dtype(), DType::F64);

    // The same computation in f64 loses the low component, which is what the
    // extended set exists to prevent.
    assert_eq!((1.0_f64 + low) - 1.0, 0.0);
}

#[test]
fn a_directed_conversion_uses_the_low_component_and_declares_its_rounding() {
    // The low component participates, so a value the plain high component would
    // not represent reaches the destination.
    let participates = external_df64(&[Df64 {
        hi: 1.0,
        lo: 2f64.powi(-52),
    }]);
    assert_eq!(
        conversion::to_f64(&participates)
            .expect("directed conversion")
            .as_slice::<f64>()
            .expect("f64 slice"),
        &[1.0 + 2f64.powi(-52)]
    );

    // A low component that cannot be represented after rounding is rounded, not
    // folded into a second destination value.
    let rounds_away = external_df64(&[Df64 {
        hi: 1.0,
        lo: 2f64.powi(-80),
    }]);
    assert_eq!(
        conversion::to_f64(&rounds_away)
            .expect("directed conversion")
            .as_slice::<f64>()
            .expect("f64 slice"),
        &[1.0]
    );

    // Exactly half an ulp rounds to even, matching the declared rule.
    let tie = external_df64(&[Df64 {
        hi: 1.0,
        lo: 2f64.powi(-53),
    }]);
    assert_eq!(
        conversion::to_f64(&tie)
            .expect("directed conversion")
            .as_slice::<f64>()
            .expect("f64 slice"),
        &[1.0]
    );
}

#[test]
fn the_two_directions_are_exact_where_the_value_is_representable() {
    // A value that is exactly representable in both types survives the round trip.
    let low = 2f64.powi(-80);
    let widened = conversion::to_df64(
        &Tensor::from_vec_col_major(vec![1], vec![low]).expect("shape matches data"),
    )
    .expect("directed conversion");
    assert_eq!(
        widened.dtype(),
        DType::External(std::any::TypeId::of::<Df64>())
    );
    assert_eq!(values_of(&widened), &[Df64 { hi: low, lo: 0.0 }]);

    let narrowed = conversion::to_f64(&widened).expect("directed conversion");
    assert_eq!(narrowed.as_slice::<f64>().expect("f64 slice"), &[low]);

    // A value only the extended type holds loses its low component in the f64
    // direction, and the widening does not recover what the narrowing discarded.
    let exact = external_df64(&[Df64 { hi: 1.0, lo: low }]);
    let lost = conversion::to_f64(&exact).expect("directed conversion");
    assert_eq!(lost.as_slice::<f64>().expect("f64 slice"), &[1.0]);
    assert_eq!(
        values_of(&conversion::to_df64(&lost).expect("directed conversion")),
        &[Df64::from_f64(1.0)]
    );
}

#[test]
fn a_directed_conversion_rejects_a_source_it_does_not_declare() {
    let ordinary = Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).expect("shape matches data");
    // The f64 -> Df64 direction takes f64, and the Df64 -> f64 direction takes the
    // external type; neither coerces the other silently.
    assert!(conversion::to_df64(&external_df64(&[Df64::from_f64(1.0)])).is_err());
    assert!(conversion::to_f64(&ordinary).is_err());
}

#[test]
fn a_directed_conversion_rejects_a_payload_of_another_element_type() {
    // The narrowing direction takes the contribution's element type, so a payload of a
    // different element type is neither the scalar it declares nor an ordinary f64 tensor.
    let other = external_df64(&[Df64::from_f64(1.0)]);
    assert!(conversion::to_f64(&other).is_ok());

    let foreign = Tensor::external(ErasedHostTensor::new(
        HostTensor::from_vec_col_major(vec![1], vec![7_i64]).expect("shape matches data"),
    ));
    assert!(conversion::to_f64(&foreign).is_err());
    assert!(conversion::to_df64(&foreign).is_err());
}
