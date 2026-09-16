//! The public surface of the bfloat16 contribution, exercised from outside the crate.
//!
//! The crate's examples are doctests, and the coverage harness does not instrument doctests, so
//! this file drives the same surface from an integration test. That keeps the reported coverage of
//! the crate's own lines honest rather than relying on documentation examples to cover them.

use tenferro_bf16_proof::reduction::product_in_f32_accumulation;
use tenferro_bf16_proof::{bf16, Bf16, Bf16Add, Bf16Mul, Bf16Set, Bf16Sub, Bf16Tag};
use tenferro_cpu::BinaryScalarOp;
use tenferro_tensor_core::{HostTensor, Scalar, ScalarArithmetic, ScalarDomain, ScalarSet};

fn tensor(values: &[f32]) -> HostTensor<Bf16> {
    HostTensor::from_vec_col_major(
        vec![values.len()],
        values.iter().copied().map(Bf16::from_f32).collect(),
    )
    .expect("shape matches data")
}

#[test]
fn the_constructors_and_accessors_agree_with_the_stored_representation() {
    let wrapped = Bf16::of(bf16::from_f32(2.0));
    assert_eq!(wrapped.narrow(), bf16::from_f32(2.0));
    assert_eq!(wrapped.to_f32(), 2.0);
    assert_eq!(wrapped.to_f64(), 2.0);

    assert_eq!(Bf16::from_f64(0.5).to_f64(), 0.5);
    assert_eq!(Bf16::from_f32(-0.5).to_f32(), -0.5);
    assert_eq!(Bf16::default().to_f32(), 0.0);
}

#[test]
fn the_arithmetic_trait_provides_the_identities_and_operations() {
    assert_eq!(Bf16::zero().to_f32(), 0.0);
    assert_eq!(Bf16::one().to_f32(), 1.0);
    assert_eq!(<Bf16 as ScalarArithmetic>::scalar_zero().to_f32(), 0.0);
    assert_eq!(<Bf16 as ScalarArithmetic>::scalar_one().to_f32(), 1.0);
    assert_eq!(<Bf16 as Scalar>::DOMAIN, ScalarDomain::Field);

    let two = Bf16::from_f32(2.0);
    let three = Bf16::from_f32(3.0);
    assert_eq!(two.scalar_add(three).to_f32(), 5.0);
    assert_eq!(three.scalar_sub(two).to_f32(), 1.0);
    assert_eq!(two.scalar_mul(three).to_f32(), 6.0);

    // The standard operators the shared kernels reach through the trait agree with it.
    assert_eq!((two + three).to_f32(), 5.0);
    assert_eq!((three - two).to_f32(), 1.0);
    assert_eq!((two * three).to_f32(), 6.0);
    assert_eq!((-two).to_f32(), -2.0);
}

#[test]
fn the_operation_types_apply_through_the_shared_trait() {
    let two = Bf16::from_f32(2.0);
    let three = Bf16::from_f32(3.0);
    assert_eq!(
        <Bf16Add as BinaryScalarOp<Bf16>>::apply(two, three).to_f32(),
        5.0
    );
    assert_eq!(
        <Bf16Sub as BinaryScalarOp<Bf16>>::apply(three, two).to_f32(),
        1.0
    );
    assert_eq!(
        <Bf16Mul as BinaryScalarOp<Bf16>>::apply(two, three).to_f32(),
        6.0
    );
}

#[test]
fn the_set_carries_both_members_and_reports_their_tags() {
    let narrow = Bf16Set::Bf16(tensor(&[1.0, 2.0]));
    let wide = Bf16Set::F32(
        HostTensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]).expect("shape matches data"),
    );

    assert_eq!(narrow.tag(), Bf16Tag::Bf16);
    assert_eq!(wide.tag(), Bf16Tag::F32);
    assert_eq!(Bf16Set::TAGS, &[Bf16Tag::F32, Bf16Tag::Bf16]);
}

#[test]
fn the_product_accumulates_in_f32() {
    // Powers of two are exactly representable in bfloat16, so the product's contract is an
    // equality rather than a tolerance, and the accumulation is visible in the result.
    let factors = tensor(&[2.0, 3.0, 4.0]);
    assert_eq!(
        product_in_f32_accumulation(factors.as_slice()).to_f32(),
        24.0
    );

    let halves = tensor(&[0.5, 0.5, 0.5]);
    assert_eq!(
        product_in_f32_accumulation(halves.as_slice()).to_f32(),
        0.125
    );
}
