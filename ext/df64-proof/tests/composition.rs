use std::ops::Add;

use tenferro_cpu::{scalar_binary_into, scalar_fold};
use tenferro_df64_proof::Df64;
use tenferro_tensor_core::HostTensor;
use tenferro_tensor_core::{ad_admission, AdAdmissionError, ScalarArithmetic, ScalarDomain};

fn df64(values: &[f64]) -> HostTensor<Df64> {
    HostTensor::from_vec_col_major(
        vec![values.len()],
        values.iter().copied().map(Df64::from_f64).collect(),
    )
    .expect("shape matches data")
}

#[test]
fn external_scalar_sum_retains_low_order_information() {
    let low = 2f64.powi(-80);
    let values = df64(&[1.0, low]);

    let total = scalar_fold("sum", &values, Df64::zero(), Df64::add).expect("reduction succeeds");
    let difference = total - Df64::from_f64(1.0);

    assert_eq!(difference, Df64 { hi: low, lo: 0.0 });
    assert_eq!(difference.narrow_to_f64(), low);

    // Control: the same computation in f64 loses the low component entirely.
    let f64_total = (1.0_f64 + low) - 1.0;
    assert_eq!(f64_total, 0.0);
}

#[test]
fn external_scalar_runs_through_the_elementwise_entry_point() {
    let lhs = df64(&[1.0, 3.0]);
    let rhs = df64(&[2.0, 4.0]);
    let mut destination = df64(&[0.0, 0.0]);

    scalar_binary_into("add", &mut destination, &lhs, &rhs, Df64::add).expect("addition succeeds");

    assert_eq!(
        destination.as_slice(),
        &[Df64::from_f64(3.0), Df64::from_f64(7.0)]
    );
}

#[test]
fn external_scalar_keeps_low_order_information_through_the_elementwise_entry_point() {
    let low = 2f64.powi(-80);
    let lhs = df64(&[1.0]);
    let rhs = df64(&[low]);
    let mut destination = df64(&[0.0]);

    scalar_binary_into("add", &mut destination, &lhs, &rhs, Df64::add).expect("addition succeeds");

    assert_eq!(destination.as_slice()[0], Df64 { hi: 1.0, lo: low });
    assert_eq!(destination.as_slice()[0].narrow_to_f64(), 1.0);
}

#[test]
fn preset_scalar_f64_uses_the_same_entry_points() {
    let lhs = HostTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let rhs = HostTensor::from_vec_col_major(vec![2], vec![10.0_f64, 20.0]).unwrap();
    let mut destination = HostTensor::from_vec_col_major(vec![2], vec![0.0_f64, 0.0]).unwrap();

    scalar_binary_into("add", &mut destination, &lhs, &rhs, |a, b| a + b).unwrap();
    assert_eq!(destination.as_slice(), &[11.0, 22.0]);

    let total = scalar_fold("sum", &destination, 0.0_f64, |a, b| a + b).unwrap();
    assert_eq!(total, 33.0);
}

#[test]
fn mutable_borrow_through_the_public_tensor_type_reaches_the_entry_point() {
    let mut values = df64(&[1.0, 2.0]);
    for element in values.as_mut_slice() {
        *element = *element + Df64::from_f64(1.0);
    }

    let lhs = df64(&[10.0, 10.0]);
    let mut destination = df64(&[0.0, 0.0]);
    scalar_binary_into("add", &mut destination, &values, &lhs, Df64::add).unwrap();

    assert_eq!(
        destination.as_slice(),
        &[Df64::from_f64(12.0), Df64::from_f64(13.0)]
    );
}

#[test]
fn mismatched_shapes_are_rejected_without_touching_the_destination() {
    let lhs = df64(&[1.0, 2.0]);
    let rhs = df64(&[1.0]);
    let mut destination = df64(&[9.0, 9.0]);

    let error = scalar_binary_into("add", &mut destination, &lhs, &rhs, Df64::add)
        .expect_err("shape mismatch is rejected");

    assert_eq!(
        error.kind(),
        tenferro_tensor_core::ErrorKind::Validation(
            tenferro_tensor_core::ValidationKind::ShapeMismatch
        )
    );
    assert_eq!(
        destination.as_slice(),
        &[Df64::from_f64(9.0), Df64::from_f64(9.0)]
    );
}

#[test]
fn empty_input_folds_to_the_initial_value() {
    let empty: HostTensor<Df64> = HostTensor::from_vec_col_major(vec![0], Vec::new()).unwrap();
    let total = scalar_fold("sum", &empty, Df64::zero(), Df64::add).unwrap();
    assert_eq!(total, Df64::zero());
}

#[test]
fn external_scalar_implements_the_public_contract() {
    assert_eq!(
        <Df64 as tenferro_tensor_core::Scalar>::DOMAIN,
        ScalarDomain::Field
    );
    assert_eq!(
        Df64::scalar_add(Df64::from_f64(1.0), Df64::from_f64(2.0)),
        Df64::from_f64(3.0)
    );
    assert_eq!(
        Df64::scalar_mul(Df64::from_f64(3.0), Df64::from_f64(4.0)),
        Df64::from_f64(12.0)
    );
    assert_eq!(
        Df64::scalar_sub(Df64::from_f64(3.0), Df64::from_f64(4.0)),
        Df64::from_f64(-1.0)
    );
    assert_eq!(Df64::scalar_one(), Df64::from_f64(1.0));
    assert_eq!(Df64::scalar_zero(), Df64::zero());
}

#[test]
fn external_scalar_reaches_the_shared_admission_query() {
    // First order is admissible in principle but has no rules yet: the shared
    // query rejects it explicitly instead of producing a zero gradient.
    assert_eq!(
        ad_admission::<Df64>(1),
        Err(AdAdmissionError::AdRuleUnavailable)
    );
    assert_eq!(
        ad_admission::<Df64>(2),
        Err(AdAdmissionError::UnsupportedAdOrder { order: 2 })
    );
    // A non-field preset scalar is rejected on the same query.
    assert_eq!(
        ad_admission::<bool>(1),
        Err(AdAdmissionError::NonFieldScalar)
    );
}
