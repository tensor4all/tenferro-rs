//! Two sets that both contain the contribution reuse its numerical kernel.
//!
//! #1785 requires that the contribution-owned Df64 kernel be reused by two sets
//! containing that contribution, using practical combinations of the supported types. The
//! contribution's set pairs the external scalar with `f64`; the application's set pairs it
//! with `f32` and `f64`. Both hand out the same Rust value, and both drive it through the
//! same generic entry point, which is parameterized by the contribution's scalar and
//! operation type. A second containing set therefore adds membership and dispatch rather
//! than another numerical specialization. The object-level check counts the instantiations
//! each target emits.

use tenferro_cpu::scalar_binary_into;
use tenferro_df64_proof::{Df64, Df64Add, ExtendedSet};
use tenferro_scalar_consumer_application::ApplicationSet;
use tenferro_tensor_core::{HostTensor, ScalarSet};

fn vector(values: Vec<Df64>) -> HostTensor<Df64> {
    HostTensor::from_vec_col_major(vec![values.len()], values).expect("shape matches data")
}

/// The member the contribution's own set stores.
fn member_of_the_contribution_set() -> HostTensor<Df64> {
    match ExtendedSet::Df64(vector(vec![Df64::from_f64(1.0)])) {
        ExtendedSet::Df64(value) => value,
        other => panic!(
            "the contribution's set must contain the contribution, got {:?}",
            other.tag()
        ),
    }
}

/// The same member as the application's set stores it under its own tag.
fn member_of_the_application_set() -> HostTensor<Df64> {
    match ApplicationSet::Df64(vector(vec![Df64::from_f64(1.0)])) {
        ApplicationSet::Df64(value) => value,
        other => panic!(
            "the application's set must contain the contribution, got {:?}",
            other.tag()
        ),
    }
}

#[test]
fn two_containing_sets_reuse_the_contribution_entry_point() {
    let low = 2f64.powi(-80);
    let increment = vector(vec![Df64 { hi: low, lo: 0.0 }]);

    // Each set stores the contribution, so the member it hands out is the same Rust value.
    assert_eq!(
        member_of_the_contribution_set().as_slice(),
        member_of_the_application_set().as_slice()
    );

    // The same generic entry point serves both sets, so the contribution's kernel has one
    // instantiation for the program rather than one per containing set.
    let mut through_contribution = vector(vec![Df64::from_f64(1.0)]);
    let mut through_application = vector(vec![Df64::from_f64(1.0)]);
    scalar_binary_into::<Df64, Df64Add>(
        "add",
        &mut through_contribution,
        &member_of_the_contribution_set(),
        &increment,
    )
    .expect("addition through the contribution's set succeeds");
    scalar_binary_into::<Df64, Df64Add>(
        "add",
        &mut through_application,
        &member_of_the_application_set(),
        &increment,
    )
    .expect("addition through the application's set succeeds");

    // Both sets produce the same extended-precision result.
    assert_eq!(
        through_contribution.as_slice(),
        through_application.as_slice()
    );
    assert_eq!(
        through_contribution.as_slice()[0],
        Df64 { hi: 1.0, lo: low }
    );

    // Control: the same sum in f64 cannot carry the low component at all.
    assert_eq!(through_contribution.as_slice()[0].narrow_to_f64(), 1.0);
    assert_eq!((1.0_f64 + low) - 1.0, 0.0);
}
