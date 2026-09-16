//! The control for [`contribution_reuse`](../contribution_reuse.rs): one containing set.
//!
//! Both files run the same arithmetic on the same shapes through the same generic entry
//! point. The only difference is that the other file also uses a second ScalarSet that
//! contains the contribution. Comparing the instantiations the two targets emit therefore
//! isolates what a second containing set adds: the check that reads this pair requires the
//! instantiation sets to be equal, so the second set contributes membership and dispatch
//! rather than another numerical specialization.

use tenferro_cpu::scalar_binary_into;
use tenferro_df64_proof::{Df64, Df64Add, ExtendedSet};
use tenferro_tensor_core::{HostTensor, ScalarSet};

fn vector(values: Vec<Df64>) -> HostTensor<Df64> {
    HostTensor::from_vec_col_major(vec![values.len()], values).expect("shape matches data")
}

fn member_of_the_contribution_set() -> HostTensor<Df64> {
    match ExtendedSet::Df64(vector(vec![Df64::from_f64(1.0)])) {
        ExtendedSet::Df64(value) => value,
        other => panic!(
            "the contribution's set must contain the contribution, got {:?}",
            other.tag()
        ),
    }
}

#[test]
fn one_containing_set_runs_the_same_arithmetic() {
    let low = 2f64.powi(-80);
    let increment = vector(vec![Df64 { hi: low, lo: 0.0 }]);

    let mut destination = vector(vec![Df64::from_f64(1.0)]);
    scalar_binary_into::<Df64, Df64Add>(
        "add",
        &mut destination,
        &member_of_the_contribution_set(),
        &increment,
    )
    .expect("addition through the contribution's set succeeds");

    assert_eq!(destination.as_slice()[0], Df64 { hi: 1.0, lo: low });
    assert_eq!(destination.as_slice()[0].narrow_to_f64(), 1.0);
    assert_eq!((1.0_f64 + low) - 1.0, 0.0);
}
