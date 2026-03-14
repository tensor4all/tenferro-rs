use std::sync::{Arc, Mutex};

use chainrules::{AutogradGraph, DualValue, Tape, TrackedValue, Variable};

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn chainrules_public_handles_are_send_sync() {
    assert_send_sync::<Tape<f64>>();
    assert_send_sync::<TrackedValue<f64>>();
    assert_send_sync::<DualValue<f64>>();
    assert_send_sync::<Variable<f64>>();
    assert_send_sync::<Arc<Mutex<AutogradGraph<f64>>>>();
}
