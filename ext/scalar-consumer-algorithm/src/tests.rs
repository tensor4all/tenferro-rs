//! The algorithm's capability contract, exercised with a recording binding.

use std::cell::RefCell;

use tenferro_runtime::{Error, ErrorPhase, TracedTensor};

use crate::{squared_factor_norm, ScalarSupport};

/// A binding that records the calls the algorithm makes.
#[derive(Default)]
struct Recording {
    calls: RefCell<Vec<&'static str>>,
}

impl ScalarSupport for Recording {
    fn qr(&self, _input: &TracedTensor) -> Result<(TracedTensor, TracedTensor), Error> {
        self.calls.borrow_mut().push("qr");
        Err(Error::runtime_state(
            "recording",
            ErrorPhase::GraphBuild,
            "the shape is what matters here",
        ))
    }

    fn to_f64(&self, _input: &TracedTensor) -> Result<TracedTensor, Error> {
        self.calls.borrow_mut().push("to_f64");
        Err(Error::runtime_state(
            "recording",
            ErrorPhase::GraphBuild,
            "the shape is what matters here",
        ))
    }
}

#[test]
fn the_algorithm_asks_only_for_the_declared_capabilities() {
    let support = Recording::default();
    let input = TracedTensor::input_concrete_shape(tenferro_tensor::DType::F64, &[2, 1])
        .expect("traced input");

    // The factorization is the first capability the algorithm needs, so a binding that
    // refuses it stops the program before any other call.
    assert!(squared_factor_norm(&input, &support).is_err());
    assert_eq!(support.calls.borrow().as_slice(), &["qr"]);
}
