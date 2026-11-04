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

#[test]
fn the_algorithm_reports_a_loss_that_does_not_depend_on_its_input() {
    use tenferro_ad::AdContext;
    use tenferro_runtime::DType;

    use crate::factor_norm_gradient;

    /// A binding whose presentation ignores the value it is given.
    struct ConstantLoss;

    impl ScalarSupport for ConstantLoss {
        fn qr(&self, input: &TracedTensor) -> Result<(TracedTensor, TracedTensor), Error> {
            Ok((input.clone(), input.clone()))
        }

        fn to_f64(&self, _input: &TracedTensor) -> Result<TracedTensor, Error> {
            // The presented value does not depend on the input, so the input is inactive.
            TracedTensor::from_tensor_concrete_shape(
                tenferro_tensor::Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).map_err(
                    |source| {
                        Error::runtime_state_source("constant loss", ErrorPhase::GraphBuild, source)
                    },
                )?,
            )
        }
    }

    let ad = AdContext::builder().build().expect("ad context");
    let input = TracedTensor::input_concrete_shape(DType::F64, &[1]).expect("traced input");
    let seed = TracedTensor::input_concrete_shape(DType::F64, &[]).expect("traced seed");

    // The algorithm refuses to report nothing when the input does not reach the loss.
    assert!(factor_norm_gradient(&ad, &input, &seed, &ConstantLoss).is_err());
}
