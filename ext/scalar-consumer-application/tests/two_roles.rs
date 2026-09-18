//! The application role: one algorithm, two bindings.
//!
//! Both tests call the same function from the algorithm crate and differ only in the
//! binding they install: canonical standard support, and standard support with the
//! external scalar contribution. No source line of the algorithm changes between them.

use std::sync::Arc;

use tenferro_ad::semantic_extension::SemanticExtensionRuleSet;
use tenferro_ad::AdContext;
use tenferro_cpu::CpuBackend;
use tenferro_df64_proof::ad::Df64VjpRule;
use tenferro_df64_proof::extension::{module, Df64Qr, Df64ToF64, DF64_SCALAR_IDENTITY};
use tenferro_df64_proof::Df64;
use tenferro_linalg::TracedTensorLinalgExt;
use tenferro_runtime::extension::apply;
use tenferro_runtime::{Error, ErrorPhase, GraphCompiler, Runtime, TracedTensor};
use tenferro_scalar_consumer_algorithm::{factor_norm_gradient, ScalarSupport};
use tenferro_tensor::{DType, Tensor};
use tenferro_tensor_core::{ErasedHostTensor, HostTensor};

/// Canonical standard support: the linalg factorization, with values already `f64`.
struct Standard;

impl ScalarSupport for Standard {
    fn qr(&self, input: &TracedTensor) -> Result<(TracedTensor, TracedTensor), Error> {
        input.qr()
    }

    fn to_f64(&self, input: &TracedTensor) -> Result<TracedTensor, Error> {
        // The supported scalar is already ordinary `f64`, so no conversion is needed.
        Ok(input.clone())
    }
}

/// Standard support plus the external scalar contribution.
struct Extended;

impl ScalarSupport for Extended {
    fn qr(&self, input: &TracedTensor) -> Result<(TracedTensor, TracedTensor), Error> {
        let outputs = apply(Arc::new(Df64Qr), &[input])?;
        let mut outputs = outputs.into_iter();
        let q = outputs.next().ok_or_else(|| {
            Error::runtime_state(
                "Extended::qr",
                ErrorPhase::GraphBuild,
                "the factorization returned no factor",
            )
        })?;
        let r = outputs.next().ok_or_else(|| {
            Error::runtime_state(
                "Extended::qr",
                ErrorPhase::GraphBuild,
                "the factorization returned no triangular factor",
            )
        })?;
        Ok((q, r))
    }

    fn to_f64(&self, input: &TracedTensor) -> Result<TracedTensor, Error> {
        Ok(apply(Arc::new(Df64ToF64), &[input])?.remove(0))
    }
}

/// Build the runtime the application composes: the CPU engine, the canonical linalg
/// support, and the external scalar contribution.
fn runtime_with_support(contribution: bool) -> Runtime {
    let backend = CpuBackend::new();
    let mut builder = Runtime::builder();
    builder
        .register_engine(tenferro_cpu::runtime_engine_registration(&backend).expect("engine"))
        .expect("register the CPU engine");
    builder
        .install_extension_module(
            tenferro_linalg::extension_module::<CpuBackend>(
                tenferro_cpu::runtime_engine_id().expect("engine id"),
            )
            .expect("linalg module"),
        )
        .expect("install the linalg module");
    if contribution {
        builder
            .install_extension_module(module().expect("module"))
            .expect("install the Df64 module");
    }
    builder.build().expect("runtime with support")
}

fn run<S: ScalarSupport>(
    support: &S,
    contribution: bool,
    ad: &AdContext,
    input: TracedTensor,
) -> Tensor {
    let seed = TracedTensor::from_tensor_concrete_shape(
        Tensor::from_vec_col_major(vec![], vec![1.0_f64]).expect("shape matches data"),
    )
    .expect("traced seed");
    let gradient = factor_norm_gradient(ad, &input, &seed, support).expect("the program runs");

    let runtime = runtime_with_support(contribution);
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&gradient).expect("compiled program");
    let results = runtime
        .run_compiled(&program, &[])
        .expect("executed the program");
    assert_eq!(results.len(), 1);
    results.into_iter().next().expect("one result")
}

#[test]
fn the_same_algorithm_runs_with_standard_support() {
    let ad = AdContext::builder()
        .with_semantic_extension_rules(tenferro_linalg::semantic_ad_rules().expect("rules"))
        .expect("extension rules")
        .build()
        .expect("ad context");
    let input = TracedTensor::from_tensor_concrete_shape(
        Tensor::from_vec_col_major(vec![2, 1], vec![3.0_f64, 4.0]).expect("shape matches data"),
    )
    .expect("traced input");

    let gradient = run(&Standard, false, &ad, input);
    assert_eq!(gradient.dtype(), DType::F64);
    let gradient = gradient.as_slice::<f64>().expect("f64 slice");
    assert!((gradient[0] - 6.0).abs() < 1e-12, "{gradient:?}");
    assert!((gradient[1] - 8.0).abs() < 1e-12, "{gradient:?}");
}

#[test]
fn the_same_algorithm_runs_with_the_contribution() {
    let rules = SemanticExtensionRuleSet::new()
        .with_primal_vjp(Arc::new(Df64VjpRule))
        .expect("one rule per family");
    let ad = AdContext::builder()
        .with_semantic_extension_rules(rules)
        .expect("extension rules")
        .build()
        .expect("ad context");

    // The application names the scalar and its canonical identity; the algorithm does not.
    let input = TracedTensor::from_tensor_concrete_shape_declaring_scalar(
        Tensor::external(ErasedHostTensor::new(
            HostTensor::from_vec_col_major(
                vec![2, 1],
                vec![Df64::from_f64(3.0), Df64::from_f64(4.0)],
            )
            .expect("shape matches data"),
        )),
        DF64_SCALAR_IDENTITY,
    )
    .expect("traced input");

    let gradient = run(&Extended, true, &ad, input);
    assert_eq!(
        gradient.dtype(),
        DType::External(std::any::TypeId::of::<Df64>())
    );
    let Some(payload) = gradient.external_payload() else {
        panic!("expected an externally defined gradient");
    };
    let values = payload
        .downcast_ref::<Df64>()
        .expect("external element type")
        .as_slice();
    assert!(
        (values[0] - Df64::from_f64(6.0)).abs_hi() < 1e-30,
        "{values:?}"
    );
    assert!(
        (values[1] - Df64::from_f64(8.0)).abs_hi() < 1e-30,
        "{values:?}"
    );
}
