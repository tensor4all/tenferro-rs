//! Connected programs that mix ordinary `f64` work with the external scalar.
//!
//! The point of these tests is that one graph holds both kinds of value and one
//! reverse pass crosses the conversion, which is what #1790's connected programs
//! need before QR joins them.

use std::sync::Arc;

use tenferro_ad::semantic_extension::SemanticExtensionRuleSet;
use tenferro_ad::AdContext;
use tenferro_cpu::CpuBackend;
use tenferro_df64_proof::ad::Df64VjpRule;
use tenferro_df64_proof::extension::{module, Df64FromF64, Df64ToF64, DF64_SCALAR_IDENTITY};
use tenferro_df64_proof::Df64;
use tenferro_runtime::extension::apply;
use tenferro_runtime::{GraphCompiler, Runtime, TracedTensor};
use tenferro_tensor::{DType, Tensor};
use tenferro_tensor_core::{ErasedHostTensor, HostTensor};

fn external(values: Vec<Df64>, shape: Vec<usize>) -> Tensor {
    Tensor::external(ErasedHostTensor::new(
        HostTensor::from_vec_col_major(shape, values).expect("shape matches data"),
    ))
}

fn payload(tensor: &Tensor) -> Vec<Df64> {
    match tensor {
        Tensor::External(value, _) => value
            .downcast_ref::<Df64>()
            .expect("external element type")
            .as_slice()
            .to_vec(),
        other => panic!(
            "expected an externally defined payload, found {:?}",
            other.dtype()
        ),
    }
}

fn runtime_with_module() -> Runtime {
    let backend = CpuBackend::new();
    let mut builder = Runtime::builder();
    builder
        .register_engine(tenferro_cpu::runtime_engine_registration(&backend).expect("engine"))
        .expect("register the CPU engine");
    builder
        .install_extension_module(module().expect("module"))
        .expect("install the Df64 module");
    builder.build().expect("runtime with the module")
}

fn ad_context() -> AdContext {
    let rules = SemanticExtensionRuleSet::new()
        .with_primal_vjp(Arc::new(Df64VjpRule))
        .expect("one rule per family");
    AdContext::builder()
        .with_semantic_extension_rules(rules)
        .expect("extension rules")
        .build()
        .expect("ad context")
}

#[test]
fn a_narrowing_program_differentiates_and_does_not_recover_discarded_information() {
    let low = 2f64.powi(-80);
    let source = vec![
        Df64::from_f64(1.0) + Df64::from_f64(low),
        Df64::from_f64(2.0),
    ];
    let source_input = external(source.clone(), vec![2]);

    // The forward program narrows the external scalar and then does ordinary f64 work.
    let input = TracedTensor::from_tensor_concrete_shape_declaring_scalar(
        source_input,
        DF64_SCALAR_IDENTITY,
    )
    .expect("traced external input");
    let narrowed = apply(Arc::new(Df64ToF64), &[&input])
        .expect("traced narrowing")
        .remove(0);
    let loss = narrowed
        .reduce_sum_squares(&[0])
        .expect("ordinary f64 loss");

    // The reverse pass crosses the conversion back into the external scalar.
    let seed = TracedTensor::from_tensor_concrete_shape(
        Tensor::from_vec_col_major(vec![], vec![1.0_f64]).expect("shape matches data"),
    )
    .expect("traced seed");
    let gradients = ad_context()
        .vjp_many(&loss, &[&input], &seed)
        .expect("the connected reverse pass runs");
    let gradient = gradients[0].as_ref().expect("the input is active");

    let runtime = runtime_with_module();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile(gradient)
        .expect("compiled reverse program");
    let results = runtime
        .run_compiled(&program, &[])
        .expect("executed reverse program");
    assert_eq!(results.len(), 1);

    // d/dx sum(x^2) = 2x, evaluated at the *narrowed* values, so the low component
    // the narrowing discarded is not recovered by widening it back.
    let expected = vec![Df64::from_f64(2.0 * (1.0 + low)), Df64::from_f64(4.0)];
    assert_eq!(payload(&results[0]), expected);
    assert_eq!(
        payload(&results[0])[0].lo,
        0.0,
        "the discarded part is gone"
    );
    assert_eq!(
        results[0].dtype(),
        DType::External(std::any::TypeId::of::<Df64>())
    );
}

#[test]
fn a_widening_program_differentiates_across_the_conversion() {
    let source = vec![3.0_f64, 4.0];
    let input = TracedTensor::from_tensor_concrete_shape(
        Tensor::from_vec_col_major(vec![2], source.clone()).expect("shape matches data"),
    )
    .expect("traced f64 input");
    let widened = apply(Arc::new(Df64FromF64), &[&input])
        .expect("traced widening")
        .remove(0);
    // The loss is the square of the second component, computed after narrowing back.
    let narrowed = apply(Arc::new(Df64ToF64), &[&widened])
        .expect("traced narrowing")
        .remove(0);
    let loss = narrowed
        .reduce_sum_squares(&[0])
        .expect("ordinary f64 loss");

    let seed = TracedTensor::from_tensor_concrete_shape(
        Tensor::from_vec_col_major(vec![], vec![1.0_f64]).expect("shape matches data"),
    )
    .expect("traced seed");
    let gradients = ad_context()
        .vjp_many(&loss, &[&input], &seed)
        .expect("the connected reverse pass runs");
    let gradient = gradients[0].as_ref().expect("the input is active");

    let runtime = runtime_with_module();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile(gradient)
        .expect("compiled reverse program");
    let results = runtime
        .run_compiled(&program, &[])
        .expect("executed reverse program");
    assert_eq!(results.len(), 1);
    // d/dx sum(x^2) = 2x in ordinary f64, so the gradient carries no external dtype.
    assert_eq!(results[0].dtype(), DType::F64);
    assert_eq!(
        results[0].as_slice::<f64>().expect("f64 slice"),
        &[6.0, 8.0]
    );
}
