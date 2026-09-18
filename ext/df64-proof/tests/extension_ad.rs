//! First-order AD for the extension-owned operation.
//!
//! The adjoint of the total sum is a broadcast of the output cotangent, and the
//! broadcast is the contribution's own operation, so the rule emits one extension
//! node and the runtime executes it through the same registered module.

use std::sync::Arc;

use tenferro_ad::semantic_extension::SemanticExtensionRuleSet;
use tenferro_ad::AdContext;
use tenferro_cpu::CpuBackend;
use tenferro_df64_proof::ad::Df64VjpRule;
use tenferro_df64_proof::extension::{module, Df64Total, DF64_SCALAR_IDENTITY};
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
    match tensor.external_payload() {
        Some(value) => value
            .downcast_ref::<Df64>()
            .expect("external element type")
            .as_slice()
            .to_vec(),
        _ => panic!("expected an externally defined payload"),
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

#[test]
fn the_total_sum_adjoint_broadcasts_the_cotangent() {
    let rules = SemanticExtensionRuleSet::new()
        .with_primal_vjp(Arc::new(Df64VjpRule))
        .expect("one rule per family");
    let ad = AdContext::builder()
        .with_semantic_extension_rules(rules)
        .expect("extension rules")
        .build()
        .expect("ad context");

    let dtype = DType::External(std::any::TypeId::of::<Df64>());
    let input =
        TracedTensor::input_concrete_shape_declaring_scalar(dtype, &[2], DF64_SCALAR_IDENTITY)
            .expect("traced input");
    let total = apply(Arc::new(Df64Total), &[&input])
        .expect("traced total sum")
        .remove(0);
    let cotangent_value = external(vec![Df64::from_f64(3.0)], vec![]);
    let cotangent_input = cotangent_value.duplicate().expect("independent copy");
    let cotangent = TracedTensor::from_tensor_concrete_shape_declaring_scalar(
        cotangent_value,
        DF64_SCALAR_IDENTITY,
    )
    .expect("traced cotangent");

    let gradients = ad
        .vjp_many(&total, &[&input], &cotangent)
        .expect("the registered rule emits the adjoint");
    let input_gradient = gradients[0].as_ref().expect("the input is active");

    // The adjoint is the cotangent placed into the input's shape, so the backward
    // program is one extension broadcast and executes on the same runtime.
    let runtime = runtime_with_module();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile(input_gradient)
        .expect("compiled backward program");
    let results = runtime
        .run_compiled(&program, &[&cotangent_input])
        .expect("executed adjoint");
    assert_eq!(results.len(), 1);
    assert_eq!(
        payload(&results[0]),
        vec![Df64::from_f64(3.0), Df64::from_f64(3.0)]
    );
}

#[test]
fn the_adjoint_retains_low_order_information() {
    let rules = SemanticExtensionRuleSet::new()
        .with_primal_vjp(Arc::new(Df64VjpRule))
        .expect("one rule per family");
    let ad = AdContext::builder()
        .with_semantic_extension_rules(rules)
        .expect("extension rules")
        .build()
        .expect("ad context");

    let low = 2f64.powi(-80);
    let dtype = DType::External(std::any::TypeId::of::<Df64>());
    let input =
        TracedTensor::input_concrete_shape_declaring_scalar(dtype, &[2], DF64_SCALAR_IDENTITY)
            .expect("traced input");
    let total = apply(Arc::new(Df64Total), &[&input])
        .expect("traced total sum")
        .remove(0);
    let cotangent_value = external(vec![Df64::from_f64(1.0) + Df64::from_f64(low)], vec![]);
    let cotangent_input = cotangent_value.duplicate().expect("independent copy");
    let cotangent = TracedTensor::from_tensor_concrete_shape_declaring_scalar(
        cotangent_value,
        DF64_SCALAR_IDENTITY,
    )
    .expect("traced cotangent");
    let gradients = ad
        .vjp_many(&total, &[&input], &cotangent)
        .expect("the registered rule emits the adjoint");
    let input_gradient = gradients[0].as_ref().expect("the input is active");

    let runtime = runtime_with_module();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile(input_gradient)
        .expect("compiled backward program");
    let results = runtime
        .run_compiled(&program, &[&cotangent_input])
        .expect("executed adjoint");
    let expected = Df64::from_f64(1.0) + Df64::from_f64(low);
    assert_eq!(payload(&results[0]), vec![expected, expected]);
    assert_eq!(expected.lo, low, "the low component survives the adjoint");
}

#[test]
fn the_total_sum_linearization_sums_the_tangents() {
    use tenferro_df64_proof::ad::Df64LinearizeRule;

    let rules = SemanticExtensionRuleSet::new()
        .with_linearize(Arc::new(Df64LinearizeRule))
        .expect("one linearize rule per family");
    let ad = AdContext::builder()
        .with_semantic_extension_rules(rules)
        .expect("extension rules")
        .build()
        .expect("ad context");

    let low = 2f64.powi(-80);
    let dtype = DType::External(std::any::TypeId::of::<Df64>());
    let input =
        TracedTensor::input_concrete_shape_declaring_scalar(dtype, &[2], DF64_SCALAR_IDENTITY)
            .expect("traced input");
    let total = apply(Arc::new(Df64Total), &[&input])
        .expect("traced total sum")
        .remove(0);
    let tangent_value = external(vec![Df64::from_f64(1.0), Df64::from_f64(low)], vec![2]);
    let tangent_input = tangent_value.duplicate().expect("independent copy");
    let tangent = TracedTensor::from_tensor_concrete_shape_declaring_scalar(
        tangent_value,
        DF64_SCALAR_IDENTITY,
    )
    .expect("traced tangent");

    let jvp = ad
        .jvp(&total, &input, &tangent)
        .expect("the registered rule emits the linearization");
    let runtime = runtime_with_module();
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&jvp).expect("compiled forward program");
    let results = runtime
        .run_compiled(&program, &[&tangent_input])
        .expect("executed linearization");
    assert_eq!(results.len(), 1);
    assert_eq!(
        payload(&results[0]),
        vec![Df64::from_f64(1.0) + Df64::from_f64(low)]
    );
}
