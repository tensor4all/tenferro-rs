//! The typed boundaries of the contribution's operations.
//!
//! Each test drives a shape the contribution must refuse and asserts the refusal, so the
//! error branches are behavior rather than uncovered lines.

use std::sync::Arc;

use tenferro_ad::semantic_extension::SemanticExtensionRuleSet;
use tenferro_ad::AdContext;
use tenferro_cpu::CpuBackend;
use tenferro_df64_proof::ad::Df64VjpRule;
use tenferro_df64_proof::extension::{
    module, Df64Expand, Df64FromF64, Df64Qr, Df64QrJvp, Df64QrVjp, Df64ToF64, DF64_SCALAR_IDENTITY,
};
use tenferro_df64_proof::Df64;
use tenferro_runtime::extension::apply;
use tenferro_runtime::{GraphCompiler, Runtime, TracedTensor};
use tenferro_tensor::Tensor;
use tenferro_tensor_core::{ErasedHostTensor, HostTensor};

fn external(values: Vec<Df64>, shape: Vec<usize>) -> Tensor {
    Tensor::external(ErasedHostTensor::new(
        HostTensor::from_vec_col_major(shape, values).expect("shape matches data"),
    ))
}

fn preset(values: Vec<f64>, shape: Vec<usize>) -> Tensor {
    Tensor::from_vec_col_major(shape, values).expect("shape matches data")
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

/// Build one traced leaf whose data the program reads.
fn leaf(values: &[Df64], shape: &[usize]) -> TracedTensor {
    TracedTensor::from_tensor_concrete_shape_declaring_scalar(
        external(values.to_vec(), shape.to_vec()),
        DF64_SCALAR_IDENTITY,
    )
    .expect("traced leaf")
}

fn preset_leaf(values: &[f64], shape: &[usize]) -> TracedTensor {
    TracedTensor::from_tensor_concrete_shape(preset(values.to_vec(), shape.to_vec()))
        .expect("traced leaf")
}

fn execute(graph: &TracedTensor) -> tenferro_runtime::Result<Vec<Tensor>> {
    let runtime = runtime_with_module();
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(graph).expect("compiled program");
    runtime.run_compiled(&program, &[])
}

#[test]
fn the_factorization_refuses_inputs_it_cannot_factor() {
    // A rank-one input is not a matrix.
    let vector = leaf(&[Df64::from_f64(1.0), Df64::from_f64(2.0)], &[2]);
    let error = apply(Arc::new(Df64Qr), &[&vector]).expect_err("a vector is not a matrix");
    assert!(error.to_string().contains("rank-2"), "{error}");

    // A wide matrix has more columns than rows.
    let wide = leaf(
        &[
            Df64::from_f64(1.0),
            Df64::from_f64(0.0),
            Df64::from_f64(0.0),
            Df64::from_f64(1.0),
            Df64::from_f64(1.0),
            Df64::from_f64(1.0),
        ],
        &[2, 3],
    );
    let outputs = apply(Arc::new(Df64Qr), &[&wide]).expect("the family plans this shape");
    let error = execute(&outputs[0]).expect_err("the body refuses a wide matrix");
    assert!(error.to_string().contains("rows as columns"), "{error}");

    // A zero column has no unit vector.
    // The first column is zero, so the factorization has no unit vector for it.
    let zero_column = leaf(
        &[
            Df64::zero(),
            Df64::zero(),
            Df64::from_f64(1.0),
            Df64::from_f64(1.0),
        ],
        &[2, 2],
    );
    let outputs = apply(Arc::new(Df64Qr), &[&zero_column]).expect("the family plans this shape");
    let error = execute(&outputs[0]).expect_err("the body refuses a zero column");
    assert!(error.to_string().contains("zero column"), "{error}");
}

#[test]
fn the_conversions_refuse_the_scalar_they_do_not_declare() {
    // Narrowing takes the external scalar, not a preset tensor.
    let ordinary = preset_leaf(&[1.0], &[1]);
    let error = apply(Arc::new(Df64ToF64), &[&ordinary]).expect_err("preset input");
    assert!(error.to_string().contains("externally defined"), "{error}");

    // Widening takes a preset f64 tensor, not the external scalar.
    let extended = leaf(&[Df64::from_f64(1.0)], &[1]);
    let error = apply(Arc::new(Df64FromF64), &[&extended]).expect_err("external input");
    assert!(error.to_string().contains("preset f64"), "{error}");
}

#[test]
fn the_factorization_derivatives_refuse_a_singular_factor() {
    // A zero triangular factor cannot be solved against, in either direction.
    let q = leaf(&[Df64::from_f64(1.0)], &[1, 1]);
    let singular = leaf(&[Df64::zero()], &[1, 1]);
    let cotangent = leaf(&[Df64::from_f64(1.0)], &[1, 1]);

    let adjoint = apply(
        Arc::new(Df64QrVjp::of(false, true)),
        &[&q, &singular, &cotangent],
    )
    .expect("the family plans this shape");
    let error = execute(&adjoint[0]).expect_err("the body refuses a singular factor");
    assert!(error.to_string().contains("invertible"), "{error}");

    let tangent = apply(Arc::new(Df64QrJvp), &[&q, &singular, &cotangent])
        .expect("the family plans this shape");
    let error = execute(&tangent[0]).expect_err("the body refuses a singular factor");
    assert!(error.to_string().contains("invertible"), "{error}");
}

#[test]
fn a_derivative_rule_refuses_an_operation_outside_its_domain() {
    let rules = SemanticExtensionRuleSet::new()
        .with_primal_vjp(Arc::new(Df64VjpRule))
        .expect("one rule per family");
    let ad = AdContext::builder()
        .with_semantic_extension_rules(rules)
        .expect("extension rules")
        .build()
        .expect("ad context");

    // The broadcast is the total sum's adjoint, not a primal operation, so asking for its
    // adjoint is outside the rule's domain.
    let input = leaf(&[Df64::from_f64(1.0)], &[1]);
    let expanded = apply(Arc::new(Df64Expand::new(vec![2])), &[&input])
        .expect("traced broadcast")
        .remove(0);
    let seed = preset_leaf(&[1.0, 1.0], &[2]);
    let error = ad
        .vjp_many(&expanded, &[&input], &seed)
        .expect_err("the broadcast has no adjoint rule");
    assert!(error.to_string().contains("df64_ops"), "{error}");
}
