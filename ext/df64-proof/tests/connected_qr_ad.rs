//! The connected QR program of #1790.
//!
//! `A -> QR(A) -> narrow R to f64 -> ordinary f64 loss` differentiates back into the
//! external scalar, and the forward tangent passes through the same graph. This is the
//! checkpoint that needs the factorization, the conversion, and the extension AD rules
//! at once.

use std::sync::Arc;

use tenferro_ad::semantic_extension::SemanticExtensionRuleSet;
use tenferro_ad::AdContext;
use tenferro_cpu::CpuBackend;
use tenferro_df64_proof::ad::{Df64LinearizeRule, Df64VjpRule};
use tenferro_df64_proof::extension::{module, Df64Qr, Df64ToF64, DF64_SCALAR_IDENTITY};
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

fn ad_context(rules: SemanticExtensionRuleSet) -> AdContext {
    AdContext::builder()
        .with_semantic_extension_rules(rules)
        .expect("extension rules")
        .build()
        .expect("ad context")
}

/// `A -> QR -> R -> f64`, with a loss over the narrowed factor.
fn connected_program(
    input: Tensor,
    tangent: Option<Tensor>,
) -> (TracedTensor, TracedTensor, TracedTensor) {
    let dtype = DType::External(std::any::TypeId::of::<Df64>());
    let input =
        TracedTensor::from_tensor_concrete_shape_declaring_scalar(input, DF64_SCALAR_IDENTITY)
            .expect("traced input");
    assert_eq!(input.dtype, dtype);
    let outputs = apply(Arc::new(Df64Qr), &[&input]).expect("traced QR");
    let factor = outputs[1].clone();
    let narrowed = apply(Arc::new(Df64ToF64), &[&factor])
        .expect("traced narrowing")
        .remove(0);
    // The loss is R[0,0]^2, an ordinary f64 operation.
    let loss = narrowed.mul(&narrowed).expect("ordinary f64 loss");
    let tangent_leaf = match tangent {
        Some(tensor) => {
            TracedTensor::from_tensor_concrete_shape_declaring_scalar(tensor, DF64_SCALAR_IDENTITY)
                .expect("traced tangent")
        }
        None => input.clone(),
    };
    (input, loss, tangent_leaf)
}

#[test]
fn the_connected_qr_program_gives_the_expected_gradient() {
    let rules = SemanticExtensionRuleSet::new()
        .with_primal_vjp(Arc::new(Df64VjpRule))
        .expect("one rule per family");
    let ad = ad_context(rules);
    // #1790's orientation case as the traced program's own data.
    let (input, loss, _) = connected_program(
        external(vec![Df64::from_f64(3.0), Df64::from_f64(4.0)], vec![2, 1]),
        None,
    );

    // The seed is the derivative of the loss with respect to itself.
    let seed = TracedTensor::from_tensor_concrete_shape(
        Tensor::from_vec_col_major(vec![1, 1], vec![1.0_f64]).expect("shape matches data"),
    )
    .expect("traced seed");
    let gradients = ad
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

    // #1790's orientation case: A = [[3], [4]] has R = [[5]] and L = R^2 = 25, so
    // dL/dA = 2 R A / |A| = [[6], [8]].
    let gradient = payload(&results[0]);
    assert_eq!(gradient.len(), 2);
    assert!(
        (gradient[0] - Df64::from_f64(6.0)).abs_hi() < 1e-30,
        "{gradient:?}"
    );
    assert!(
        (gradient[1] - Df64::from_f64(8.0)).abs_hi() < 1e-30,
        "{gradient:?}"
    );
}

#[test]
fn the_connected_qr_program_forward_tangent_matches_an_independent_derivative() {
    let rules = SemanticExtensionRuleSet::new()
        .with_linearize(Arc::new(Df64LinearizeRule))
        .expect("one linearize rule per family");
    let ad = ad_context(rules);

    // The tangent direction is the first unit vector, so the forward pass is the first
    // column of the Jacobian.
    let tangent = external(vec![Df64::from_f64(1.0), Df64::zero()], vec![2, 1]);
    let (input, _, tangent_leaf) = connected_program(
        external(vec![Df64::from_f64(3.0), Df64::from_f64(4.0)], vec![2, 1]),
        Some(tangent),
    );
    let qr = apply(Arc::new(Df64Qr), &[&input]).expect("traced QR");
    let factor = qr[1].clone();
    let narrowed = apply(Arc::new(Df64ToF64), &[&factor])
        .expect("traced narrowing")
        .remove(0);

    let forward = ad
        .jvp(&narrowed, &input, &tangent_leaf)
        .expect("the connected forward pass runs");
    let runtime = runtime_with_module();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile(&forward)
        .expect("compiled forward program");
    let results = runtime
        .run_compiled(&program, &[])
        .expect("executed forward program");

    // R = |A|, so the tangent of R is (Q^T A_dot) R = (3/5)(5) = 3 for the first unit
    // direction; the graph narrows it to f64 on the way out.
    assert_eq!(results[0].dtype(), DType::F64);
    let tangent = results[0].as_slice::<f64>().expect("f64 slice");
    assert!(
        (tangent[0] - 3.0).abs() < 1e-15,
        "the forward tangent is {tangent:?}"
    );
}
