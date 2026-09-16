//! Independent references for the QR derivatives.
//!
//! #1788 requires both factor derivatives to be checked with independent precision-appropriate
//! references, finite differences, and JVP/VJP duality, and #1790 repeats the requirement for the
//! connected programs: re-running the same kernel, or checking a standalone derivative, is not
//! that evidence. This file supplies the two references.
//!
//! The duality identity is `sum(JVP(v) * w) == sum(v * VJP(w))` for a tangent `v` and a cotangent
//! `w`. Reverse mode is linear in the cotangent, so the factor cotangents are taken one at a time
//! and the resulting input cotangents are added, which also exercises both availability paths of
//! the contributed adjoint. Both sides are computed in the extended scalar, so they should agree
//! far below what an `f64` reference could resolve.
//!
//! The second reference differentiates a scalar loss by central differences at a step small enough
//! that an `f64` intermediate could not resolve the change, and compares the directional
//! derivative with the one the VJP produces. That case first reproduces the orientation result
//! `dL/dA = [[6], [8]]` for `A = [[3], [4]]`, which is also what confirms the inputs were bound in
//! the order the compiled program expects.

use std::sync::Arc;

use tenferro_ad::semantic_extension::SemanticExtensionRuleSet;
use tenferro_ad::AdContext;
use tenferro_cpu::CpuBackend;
use tenferro_df64_proof::ad::{Df64LinearizeRule, Df64VjpRule};
use tenferro_df64_proof::extension::{module, Df64Qr, DF64_SCALAR_IDENTITY};
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

fn linearize_context() -> AdContext {
    let rules = SemanticExtensionRuleSet::new()
        .with_linearize(Arc::new(Df64LinearizeRule))
        .expect("one linearize rule per family");
    AdContext::builder()
        .with_semantic_extension_rules(rules)
        .expect("extension rules")
        .build()
        .expect("ad context")
}

fn cotangent_context() -> AdContext {
    let rules = SemanticExtensionRuleSet::new()
        .with_primal_vjp(Arc::new(Df64VjpRule))
        .expect("one adjoint rule per family");
    AdContext::builder()
        .with_semantic_extension_rules(rules)
        .expect("extension rules")
        .build()
        .expect("ad context")
}

fn leaf(values: Vec<Df64>, shape: Vec<usize>) -> TracedTensor {
    TracedTensor::from_tensor_concrete_shape_declaring_scalar(
        external(values, shape),
        DF64_SCALAR_IDENTITY,
    )
    .expect("traced leaf")
}

/// `A = [[3], [4]]`, the orientation case, so the results can also be read by hand.
fn orientation_input() -> Vec<Df64> {
    vec![Df64::from_f64(3.0), Df64::from_f64(4.0)]
}

/// Run `A -> QR` and return `(Q, R)` as `Df64` vectors.
fn factorize(values: &[Df64], rows: usize, columns: usize) -> (Vec<Df64>, Vec<Df64>) {
    let input = leaf(values.to_vec(), vec![rows, columns]);
    let outputs = apply(Arc::new(Df64Qr), &[&input]).expect("traced QR");
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_many(&[&outputs[0], &outputs[1]])
        .expect("compiled QR");
    let results = runtime_with_module()
        .run_compiled(&program, &[&external(values.to_vec(), vec![rows, columns])])
        .expect("QR execution");
    (payload(&results[0]), payload(&results[1]))
}

fn sum_of_squares(values: &[Df64]) -> Df64 {
    values.iter().fold(Df64::zero(), |accumulator, value| {
        accumulator + *value * *value
    })
}

fn dot(lhs: &[Df64], rhs: &[Df64]) -> Df64 {
    assert_eq!(lhs.len(), rhs.len(), "dot product needs equal lengths");
    lhs.iter()
        .zip(rhs)
        .fold(Df64::zero(), |accumulator, (a, b)| accumulator + *a * *b)
}

fn relative_gap(lhs: Df64, rhs: Df64) -> f64 {
    let scale = lhs.abs_hi().max(rhs.abs_hi()).max(1.0);
    (lhs - rhs).abs_hi() / scale
}

#[test]
fn the_jvp_and_vjp_satisfy_the_duality_identity() {
    let (rows, columns) = (2, 1);
    let values = orientation_input();
    let tangent_values = vec![Df64::from_f64(1.0), Df64::from_f64(2.0)];
    let q_cotangent = vec![Df64::from_f64(0.5), Df64::from_f64(-0.25)];
    let r_cotangent = vec![Df64::from_f64(2.0)];

    let input = leaf(values.clone(), vec![rows, columns]);
    let outputs = apply(Arc::new(Df64Qr), &[&input]).expect("traced QR");

    // Forward mode, one factor at a time: the tangent of Q and of R for the same tangent of A.
    let tangent_leaf = leaf(tangent_values.clone(), vec![rows, columns]);
    let linearize = linearize_context();
    let q_dot = linearize
        .jvp(&outputs[0], &input, &tangent_leaf)
        .expect("traced JVP of the factor");
    let r_dot = linearize
        .jvp(&outputs[1], &input, &tangent_leaf)
        .expect("traced JVP of the triangular factor");

    // Reverse mode, one factor at a time. The adjoint is linear in the cotangent, so the two
    // results add up to the cotangent of the input.
    let q_leaf = leaf(q_cotangent.clone(), vec![rows, columns]);
    let r_leaf = leaf(r_cotangent.clone(), vec![columns, columns]);
    let cotangents = cotangent_context();
    let q_bar = cotangents
        .vjp(&outputs[0], &input, &q_leaf)
        .expect("traced VJP from the factor");
    let r_bar = cotangents
        .vjp(&outputs[1], &input, &r_leaf)
        .expect("traced VJP from the triangular factor");

    let runtime = runtime_with_module();
    let mut compiler = GraphCompiler::new();
    let q_dot_program = compiler.compile(&q_dot).expect("compiled factor tangent");
    let r_dot_program = compiler
        .compile(&r_dot)
        .expect("compiled triangular-factor tangent");
    let q_bar_program = compiler.compile(&q_bar).expect("compiled factor cotangent");
    let r_bar_program = compiler
        .compile(&r_bar)
        .expect("compiled triangular-factor cotangent");

    let input_value = external(values.clone(), vec![rows, columns]);
    let tangent_value = external(tangent_values.clone(), vec![rows, columns]);
    let q_dot_value = payload(
        &runtime
            .run_compiled(&q_dot_program, &[&input_value, &tangent_value])
            .expect("factor tangent execution")[0],
    );
    let r_dot_value = payload(
        &runtime
            .run_compiled(&r_dot_program, &[&input_value, &tangent_value])
            .expect("triangular-factor tangent execution")[0],
    );
    let q_bar_value = payload(
        &runtime
            .run_compiled(
                &q_bar_program,
                &[
                    &input_value,
                    &external(q_cotangent.clone(), vec![rows, columns]),
                ],
            )
            .expect("factor cotangent execution")[0],
    );
    let r_bar_value = payload(
        &runtime
            .run_compiled(
                &r_bar_program,
                &[
                    &input_value,
                    &external(r_cotangent.clone(), vec![columns, columns]),
                ],
            )
            .expect("triangular-factor cotangent execution")[0],
    );

    // sum(Q_dot * w_Q) + sum(R_dot * w_R) == sum(v * (A_bar_Q + A_bar_R))
    let forward_side = dot(&q_dot_value, &q_cotangent) + dot(&r_dot_value, &r_cotangent);
    let input_bar: Vec<Df64> = q_bar_value
        .iter()
        .zip(&r_bar_value)
        .map(|(q, r)| *q + *r)
        .collect();
    let backward_side = dot(&tangent_values, &input_bar);
    println!("Q_dot {q_dot_value:?}");
    println!("R_dot {r_dot_value:?}");
    println!("A_bar_Q {q_bar_value:?}");
    println!("A_bar_R {r_bar_value:?}");
    println!("input_bar {input_bar:?}");
    let gap = relative_gap(forward_side, backward_side);
    println!(
        "duality: forward side {forward_side:?}, backward side {backward_side:?}, \
         relative difference {gap:.3e}"
    );
    assert!(
        gap < 1e-20,
        "the JVP and VJP disagree on the duality identity: {forward_side:?} against \
         {backward_side:?}"
    );
}

#[test]
fn the_factor_gradient_matches_a_central_difference_of_the_loss() {
    // L(A) = sum(R * R), so dL/dR = 2R and the adjoint with that cotangent gives dL/dA, which is
    // [[6], [8]] for the orientation input.
    let (rows, columns) = (2, 1);
    let values = orientation_input();
    let direction = vec![Df64::from_f64(1.0), Df64::from_f64(-0.5)];
    // Small enough that an f64 intermediate could not resolve the change of the loss, large
    // enough that the extended scalar resolves it many digits above its own rounding.
    let step = Df64::from_f64(1e-16);

    let input = leaf(values.clone(), vec![rows, columns]);
    let outputs = apply(Arc::new(Df64Qr), &[&input]).expect("traced QR");
    let (_, r) = factorize(&values, rows, columns);
    let r_cotangent: Vec<Df64> = r.iter().map(|value| *value * Df64::from_f64(2.0)).collect();

    let r_leaf = leaf(r_cotangent.clone(), vec![columns, columns]);
    let gradient = cotangent_context()
        .vjp(&outputs[1], &input, &r_leaf)
        .expect("traced VJP of the loss");

    let runtime = runtime_with_module();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile(&gradient)
        .expect("compiled input cotangent");
    let gradient_values = payload(
        &runtime
            .run_compiled(
                &program,
                &[
                    &external(values.clone(), vec![rows, columns]),
                    &external(r_cotangent.clone(), vec![columns, columns]),
                ],
            )
            .expect("input cotangent execution")[0],
    );

    println!("gradient {gradient_values:?}");
    // The orientation case is the guard that the inputs were bound as the program expects. The
    // comparison is on the represented value rather than exact equality, because the gradient
    // carries a low component at the level of the scalar's own rounding.
    let expected = [Df64::from_f64(6.0), Df64::from_f64(8.0)];
    for (index, (value, wanted)) in gradient_values.iter().zip(&expected).enumerate() {
        let gap = relative_gap(*value, *wanted);
        assert!(
            gap < 1e-30,
            "the gradient of the orientation case is not [[6], [8]] at entry {index}: {value:?}              against {wanted:?}"
        );
    }

    // Central difference of the loss along the direction the cotangent is contracted with.
    let forward_values: Vec<Df64> = values
        .iter()
        .zip(&direction)
        .map(|(value, offset)| *value + step * *offset)
        .collect();
    let backward_values: Vec<Df64> = values
        .iter()
        .zip(&direction)
        .map(|(value, offset)| *value - step * *offset)
        .collect();
    let (_, r_forward) = factorize(&forward_values, rows, columns);
    let (_, r_backward) = factorize(&backward_values, rows, columns);
    let finite_difference =
        (sum_of_squares(&r_forward) - sum_of_squares(&r_backward)) / (Df64::from_f64(2.0) * step);

    let analytic = dot(&gradient_values, &direction);
    let gap = relative_gap(analytic, finite_difference);
    println!(
        "central difference: analytic {analytic:?}, finite difference {finite_difference:?}, \
         relative difference {gap:.3e}"
    );
    assert!(
        gap < 1e-20,
        "the analytic gradient disagrees with the central difference: {analytic:?} against \
         {finite_difference:?}"
    );
}
