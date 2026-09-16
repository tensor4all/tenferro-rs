//! The adjoint of the contraction, with independent references.
//!
//! #1793 asks for canonical first-order AD for the contraction, and #1788's requirement that the
//! derivative be checked against something independent applies here too: re-running the same kernel
//! is not evidence. So this file checks the adjoint against hand-written products for the matrix
//! case, against the extended scalar's precision for a contraction whose cotangent carries a low
//! component, and against a central difference of a scalar loss. The forward tangent stays refused,
//! which the last test asserts rather than assumes.

use std::sync::Arc;

use tenferro_ad::semantic_extension::SemanticExtensionRuleSet;
use tenferro_ad::AdContext;
use tenferro_cpu::CpuBackend;
use tenferro_df64_proof::ad::{Df64LinearizeRule, Df64VjpRule};
use tenferro_df64_proof::extension::{module, Df64Einsum, DF64_SCALAR_IDENTITY};
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

fn numbers(values: &[f64]) -> Vec<Df64> {
    values.iter().copied().map(Df64::from_f64).collect()
}

/// Contract `lhs` with `rhs` through the runtime and return the result's elements.
fn contract(
    lhs: Vec<Df64>,
    lhs_shape: [usize; 2],
    rhs: Vec<Df64>,
    rhs_shape: [usize; 2],
) -> Vec<Df64> {
    let op = Df64Einsum::new(&[0, 1], &[1, 2], &[0, 2]).expect("a contraction");
    let lhs_leaf = leaf(lhs.clone(), lhs_shape.to_vec());
    let rhs_leaf = leaf(rhs.clone(), rhs_shape.to_vec());
    let output = apply(Arc::new(op), &[&lhs_leaf, &rhs_leaf]).expect("traced contraction");
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&output[0]).expect("compiled contraction");
    let results = runtime_with_module()
        .run_compiled(
            &program,
            &[
                &external(lhs, lhs_shape.to_vec()),
                &external(rhs, rhs_shape.to_vec()),
            ],
        )
        .expect("contraction execution");
    payload(&results[0])
}

/// The cotangent of `wrt` for a contraction with the given cotangent on its output.
fn adjoint(
    lhs: Vec<Df64>,
    lhs_shape: [usize; 2],
    rhs: Vec<Df64>,
    rhs_shape: [usize; 2],
    cotangent: Vec<Df64>,
    cotangent_shape: [usize; 2],
    wrt_lhs: bool,
) -> Vec<Df64> {
    let op = Df64Einsum::new(&[0, 1], &[1, 2], &[0, 2]).expect("a contraction");
    let lhs_leaf = leaf(lhs.clone(), lhs_shape.to_vec());
    let rhs_leaf = leaf(rhs.clone(), rhs_shape.to_vec());
    let output = apply(Arc::new(op), &[&lhs_leaf, &rhs_leaf]).expect("traced contraction");
    let cotangent_leaf = leaf(cotangent.clone(), cotangent_shape.to_vec());
    let wrt = if wrt_lhs { &lhs_leaf } else { &rhs_leaf };
    let gradient = cotangent_context()
        .vjp(&output[0], wrt, &cotangent_leaf)
        .expect("traced adjoint");
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&gradient).expect("compiled adjoint");
    let results = runtime_with_module()
        .run_compiled(
            &program,
            &[
                &external(lhs, lhs_shape.to_vec()),
                &external(rhs, rhs_shape.to_vec()),
                &external(cotangent, cotangent_shape.to_vec()),
            ],
        )
        .expect("adjoint execution");
    payload(&results[0])
}

#[test]
fn the_adjoint_matches_the_hand_written_products() {
    // A = [[1, 2], [3, 4]] in column-major order, B = [[5, 6], [7, 8]], cotangent [[1, 0], [0, 1]].
    let a = numbers(&[1.0, 3.0, 2.0, 4.0]);
    let b = numbers(&[5.0, 7.0, 6.0, 8.0]);
    let cotangent = numbers(&[1.0, 0.0, 0.0, 1.0]);

    // A_bar = C_bar * B^T = [[5, 7], [6, 8]] and B_bar = A^T * C_bar = [[1, 2], [3, 4]].
    assert_eq!(
        adjoint(a.clone(), [2, 2], b.clone(), [2, 2], cotangent.clone(), [2, 2], true),
        numbers(&[5.0, 6.0, 7.0, 8.0]),
        "the adjoint of the first operand must be the cotangent times the other operand's transpose"
    );
    // B_bar = A^T * C_bar = [[1, 3], [2, 4]], which is [1, 2, 3, 4] in column-major order.
    assert_eq!(
        adjoint(a, [2, 2], b, [2, 2], cotangent, [2, 2], false),
        numbers(&[1.0, 2.0, 3.0, 4.0]),
        "the adjoint of the second operand must be the first operand's transpose times the cotangent"
    );
}

#[test]
fn the_adjoint_keeps_the_low_component_in_the_cotangent() {
    let low = 2f64.powi(-80);
    // A = [[1], [1]] (2x1) and B = [[1, 2^-80]] (1x2) contract to a 2x2 result. The adjoint of A
    // sums B's entries for each cotangent row, so every entry of A's cotangent is 1 + 2^-80: an
    // f64 accumulator would return 1.0 and drop the low component.
    let results = adjoint(
        numbers(&[1.0, 1.0]),
        [2, 1],
        vec![Df64::from_f64(1.0), Df64 { hi: low, lo: 0.0 }],
        [1, 2],
        vec![Df64::from_f64(1.0); 4],
        [2, 2],
        true,
    );

    let expected = Df64 { hi: 1.0, lo: low };
    assert_eq!(results.len(), 2);
    assert_eq!(
        results[0], expected,
        "the first cotangent entry lost the low component"
    );
    assert_eq!(
        results[1], expected,
        "the second cotangent entry lost the low component"
    );
    assert_eq!(
        (results[0] - Df64::from_f64(1.0)).narrow_to_f64(),
        low,
        "subtracting one must retain 2^-80 in the cotangent"
    );
    // Control: the same accumulation in f64 cannot carry the low component at all.
    assert_eq!(1.0_f64 + low, 1.0);
}

#[test]
fn the_adjoint_matches_a_central_difference_of_the_loss() {
    // L(A) = sum((A * B)^2), so dL/dA = 2 * (A * B) * B^T. The comparison is against a central
    // difference at a step small enough that an f64 intermediate could not resolve the change,
    // which is the independent reference rather than a second evaluation of the same kernel.
    let a = numbers(&[0.5, -0.25, 0.75, 0.125]);
    let b = numbers(&[2.0, 0.5, -1.0, 1.5]);
    let direction = numbers(&[1.0, 0.0, 0.0, 1.0]);
    // The step is chosen against both errors: the central difference truncates at h^2, and dividing
    // by 2h amplifies the loss's own rounding, which is why 1e-16 would leave only about 1e-16 of
    // relative agreement. At 1e-12 both are far below the tolerance, which an f64 loss could not
    // reach at any step because its rounding is 1e-16 of the loss rather than of the difference.
    let step = Df64::from_f64(1e-12);

    let product = contract(a.clone(), [2, 2], b.clone(), [2, 2]);
    // dL/dC = 2C, which is the cotangent the adjoint takes.
    let cotangent: Vec<Df64> = product
        .iter()
        .map(|value| *value * Df64::from_f64(2.0))
        .collect();
    let analytic = {
        let gradient = adjoint(
            a.clone(),
            [2, 2],
            b.clone(),
            [2, 2],
            cotangent,
            [2, 2],
            true,
        );
        gradient
            .iter()
            .zip(&direction)
            .fold(Df64::zero(), |sum, (g, d)| sum + *g * *d)
    };

    let loss = |values: Vec<Df64>| -> Df64 {
        contract(values, [2, 2], b.clone(), [2, 2])
            .iter()
            .fold(Df64::zero(), |sum, value| sum + *value * *value)
    };
    let forward: Vec<Df64> = a
        .iter()
        .zip(&direction)
        .map(|(value, offset)| *value + step * *offset)
        .collect();
    let backward: Vec<Df64> = a
        .iter()
        .zip(&direction)
        .map(|(value, offset)| *value - step * *offset)
        .collect();
    let finite_difference = (loss(forward) - loss(backward)) / (Df64::from_f64(2.0) * step);

    let gap = (analytic - finite_difference).abs_hi()
        / analytic.abs_hi().max(finite_difference.abs_hi()).max(1.0);
    println!(
        "contraction adjoint: analytic {analytic:?}, central difference {finite_difference:?}, \
         relative difference {gap:.3e}"
    );
    assert!(
        gap < 1e-20,
        "the adjoint disagrees with the central difference: {analytic:?} against {finite_difference:?}"
    );
}

#[test]
fn the_forward_tangent_is_still_refused() {
    let rules = SemanticExtensionRuleSet::new()
        .with_linearize(Arc::new(Df64LinearizeRule))
        .expect("one linearize rule per family");
    let context = AdContext::builder()
        .with_semantic_extension_rules(rules)
        .expect("extension rules")
        .build()
        .expect("ad context");

    let op = Df64Einsum::new(&[0, 1], &[1, 2], &[0, 2]).expect("a contraction");
    let lhs = leaf(numbers(&[1.0, 1.0]), vec![1, 2]);
    let rhs = leaf(numbers(&[1.0, 1.0]), vec![2, 1]);
    let output = apply(Arc::new(op), &[&lhs, &rhs]).expect("traced contraction");
    let tangent = leaf(numbers(&[1.0, 1.0]), vec![1, 2]);

    let error = context
        .jvp(&output[0], &lhs, &tangent)
        .expect_err("the forward tangent of the contraction is not implemented");
    assert!(
        error.to_string().contains("Linearize") || error.to_string().contains("unsupported"),
        "the refusal must name the missing rule: {error}"
    );
}
