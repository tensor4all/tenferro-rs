//! The derivative rules' own boundaries.
//!
//! #1789 and the design's rule both require an explicit typed refusal when a rule is asked
//! about something outside its domain, and the supported combinations have to keep working.
//! These tests drive the public AD entry points rather than the rules directly, because a
//! request is built by the transform itself.

use std::sync::Arc;

use tenferro_ad::semantic_extension::SemanticExtensionRuleSet;
use tenferro_ad::AdContext;
use tenferro_df64_proof::ad::{Df64LinearizeRule, Df64VjpRule};
use tenferro_df64_proof::extension::{
    Df64Expand, Df64Qr, Df64QrJvp, Df64QrVjp, DF64_SCALAR_IDENTITY,
};
use tenferro_df64_proof::Df64;
use tenferro_runtime::extension::apply;
use tenferro_runtime::TracedTensor;
use tenferro_tensor::{DType, Tensor};
use tenferro_tensor_core::{ErasedHostTensor, HostTensor};

fn external(values: Vec<Df64>, shape: Vec<usize>) -> Tensor {
    Tensor::external(ErasedHostTensor::new(
        HostTensor::from_vec_col_major(shape, values).expect("shape matches data"),
    ))
}

fn leaf(values: Vec<Df64>, shape: Vec<usize>) -> TracedTensor {
    TracedTensor::from_tensor_concrete_shape_declaring_scalar(
        external(values, shape),
        DF64_SCALAR_IDENTITY,
    )
    .expect("traced leaf")
}

fn external_dtype() -> DType {
    DType::External(std::any::TypeId::of::<Df64>())
}

fn vjp_context() -> AdContext {
    let rules = SemanticExtensionRuleSet::new()
        .with_primal_vjp(Arc::new(Df64VjpRule))
        .expect("one rule per family");
    AdContext::builder()
        .with_semantic_extension_rules(rules)
        .expect("extension rules")
        .build()
        .expect("ad context")
}

fn jvp_context() -> AdContext {
    let rules = SemanticExtensionRuleSet::new()
        .with_linearize(Arc::new(Df64LinearizeRule))
        .expect("one linearize rule per family");
    AdContext::builder()
        .with_semantic_extension_rules(rules)
        .expect("extension rules")
        .build()
        .expect("ad context")
}

fn matrix(order: usize, value: f64) -> TracedTensor {
    leaf(
        vec![Df64::from_f64(value); order * order],
        vec![order, order],
    )
}

/// A rule asked about an operation it does not differentiate refuses instead of guessing.
#[test]
fn the_vjp_rule_refuses_an_operation_outside_its_domain() {
    let ad = vjp_context();
    let identity = matrix(1, 0.0);
    let seed = leaf(vec![Df64::from_f64(1.0)], vec![1, 1]);

    // The broadcast is the total sum's adjoint, and the two derivative operations are
    // the factorization's own bodies: none of them is a primal operation to differentiate.
    let expanded = apply(Arc::new(Df64Expand::new(vec![1, 1])), &[&identity])
        .expect("traced broadcast")
        .remove(0);
    assert!(
        ad.vjp_many(&expanded, &[&identity], &seed).is_err(),
        "the broadcast has no adjoint"
    );

    let factor = matrix(1, 1.0);
    let triangular = matrix(1, 2.0);
    let adjoint = apply(
        Arc::new(Df64QrVjp::of(false, true)),
        &[&factor, &triangular, &seed],
    )
    .expect("traced adjoint")
    .remove(0);
    assert!(
        ad.vjp_many(&adjoint, &[&factor], &seed).is_err(),
        "the adjoint is not a primal operation"
    );

    let tangent = apply(Arc::new(Df64QrJvp), &[&factor, &triangular, &seed])
        .expect("traced tangent")
        .remove(0);
    assert!(
        ad.vjp_many(&tangent, &[&factor], &seed).is_err(),
        "the tangent is not a primal operation"
    );
}

/// A loss that depends on the factor `Q` alone differentiates through the factorization.
///
/// The loss's cotangent reaches the factorization as the factor's alone, so the adjoint has
/// to work with one cotangent present and the other absent.
#[test]
fn the_factorization_adjoint_accepts_the_factor_cotangent_alone() {
    let ad = vjp_context();

    let source = leaf(vec![Df64::from_f64(3.0), Df64::from_f64(4.0)], vec![2, 1]);
    let factors = apply(Arc::new(Df64Qr), &[&source]).expect("traced QR");
    // A loss over `Q` alone: the triangular factor's cotangent is the absent one.
    let narrowed = apply(
        Arc::new(tenferro_df64_proof::extension::Df64ToF64),
        &[&factors[0]],
    )
    .expect("traced narrowing")
    .remove(0);
    let loss = narrowed.mul(&narrowed).expect("ordinary f64 loss");

    let seed = TracedTensor::from_tensor_concrete_shape(
        Tensor::from_vec_col_major(vec![2, 1], vec![1.0_f64, 0.0]).expect("shape matches data"),
    )
    .expect("traced seed");
    let gradients = ad
        .vjp_many(&loss, &[&source], &seed)
        .expect("the factor's cotangent alone is accepted");
    assert_eq!(gradients.len(), 1);
    assert!(gradients[0].is_some(), "the source is active");
}

/// The linearization rule refuses the operations that are not primal.
#[test]
fn the_linearize_rule_refuses_an_operation_outside_its_domain() {
    let ad = jvp_context();
    let factor = matrix(1, 1.0);
    let tangent = matrix(1, 1.0);

    let expanded = apply(Arc::new(Df64Expand::new(vec![1, 1])), &[&factor])
        .expect("traced broadcast")
        .remove(0);
    assert!(
        ad.jvp(&expanded, &factor, &tangent).is_err(),
        "the broadcast has no linearization rule"
    );
}

/// The conversions and the factorization linearize through their own operations.
#[test]
fn the_conversions_and_the_factorization_linearize() {
    let ad = jvp_context();

    // The widening conversion: the tangent passes through the same conversion.
    let source = leaf(vec![Df64::from_f64(3.0), Df64::from_f64(4.0)], vec![2, 1]);
    let narrowed = apply(
        Arc::new(tenferro_df64_proof::extension::Df64ToF64),
        &[&source],
    )
    .expect("traced narrowing")
    .remove(0);
    let tangent = leaf(vec![Df64::from_f64(1.0), Df64::from_f64(0.0)], vec![2, 1]);
    let forward = ad
        .jvp(&narrowed, &source, &tangent)
        .expect("the narrowing linearizes");
    assert_eq!(forward.dtype(), DType::F64);

    // The factorization: the tangent of the two factors comes from their own operation.
    let matrix_input = matrix(2, 2.0);
    let factors = apply(Arc::new(Df64Qr), &[&matrix_input]).expect("traced QR");
    let factor_tangent = matrix(2, 1.0);
    let factor_forward = ad
        .jvp(&factors[1], &matrix_input, &factor_tangent)
        .expect("the factorization linearizes");
    assert_eq!(factor_forward.dtype(), external_dtype());
}
