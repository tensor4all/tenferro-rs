//! Integration tests for the fused tropical `ExtensionOp`.
//!
//! These tests are the extension contract self-test: they exercise the full
//! path from `register_extension` through `TracedTensor` graph building,
//! primal execution, and both linearize- and transpose-based AD. They also
//! cross-check the fused path against the composition-based wrapper to confirm
//! the forward primal and AD gradient agree (modulo tie-breaking).

use std::sync::Arc;

use tenferro::extension::{register_extension, ExtensionFactory, ExtensionRegistryError};
use tenferro::{CpuBackend, Engine, Tensor, TracedTensor};
use tenferro_ext_tropical::fused::{
    register_fused_tropical, FusedTropicalDotGeneralFactory, FusedTropicalDotGeneralOp,
    TropicalKind, FUSED_TROPICAL_DOT_GENERAL_FAMILY_ID,
};
use tenferro_ext_tropical::traced::{
    min_plus_dot_general, min_plus_dot_general_fused, tropical_dot_general,
    tropical_dot_general_fused,
};
use tenferro_ops::ext_op::ExtensionOp;
use tenferro_tensor::DType;

fn engine() -> Engine<CpuBackend> {
    Engine::new(CpuBackend::new())
}

fn f64_data(tensor: &Tensor) -> &[f64] {
    tensor.as_slice::<f64>().expect("expected f64 tensor")
}

fn col_major(data: Vec<f64>, shape: Vec<usize>) -> TracedTensor {
    TracedTensor::from_vec(shape, data)
}

fn assert_f64_close(got: &[f64], expected: &[f64], label: &str) {
    assert_eq!(
        got.len(),
        expected.len(),
        "{label}: length mismatch got={} expected={}",
        got.len(),
        expected.len()
    );
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-9,
            "{label}: at flat index {i}: got {g}, want {e}"
        );
    }
}

// ---------------------------------------------------------------------------
// Forward numerics: fused vs composition must agree exactly for a case
// with no ties (every (i, j) has a unique argmax k).
// ---------------------------------------------------------------------------

#[test]
fn fused_forward_matches_composition_max_plus_2x2() {
    // a = [[1, 3], [2, 4]]  (col-major [1, 2, 3, 4])
    // b = [[10, 30], [20, 40]]  (col-major [10, 20, 30, 40])
    let a = col_major(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = col_major(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);

    let mut c_fused = tropical_dot_general_fused(&a, &b);
    let mut c_comp = tropical_dot_general(&a, &b);

    let mut eng = engine();
    let out_fused = c_fused.eval(&mut eng).unwrap().clone();
    let out_comp = c_comp.eval(&mut eng).unwrap().clone();

    assert_eq!(out_fused.shape(), &[2, 2]);
    assert_eq!(out_comp.shape(), &[2, 2]);
    // Hand-computed reference (see tropical_ad.rs composition test):
    // out = [23, 24, 43, 44] in col-major order.
    let expected = [23.0, 24.0, 43.0, 44.0];
    assert_f64_close(f64_data(&out_fused), &expected, "fused");
    assert_f64_close(f64_data(&out_comp), &expected, "composition");
}

#[test]
fn fused_forward_matches_composition_non_square() {
    // a: [2, 3] col-major, b: [3, 1] col-major.
    let a = col_major(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]);
    let b = col_major(vec![10.0, 20.0, 30.0], vec![3, 1]);

    let mut c_fused = tropical_dot_general_fused(&a, &b);
    let mut c_comp = tropical_dot_general(&a, &b);

    let mut eng = engine();
    let out_fused = c_fused.eval(&mut eng).unwrap().clone();
    let out_comp = c_comp.eval(&mut eng).unwrap().clone();

    assert_eq!(out_fused.shape(), &[2, 1]);
    let expected = [33.0, 36.0];
    assert_f64_close(f64_data(&out_fused), &expected, "fused non-square");
    assert_f64_close(f64_data(&out_comp), &expected, "composition non-square");
}

#[test]
fn fused_forward_matches_composition_min_plus_2x2() {
    let a = col_major(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = col_major(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);

    let mut c_fused = min_plus_dot_general_fused(&a, &b);
    let mut c_comp = min_plus_dot_general(&a, &b);

    let mut eng = engine();
    let out_fused = c_fused.eval(&mut eng).unwrap().clone();
    let out_comp = c_comp.eval(&mut eng).unwrap().clone();

    // min-plus: out = [11, 12, 31, 32].
    let expected = [11.0, 12.0, 31.0, 32.0];
    assert_f64_close(f64_data(&out_fused), &expected, "fused min-plus");
    assert_f64_close(f64_data(&out_comp), &expected, "composition min-plus");
}

// ---------------------------------------------------------------------------
// Backward (AD) numerics: the fused path's `transpose_rule` must produce
// the same gradient as the composition path's AD — the "same math"
// invariant.
// ---------------------------------------------------------------------------

#[test]
fn fused_backward_matches_composition_gradient_max_plus() {
    let a = col_major(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = col_major(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);

    // Composition grad
    let c_comp = tropical_dot_general(&a, &b);
    let loss_comp = c_comp.reduce_sum(&[0, 1]);
    let mut g_comp_a = loss_comp.grad(&a).expect("comp grad a");
    let mut g_comp_b = loss_comp.grad(&b).expect("comp grad b");

    // Fused grad
    let c_fused = tropical_dot_general_fused(&a, &b);
    let loss_fused = c_fused.reduce_sum(&[0, 1]);
    let mut g_fused_a = loss_fused.grad(&a).expect("fused grad a");
    let mut g_fused_b = loss_fused.grad(&b).expect("fused grad b");

    let mut eng = engine();
    let comp_a = g_comp_a.eval(&mut eng).unwrap().clone();
    let comp_b = g_comp_b.eval(&mut eng).unwrap().clone();
    let fused_a = g_fused_a.eval(&mut eng).unwrap().clone();
    let fused_b = g_fused_b.eval(&mut eng).unwrap().clone();

    assert_eq!(fused_a.shape(), comp_a.shape());
    assert_eq!(fused_b.shape(), comp_b.shape());
    assert_f64_close(f64_data(&fused_a), f64_data(&comp_a), "grad a");
    assert_f64_close(f64_data(&fused_b), f64_data(&comp_b), "grad b");

    // Sanity: total grad mass matches M*N (no ties in this numerics).
    let total_a: f64 = f64_data(&fused_a).iter().sum();
    let total_b: f64 = f64_data(&fused_b).iter().sum();
    assert!((total_a - 4.0).abs() < 1e-9, "grad a mass {total_a}");
    assert!((total_b - 4.0).abs() < 1e-9, "grad b mass {total_b}");
}

#[test]
fn fused_backward_matches_composition_gradient_min_plus() {
    let a = col_major(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = col_major(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);

    let c_comp = min_plus_dot_general(&a, &b);
    let loss_comp = c_comp.reduce_sum(&[0, 1]);
    let mut g_comp_a = loss_comp.grad(&a).expect("comp grad a");

    let c_fused = min_plus_dot_general_fused(&a, &b);
    let loss_fused = c_fused.reduce_sum(&[0, 1]);
    let mut g_fused_a = loss_fused.grad(&a).expect("fused grad a");

    let mut eng = engine();
    let comp_a = g_comp_a.eval(&mut eng).unwrap().clone();
    let fused_a = g_fused_a.eval(&mut eng).unwrap().clone();
    assert_f64_close(f64_data(&fused_a), f64_data(&comp_a), "min-plus grad a");
}

#[test]
fn fused_backward_only_through_a() {
    // loss only reaches a: y = sum(fused(a, b_const)).
    // We still build b as a Traced leaf (no constants special-case needed here),
    // but ask only for grad wrt a. This exercises the transpose_rule active_mask
    // pathway with [true, false].
    let a = col_major(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = col_major(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);

    let c_fused = tropical_dot_general_fused(&a, &b);
    let loss = c_fused.reduce_sum(&[0, 1]);
    let mut grad_a = loss.grad(&a).expect("grad a");

    let mut eng = engine();
    let ga = grad_a.eval(&mut eng).unwrap().clone();
    assert_eq!(ga.shape(), &[2, 2]);
    let vals = f64_data(&ga);
    let total: f64 = vals.iter().sum();
    assert!((total - 4.0).abs() < 1e-9, "grad total {total}");
}

// ---------------------------------------------------------------------------
// Symbolic-shape forward / backward: the fused op's `infer_output_meta`
// must be total over symbolic inputs, and the emitted AD subgraph must
// remain shape-polymorphic.
// ---------------------------------------------------------------------------

#[test]
fn fused_forward_symbolic_2x2() {
    let a_sym = TracedTensor::input_symbolic_shape(DType::F64, 2);
    let b_sym = TracedTensor::input_symbolic_shape(DType::F64, 2);

    let mut c = tropical_dot_general_fused(&a_sym, &b_sym);

    let a_data = Tensor::from_vec(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);
    let b_data = Tensor::from_vec(vec![2, 2], vec![10.0_f64, 20.0, 30.0, 40.0]);
    let expected = [23.0_f64, 24.0, 43.0, 44.0];

    let mut eng = engine();
    let out = c
        .eval_with_inputs(&mut eng, &[(&a_sym, &a_data), (&b_sym, &b_data)])
        .expect("forward");
    assert_eq!(out.shape(), &[2, 2]);
    assert_f64_close(f64_data(out), &expected, "fused symbolic forward");
}

#[test]
fn fused_backward_symbolic_2x2() {
    let a_sym = TracedTensor::input_symbolic_shape(DType::F64, 2);
    let b_sym = TracedTensor::input_symbolic_shape(DType::F64, 2);

    let c = tropical_dot_general_fused(&a_sym, &b_sym);
    let loss = c.reduce_sum(&[0, 1]);
    let mut grad_a = loss.grad(&a_sym).expect("grad a");
    let mut grad_b = loss.grad(&b_sym).expect("grad b");

    // Cross-check against composition.
    let c_comp = tropical_dot_general(&a_sym, &b_sym);
    let loss_comp = c_comp.reduce_sum(&[0, 1]);
    let mut grad_a_comp = loss_comp.grad(&a_sym).expect("comp grad a");
    let mut grad_b_comp = loss_comp.grad(&b_sym).expect("comp grad b");

    let a_data = Tensor::from_vec(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);
    let b_data = Tensor::from_vec(vec![2, 2], vec![10.0_f64, 20.0, 30.0, 40.0]);

    let mut eng = engine();
    let ga_fused = grad_a
        .eval_with_inputs(&mut eng, &[(&a_sym, &a_data), (&b_sym, &b_data)])
        .expect("grad a eval");
    let gb_fused = grad_b
        .eval_with_inputs(&mut eng, &[(&a_sym, &a_data), (&b_sym, &b_data)])
        .expect("grad b eval");
    let ga_comp = grad_a_comp
        .eval_with_inputs(&mut eng, &[(&a_sym, &a_data), (&b_sym, &b_data)])
        .expect("comp grad a eval");
    let gb_comp = grad_b_comp
        .eval_with_inputs(&mut eng, &[(&a_sym, &a_data), (&b_sym, &b_data)])
        .expect("comp grad b eval");

    assert_f64_close(f64_data(ga_fused), f64_data(ga_comp), "symbolic grad a");
    assert_f64_close(f64_data(gb_fused), f64_data(gb_comp), "symbolic grad b");
}

// ---------------------------------------------------------------------------
// Extension contract self-test: identity, registry, and failure modes.
// ---------------------------------------------------------------------------

#[test]
fn family_id_matches_spec_format() {
    // The spec requires `<crate>.<op>.v<major>`.
    assert_eq!(
        FUSED_TROPICAL_DOT_GENERAL_FAMILY_ID,
        "tenferro-ext-tropical.fused_dot_general.v1"
    );
}

#[test]
fn duplicate_factory_registration_is_rejected() {
    // Install the real factory first (tolerating any prior registration
    // from other parallel tests in the same binary).
    register_fused_tropical().expect("register");

    // Now try to register another factory with the same family_id — this
    // must fail with Duplicate, regardless of prior state.
    struct DupFactory;
    impl ExtensionFactory for DupFactory {
        fn family_id(&self) -> &'static str {
            FUSED_TROPICAL_DOT_GENERAL_FAMILY_ID
        }
        fn version(&self) -> u32 {
            1
        }
    }
    let err = register_extension(Arc::new(DupFactory)).expect_err("duplicate family_id must error");
    assert!(matches!(
        err,
        ExtensionRegistryError::Duplicate {
            family_id: "tenferro-ext-tropical.fused_dot_general.v1"
        }
    ));
}

#[test]
fn malformed_family_id_is_rejected() {
    struct BadFactory;
    impl ExtensionFactory for BadFactory {
        fn family_id(&self) -> &'static str {
            "no-version-suffix"
        }
        fn version(&self) -> u32 {
            1
        }
    }
    let err = register_extension(Arc::new(BadFactory)).expect_err("malformed family_id must error");
    assert!(matches!(
        err,
        ExtensionRegistryError::MalformedFamilyId {
            family_id: "no-version-suffix"
        }
    ));
}

#[test]
fn payload_eq_requires_kind_match() {
    let max =
        Arc::new(FusedTropicalDotGeneralOp::new(TropicalKind::MaxPlus)) as Arc<dyn ExtensionOp>;
    let max2 =
        Arc::new(FusedTropicalDotGeneralOp::new(TropicalKind::MaxPlus)) as Arc<dyn ExtensionOp>;
    let min =
        Arc::new(FusedTropicalDotGeneralOp::new(TropicalKind::MinPlus)) as Arc<dyn ExtensionOp>;

    // Same kind => payload_eq is true.
    assert!(max.payload_eq(max2.as_ref()));
    // Different kind => payload_eq is false.
    assert!(!max.payload_eq(min.as_ref()));
}

#[test]
fn factory_reports_version_and_family_id() {
    let f = FusedTropicalDotGeneralFactory;
    assert_eq!(f.family_id(), FUSED_TROPICAL_DOT_GENERAL_FAMILY_ID);
    assert_eq!(f.version(), 1);
}

#[test]
fn explicit_register_is_idempotent() {
    // Calling register_fused_tropical multiple times must never error —
    // this is what lets the fused traced wrappers call it on every build
    // without leaking registry state into user code.
    register_fused_tropical().expect("first");
    register_fused_tropical().expect("second");
    register_fused_tropical().expect("third");
}
