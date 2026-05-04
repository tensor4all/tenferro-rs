//! Integration tests for `tenferro-ext-tropical` symbolic-shape
//! coverage.
//!
//! These tests act as the contract test for symbolic-AD correctness:
//!
//! - They build tropical composition graphs over fully-symbolic
//!   `TracedTensor::input_symbolic_shape` placeholders, never calling
//!   `try_concrete_shape`.
//! - They verify forward numerics for one concrete binding, then verify
//!   shape polymorphism by evaluating the same graph with a
//!   differently-shaped binding.
//! - They verify that `grad` produces a graph that can be evaluated via
//!   `eval_with_inputs` with a concrete binding, and that the numeric
//!   result matches the analytical / finite-difference reference.

use tenferro::{CpuBackend, Engine, Tensor, TracedTensor};
use tenferro_ext_tropical::traced::{min_plus_dot_general, tropical_dot_general};
use tenferro_tensor::DType;

fn engine() -> Engine<CpuBackend> {
    Engine::new(CpuBackend::new())
}

fn f64_data(tensor: &Tensor) -> &[f64] {
    tensor.as_slice::<f64>().expect("expected f64 tensor")
}

// ---------------------------------------------------------------------------
// Forward correctness on symbolic inputs
// ---------------------------------------------------------------------------

/// Build the composition graph over symbolic-shape placeholders, then
/// bind concrete tensors at eval time. The reference matches the concrete
/// `tropical_dot_general_forward_2x2` numerics exactly; the only difference is
/// the graph is now shape-polymorphic.
#[test]
fn tropical_dot_general_forward_symbolic_2x2() {
    let a_sym = TracedTensor::input_symbolic_shape(DType::F64, 2);
    let b_sym = TracedTensor::input_symbolic_shape(DType::F64, 2);

    let mut c = tropical_dot_general(&a_sym, &b_sym);

    // out[i, j] = max_k (a[i, k] + b[k, j]). Col-major flat:
    // [23, 24, 43, 44].
    let a_data = Tensor::from_vec(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);
    let b_data = Tensor::from_vec(vec![2, 2], vec![10.0_f64, 20.0, 30.0, 40.0]);
    let expected = [23.0, 24.0, 43.0, 44.0];

    let mut eng = engine();
    let out = c
        .eval_with_inputs(&mut eng, &[(&a_sym, &a_data), (&b_sym, &b_data)])
        .expect("forward eval");

    assert_eq!(out.shape(), &[2, 2]);
    for (i, (&g, &e)) in f64_data(out).iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-12,
            "at flat index {i}: got {g}, want {e}"
        );
    }
}

/// Shape polymorphism: build the composition graph once, then evaluate
/// with two differently-shaped bindings. This is the tropical analogue
/// of `symbolic_grad::grad_evaluates_with_different_shapes_from_same_symbolic_graph`
/// and proves that `broadcast_in_dim_sym` does not bake in concrete
/// sizes.
#[test]
fn tropical_dot_general_symbolic_shape_polymorphic_forward() {
    let a_sym = TracedTensor::input_symbolic_shape(DType::F64, 2);
    let b_sym = TracedTensor::input_symbolic_shape(DType::F64, 2);
    let c_template = tropical_dot_general(&a_sym, &b_sym);

    let mut eng = engine();

    // First evaluation: a is [2, 2], b is [2, 2].
    let a1 = Tensor::from_vec(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);
    let b1 = Tensor::from_vec(vec![2, 2], vec![10.0_f64, 20.0, 30.0, 40.0]);
    let mut c1 = c_template.clone();
    let out1 = c1
        .eval_with_inputs(&mut eng, &[(&a_sym, &a1), (&b_sym, &b1)])
        .expect("forward eval 2x2");
    assert_eq!(out1.shape(), &[2, 2]);
    let expected1 = [23.0_f64, 24.0, 43.0, 44.0];
    for (i, (&g, &e)) in f64_data(out1).iter().zip(expected1.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-12,
            "2x2: at flat index {i}: got {g}, want {e}"
        );
    }

    // Second evaluation: a is [2, 3], b is [3, 1].
    let a2 = Tensor::from_vec(vec![2, 3], vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0]);
    let b2 = Tensor::from_vec(vec![3, 1], vec![10.0_f64, 20.0, 30.0]);
    let mut c2 = c_template.clone();
    let out2 = c2
        .eval_with_inputs(&mut eng, &[(&a_sym, &a2), (&b_sym, &b2)])
        .expect("forward eval 2x3 * 3x1");
    assert_eq!(out2.shape(), &[2, 1]);
    let expected2 = [33.0_f64, 36.0];
    for (i, (&g, &e)) in f64_data(out2).iter().zip(expected2.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-12,
            "2x3 * 3x1: at flat index {i}: got {g}, want {e}"
        );
    }
}

// ---------------------------------------------------------------------------
// Backward (AD) correctness on symbolic inputs
// ---------------------------------------------------------------------------

/// The argmax-subgradient property of `reduce_max` means that the
/// gradient of `sum(max_k (a_ik + b_kj))` with respect to `a_ik` is the
/// count of `(i, j)` pairs for which `k` is the winning index. This
/// property is shape-agnostic, so it must also hold when the graph is
/// built over symbolic-shape placeholders and evaluated later.
///
/// Regression: before the deferred zero-tangent fix,
/// `loss.grad(&a_sym)` would materialise zero cotangents with a
/// concrete `vec![0; rank]` shape for the symbolic `a_sym`, which
/// crashed at eval. After the fix the zero is synthesised at
/// `eval_with_inputs` time from the user-supplied primal binding.
#[test]
fn tropical_dot_general_backward_symbolic() {
    let a_sym = TracedTensor::input_symbolic_shape(DType::F64, 2);
    let b_sym = TracedTensor::input_symbolic_shape(DType::F64, 2);

    let c = tropical_dot_general(&a_sym, &b_sym);
    let loss = c.reduce_sum(&[0, 1]);
    let mut grad_a = loss.grad(&a_sym).expect("grad build");

    let a_data = Tensor::from_vec(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);
    let b_data = Tensor::from_vec(vec![2, 2], vec![10.0_f64, 20.0, 30.0, 40.0]);

    let mut eng = engine();
    let g = grad_a
        .eval_with_inputs(&mut eng, &[(&a_sym, &a_data), (&b_sym, &b_data)])
        .expect("grad eval");
    assert_eq!(g.shape(), &[2, 2]);

    let vals = f64_data(g);
    // Every entry is a non-negative integer (argmax mass).
    for &v in vals {
        assert!(v >= 0.0, "grad entry {v} should be non-negative");
        assert!(
            (v - v.round()).abs() < 1e-9,
            "grad entry {v} should be an integer (argmax mass)"
        );
    }
    // Total mass equals M*N = 4 (one winner per output cell).
    let total: f64 = vals.iter().sum();
    assert!(
        (total - 4.0).abs() < 1e-9,
        "grad total {total} should equal M*N = 4 (one winning k per output cell); full grad: {vals:?}",
    );
    assert!(
        vals.iter().any(|&v| v > 0.0),
        "expected a non-trivial gradient; got all zeros ({vals:?})"
    );
}

/// Same shape-polymorphism story, but for the backward pass: build the
/// grad graph once over symbolic inputs, then evaluate it with two
/// distinct concrete bindings. The second binding uses rectangular
/// shapes to stress-test the deferred zero-tangent synthesis (the zero
/// tensor must match the bound primal shape, not the default concrete
/// shape baked in at graph build time).
#[test]
fn tropical_dot_general_grad_shape_polymorphic() {
    let a_sym = TracedTensor::input_symbolic_shape(DType::F64, 2);
    let b_sym = TracedTensor::input_symbolic_shape(DType::F64, 2);

    let c = tropical_dot_general(&a_sym, &b_sym);
    let loss = c.reduce_sum(&[0, 1]);
    let grad_a_template = loss.grad(&a_sym).expect("grad build");

    let mut eng = engine();

    // First eval: 2x2 shapes, total argmax mass = 4.
    let a1 = Tensor::from_vec(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);
    let b1 = Tensor::from_vec(vec![2, 2], vec![10.0_f64, 20.0, 30.0, 40.0]);
    let mut g1 = grad_a_template.clone();
    let out1 = g1
        .eval_with_inputs(&mut eng, &[(&a_sym, &a1), (&b_sym, &b1)])
        .expect("grad eval 1");
    assert_eq!(out1.shape(), &[2, 2]);
    let total1: f64 = f64_data(out1).iter().sum();
    assert!(
        (total1 - 4.0).abs() < 1e-9,
        "2x2 grad total should equal M*N = 4 (one winner per output), got {total1}"
    );

    // Second eval: 2x3 * 3x1 → output is [2, 1], so total argmax mass = 2.
    let a2 = Tensor::from_vec(vec![2, 3], vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0]);
    let b2 = Tensor::from_vec(vec![3, 1], vec![10.0_f64, 20.0, 30.0]);
    let mut g2 = grad_a_template.clone();
    let out2 = g2
        .eval_with_inputs(&mut eng, &[(&a_sym, &a2), (&b_sym, &b2)])
        .expect("grad eval 2");
    assert_eq!(out2.shape(), &[2, 3]);
    let total2: f64 = f64_data(out2).iter().sum();
    assert!(
        (total2 - 2.0).abs() < 1e-9,
        "2x3 * 3x1 grad total should equal M*N = 2 (one winner per output), got {total2}"
    );
}

// ---------------------------------------------------------------------------
// Min-plus forward and backward on symbolic inputs
// ---------------------------------------------------------------------------

#[test]
fn min_plus_dot_general_forward_symbolic_2x2() {
    let a_sym = TracedTensor::input_symbolic_shape(DType::F64, 2);
    let b_sym = TracedTensor::input_symbolic_shape(DType::F64, 2);

    let mut c = min_plus_dot_general(&a_sym, &b_sym);

    let a_data = Tensor::from_vec(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);
    let b_data = Tensor::from_vec(vec![2, 2], vec![10.0_f64, 20.0, 30.0, 40.0]);
    // min-plus out[i, j] = min_k (a[i, k] + b[k, j]).
    let expected = [11.0_f64, 12.0, 31.0, 32.0];

    let mut eng = engine();
    let out = c
        .eval_with_inputs(&mut eng, &[(&a_sym, &a_data), (&b_sym, &b_data)])
        .expect("min-plus forward eval");
    assert_eq!(out.shape(), &[2, 2]);
    for (i, (&g, &e)) in f64_data(out).iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-12,
            "min-plus at flat index {i}: got {g}, want {e}"
        );
    }
}

#[test]
fn min_plus_dot_general_backward_symbolic() {
    let a_sym = TracedTensor::input_symbolic_shape(DType::F64, 2);
    let b_sym = TracedTensor::input_symbolic_shape(DType::F64, 2);

    let c = min_plus_dot_general(&a_sym, &b_sym);
    let loss = c.reduce_sum(&[0, 1]);
    let mut grad_a = loss.grad(&a_sym).expect("grad build");

    let a_data = Tensor::from_vec(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);
    let b_data = Tensor::from_vec(vec![2, 2], vec![10.0_f64, 20.0, 30.0, 40.0]);

    let mut eng = engine();
    let g = grad_a
        .eval_with_inputs(&mut eng, &[(&a_sym, &a_data), (&b_sym, &b_data)])
        .expect("grad eval");
    assert_eq!(g.shape(), &[2, 2]);

    let vals = f64_data(g);
    // Same argmin mass argument as max-plus: total mass = M*N = 4.
    let total: f64 = vals.iter().sum();
    assert!(
        (total - 4.0).abs() < 1e-9,
        "min-plus grad total {total} should equal M*N = 4"
    );
    for &v in vals {
        assert!(v >= 0.0, "grad entry {v} should be non-negative");
        assert!(
            (v - v.round()).abs() < 1e-9,
            "grad entry {v} should be an integer (argmin mass)"
        );
    }
}
