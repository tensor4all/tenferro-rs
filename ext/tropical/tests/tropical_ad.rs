//! Integration tests for `tenferro-ext-tropical` composition coverage.
//!
//! These tests:
//! - exercise the public `tenferro` facade only,
//! - verify the forward numerics of the concrete-shape tropical
//!   compositions, and
//! - verify that automatic differentiation flows through the core AD
//!   rules of the underlying primitives (so no tropical-specific AD rule
//!   is required).

use tenferro::{CpuBackend, Engine, TracedTensor};
use tenferro_ext_tropical::newtype::{MaxMul, MaxPlus, MinPlus};
use tenferro_ext_tropical::traced::{
    min_plus_dot_general, tropical_dot_general, tropical_reduce_sum,
};

fn engine() -> Engine<CpuBackend> {
    Engine::new(CpuBackend::new())
}

/// Column-major `f64` values: tensor stores data with the leftmost dim
/// varying fastest. We use this helper to build row-major-style test
/// matrices by building them in the transposed layout.
fn col_major(data: Vec<f64>, shape: Vec<usize>) -> TracedTensor {
    TracedTensor::from_vec(shape, data)
}

// ---------------------------------------------------------------------------
// Forward correctness for tropical (max-plus) matmul
// ---------------------------------------------------------------------------

#[test]
fn tropical_dot_general_forward_2x2() {
    // Col-major layout for `tenferro`: index [i, j] at offset i + j*rows.
    //
    //         j=0 j=1
    //   a = [ 1.0  3.0 ]
    //       [ 2.0  4.0 ]
    // stored as [1.0, 2.0, 3.0, 4.0] for shape [2, 2].
    let a = col_major(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    //         j=0  j=1
    //   b = [ 10.0  30.0 ]
    //       [ 20.0  40.0 ]
    let b = col_major(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);

    // out[i, j] = max_k (a[i, k] + b[k, j])
    //
    // out[0, 0] = max(a[0,0]+b[0,0], a[0,1]+b[1,0]) = max(1+10, 3+20) = 23
    // out[0, 1] = max(a[0,0]+b[0,1], a[0,1]+b[1,1]) = max(1+30, 3+40) = 43
    // out[1, 0] = max(a[1,0]+b[0,0], a[1,1]+b[1,0]) = max(2+10, 4+20) = 24
    // out[1, 1] = max(a[1,0]+b[0,1], a[1,1]+b[1,1]) = max(2+30, 4+40) = 44
    // Column-major flat: [out[0,0], out[1,0], out[0,1], out[1,1]].
    let expected = [23.0, 24.0, 43.0, 44.0];

    let mut c = tropical_dot_general(&a, &b);
    let mut eng = engine();
    let out = c.eval(&mut eng).unwrap();
    assert_eq!(out.shape(), &[2, 2]);
    let got = out.as_slice::<f64>().unwrap();
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-12,
            "at flat index {i}: got {g}, want {e}"
        );
    }
}

#[test]
fn tropical_dot_general_non_square() {
    // a: [2, 3], b: [3, 1]. out: [2, 1].
    // Column-major: a[i, k] at offset i + k * 2.
    //   a = [ 1.0  2.0  3.0 ]
    //       [ 4.0  5.0  6.0 ]
    let a = col_major(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]);
    //   b = [ 10.0 ]
    //       [ 20.0 ]
    //       [ 30.0 ]
    let b = col_major(vec![10.0, 20.0, 30.0], vec![3, 1]);
    // out[0, 0] = max(1+10, 2+20, 3+30) = 33
    // out[1, 0] = max(4+10, 5+20, 6+30) = 36
    let expected = [33.0, 36.0];

    let mut c = tropical_dot_general(&a, &b);
    let mut eng = engine();
    let out = c.eval(&mut eng).unwrap();
    assert_eq!(out.shape(), &[2, 1]);
    let got = out.as_slice::<f64>().unwrap();
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-12,
            "at flat index {i}: got {g}, want {e}"
        );
    }
}

// ---------------------------------------------------------------------------
// Forward correctness for min-plus matmul
// ---------------------------------------------------------------------------

#[test]
fn min_plus_dot_general_forward_2x2() {
    let a = col_major(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = col_major(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);

    // out[i, j] = min_k (a[i, k] + b[k, j])
    // out[0, 0] = min(1+10, 3+20) = 11
    // out[0, 1] = min(1+30, 3+40) = 31
    // out[1, 0] = min(2+10, 4+20) = 12
    // out[1, 1] = min(2+30, 4+40) = 32
    let expected = [11.0, 12.0, 31.0, 32.0];

    let mut c = min_plus_dot_general(&a, &b);
    let mut eng = engine();
    let out = c.eval(&mut eng).unwrap();
    assert_eq!(out.shape(), &[2, 2]);
    let got = out.as_slice::<f64>().unwrap();
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-12,
            "at flat index {i}: got {g}, want {e}"
        );
    }
}

// ---------------------------------------------------------------------------
// tropical_reduce_sum wraps ReduceMax
// ---------------------------------------------------------------------------

#[test]
fn tropical_reduce_sum_takes_max() {
    let x = TracedTensor::from_vec(vec![4], vec![1.0_f64, 5.0, 2.0, 3.0]);
    let mut y = tropical_reduce_sum(&x, &[0]);
    let mut eng = engine();
    let out = y.eval(&mut eng).unwrap();
    assert_eq!(out.shape(), &[] as &[usize]);
    assert_eq!(out.as_slice::<f64>().unwrap(), &[5.0]);
}

// ---------------------------------------------------------------------------
// Backward (AD) — composition-based derivative flows through core rules
// ---------------------------------------------------------------------------

#[test]
fn tropical_dot_general_backward_flows_through_core_ad() {
    // Build a scalar loss = sum(tropical_dot_general(a, b)) and ask for
    // d loss / d a.  The backward must succeed and return a tensor whose
    // shape matches `a` — composition AD via BroadcastInDim, Add, and
    // ReduceMax should Just Work.
    let a = col_major(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = col_major(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);

    let c = tropical_dot_general(&a, &b);
    let loss = c.reduce_sum(&[0, 1]);
    let mut grad_a = loss.grad(&a).expect("grad should succeed");

    let mut eng = engine();
    let g = grad_a.eval(&mut eng).unwrap();
    assert_eq!(g.shape(), &[2, 2]);

    // The gradient of sum(max_k (a_ik + b_kj)) w.r.t. a_ik is the count of
    // (i, j) pairs for which k was the winning (arg-max) index. Because
    // every winning contribution flows through ReduceMax's subgradient
    // (which puts mass 1 on the argmax and 0 elsewhere), every entry in
    // the gradient should be a non-negative integer, and the total must
    // equal the number of (i, j) output cells, i.e. M*N = 4.
    let vals = g.as_slice::<f64>().unwrap();
    let total: f64 = vals.iter().sum();
    assert!(
        (total - 4.0).abs() < 1e-9,
        "grad total {total} should equal M*N = 4 (one winning k per output cell); full grad: {vals:?}",
    );
    // Each entry should be a non-negative integer (0, 1, 2, 3, or 4).
    for &v in vals {
        assert!(v >= 0.0, "grad entry {v} should be non-negative");
        assert!(
            (v - v.round()).abs() < 1e-9,
            "grad entry {v} should be an integer (argmax mass)"
        );
    }

    // At least one entry must be strictly positive — otherwise AD did not
    // propagate through the composition at all.
    let positive_entries = vals.iter().filter(|&&v| v > 0.0).count();
    assert!(
        positive_entries > 0,
        "expected a non-trivial gradient; got all zeros ({vals:?})"
    );
}

// ---------------------------------------------------------------------------
// Scalar newtype round-trip (not via TypedTensor — see newtype module docs
// for why `TypedTensor<MaxPlus<T>>` is not reachable through the public
// facade).
// ---------------------------------------------------------------------------

#[test]
fn scalar_newtype_round_trip() {
    // MaxPlus arithmetic
    assert_eq!(MaxPlus(3.0_f64) + MaxPlus(5.0_f64), MaxPlus(5.0_f64));
    assert_eq!(MaxPlus(3.0_f64) * MaxPlus(5.0_f64), MaxPlus(8.0_f64));
    let max_plus_zero: MaxPlus<f64> = MaxPlus::default();
    assert_eq!(max_plus_zero + MaxPlus(4.0_f64), MaxPlus(4.0_f64));

    // MinPlus arithmetic
    assert_eq!(MinPlus(3.0_f64) + MinPlus(5.0_f64), MinPlus(3.0_f64));
    assert_eq!(MinPlus(3.0_f64) * MinPlus(5.0_f64), MinPlus(8.0_f64));
    let min_plus_zero: MinPlus<f64> = MinPlus::default();
    assert_eq!(min_plus_zero + MinPlus(4.0_f64), MinPlus(4.0_f64));

    // MaxMul arithmetic
    assert_eq!(MaxMul(0.3_f64) + MaxMul(0.5_f64), MaxMul(0.5_f64));
    assert_eq!(MaxMul(0.3_f64) * MaxMul(0.5_f64), MaxMul(0.15_f64));
    let max_mul_zero: MaxMul<f64> = MaxMul::default();
    assert_eq!(max_mul_zero + MaxMul(0.4_f64), MaxMul(0.4_f64));
}
