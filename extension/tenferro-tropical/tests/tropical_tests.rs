//! Tests for tenferro-tropical: scalar types, algebra, argmax, and TensorPrims.
//!
//! TDD approach: tests written before implementation.

use num_traits::{One, Zero};
use tenferro_tropical::{MaxMul, MaxPlus, MinPlus};

// ============================================================================
// Scalar arithmetic tests
// ============================================================================

#[test]
fn maxplus_add_is_max() {
    let a = MaxPlus(3.0_f64);
    let b = MaxPlus(5.0_f64);
    let c = a + b;
    assert_eq!(c.0, 5.0); // max(3, 5) = 5
}

#[test]
fn maxplus_add_commutative() {
    let a = MaxPlus(3.0_f64);
    let b = MaxPlus(5.0_f64);
    assert_eq!((a + b).0, (b + a).0);
}

#[test]
fn maxplus_add_associative() {
    let a = MaxPlus(1.0_f64);
    let b = MaxPlus(3.0_f64);
    let c = MaxPlus(2.0_f64);
    assert_eq!(((a + b) + c).0, (a + (b + c)).0);
}

#[test]
fn maxplus_mul_is_ordinary_add() {
    let a = MaxPlus(3.0_f64);
    let b = MaxPlus(5.0_f64);
    let d = a * b;
    assert_eq!(d.0, 8.0); // 3 + 5 = 8
}

#[test]
fn maxplus_mul_associative() {
    let a = MaxPlus(1.0_f64);
    let b = MaxPlus(3.0_f64);
    let c = MaxPlus(2.0_f64);
    assert_eq!(((a * b) * c).0, (a * (b * c)).0);
}

#[test]
fn maxplus_zero_is_neg_inf() {
    let z = MaxPlus::<f64>::zero();
    assert_eq!(z.0, f64::NEG_INFINITY);
    assert!(z.is_zero());
}

#[test]
fn maxplus_one_is_zero() {
    let o = MaxPlus::<f64>::one();
    assert_eq!(o.0, 0.0);
}

#[test]
fn maxplus_zero_is_additive_identity() {
    let a = MaxPlus(3.0_f64);
    let z = MaxPlus::<f64>::zero();
    assert_eq!((a + z).0, a.0); // max(3, -inf) = 3
    assert_eq!((z + a).0, a.0);
}

#[test]
fn maxplus_one_is_multiplicative_identity() {
    let a = MaxPlus(3.0_f64);
    let o = MaxPlus::<f64>::one();
    assert_eq!((a * o).0, a.0); // 3 + 0 = 3
    assert_eq!((o * a).0, a.0);
}

#[test]
fn maxplus_add_idempotent() {
    // a ⊕ a = max(a, a) = a
    let a = MaxPlus(3.0_f64);
    assert_eq!((a + a).0, a.0);
}

#[test]
fn maxplus_distributive() {
    // a ⊗ (b ⊕ c) = (a ⊗ b) ⊕ (a ⊗ c)
    // a + max(b, c) = max(a + b, a + c)
    let a = MaxPlus(2.0_f64);
    let b = MaxPlus(3.0_f64);
    let c = MaxPlus(5.0_f64);
    let lhs = a * (b + c); // 2 + max(3, 5) = 2 + 5 = 7
    let rhs = (a * b) + (a * c); // max(2+3, 2+5) = max(5, 7) = 7
    assert_eq!(lhs.0, rhs.0);
}

#[test]
fn maxplus_zero_annihilates_mul() {
    // zero ⊗ a = zero (absorbing element)
    // (-inf) + a = -inf
    let a = MaxPlus(3.0_f64);
    let z = MaxPlus::<f64>::zero();
    assert!((z * a).is_zero());
    assert!((a * z).is_zero());
}

#[test]
fn maxplus_display() {
    let a = MaxPlus(3.5_f64);
    assert_eq!(format!("{}", a), "MaxPlus(3.5)");
}

// ---------------------------------------------------------------------------
// MinPlus tests
// ---------------------------------------------------------------------------

#[test]
fn minplus_add_is_min() {
    let a = MinPlus(3.0_f64);
    let b = MinPlus(5.0_f64);
    let c = a + b;
    assert_eq!(c.0, 3.0); // min(3, 5) = 3
}

#[test]
fn minplus_add_commutative() {
    let a = MinPlus(3.0_f64);
    let b = MinPlus(5.0_f64);
    assert_eq!((a + b).0, (b + a).0);
}

#[test]
fn minplus_add_associative() {
    let a = MinPlus(1.0_f64);
    let b = MinPlus(3.0_f64);
    let c = MinPlus(2.0_f64);
    assert_eq!(((a + b) + c).0, (a + (b + c)).0);
}

#[test]
fn minplus_mul_is_ordinary_add() {
    let a = MinPlus(3.0_f64);
    let b = MinPlus(5.0_f64);
    let d = a * b;
    assert_eq!(d.0, 8.0); // 3 + 5 = 8
}

#[test]
fn minplus_zero_is_pos_inf() {
    let z = MinPlus::<f64>::zero();
    assert_eq!(z.0, f64::INFINITY);
    assert!(z.is_zero());
}

#[test]
fn minplus_one_is_zero() {
    let o = MinPlus::<f64>::one();
    assert_eq!(o.0, 0.0);
}

#[test]
fn minplus_zero_is_additive_identity() {
    let a = MinPlus(3.0_f64);
    let z = MinPlus::<f64>::zero();
    assert_eq!((a + z).0, a.0); // min(3, +inf) = 3
    assert_eq!((z + a).0, a.0);
}

#[test]
fn minplus_one_is_multiplicative_identity() {
    let a = MinPlus(3.0_f64);
    let o = MinPlus::<f64>::one();
    assert_eq!((a * o).0, a.0); // 3 + 0 = 3
    assert_eq!((o * a).0, a.0);
}

#[test]
fn minplus_add_idempotent() {
    let a = MinPlus(3.0_f64);
    assert_eq!((a + a).0, a.0);
}

#[test]
fn minplus_distributive() {
    // a ⊗ (b ⊕ c) = (a ⊗ b) ⊕ (a ⊗ c)
    // a + min(b, c) = min(a + b, a + c)
    let a = MinPlus(2.0_f64);
    let b = MinPlus(3.0_f64);
    let c = MinPlus(5.0_f64);
    let lhs = a * (b + c); // 2 + min(3, 5) = 2 + 3 = 5
    let rhs = (a * b) + (a * c); // min(2+3, 2+5) = min(5, 7) = 5
    assert_eq!(lhs.0, rhs.0);
}

#[test]
fn minplus_zero_annihilates_mul() {
    let a = MinPlus(3.0_f64);
    let z = MinPlus::<f64>::zero();
    assert!((z * a).is_zero()); // +inf + 3 = +inf
    assert!((a * z).is_zero());
}

#[test]
fn minplus_display() {
    let a = MinPlus(3.5_f64);
    assert_eq!(format!("{}", a), "MinPlus(3.5)");
}

// ---------------------------------------------------------------------------
// MaxMul tests
// ---------------------------------------------------------------------------

#[test]
fn maxmul_add_is_max() {
    let a = MaxMul(0.3_f64);
    let b = MaxMul(0.7_f64);
    let c = a + b;
    assert_eq!(c.0, 0.7); // max(0.3, 0.7) = 0.7
}

#[test]
fn maxmul_add_commutative() {
    let a = MaxMul(0.3_f64);
    let b = MaxMul(0.7_f64);
    assert_eq!((a + b).0, (b + a).0);
}

#[test]
fn maxmul_mul_is_ordinary_mul() {
    let a = MaxMul(0.3_f64);
    let b = MaxMul(0.7_f64);
    let d = a * b;
    assert!((d.0 - 0.21).abs() < 1e-15); // 0.3 * 0.7 = 0.21
}

#[test]
fn maxmul_zero_is_zero() {
    let z = MaxMul::<f64>::zero();
    assert_eq!(z.0, 0.0);
    assert!(z.is_zero());
}

#[test]
fn maxmul_one_is_one() {
    let o = MaxMul::<f64>::one();
    assert_eq!(o.0, 1.0);
}

#[test]
fn maxmul_zero_is_additive_identity() {
    let a = MaxMul(0.5_f64);
    let z = MaxMul::<f64>::zero();
    assert_eq!((a + z).0, a.0); // max(0.5, 0) = 0.5
    assert_eq!((z + a).0, a.0);
}

#[test]
fn maxmul_one_is_multiplicative_identity() {
    let a = MaxMul(0.5_f64);
    let o = MaxMul::<f64>::one();
    assert_eq!((a * o).0, a.0); // 0.5 * 1 = 0.5
    assert_eq!((o * a).0, a.0);
}

#[test]
fn maxmul_zero_annihilates_mul() {
    let a = MaxMul(0.5_f64);
    let z = MaxMul::<f64>::zero();
    assert!((z * a).is_zero()); // 0 * 0.5 = 0
    assert!((a * z).is_zero());
}

#[test]
fn maxmul_distributive() {
    // a * max(b, c) = max(a*b, a*c) for non-negative values
    let a = MaxMul(0.5_f64);
    let b = MaxMul(0.3_f64);
    let c = MaxMul(0.7_f64);
    let lhs = a * (b + c); // 0.5 * max(0.3, 0.7) = 0.5 * 0.7 = 0.35
    let rhs = (a * b) + (a * c); // max(0.5*0.3, 0.5*0.7) = max(0.15, 0.35) = 0.35
    assert!((lhs.0 - rhs.0).abs() < 1e-15);
}

#[test]
fn maxmul_display() {
    let a = MaxMul(0.5_f64);
    assert_eq!(format!("{}", a), "MaxMul(0.5)");
}

// ---------------------------------------------------------------------------
// f32 support
// ---------------------------------------------------------------------------

#[test]
fn maxplus_f32_arithmetic() {
    let a = MaxPlus(3.0_f32);
    let b = MaxPlus(5.0_f32);
    assert_eq!((a + b).0, 5.0_f32);
    assert_eq!((a * b).0, 8.0_f32);
    assert_eq!(MaxPlus::<f32>::zero().0, f32::NEG_INFINITY);
    assert_eq!(MaxPlus::<f32>::one().0, 0.0_f32);
}

#[test]
fn minplus_f32_arithmetic() {
    let a = MinPlus(3.0_f32);
    let b = MinPlus(5.0_f32);
    assert_eq!((a + b).0, 3.0_f32);
    assert_eq!((a * b).0, 8.0_f32);
    assert_eq!(MinPlus::<f32>::zero().0, f32::INFINITY);
    assert_eq!(MinPlus::<f32>::one().0, 0.0_f32);
}

#[test]
fn maxmul_f32_arithmetic() {
    let a = MaxMul(0.3_f32);
    let b = MaxMul(0.7_f32);
    assert_eq!((a + b).0, 0.7_f32);
    assert!((a * b).0 - 0.21_f32 < 1e-6);
    assert_eq!(MaxMul::<f32>::zero().0, 0.0_f32);
    assert_eq!(MaxMul::<f32>::one().0, 1.0_f32);
}

// ============================================================================
// HasAlgebra mapping tests
// ============================================================================

#[test]
fn has_algebra_maxplus_f64() {
    use tenferro_algebra::HasAlgebra;
    use tenferro_tropical::MaxPlusAlgebra;
    fn check<T: HasAlgebra<Algebra = MaxPlusAlgebra>>() {}
    check::<MaxPlus<f64>>();
}

#[test]
fn has_algebra_maxplus_f32() {
    use tenferro_algebra::HasAlgebra;
    use tenferro_tropical::MaxPlusAlgebra;
    fn check<T: HasAlgebra<Algebra = MaxPlusAlgebra>>() {}
    check::<MaxPlus<f32>>();
}

#[test]
fn has_algebra_minplus_f64() {
    use tenferro_algebra::HasAlgebra;
    use tenferro_tropical::MinPlusAlgebra;
    fn check<T: HasAlgebra<Algebra = MinPlusAlgebra>>() {}
    check::<MinPlus<f64>>();
}

#[test]
fn has_algebra_maxmul_f64() {
    use tenferro_algebra::HasAlgebra;
    use tenferro_tropical::MaxMulAlgebra;
    fn check<T: HasAlgebra<Algebra = MaxMulAlgebra>>() {}
    check::<MaxMul<f64>>();
}

// ============================================================================
// Semiring tests
// ============================================================================

#[test]
fn semiring_maxplus() {
    use tenferro_algebra::Semiring;
    use tenferro_tropical::MaxPlusAlgebra;

    let z = MaxPlusAlgebra::zero();
    let o = MaxPlusAlgebra::one();
    assert_eq!(z.0, f64::NEG_INFINITY);
    assert_eq!(o.0, 0.0);

    let a = MaxPlus(3.0_f64);
    let b = MaxPlus(5.0_f64);
    assert_eq!(MaxPlusAlgebra::add(a, b).0, 5.0); // max(3, 5) = 5
    assert_eq!(MaxPlusAlgebra::mul(a, b).0, 8.0); // 3 + 5 = 8
}

#[test]
fn semiring_minplus() {
    use tenferro_algebra::Semiring;
    use tenferro_tropical::MinPlusAlgebra;

    let z = MinPlusAlgebra::zero();
    let o = MinPlusAlgebra::one();
    assert_eq!(z.0, f64::INFINITY);
    assert_eq!(o.0, 0.0);

    let a = MinPlus(3.0_f64);
    let b = MinPlus(5.0_f64);
    assert_eq!(MinPlusAlgebra::add(a, b).0, 3.0); // min(3, 5) = 3
    assert_eq!(MinPlusAlgebra::mul(a, b).0, 8.0); // 3 + 5 = 8
}

#[test]
fn semiring_maxmul() {
    use tenferro_algebra::Semiring;
    use tenferro_tropical::MaxMulAlgebra;

    let z = MaxMulAlgebra::zero();
    let o = MaxMulAlgebra::one();
    assert_eq!(z.0, 0.0);
    assert_eq!(o.0, 1.0);

    let a = MaxMul(0.3_f64);
    let b = MaxMul(0.7_f64);
    assert_eq!(MaxMulAlgebra::add(a, b).0, 0.7); // max(0.3, 0.7) = 0.7
    assert!((MaxMulAlgebra::mul(a, b).0 - 0.21).abs() < 1e-15); // 0.3 * 0.7 = 0.21
}

// ============================================================================
// ScalarBase / Scalar trait satisfaction tests
// ============================================================================

#[test]
fn tropical_satisfies_scalar_base() {
    use strided_traits::ScalarBase;
    fn check<T: ScalarBase>() {}
    check::<MaxPlus<f64>>();
    check::<MaxPlus<f32>>();
    check::<MinPlus<f64>>();
    check::<MinPlus<f32>>();
    check::<MaxMul<f64>>();
    check::<MaxMul<f32>>();
}

#[test]
fn tropical_satisfies_scalar() {
    use tenferro_algebra::Scalar;
    fn check<T: Scalar>() {}
    check::<MaxPlus<f64>>();
    check::<MaxPlus<f32>>();
    check::<MinPlus<f64>>();
    check::<MinPlus<f32>>();
    check::<MaxMul<f64>>();
    check::<MaxMul<f32>>();
}

// ============================================================================
// ArgmaxTracker tests
// ============================================================================

#[test]
fn argmax_tracker_new() {
    use tenferro_tropical::ArgmaxTracker;
    let tracker = ArgmaxTracker::new(&[3, 5]);
    assert_eq!(tracker.output_shape(), &[3, 5]);
    assert_eq!(tracker.indices().len(), 15); // 3 * 5
    assert!(tracker.indices().iter().all(|&i| i == 0));
}

#[test]
fn argmax_tracker_winner_index() {
    use tenferro_tropical::ArgmaxTracker;
    let mut tracker = ArgmaxTracker::new(&[2, 3]);
    // Manually set some winner indices
    // Layout is column-major: (0,0), (1,0), (0,1), (1,1), (0,2), (1,2)
    tracker.indices_mut()[0] = 5; // winner for (0,0)
    tracker.indices_mut()[3] = 7; // winner for (1,1)
    assert_eq!(tracker.winner_index(&[0, 0]), 5);
    assert_eq!(tracker.winner_index(&[1, 1]), 7);
}

// ============================================================================
// Tropical matmul tests (hand-computed)
// ============================================================================

#[test]
fn maxplus_matmul_2x2() {
    use tenferro_prims::{CpuBackend, CpuContext, PrimDescriptor, TensorPrims};

    // A = [[1, 3],    B = [[0, 2],
    //      [2, 4]]         [1, 0]]
    //
    // MaxPlus C[i,j] = max_k(A[i,k] + B[k,j])
    // C[0,0] = max(1+0, 3+1) = max(1, 4) = 4
    // C[0,1] = max(1+2, 3+0) = max(3, 3) = 3
    // C[1,0] = max(2+0, 4+1) = max(2, 5) = 5
    // C[1,1] = max(2+2, 4+0) = max(4, 4) = 4
    let mut ctx = CpuContext::new(1);

    // Column-major: A stored as [1, 2, 3, 4]
    let a_data = [MaxPlus(1.0), MaxPlus(2.0), MaxPlus(3.0), MaxPlus(4.0)];
    // Column-major: B stored as [0, 1, 2, 0]
    let b_data = [MaxPlus(0.0), MaxPlus(1.0), MaxPlus(2.0), MaxPlus(0.0)];
    let mut c_data = [MaxPlus::<f64>::zero(); 4];

    let a_view = strided_view::StridedView::new(&a_data, &[2, 2], &[1, 2], 0).unwrap();
    let b_view = strided_view::StridedView::new(&b_data, &[2, 2], &[1, 2], 0).unwrap();
    let mut c_view = strided_view::StridedViewMut::new(&mut c_data, &[2, 2], &[1, 2], 0).unwrap();

    let desc = PrimDescriptor::BatchedGemm {
        batch_dims: vec![],
        m: 2,
        n: 2,
        k: 2,
    };
    let plan =
        <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::plan::<MaxPlus<f64>>(
            &mut ctx,
            &desc,
            &[&[2, 2], &[2, 2], &[2, 2]],
        )
        .unwrap();
    <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::execute(
        &mut ctx,
        &plan,
        MaxPlus::one(),
        &[&a_view, &b_view],
        MaxPlus::zero(),
        &mut c_view,
    )
    .unwrap();

    // Expected column-major: [4, 5, 3, 4]
    assert_eq!(c_data[0].0, 4.0); // C[0,0]
    assert_eq!(c_data[1].0, 5.0); // C[1,0]
    assert_eq!(c_data[2].0, 3.0); // C[0,1]
    assert_eq!(c_data[3].0, 4.0); // C[1,1]
}

#[test]
fn minplus_matmul_2x2() {
    use tenferro_prims::{CpuBackend, CpuContext, PrimDescriptor, TensorPrims};

    // A = [[1, 3],    B = [[5, 2],
    //      [2, 4]]         [1, 6]]
    //
    // MinPlus C[i,j] = min_k(A[i,k] + B[k,j])
    // C[0,0] = min(1+5, 3+1) = min(6, 4) = 4
    // C[0,1] = min(1+2, 3+6) = min(3, 9) = 3
    // C[1,0] = min(2+5, 4+1) = min(7, 5) = 5
    // C[1,1] = min(2+2, 4+6) = min(4, 10) = 4
    let mut ctx = CpuContext::new(1);

    let a_data = [MinPlus(1.0), MinPlus(2.0), MinPlus(3.0), MinPlus(4.0)];
    let b_data = [MinPlus(5.0), MinPlus(1.0), MinPlus(2.0), MinPlus(6.0)];
    let mut c_data = [MinPlus::<f64>::zero(); 4];

    let a_view = strided_view::StridedView::new(&a_data, &[2, 2], &[1, 2], 0).unwrap();
    let b_view = strided_view::StridedView::new(&b_data, &[2, 2], &[1, 2], 0).unwrap();
    let mut c_view = strided_view::StridedViewMut::new(&mut c_data, &[2, 2], &[1, 2], 0).unwrap();

    let desc = PrimDescriptor::BatchedGemm {
        batch_dims: vec![],
        m: 2,
        n: 2,
        k: 2,
    };
    let plan =
        <CpuBackend as TensorPrims<tenferro_tropical::MinPlusAlgebra>>::plan::<MinPlus<f64>>(
            &mut ctx,
            &desc,
            &[&[2, 2], &[2, 2], &[2, 2]],
        )
        .unwrap();
    <CpuBackend as TensorPrims<tenferro_tropical::MinPlusAlgebra>>::execute(
        &mut ctx,
        &plan,
        MinPlus::one(),
        &[&a_view, &b_view],
        MinPlus::zero(),
        &mut c_view,
    )
    .unwrap();

    assert_eq!(c_data[0].0, 4.0); // C[0,0]
    assert_eq!(c_data[1].0, 5.0); // C[1,0]
    assert_eq!(c_data[2].0, 3.0); // C[0,1]
    assert_eq!(c_data[3].0, 4.0); // C[1,1]
}

#[test]
fn maxmul_matmul_2x2() {
    use tenferro_prims::{CpuBackend, CpuContext, PrimDescriptor, TensorPrims};

    // A = [[0.5, 0.3],    B = [[0.8, 0.1],
    //      [0.2, 0.9]]         [0.4, 0.6]]
    //
    // MaxMul C[i,j] = max_k(A[i,k] * B[k,j])
    // C[0,0] = max(0.5*0.8, 0.3*0.4) = max(0.40, 0.12) = 0.40
    // C[0,1] = max(0.5*0.1, 0.3*0.6) = max(0.05, 0.18) = 0.18
    // C[1,0] = max(0.2*0.8, 0.9*0.4) = max(0.16, 0.36) = 0.36
    // C[1,1] = max(0.2*0.1, 0.9*0.6) = max(0.02, 0.54) = 0.54
    let mut ctx = CpuContext::new(1);

    let a_data = [MaxMul(0.5), MaxMul(0.2), MaxMul(0.3), MaxMul(0.9)];
    let b_data = [MaxMul(0.8), MaxMul(0.4), MaxMul(0.1), MaxMul(0.6)];
    let mut c_data = [MaxMul::<f64>::zero(); 4];

    let a_view = strided_view::StridedView::new(&a_data, &[2, 2], &[1, 2], 0).unwrap();
    let b_view = strided_view::StridedView::new(&b_data, &[2, 2], &[1, 2], 0).unwrap();
    let mut c_view = strided_view::StridedViewMut::new(&mut c_data, &[2, 2], &[1, 2], 0).unwrap();

    let desc = PrimDescriptor::BatchedGemm {
        batch_dims: vec![],
        m: 2,
        n: 2,
        k: 2,
    };
    let plan = <CpuBackend as TensorPrims<tenferro_tropical::MaxMulAlgebra>>::plan::<MaxMul<f64>>(
        &mut ctx,
        &desc,
        &[&[2, 2], &[2, 2], &[2, 2]],
    )
    .unwrap();
    <CpuBackend as TensorPrims<tenferro_tropical::MaxMulAlgebra>>::execute(
        &mut ctx,
        &plan,
        MaxMul::one(),
        &[&a_view, &b_view],
        MaxMul::zero(),
        &mut c_view,
    )
    .unwrap();

    assert!((c_data[0].0 - 0.40).abs() < 1e-15); // C[0,0]
    assert!((c_data[1].0 - 0.36).abs() < 1e-15); // C[1,0]
    assert!((c_data[2].0 - 0.18).abs() < 1e-15); // C[0,1]
    assert!((c_data[3].0 - 0.54).abs() < 1e-15); // C[1,1]
}

// ============================================================================
// Application pattern tests
// ============================================================================

#[test]
fn shortest_path_minplus_bellman_ford_step() {
    use tenferro_prims::{CpuBackend, CpuContext, PrimDescriptor, TensorPrims};

    // Graph adjacency matrix (edge weights, +inf = no edge):
    // W = [[ 0,  1, +inf],
    //      [+inf, 0,  3 ],
    //      [ 2, +inf, 0 ]]
    //
    // Distance vector d = [0, +inf, +inf] (source = node 0)
    //
    // After one MinPlus mat-vec (d' = W ⊗ d):
    // d'[0] = min(0+0, 1+inf, inf+inf) = 0
    // d'[1] = min(inf+0, 0+inf, 3+inf) = +inf... wait
    //
    // Actually, for Bellman-Ford: d' = d ⊕ (W^T ⊗ d)
    // Let's use a 3x3 matmul W * d_col where d_col is a 3x1 matrix:
    //
    // W^T (transpose to get min over incoming edges):
    // W^T = [[ 0, +inf,  2 ],
    //        [ 1,   0, +inf],
    //        [+inf, 3,   0 ]]
    //
    // W^T ⊗ d = min over cols:
    // result[0] = min(0+0, inf+inf, 2+inf) = 0
    // result[1] = min(1+0, 0+inf, inf+inf) = 1
    // result[2] = min(inf+0, 3+inf, 0+inf) = +inf
    //
    // After d' = d ⊕ result: d' = min(d, result) = [0, 1, +inf]
    //
    // Second iteration with d = [0, 1, +inf]:
    // W^T ⊗ d:
    // result[0] = min(0+0, inf+1, 2+inf) = 0
    // result[1] = min(1+0, 0+1, inf+inf) = 1
    // result[2] = min(inf+0, 3+1, 0+inf) = 4
    //
    // d' = [0, 1, 4] — correct shortest distances from node 0!

    let inf = f64::INFINITY;
    let mut ctx = CpuContext::new(1);

    // W^T in column-major (3x3)
    let wt_data: Vec<MinPlus<f64>> = vec![
        MinPlus(0.0),
        MinPlus(1.0),
        MinPlus(inf),
        MinPlus(inf),
        MinPlus(0.0),
        MinPlus(3.0),
        MinPlus(2.0),
        MinPlus(inf),
        MinPlus(0.0),
    ];

    // d as column vector (3x1)
    let d_data: Vec<MinPlus<f64>> = vec![MinPlus(0.0), MinPlus(inf), MinPlus(inf)];

    let mut result = vec![MinPlus::<f64>::zero(); 3];

    let wt_view = strided_view::StridedView::new(&wt_data, &[3, 3], &[1, 3], 0).unwrap();
    let d_view = strided_view::StridedView::new(&d_data, &[3, 1], &[1, 3], 0).unwrap();
    let mut r_view = strided_view::StridedViewMut::new(&mut result, &[3, 1], &[1, 3], 0).unwrap();

    let desc = PrimDescriptor::BatchedGemm {
        batch_dims: vec![],
        m: 3,
        n: 1,
        k: 3,
    };
    let plan =
        <CpuBackend as TensorPrims<tenferro_tropical::MinPlusAlgebra>>::plan::<MinPlus<f64>>(
            &mut ctx,
            &desc,
            &[&[3, 3], &[3, 1], &[3, 1]],
        )
        .unwrap();
    <CpuBackend as TensorPrims<tenferro_tropical::MinPlusAlgebra>>::execute(
        &mut ctx,
        &plan,
        MinPlus::one(),
        &[&wt_view, &d_view],
        MinPlus::zero(),
        &mut r_view,
    )
    .unwrap();

    // First iteration: d' = [0, 1, +inf]
    assert_eq!(result[0].0, 0.0);
    assert_eq!(result[1].0, 1.0);
    assert_eq!(result[2].0, inf);

    // Second iteration with d = result
    let d2_data = result.clone();
    let mut result2 = vec![MinPlus::<f64>::zero(); 3];
    let d2_view = strided_view::StridedView::new(&d2_data, &[3, 1], &[1, 3], 0).unwrap();
    let mut r2_view = strided_view::StridedViewMut::new(&mut result2, &[3, 1], &[1, 3], 0).unwrap();
    <CpuBackend as TensorPrims<tenferro_tropical::MinPlusAlgebra>>::execute(
        &mut ctx,
        &plan,
        MinPlus::one(),
        &[&wt_view, &d2_view],
        MinPlus::zero(),
        &mut r2_view,
    )
    .unwrap();

    // Second iteration: d' = [0, 1, 4]
    assert_eq!(result2[0].0, 0.0);
    assert_eq!(result2[1].0, 1.0);
    assert_eq!(result2[2].0, 4.0);
}

#[test]
fn viterbi_maxmul_hmm_step() {
    use tenferro_prims::{CpuBackend, CpuContext, PrimDescriptor, TensorPrims};

    // Transition matrix (2 states):
    // T = [[0.7, 0.3],
    //      [0.4, 0.6]]
    //
    // State probabilities: p = [0.5, 0.5]
    //
    // Viterbi step: p' = T^T ⊗ p (MaxMul matmul)
    // p'[0] = max(T[0,0]*p[0], T[1,0]*p[1]) = max(0.7*0.5, 0.4*0.5) = max(0.35, 0.2) = 0.35
    // p'[1] = max(T[0,1]*p[0], T[1,1]*p[1]) = max(0.3*0.5, 0.6*0.5) = max(0.15, 0.3) = 0.3
    let mut ctx = CpuContext::new(1);

    // T^T in column-major (2x2)
    let tt_data = [MaxMul(0.7), MaxMul(0.3), MaxMul(0.4), MaxMul(0.6)];
    let p_data = [MaxMul(0.5), MaxMul(0.5)];
    let mut result = [MaxMul::<f64>::zero(); 2];

    let tt_view = strided_view::StridedView::new(&tt_data, &[2, 2], &[1, 2], 0).unwrap();
    let p_view = strided_view::StridedView::new(&p_data, &[2, 1], &[1, 2], 0).unwrap();
    let mut r_view = strided_view::StridedViewMut::new(&mut result, &[2, 1], &[1, 2], 0).unwrap();

    let desc = PrimDescriptor::BatchedGemm {
        batch_dims: vec![],
        m: 2,
        n: 1,
        k: 2,
    };
    let plan = <CpuBackend as TensorPrims<tenferro_tropical::MaxMulAlgebra>>::plan::<MaxMul<f64>>(
        &mut ctx,
        &desc,
        &[&[2, 2], &[2, 1], &[2, 1]],
    )
    .unwrap();
    <CpuBackend as TensorPrims<tenferro_tropical::MaxMulAlgebra>>::execute(
        &mut ctx,
        &plan,
        MaxMul::one(),
        &[&tt_view, &p_view],
        MaxMul::zero(),
        &mut r_view,
    )
    .unwrap();

    assert!((result[0].0 - 0.35).abs() < 1e-15);
    assert!((result[1].0 - 0.3).abs() < 1e-15);
}

// ============================================================================
// Tropical reduce tests
// ============================================================================

#[test]
fn maxplus_reduce_sum() {
    use tenferro_prims::{CpuBackend, CpuContext, PrimDescriptor, ReduceOp, TensorPrims};

    // A = [[1, 3],    (2x2, col-major: [1, 2, 3, 4])
    //      [2, 4]]
    //
    // "Sum" over axis 1 (columns) under MaxPlus = max over columns:
    // C[0] = max(1, 3) = 3
    // C[1] = max(2, 4) = 4
    let mut ctx = CpuContext::new(1);

    let a_data = [MaxPlus(1.0), MaxPlus(2.0), MaxPlus(3.0), MaxPlus(4.0)];
    let mut c_data = [MaxPlus::<f64>::zero(); 2];

    let a_view = strided_view::StridedView::new(&a_data, &[2, 2], &[1, 2], 0).unwrap();
    let mut c_view = strided_view::StridedViewMut::new(&mut c_data, &[2], &[1], 0).unwrap();

    let desc = PrimDescriptor::Reduce {
        modes_a: vec![0, 1],
        modes_c: vec![0],
        op: ReduceOp::Sum,
    };
    let plan =
        <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::plan::<MaxPlus<f64>>(
            &mut ctx,
            &desc,
            &[&[2, 2], &[2]],
        )
        .unwrap();
    <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::execute(
        &mut ctx,
        &plan,
        MaxPlus::one(),
        &[&a_view],
        MaxPlus::zero(),
        &mut c_view,
    )
    .unwrap();

    assert_eq!(c_data[0].0, 3.0); // max(1, 3)
    assert_eq!(c_data[1].0, 4.0); // max(2, 4)
}

// ============================================================================
// Tropical permute test
// ============================================================================

#[test]
fn maxplus_permute_transpose() {
    use tenferro_prims::{CpuBackend, CpuContext, PrimDescriptor, TensorPrims};

    // A = [[1, 3],    (2x2, col-major: [1, 2, 3, 4])
    //      [2, 4]]
    // Transpose: A^T = [[1, 2], [3, 4]]  (col-major: [1, 3, 2, 4])
    let mut ctx = CpuContext::new(1);

    let a_data = [MaxPlus(1.0), MaxPlus(2.0), MaxPlus(3.0), MaxPlus(4.0)];
    let mut c_data = [MaxPlus::<f64>::zero(); 4];

    let a_view = strided_view::StridedView::new(&a_data, &[2, 2], &[1, 2], 0).unwrap();
    let mut c_view = strided_view::StridedViewMut::new(&mut c_data, &[2, 2], &[1, 2], 0).unwrap();

    let desc = PrimDescriptor::Permute {
        modes_a: vec![0, 1],
        modes_b: vec![1, 0],
    };
    let plan =
        <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::plan::<MaxPlus<f64>>(
            &mut ctx,
            &desc,
            &[&[2, 2], &[2, 2]],
        )
        .unwrap();
    <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::execute(
        &mut ctx,
        &plan,
        MaxPlus::one(),
        &[&a_view],
        MaxPlus::zero(),
        &mut c_view,
    )
    .unwrap();

    // Transposed col-major: [1, 3, 2, 4]
    assert_eq!(c_data[0].0, 1.0);
    assert_eq!(c_data[1].0, 3.0);
    assert_eq!(c_data[2].0, 2.0);
    assert_eq!(c_data[3].0, 4.0);
}

// ============================================================================
// Tropical trace execution test
// ============================================================================

#[test]
fn maxplus_trace_3x3() {
    use tenferro_prims::{CpuBackend, CpuContext, PrimDescriptor, TensorPrims};

    // A = [[10, 4, 7],   (3x3, column-major)
    //      [ 2, 8, 5],
    //      [ 6, 3, 1]]
    //
    // Trace = A[0,0] ⊕ A[1,1] ⊕ A[2,2] = max(10, 8, 1) = 10
    let mut ctx = CpuContext::new(1);

    // Column-major: data[i + 3*j]
    let a_data = [
        MaxPlus(10.0),
        MaxPlus(2.0),
        MaxPlus(6.0), // col 0
        MaxPlus(4.0),
        MaxPlus(8.0),
        MaxPlus(3.0), // col 1
        MaxPlus(7.0),
        MaxPlus(5.0),
        MaxPlus(1.0), // col 2
    ];
    let mut c_data = [MaxPlus::<f64>::zero()]; // scalar output

    let a_view = strided_view::StridedView::new(&a_data, &[3, 3], &[1, 3], 0).unwrap();
    let mut c_view = strided_view::StridedViewMut::new(&mut c_data, &[], &[], 0).unwrap();

    let desc = PrimDescriptor::Trace {
        modes_a: vec![0, 1],
        modes_c: vec![],
        paired: vec![(0, 1)],
    };
    let plan =
        <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::plan::<MaxPlus<f64>>(
            &mut ctx,
            &desc,
            &[&[3, 3], &[]],
        )
        .unwrap();
    <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::execute(
        &mut ctx,
        &plan,
        MaxPlus::one(),
        &[&a_view],
        MaxPlus::zero(),
        &mut c_view,
    )
    .unwrap();

    assert_eq!(c_data[0].0, 10.0); // max(10, 8, 1)
}

#[test]
fn maxplus_trace_partial_3d() {
    use tenferro_prims::{CpuBackend, CpuContext, PrimDescriptor, TensorPrims};

    // Partial trace: "iij->j" on a [2,2,3] tensor.
    // Result[j] = max_i(A[i,i,j]) = max of diagonal slices.
    //
    // A[0,0,:] = [1, 2, 3]
    // A[1,1,:] = [8, 5, 6]
    // Result = [max(1,8), max(2,5), max(3,6)] = [8, 5, 6]
    let mut ctx = CpuContext::new(1);

    // Shape [2,2,3], column-major: data[i + 2*j + 4*k]
    let a_data = [
        // k=0: [[1,?],[?,8]]
        MaxPlus(1.0),
        MaxPlus(10.0), // i=0,j=0; i=1,j=0
        MaxPlus(10.0),
        MaxPlus(8.0), // i=0,j=1; i=1,j=1
        // k=1: [[2,?],[?,5]]
        MaxPlus(2.0),
        MaxPlus(10.0),
        MaxPlus(10.0),
        MaxPlus(5.0),
        // k=2: [[3,?],[?,6]]
        MaxPlus(3.0),
        MaxPlus(10.0),
        MaxPlus(10.0),
        MaxPlus(6.0),
    ];
    let mut c_data = [MaxPlus::<f64>::zero(); 3];

    let a_view = strided_view::StridedView::new(&a_data, &[2, 2, 3], &[1, 2, 4], 0).unwrap();
    let mut c_view = strided_view::StridedViewMut::new(&mut c_data, &[3], &[1], 0).unwrap();

    let desc = PrimDescriptor::Trace {
        modes_a: vec![0, 1, 2],
        modes_c: vec![2],
        paired: vec![(0, 1)],
    };
    let plan =
        <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::plan::<MaxPlus<f64>>(
            &mut ctx,
            &desc,
            &[&[2, 2, 3], &[3]],
        )
        .unwrap();
    <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::execute(
        &mut ctx,
        &plan,
        MaxPlus::one(),
        &[&a_view],
        MaxPlus::zero(),
        &mut c_view,
    )
    .unwrap();

    // Off-diagonal values are 10, but those are not summed in trace
    assert_eq!(c_data[0].0, 8.0); // max(1, 8)
    assert_eq!(c_data[1].0, 5.0); // max(2, 5)
    assert_eq!(c_data[2].0, 6.0); // max(3, 6)
}

// ============================================================================
// Anti-trace: embed scalar/vector into diagonal of a matrix
// (AD backward of trace operation)
// ============================================================================

#[test]
fn maxplus_anti_trace_scalar_to_diag() {
    use tenferro_prims::{CpuBackend, CpuContext, PrimDescriptor, TensorPrims};

    // AntiTrace: scalar → 2x2 matrix with value on diagonal
    // input = MaxPlus(5.0) (scalar)
    // output[i,j] = alpha * input if i==j, else unchanged (beta * old)
    //
    // With alpha=1, beta=0:
    // output = [[5, -inf],
    //           [-inf, 5]]
    let mut ctx = CpuContext::new(1);

    let in_data = [MaxPlus(5.0)];
    let mut out_data = [MaxPlus::<f64>::zero(); 4]; // 2x2

    let in_view = strided_view::StridedView::new(&in_data, &[], &[], 0).unwrap();
    let mut out_view =
        strided_view::StridedViewMut::new(&mut out_data, &[2, 2], &[1, 2], 0).unwrap();

    let desc = PrimDescriptor::AntiTrace {
        modes_a: vec![],
        modes_c: vec![0, 1],
        paired: vec![(0, 1)],
    };
    let plan =
        <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::plan::<MaxPlus<f64>>(
            &mut ctx,
            &desc,
            &[&[], &[2, 2]],
        )
        .unwrap();
    <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::execute(
        &mut ctx,
        &plan,
        MaxPlus::one(),
        &[&in_view],
        MaxPlus::zero(),
        &mut out_view,
    )
    .unwrap();

    // Column-major: [out[0,0], out[1,0], out[0,1], out[1,1]]
    // Diagonal gets 5.0, off-diagonal stays at zero (-inf) + 5.0 = 5.0...
    // Actually anti-trace adds val to each diagonal position:
    // out[d,d] += alpha * input for d in 0..diag_dim
    // After scale_output with beta=0, all entries are -inf, then add 5.0 to diag
    assert_eq!(out_data[0].0, 5.0); // [0,0] diagonal
    assert_eq!(out_data[1].0, f64::NEG_INFINITY); // [1,0] off-diagonal
    assert_eq!(out_data[2].0, f64::NEG_INFINITY); // [0,1] off-diagonal
    assert_eq!(out_data[3].0, 5.0); // [1,1] diagonal
}

// ============================================================================
// f32 matmul tests (verify SIMD dispatch for f32 types)
// ============================================================================

#[test]
fn maxplus_matmul_2x2_f32() {
    use tenferro_prims::{CpuBackend, CpuContext, PrimDescriptor, TensorPrims};

    // Same as maxplus_matmul_2x2 but with f32 — verifies f32 SIMD dispatch.
    // A = [[1, 3],    B = [[0, 2],
    //      [2, 4]]         [1, 0]]
    // C[0,0] = max(1+0, 3+1) = 4
    // C[0,1] = max(1+2, 3+0) = 3
    // C[1,0] = max(2+0, 4+1) = 5
    // C[1,1] = max(2+2, 4+0) = 4
    let mut ctx = CpuContext::new(1);

    let a_data = [
        MaxPlus(1.0_f32),
        MaxPlus(2.0_f32),
        MaxPlus(3.0_f32),
        MaxPlus(4.0_f32),
    ];
    let b_data = [
        MaxPlus(0.0_f32),
        MaxPlus(1.0_f32),
        MaxPlus(2.0_f32),
        MaxPlus(0.0_f32),
    ];
    let mut c_data = [MaxPlus::<f32>::zero(); 4];

    let a_view = strided_view::StridedView::new(&a_data, &[2, 2], &[1, 2], 0).unwrap();
    let b_view = strided_view::StridedView::new(&b_data, &[2, 2], &[1, 2], 0).unwrap();
    let mut c_view = strided_view::StridedViewMut::new(&mut c_data, &[2, 2], &[1, 2], 0).unwrap();

    let desc = PrimDescriptor::BatchedGemm {
        batch_dims: vec![],
        m: 2,
        n: 2,
        k: 2,
    };
    let plan =
        <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::plan::<MaxPlus<f32>>(
            &mut ctx,
            &desc,
            &[&[2, 2], &[2, 2], &[2, 2]],
        )
        .unwrap();
    <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::execute(
        &mut ctx,
        &plan,
        MaxPlus::one(),
        &[&a_view, &b_view],
        MaxPlus::zero(),
        &mut c_view,
    )
    .unwrap();

    assert_eq!(c_data[0].0, 4.0_f32);
    assert_eq!(c_data[1].0, 5.0_f32);
    assert_eq!(c_data[2].0, 3.0_f32);
    assert_eq!(c_data[3].0, 4.0_f32);
}

#[test]
fn minplus_matmul_2x2_f32() {
    use tenferro_prims::{CpuBackend, CpuContext, PrimDescriptor, TensorPrims};

    // MinPlus f32 SIMD dispatch.
    // A = [[1, 3],    B = [[5, 2],
    //      [2, 4]]         [1, 6]]
    // C[0,0] = min(1+5, 3+1) = 4
    // C[0,1] = min(1+2, 3+6) = 3
    // C[1,0] = min(2+5, 4+1) = 5
    // C[1,1] = min(2+2, 4+6) = 4
    let mut ctx = CpuContext::new(1);

    let a_data = [
        MinPlus(1.0_f32),
        MinPlus(2.0_f32),
        MinPlus(3.0_f32),
        MinPlus(4.0_f32),
    ];
    let b_data = [
        MinPlus(5.0_f32),
        MinPlus(1.0_f32),
        MinPlus(2.0_f32),
        MinPlus(6.0_f32),
    ];
    let mut c_data = [MinPlus::<f32>::zero(); 4];

    let a_view = strided_view::StridedView::new(&a_data, &[2, 2], &[1, 2], 0).unwrap();
    let b_view = strided_view::StridedView::new(&b_data, &[2, 2], &[1, 2], 0).unwrap();
    let mut c_view = strided_view::StridedViewMut::new(&mut c_data, &[2, 2], &[1, 2], 0).unwrap();

    let desc = PrimDescriptor::BatchedGemm {
        batch_dims: vec![],
        m: 2,
        n: 2,
        k: 2,
    };
    let plan =
        <CpuBackend as TensorPrims<tenferro_tropical::MinPlusAlgebra>>::plan::<MinPlus<f32>>(
            &mut ctx,
            &desc,
            &[&[2, 2], &[2, 2], &[2, 2]],
        )
        .unwrap();
    <CpuBackend as TensorPrims<tenferro_tropical::MinPlusAlgebra>>::execute(
        &mut ctx,
        &plan,
        MinPlus::one(),
        &[&a_view, &b_view],
        MinPlus::zero(),
        &mut c_view,
    )
    .unwrap();

    assert_eq!(c_data[0].0, 4.0_f32);
    assert_eq!(c_data[1].0, 5.0_f32);
    assert_eq!(c_data[2].0, 3.0_f32);
    assert_eq!(c_data[3].0, 4.0_f32);
}

#[test]
fn maxmul_matmul_2x2_f32() {
    use tenferro_prims::{CpuBackend, CpuContext, PrimDescriptor, TensorPrims};

    // MaxMul f32 SIMD dispatch.
    // A = [[0.5, 0.3],    B = [[0.8, 0.1],
    //      [0.2, 0.9]]         [0.4, 0.6]]
    // C[0,0] = max(0.5*0.8, 0.3*0.4) = 0.40
    // C[0,1] = max(0.5*0.1, 0.3*0.6) = 0.18
    // C[1,0] = max(0.2*0.8, 0.9*0.4) = 0.36
    // C[1,1] = max(0.2*0.1, 0.9*0.6) = 0.54
    let mut ctx = CpuContext::new(1);

    let a_data = [
        MaxMul(0.5_f32),
        MaxMul(0.2_f32),
        MaxMul(0.3_f32),
        MaxMul(0.9_f32),
    ];
    let b_data = [
        MaxMul(0.8_f32),
        MaxMul(0.4_f32),
        MaxMul(0.1_f32),
        MaxMul(0.6_f32),
    ];
    let mut c_data = [MaxMul::<f32>::zero(); 4];

    let a_view = strided_view::StridedView::new(&a_data, &[2, 2], &[1, 2], 0).unwrap();
    let b_view = strided_view::StridedView::new(&b_data, &[2, 2], &[1, 2], 0).unwrap();
    let mut c_view = strided_view::StridedViewMut::new(&mut c_data, &[2, 2], &[1, 2], 0).unwrap();

    let desc = PrimDescriptor::BatchedGemm {
        batch_dims: vec![],
        m: 2,
        n: 2,
        k: 2,
    };
    let plan = <CpuBackend as TensorPrims<tenferro_tropical::MaxMulAlgebra>>::plan::<MaxMul<f32>>(
        &mut ctx,
        &desc,
        &[&[2, 2], &[2, 2], &[2, 2]],
    )
    .unwrap();
    <CpuBackend as TensorPrims<tenferro_tropical::MaxMulAlgebra>>::execute(
        &mut ctx,
        &plan,
        MaxMul::one(),
        &[&a_view, &b_view],
        MaxMul::zero(),
        &mut c_view,
    )
    .unwrap();

    assert!((c_data[0].0 - 0.40_f32).abs() < 1e-6);
    assert!((c_data[1].0 - 0.36_f32).abs() < 1e-6);
    assert!((c_data[2].0 - 0.18_f32).abs() < 1e-6);
    assert!((c_data[3].0 - 0.54_f32).abs() < 1e-6);
}

// ============================================================================
// Matmul accumulation test (C = alpha * A*B + beta * C_old)
// ============================================================================

#[test]
fn maxplus_matmul_accumulate_with_beta() {
    use tenferro_prims::{CpuBackend, CpuContext, PrimDescriptor, TensorPrims};

    // Tests accumulation: C_new = alpha * (A ⊗ B) ⊕ beta * C_old
    // For MaxPlus with alpha=2, beta=1:
    //   C_new[i,j] = max(2 + max_k(A[i,k] + B[k,j]), 1 + C_old[i,j])
    //
    // A = [[1, 3],   B = [[0, 2],
    //      [2, 4]]        [1, 0]]
    // A ⊗ B = [[4, 3], [5, 4]]  (same as maxplus_matmul_2x2)
    //
    // alpha=MaxPlus(2.0), so alpha * result = 2 + result:
    // alpha*(A⊗B) = [[6, 5], [7, 6]]
    //
    // C_old = [[10, 0], [0, 10]]
    // beta=MaxPlus(1.0), so beta * C_old = 1 + C_old:
    // beta*C_old = [[11, 1], [1, 11]]
    //
    // C_new = max(alpha*(A⊗B), beta*C_old):
    // C_new = [[max(6,11), max(5,1)], [max(7,1), max(6,11)]]
    //       = [[11, 5], [7, 11]]
    let mut ctx = CpuContext::new(1);

    let a_data = [MaxPlus(1.0), MaxPlus(2.0), MaxPlus(3.0), MaxPlus(4.0)];
    let b_data = [MaxPlus(0.0), MaxPlus(1.0), MaxPlus(2.0), MaxPlus(0.0)];
    let mut c_data = [MaxPlus(10.0), MaxPlus(0.0), MaxPlus(0.0), MaxPlus(10.0)]; // col-major

    let a_view = strided_view::StridedView::new(&a_data, &[2, 2], &[1, 2], 0).unwrap();
    let b_view = strided_view::StridedView::new(&b_data, &[2, 2], &[1, 2], 0).unwrap();
    let mut c_view = strided_view::StridedViewMut::new(&mut c_data, &[2, 2], &[1, 2], 0).unwrap();

    let desc = PrimDescriptor::BatchedGemm {
        batch_dims: vec![],
        m: 2,
        n: 2,
        k: 2,
    };
    let plan =
        <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::plan::<MaxPlus<f64>>(
            &mut ctx,
            &desc,
            &[&[2, 2], &[2, 2], &[2, 2]],
        )
        .unwrap();
    <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::execute(
        &mut ctx,
        &plan,
        MaxPlus(2.0), // alpha
        &[&a_view, &b_view],
        MaxPlus(1.0), // beta (non-zero: accumulate)
        &mut c_view,
    )
    .unwrap();

    // col-major: [C[0,0], C[1,0], C[0,1], C[1,1]]
    assert_eq!(c_data[0].0, 11.0); // max(6, 11) = 11
    assert_eq!(c_data[1].0, 7.0); // max(7, 1) = 7
    assert_eq!(c_data[2].0, 5.0); // max(5, 1) = 5
    assert_eq!(c_data[3].0, 11.0); // max(6, 11) = 11
}

// ============================================================================
// Anti-diag execution test (AD backward of off-diagonal extraction)
// ============================================================================

#[test]
fn maxplus_anti_diag_vector_to_matrix() {
    use tenferro_prims::{CpuBackend, CpuContext, PrimDescriptor, TensorPrims};

    // AntiDiag: input vector [a, b] → output matrix where column j gets a copy
    // of input[j] along the anti-diagonal pattern.
    //
    // With modes_a=[0], modes_c=[0,1], paired=[(0,1)]:
    // This maps v[i] → M[i,j] where j = i (diagonal embedding).
    // For input [5, 3]:
    // output = [[5, -inf],
    //           [-inf, 3]]
    let mut ctx = CpuContext::new(1);

    let in_data = [MaxPlus(5.0), MaxPlus(3.0)];
    let mut out_data = [MaxPlus::<f64>::zero(); 4]; // 2x2

    let in_view = strided_view::StridedView::new(&in_data, &[2], &[1], 0).unwrap();
    let mut out_view =
        strided_view::StridedViewMut::new(&mut out_data, &[2, 2], &[1, 2], 0).unwrap();

    let desc = PrimDescriptor::AntiDiag {
        modes_a: vec![0],
        modes_c: vec![0, 1],
        paired: vec![(0, 1)],
    };
    let plan =
        <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::plan::<MaxPlus<f64>>(
            &mut ctx,
            &desc,
            &[&[2], &[2, 2]],
        )
        .unwrap();
    <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::execute(
        &mut ctx,
        &plan,
        MaxPlus::one(),
        &[&in_view],
        MaxPlus::zero(),
        &mut out_view,
    )
    .unwrap();

    // col-major: [M[0,0], M[1,0], M[0,1], M[1,1]]
    assert_eq!(out_data[0].0, 5.0); // [0,0] = input[0]
    assert_eq!(out_data[1].0, f64::NEG_INFINITY); // [1,0] off-diag
    assert_eq!(out_data[2].0, f64::NEG_INFINITY); // [0,1] off-diag
    assert_eq!(out_data[3].0, 3.0); // [1,1] = input[1]
}

// ============================================================================
// MakeContiguous test
// ============================================================================

#[test]
fn maxplus_make_contiguous() {
    use tenferro_prims::{CpuBackend, CpuContext, PrimDescriptor, TensorPrims};

    // Test MakeContiguous: copy a strided (non-contiguous) view into a
    // contiguous buffer. Input is a transposed view of a 2x3 matrix.
    //
    // A = [[1, 2, 3],   stored col-major: [1, 4, 2, 5, 3, 6]
    //      [4, 5, 6]]
    //
    // Transposed view (3x2, strides=[2,1]):
    // A^T = [[1, 4],
    //        [2, 5],
    //        [3, 6]]
    //
    // MakeContiguous should copy this into a fresh [3,2] contiguous buffer.
    let mut ctx = CpuContext::new(1);

    let a_data = [
        MaxPlus(1.0),
        MaxPlus(4.0),
        MaxPlus(2.0),
        MaxPlus(5.0),
        MaxPlus(3.0),
        MaxPlus(6.0),
    ];
    let mut out_data = [MaxPlus::<f64>::zero(); 6];

    // Transposed view: shape [3,2], strides [2,1] — non-contiguous
    let a_view = strided_view::StridedView::new(&a_data, &[3, 2], &[2, 1], 0).unwrap();
    let mut out_view =
        strided_view::StridedViewMut::new(&mut out_data, &[3, 2], &[1, 3], 0).unwrap();

    let desc = PrimDescriptor::MakeContiguous;
    let plan =
        <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::plan::<MaxPlus<f64>>(
            &mut ctx,
            &desc,
            &[&[3, 2], &[3, 2]],
        )
        .unwrap();
    <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::execute(
        &mut ctx,
        &plan,
        MaxPlus::one(),
        &[&a_view],
        MaxPlus::zero(),
        &mut out_view,
    )
    .unwrap();

    // Output in col-major [3,2]: [1, 2, 3, 4, 5, 6]
    assert_eq!(out_data[0].0, 1.0);
    assert_eq!(out_data[1].0, 2.0);
    assert_eq!(out_data[2].0, 3.0);
    assert_eq!(out_data[3].0, 4.0);
    assert_eq!(out_data[4].0, 5.0);
    assert_eq!(out_data[5].0, 6.0);
}

// ============================================================================
// Anti-trace with free axes (backward of partial trace)
// ============================================================================

#[test]
fn maxplus_anti_trace_vec_to_3d() {
    use tenferro_prims::{CpuBackend, CpuContext, PrimDescriptor, TensorPrims};

    // AntiTrace with free axes: "j->iij" — backward of partial trace "iij->j".
    // Input: vector [a, b, c] (size 3, mode 2)
    // Output: tensor [2,2,3] where output[d,d,j] += alpha * input[j]
    //
    // With alpha=1, beta=0, diag_dim=2:
    // output[0,0,j] = input[j]
    // output[1,1,j] = input[j]
    // output[other] = -inf (MaxPlus zero)
    let mut ctx = CpuContext::new(1);

    let in_data = [MaxPlus(10.0), MaxPlus(20.0), MaxPlus(30.0)];
    let mut out_data = [MaxPlus::<f64>::zero(); 12]; // 2×2×3

    let in_view = strided_view::StridedView::new(&in_data, &[3], &[1], 0).unwrap();
    let mut out_view =
        strided_view::StridedViewMut::new(&mut out_data, &[2, 2, 3], &[1, 2, 4], 0).unwrap();

    let desc = PrimDescriptor::AntiTrace {
        modes_a: vec![2],       // free axis: mode 2 (k dimension)
        modes_c: vec![0, 1, 2], // output: modes 0, 1, 2
        paired: vec![(0, 1)],   // diagonal on modes 0 and 1
    };
    let plan =
        <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::plan::<MaxPlus<f64>>(
            &mut ctx,
            &desc,
            &[&[3], &[2, 2, 3]],
        )
        .unwrap();
    <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::execute(
        &mut ctx,
        &plan,
        MaxPlus::one(),
        &[&in_view],
        MaxPlus::zero(),
        &mut out_view,
    )
    .unwrap();

    // col-major [2,2,3], strides [1,2,4]:
    // index(i,j,k) = i + 2*j + 4*k
    let get = |i: usize, j: usize, k: usize| out_data[i + 2 * j + 4 * k].0;

    // Diagonal entries: output[d,d,k] = input[k]
    assert_eq!(get(0, 0, 0), 10.0);
    assert_eq!(get(0, 0, 1), 20.0);
    assert_eq!(get(0, 0, 2), 30.0);
    assert_eq!(get(1, 1, 0), 10.0);
    assert_eq!(get(1, 1, 1), 20.0);
    assert_eq!(get(1, 1, 2), 30.0);

    // Off-diagonal: stays at -inf (zero was set by scale_output)
    assert_eq!(get(1, 0, 0), f64::NEG_INFINITY);
    assert_eq!(get(0, 1, 0), f64::NEG_INFINITY);
}

// ============================================================================
// Permute with alpha/beta scaling
// ============================================================================

#[test]
fn maxplus_permute_with_alpha_beta() {
    use tenferro_prims::{CpuBackend, CpuContext, PrimDescriptor, TensorPrims};

    // Tests permute with non-trivial alpha and beta:
    // output = alpha * permute(input) ⊕ beta * output_old
    //
    // input = [[1, 3], [2, 4]]  (col-major: [1, 2, 3, 4])
    // permute = transpose → [[1, 2], [3, 4]] (col-major: [1, 3, 2, 4])
    //
    // With alpha=MaxPlus(10.0), beta=MaxPlus(0.0):
    //   alpha * permute = [11, 13, 12, 14]
    //   beta * old = [0+5, 0+0, 0+0, 0+5] = [5, 0, 0, 5]
    //   result = max of each = [11, 13, 12, 14]
    let mut ctx = CpuContext::new(1);

    let a_data = [MaxPlus(1.0), MaxPlus(2.0), MaxPlus(3.0), MaxPlus(4.0)];
    let mut c_data = [MaxPlus(5.0), MaxPlus(0.0), MaxPlus(0.0), MaxPlus(5.0)];

    let a_view = strided_view::StridedView::new(&a_data, &[2, 2], &[1, 2], 0).unwrap();
    let mut c_view = strided_view::StridedViewMut::new(&mut c_data, &[2, 2], &[1, 2], 0).unwrap();

    let desc = PrimDescriptor::Permute {
        modes_a: vec![0, 1],
        modes_b: vec![1, 0],
    };
    let plan =
        <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::plan::<MaxPlus<f64>>(
            &mut ctx,
            &desc,
            &[&[2, 2], &[2, 2]],
        )
        .unwrap();
    <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::execute(
        &mut ctx,
        &plan,
        MaxPlus(10.0), // alpha
        &[&a_view],
        MaxPlus(0.0), // beta (non-zero: accumulate)
        &mut c_view,
    )
    .unwrap();

    // alpha * permuted = 10 + [1, 3, 2, 4] = [11, 13, 12, 14]
    // beta * old = 0 + [5, 0, 0, 5] = [5, 0, 0, 5]
    // result = max(alpha*permuted, beta*old)
    assert_eq!(c_data[0].0, 11.0); // max(11, 5)
    assert_eq!(c_data[1].0, 13.0); // max(13, 0)
    assert_eq!(c_data[2].0, 12.0); // max(12, 0)
    assert_eq!(c_data[3].0, 14.0); // max(14, 5)
}

// ============================================================================
// MakeContiguous with alpha/beta scaling
// ============================================================================

#[test]
fn maxplus_make_contiguous_with_alpha_beta() {
    use tenferro_prims::{CpuBackend, CpuContext, PrimDescriptor, TensorPrims};

    // MakeContiguous with non-trivial alpha/beta exercises the slow path.
    // input = [10, 20] (contiguous)
    // alpha = MaxPlus(1.0), beta = MaxPlus(5.0)
    // output_old = [0, 0]
    //
    // result = max(1+input, 5+old) = max([11, 21], [5, 5]) = [11, 21]
    let mut ctx = CpuContext::new(1);

    let in_data = [MaxPlus(10.0), MaxPlus(20.0)];
    let mut out_data = [MaxPlus(0.0), MaxPlus(0.0)];

    let in_view = strided_view::StridedView::new(&in_data, &[2], &[1], 0).unwrap();
    let mut out_view = strided_view::StridedViewMut::new(&mut out_data, &[2], &[1], 0).unwrap();

    let desc = PrimDescriptor::MakeContiguous;
    let plan =
        <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::plan::<MaxPlus<f64>>(
            &mut ctx,
            &desc,
            &[&[2], &[2]],
        )
        .unwrap();
    <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::execute(
        &mut ctx,
        &plan,
        MaxPlus(1.0), // alpha
        &[&in_view],
        MaxPlus(5.0), // beta (non-zero)
        &mut out_view,
    )
    .unwrap();

    assert_eq!(out_data[0].0, 11.0); // max(1+10, 5+0) = max(11, 5)
    assert_eq!(out_data[1].0, 21.0); // max(1+20, 5+0) = max(21, 5)
}

// ============================================================================
// scale_output with beta != 0 and beta != 1
// ============================================================================

#[test]
fn maxplus_reduce_with_beta_accumulate() {
    use tenferro_prims::{CpuBackend, CpuContext, PrimDescriptor, ReduceOp, TensorPrims};

    // Reduce with non-zero beta tests the accumulation path in execute_reduce_sum.
    // A = [[1, 3],    (col-major: [1, 2, 3, 4])
    //      [2, 4]]
    // reduce axis 1: result = [max(1,3), max(2,4)] = [3, 4]
    // With alpha=MaxPlus(0.0), beta=MaxPlus(0.0):
    //   output = max(0+[3,4], 0+[100,100]) = max([3,4], [100,100]) = [100, 100]
    let mut ctx = CpuContext::new(1);

    let a_data = [MaxPlus(1.0), MaxPlus(2.0), MaxPlus(3.0), MaxPlus(4.0)];
    let mut c_data = [MaxPlus(100.0), MaxPlus(100.0)]; // pre-filled

    let a_view = strided_view::StridedView::new(&a_data, &[2, 2], &[1, 2], 0).unwrap();
    let mut c_view = strided_view::StridedViewMut::new(&mut c_data, &[2], &[1], 0).unwrap();

    let desc = PrimDescriptor::Reduce {
        modes_a: vec![0, 1],
        modes_c: vec![0],
        op: ReduceOp::Sum,
    };
    let plan =
        <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::plan::<MaxPlus<f64>>(
            &mut ctx,
            &desc,
            &[&[2, 2], &[2]],
        )
        .unwrap();
    <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::execute(
        &mut ctx,
        &plan,
        MaxPlus(0.0), // alpha = one
        &[&a_view],
        MaxPlus(0.0), // beta = one
        &mut c_view,
    )
    .unwrap();

    // alpha*reduce = 0+[3,4] = [3,4]
    // beta*old = 0+[100,100] = [100,100]
    // result = max([3,4], [100,100]) = [100, 100]
    assert_eq!(c_data[0].0, 100.0);
    assert_eq!(c_data[1].0, 100.0);
}

// ============================================================================
// Trace with beta accumulate
// ============================================================================

#[test]
fn maxplus_trace_with_beta_accumulate() {
    use tenferro_prims::{CpuBackend, CpuContext, PrimDescriptor, TensorPrims};

    // Trace with non-zero beta: C_new = max(alpha * trace, beta * C_old)
    // A = [[10, 4],    trace = max(10, 8) = 10
    //      [2,  8]]
    // With alpha=MaxPlus(0.0), beta=MaxPlus(0.0):
    //   alpha*trace = 0+10 = 10
    //   beta*old = 0+20 = 20
    //   result = max(10, 20) = 20
    let mut ctx = CpuContext::new(1);

    let a_data = [MaxPlus(10.0), MaxPlus(2.0), MaxPlus(4.0), MaxPlus(8.0)];
    let mut c_data = [MaxPlus(20.0)]; // pre-filled scalar

    let a_view = strided_view::StridedView::new(&a_data, &[2, 2], &[1, 2], 0).unwrap();
    let mut c_view = strided_view::StridedViewMut::new(&mut c_data, &[], &[], 0).unwrap();

    let desc = PrimDescriptor::Trace {
        modes_a: vec![0, 1],
        modes_c: vec![],
        paired: vec![(0, 1)],
    };
    let plan =
        <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::plan::<MaxPlus<f64>>(
            &mut ctx,
            &desc,
            &[&[2, 2], &[]],
        )
        .unwrap();
    <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::execute(
        &mut ctx,
        &plan,
        MaxPlus(0.0), // alpha
        &[&a_view],
        MaxPlus(0.0), // beta
        &mut c_view,
    )
    .unwrap();

    assert_eq!(c_data[0].0, 20.0); // max(0+10, 0+20) = 20
}

// ============================================================================
// MinPlus non-GEMM operations (verifies the MinPlus algebra works for all prims)
// ============================================================================

#[test]
fn minplus_reduce_column_min() {
    use tenferro_prims::{CpuBackend, CpuContext, PrimDescriptor, ReduceOp, TensorPrims};

    // MinPlus "sum" is min. Reduce axis 1 = min over columns.
    // A = [[5, 1],   → result = [min(5,1), min(2,8)] = [1, 2]
    //      [2, 8]]
    let mut ctx = CpuContext::new(1);

    let a_data = [MinPlus(5.0), MinPlus(2.0), MinPlus(1.0), MinPlus(8.0)];
    let mut c_data = [MinPlus::<f64>::zero(); 2];

    let a_view = strided_view::StridedView::new(&a_data, &[2, 2], &[1, 2], 0).unwrap();
    let mut c_view = strided_view::StridedViewMut::new(&mut c_data, &[2], &[1], 0).unwrap();

    let desc = PrimDescriptor::Reduce {
        modes_a: vec![0, 1],
        modes_c: vec![0],
        op: ReduceOp::Sum,
    };
    let plan =
        <CpuBackend as TensorPrims<tenferro_tropical::MinPlusAlgebra>>::plan::<MinPlus<f64>>(
            &mut ctx,
            &desc,
            &[&[2, 2], &[2]],
        )
        .unwrap();
    <CpuBackend as TensorPrims<tenferro_tropical::MinPlusAlgebra>>::execute(
        &mut ctx,
        &plan,
        MinPlus::one(),
        &[&a_view],
        MinPlus::zero(),
        &mut c_view,
    )
    .unwrap();

    assert_eq!(c_data[0].0, 1.0); // min(5, 1)
    assert_eq!(c_data[1].0, 2.0); // min(2, 8)
}

#[test]
fn maxmul_permute_transpose() {
    use tenferro_prims::{CpuBackend, CpuContext, PrimDescriptor, TensorPrims};

    // Verifies MaxMul algebra works for non-GEMM operations.
    // Transpose [[0.5, 0.3], [0.2, 0.9]] → [[0.5, 0.2], [0.3, 0.9]]
    let mut ctx = CpuContext::new(1);

    let a_data = [MaxMul(0.5), MaxMul(0.2), MaxMul(0.3), MaxMul(0.9)];
    let mut c_data = [MaxMul::<f64>::zero(); 4];

    let a_view = strided_view::StridedView::new(&a_data, &[2, 2], &[1, 2], 0).unwrap();
    let mut c_view = strided_view::StridedViewMut::new(&mut c_data, &[2, 2], &[1, 2], 0).unwrap();

    let desc = PrimDescriptor::Permute {
        modes_a: vec![0, 1],
        modes_b: vec![1, 0],
    };
    let plan = <CpuBackend as TensorPrims<tenferro_tropical::MaxMulAlgebra>>::plan::<MaxMul<f64>>(
        &mut ctx,
        &desc,
        &[&[2, 2], &[2, 2]],
    )
    .unwrap();
    <CpuBackend as TensorPrims<tenferro_tropical::MaxMulAlgebra>>::execute(
        &mut ctx,
        &plan,
        MaxMul::one(),
        &[&a_view],
        MaxMul::zero(),
        &mut c_view,
    )
    .unwrap();

    // Transposed col-major: [0.5, 0.3, 0.2, 0.9]
    assert!((c_data[0].0 - 0.5).abs() < 1e-15);
    assert!((c_data[1].0 - 0.3).abs() < 1e-15);
    assert!((c_data[2].0 - 0.2).abs() < 1e-15);
    assert!((c_data[3].0 - 0.9).abs() < 1e-15);
}

// ============================================================================
// Anti-trace with non-zero beta (accumulation into existing output)
// ============================================================================

#[test]
fn maxplus_anti_trace_accumulate_beta() {
    use tenferro_prims::{CpuBackend, CpuContext, PrimDescriptor, TensorPrims};

    // Anti-trace with beta != 0: output = beta * output_old, then add alpha * diag.
    // This exercises the scale_output function with beta != 0 && beta != 1.
    //
    // input = MaxPlus(5.0) (scalar)
    // output_old = [[2, 3], [4, 6]]
    // beta = MaxPlus(10.0): beta * output_old = 10 + [[2,3],[4,6]] = [[12,13],[14,16]]
    // alpha = MaxPlus(0.0): alpha * input = 0 + 5 = 5
    // Then add 5 to diagonal: output[d,d] += 5
    // output[0,0] = max(12, 5) = 12
    // output[1,0] = 14 (no addition)
    // output[0,1] = 13 (no addition)
    // output[1,1] = max(16, 5) = 16
    let mut ctx = CpuContext::new(1);

    let in_data = [MaxPlus(5.0)];
    let mut out_data = [MaxPlus(2.0), MaxPlus(4.0), MaxPlus(3.0), MaxPlus(6.0)];

    let in_view = strided_view::StridedView::new(&in_data, &[], &[], 0).unwrap();
    let mut out_view =
        strided_view::StridedViewMut::new(&mut out_data, &[2, 2], &[1, 2], 0).unwrap();

    let desc = PrimDescriptor::AntiTrace {
        modes_a: vec![],
        modes_c: vec![0, 1],
        paired: vec![(0, 1)],
    };
    let plan =
        <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::plan::<MaxPlus<f64>>(
            &mut ctx,
            &desc,
            &[&[], &[2, 2]],
        )
        .unwrap();
    <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::execute(
        &mut ctx,
        &plan,
        MaxPlus(0.0), // alpha
        &[&in_view],
        MaxPlus(10.0), // beta != 0, != 1
        &mut out_view,
    )
    .unwrap();

    // scale_output: beta * old = 10 + [2,4,3,6] = [12,14,13,16]
    // Then diag += alpha * input = 0 + 5 = 5
    // [0,0] = max(12, 5) = 12; [1,0] = 14; [0,1] = 13; [1,1] = max(16, 5) = 16
    assert_eq!(out_data[0].0, 12.0);
    assert_eq!(out_data[1].0, 14.0);
    assert_eq!(out_data[2].0, 13.0);
    assert_eq!(out_data[3].0, 16.0);
}

// ============================================================================
// Plan validation: ReduceOp::Max/Min returns error for tropical types
// ============================================================================

#[test]
fn tropical_reduce_max_op_returns_error() {
    use tenferro_device::Error;
    use tenferro_prims::{CpuBackend, CpuContext, PrimDescriptor, ReduceOp, TensorPrims};

    // For tropical types, ReduceOp::Sum already implements the correct tropical
    // reduction (max for MaxPlus, min for MinPlus). ReduceOp::Max is meaningless
    // and should return an error.
    let mut ctx = CpuContext::new(1);

    let a_data = [MaxPlus(1.0), MaxPlus(2.0)];
    let mut c_data = [MaxPlus::<f64>::zero()];

    let a_view = strided_view::StridedView::new(&a_data, &[2], &[1], 0).unwrap();
    let mut c_view = strided_view::StridedViewMut::new(&mut c_data, &[], &[], 0).unwrap();

    let desc = PrimDescriptor::Reduce {
        modes_a: vec![0],
        modes_c: vec![],
        op: ReduceOp::Sum,
    };
    let plan =
        <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::plan::<MaxPlus<f64>>(
            &mut ctx,
            &desc,
            &[&[2], &[]],
        )
        .unwrap();

    // Construct a ReduceOp::Max plan manually by planning with Sum then executing
    // Actually, plan with Max directly:
    let desc_max = PrimDescriptor::Reduce {
        modes_a: vec![0],
        modes_c: vec![],
        op: ReduceOp::Max,
    };
    let plan_max = <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::plan::<
        MaxPlus<f64>,
    >(&mut ctx, &desc_max, &[&[2], &[]])
    .unwrap();
    let err = <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::execute(
        &mut ctx,
        &plan_max,
        MaxPlus::one(),
        &[&a_view],
        MaxPlus::zero(),
        &mut c_view,
    )
    .unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(_)));

    // Also verify Sum works correctly (max(1, 2) = 2)
    <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::execute(
        &mut ctx,
        &plan,
        MaxPlus::one(),
        &[&a_view],
        MaxPlus::zero(),
        &mut c_view,
    )
    .unwrap();
    assert_eq!(c_data[0].0, 2.0);
}

// ============================================================================
// Plan validation: unsupported operations
// ============================================================================

#[test]
fn tropical_plan_contract_returns_error() {
    use tenferro_device::Error;
    use tenferro_prims::{CpuBackend, CpuContext, PrimDescriptor, TensorPrims};

    let mut ctx = CpuContext::new(1);
    let desc = PrimDescriptor::Contract {
        modes_a: vec![0, 1],
        modes_b: vec![1, 2],
        modes_c: vec![0, 2],
    };
    let result = <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::plan::<MaxPlus<f64>>(
        &mut ctx,
        &desc,
        &[&[2, 3], &[3, 4], &[2, 4]],
    );
    assert!(matches!(result, Err(Error::InvalidArgument(_))));
}

#[test]
fn tropical_plan_elementwise_mul_returns_error() {
    use tenferro_device::Error;
    use tenferro_prims::{CpuBackend, CpuContext, PrimDescriptor, TensorPrims};

    let mut ctx = CpuContext::new(1);
    let desc = PrimDescriptor::ElementwiseMul;
    let result = <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::plan::<MaxPlus<f64>>(
        &mut ctx,
        &desc,
        &[&[2], &[2], &[2]],
    );
    assert!(matches!(result, Err(Error::InvalidArgument(_))));
}

// ============================================================================
// Execute arity validation tests
// ============================================================================

#[test]
fn tropical_execute_gemm_wrong_arity_returns_error() {
    use tenferro_device::Error;
    use tenferro_prims::{CpuBackend, CpuContext, PrimDescriptor, TensorPrims};

    let mut ctx = CpuContext::new(1);
    let desc = PrimDescriptor::BatchedGemm {
        batch_dims: vec![],
        m: 2,
        n: 2,
        k: 2,
    };
    let plan =
        <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::plan::<MaxPlus<f64>>(
            &mut ctx,
            &desc,
            &[&[2, 2], &[2, 2], &[2, 2]],
        )
        .unwrap();

    let a_data = [MaxPlus(1.0_f64); 4];
    let a_view = strided_view::StridedView::new(&a_data, &[2, 2], &[1, 2], 0).unwrap();
    let mut out_data = [MaxPlus::<f64>::zero(); 4];
    let mut out_view =
        strided_view::StridedViewMut::new(&mut out_data, &[2, 2], &[1, 2], 0).unwrap();

    // Pass 1 input instead of 2
    let err = <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::execute(
        &mut ctx,
        &plan,
        MaxPlus::one(),
        &[&a_view], // wrong: need 2
        MaxPlus::zero(),
        &mut out_view,
    )
    .unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(_)));
}

#[test]
fn tropical_execute_reduce_wrong_arity_returns_error() {
    use tenferro_device::Error;
    use tenferro_prims::{CpuBackend, CpuContext, PrimDescriptor, ReduceOp, TensorPrims};

    let mut ctx = CpuContext::new(1);
    let desc = PrimDescriptor::Reduce {
        modes_a: vec![0, 1],
        modes_c: vec![0],
        op: ReduceOp::Sum,
    };
    let plan =
        <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::plan::<MaxPlus<f64>>(
            &mut ctx,
            &desc,
            &[&[2, 2], &[2]],
        )
        .unwrap();

    let mut out_data = [MaxPlus::<f64>::zero(); 2];
    let mut out_view = strided_view::StridedViewMut::new(&mut out_data, &[2], &[1], 0).unwrap();

    let no_inputs: [&strided_view::StridedView<MaxPlus<f64>>; 0] = [];
    let err = <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::execute(
        &mut ctx,
        &plan,
        MaxPlus::one(),
        &no_inputs,
        MaxPlus::zero(),
        &mut out_view,
    )
    .unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(_)));
}

#[test]
fn tropical_execute_trace_wrong_arity_returns_error() {
    use tenferro_device::Error;
    use tenferro_prims::{CpuBackend, CpuContext, PrimDescriptor, TensorPrims};

    let mut ctx = CpuContext::new(1);
    let desc = PrimDescriptor::Trace {
        modes_a: vec![0, 1],
        modes_c: vec![],
        paired: vec![(0, 1)],
    };
    let plan =
        <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::plan::<MaxPlus<f64>>(
            &mut ctx,
            &desc,
            &[&[2, 2], &[]],
        )
        .unwrap();

    let mut out_data = [MaxPlus::<f64>::zero()];
    let mut out_view = strided_view::StridedViewMut::new(&mut out_data, &[], &[], 0).unwrap();

    let no_inputs: [&strided_view::StridedView<MaxPlus<f64>>; 0] = [];
    let err = <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::execute(
        &mut ctx,
        &plan,
        MaxPlus::one(),
        &no_inputs,
        MaxPlus::zero(),
        &mut out_view,
    )
    .unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(_)));
}

#[test]
fn tropical_execute_anti_trace_wrong_arity_returns_error() {
    use tenferro_device::Error;
    use tenferro_prims::{CpuBackend, CpuContext, PrimDescriptor, TensorPrims};

    let mut ctx = CpuContext::new(1);
    let desc = PrimDescriptor::AntiTrace {
        modes_a: vec![],
        modes_c: vec![0, 1],
        paired: vec![(0, 1)],
    };
    let plan =
        <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::plan::<MaxPlus<f64>>(
            &mut ctx,
            &desc,
            &[&[], &[2, 2]],
        )
        .unwrap();

    let mut out_data = [MaxPlus::<f64>::zero(); 4];
    let mut out_view =
        strided_view::StridedViewMut::new(&mut out_data, &[2, 2], &[1, 2], 0).unwrap();

    let no_inputs: [&strided_view::StridedView<MaxPlus<f64>>; 0] = [];
    let err = <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::execute(
        &mut ctx,
        &plan,
        MaxPlus::one(),
        &no_inputs,
        MaxPlus::zero(),
        &mut out_view,
    )
    .unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(_)));
}

#[test]
fn tropical_execute_anti_diag_wrong_arity_returns_error() {
    use tenferro_device::Error;
    use tenferro_prims::{CpuBackend, CpuContext, PrimDescriptor, TensorPrims};

    let mut ctx = CpuContext::new(1);
    let desc = PrimDescriptor::AntiDiag {
        modes_a: vec![0],
        modes_c: vec![0, 1],
        paired: vec![(0, 1)],
    };
    let plan =
        <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::plan::<MaxPlus<f64>>(
            &mut ctx,
            &desc,
            &[&[2], &[2, 2]],
        )
        .unwrap();

    let mut out_data = [MaxPlus::<f64>::zero(); 4];
    let mut out_view =
        strided_view::StridedViewMut::new(&mut out_data, &[2, 2], &[1, 2], 0).unwrap();

    let no_inputs: [&strided_view::StridedView<MaxPlus<f64>>; 0] = [];
    let err = <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::execute(
        &mut ctx,
        &plan,
        MaxPlus::one(),
        &no_inputs,
        MaxPlus::zero(),
        &mut out_view,
    )
    .unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(_)));
}

#[test]
fn tropical_execute_make_contiguous_wrong_arity_returns_error() {
    use tenferro_device::Error;
    use tenferro_prims::{CpuBackend, CpuContext, PrimDescriptor, TensorPrims};

    let mut ctx = CpuContext::new(1);
    let desc = PrimDescriptor::MakeContiguous;
    let plan =
        <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::plan::<MaxPlus<f64>>(
            &mut ctx,
            &desc,
            &[&[2], &[2]],
        )
        .unwrap();

    let mut out_data = [MaxPlus::<f64>::zero(); 2];
    let mut out_view = strided_view::StridedViewMut::new(&mut out_data, &[2], &[1], 0).unwrap();

    let no_inputs: [&strided_view::StridedView<MaxPlus<f64>>; 0] = [];
    let err = <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::execute(
        &mut ctx,
        &plan,
        MaxPlus::one(),
        &no_inputs,
        MaxPlus::zero(),
        &mut out_view,
    )
    .unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(_)));
}

// ============================================================================
// Extension not supported test
// ============================================================================

#[test]
fn tropical_no_extensions() {
    use tenferro_prims::{CpuBackend, Extension, TensorPrims};

    assert!(!<CpuBackend as TensorPrims<
        tenferro_tropical::MaxPlusAlgebra,
    >>::has_extension_for::<MaxPlus<f64>>(
        Extension::Contract
    ));
    assert!(!<CpuBackend as TensorPrims<
        tenferro_tropical::MaxPlusAlgebra,
    >>::has_extension_for::<MaxPlus<f64>>(
        Extension::ElementwiseMul
    ));
    assert!(!<CpuBackend as TensorPrims<
        tenferro_tropical::MinPlusAlgebra,
    >>::has_extension_for::<MinPlus<f64>>(
        Extension::Contract
    ));
    assert!(!<CpuBackend as TensorPrims<
        tenferro_tropical::MaxMulAlgebra,
    >>::has_extension_for::<MaxMul<f64>>(
        Extension::Contract
    ));
}

// ============================================================================
// Plan validation error tests (verifying API rejects invalid descriptors)
// ============================================================================

#[test]
fn tropical_plan_reduce_mode_rank_mismatch_returns_error() {
    use tenferro_device::Error;
    use tenferro_prims::{CpuBackend, CpuContext, PrimDescriptor, ReduceOp, TensorPrims};

    let mut ctx = CpuContext::new(1);
    // modes_a has 2 labels but shape has rank 3
    let desc = PrimDescriptor::Reduce {
        modes_a: vec![0, 1],
        modes_c: vec![0],
        op: ReduceOp::Sum,
    };
    let result = <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::plan::<MaxPlus<f64>>(
        &mut ctx,
        &desc,
        &[&[2, 3, 4], &[2]],
    );
    assert!(matches!(result, Err(Error::InvalidArgument(_))));
}

#[test]
fn tropical_plan_reduce_output_shape_mismatch_returns_error() {
    use tenferro_device::Error;
    use tenferro_prims::{CpuBackend, CpuContext, PrimDescriptor, ReduceOp, TensorPrims};

    let mut ctx = CpuContext::new(1);
    let desc = PrimDescriptor::Reduce {
        modes_a: vec![0, 1],
        modes_c: vec![0],
        op: ReduceOp::Sum,
    };
    // Output shape [3] doesn't match input mode 0 which has size 2
    let result = <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::plan::<MaxPlus<f64>>(
        &mut ctx,
        &desc,
        &[&[2, 2], &[3]],
    );
    assert!(matches!(result, Err(Error::InvalidArgument(_))));
}

#[test]
fn tropical_plan_trace_paired_dims_unequal_returns_error() {
    use tenferro_device::Error;
    use tenferro_prims::{CpuBackend, CpuContext, PrimDescriptor, TensorPrims};

    let mut ctx = CpuContext::new(1);
    // Paired axes have unequal dimensions (mode 0: size 2, mode 1: size 3)
    let desc = PrimDescriptor::Trace {
        modes_a: vec![0, 1],
        modes_c: vec![],
        paired: vec![(0, 1)],
    };
    let result = <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::plan::<MaxPlus<f64>>(
        &mut ctx,
        &desc,
        &[&[2, 3], &[]],
    );
    assert!(matches!(result, Err(Error::InvalidArgument(_))));
}

#[test]
fn tropical_plan_trace_output_shape_mismatch_returns_error() {
    use tenferro_device::Error;
    use tenferro_prims::{CpuBackend, CpuContext, PrimDescriptor, TensorPrims};

    let mut ctx = CpuContext::new(1);
    // Free mode 2 has size 3 in input but output says size 4
    let desc = PrimDescriptor::Trace {
        modes_a: vec![0, 1, 2],
        modes_c: vec![2],
        paired: vec![(0, 1)],
    };
    let result = <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::plan::<MaxPlus<f64>>(
        &mut ctx,
        &desc,
        &[&[2, 2, 3], &[4]],
    );
    assert!(matches!(result, Err(Error::InvalidArgument(_))));
}

#[test]
fn tropical_plan_permute_shape_mismatch_returns_error() {
    use tenferro_device::Error;
    use tenferro_prims::{CpuBackend, CpuContext, PrimDescriptor, TensorPrims};

    let mut ctx = CpuContext::new(1);
    // Permuted output shape doesn't match: mode 0 has size 2, but output says 3
    let desc = PrimDescriptor::Permute {
        modes_a: vec![0, 1],
        modes_b: vec![1, 0],
    };
    let result = <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::plan::<MaxPlus<f64>>(
        &mut ctx,
        &desc,
        &[&[2, 3], &[2, 3]],
    );
    assert!(matches!(result, Err(Error::InvalidArgument(_))));
}

#[test]
fn tropical_plan_anti_trace_paired_dims_unequal_returns_error() {
    use tenferro_device::Error;
    use tenferro_prims::{CpuBackend, CpuContext, PrimDescriptor, TensorPrims};

    let mut ctx = CpuContext::new(1);
    // AntiTrace: paired axes in output have unequal sizes (mode 0: size 2, mode 1: size 3)
    let desc = PrimDescriptor::AntiTrace {
        modes_a: vec![],
        modes_c: vec![0, 1],
        paired: vec![(0, 1)],
    };
    let result = <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::plan::<MaxPlus<f64>>(
        &mut ctx,
        &desc,
        &[&[], &[2, 3]],
    );
    assert!(matches!(result, Err(Error::InvalidArgument(_))));
}

#[test]
fn tropical_plan_anti_trace_input_shape_mismatch_returns_error() {
    use tenferro_device::Error;
    use tenferro_prims::{CpuBackend, CpuContext, PrimDescriptor, TensorPrims};

    let mut ctx = CpuContext::new(1);
    // AntiTrace: free axis (mode 2) has size 3 in input but size 4 in output
    let desc = PrimDescriptor::AntiTrace {
        modes_a: vec![2],
        modes_c: vec![0, 1, 2],
        paired: vec![(0, 1)],
    };
    let result = <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::plan::<MaxPlus<f64>>(
        &mut ctx,
        &desc,
        &[&[3], &[2, 2, 4]],
    );
    assert!(matches!(result, Err(Error::InvalidArgument(_))));
}

#[test]
fn tropical_plan_anti_diag_paired_dims_unequal_returns_error() {
    use tenferro_device::Error;
    use tenferro_prims::{CpuBackend, CpuContext, PrimDescriptor, TensorPrims};

    let mut ctx = CpuContext::new(1);
    // AntiDiag: paired axes in output have unequal sizes
    let desc = PrimDescriptor::AntiDiag {
        modes_a: vec![0],
        modes_c: vec![0, 1],
        paired: vec![(0, 1)],
    };
    let result = <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::plan::<MaxPlus<f64>>(
        &mut ctx,
        &desc,
        &[&[2], &[2, 3]],
    );
    assert!(matches!(result, Err(Error::InvalidArgument(_))));
}

#[test]
fn tropical_plan_anti_diag_input_shape_mismatch_returns_error() {
    use tenferro_device::Error;
    use tenferro_prims::{CpuBackend, CpuContext, PrimDescriptor, TensorPrims};

    let mut ctx = CpuContext::new(1);
    // AntiDiag: free axis (mode 0) has size 2 in input but size 3 in output
    let desc = PrimDescriptor::AntiDiag {
        modes_a: vec![0],
        modes_c: vec![0, 1],
        paired: vec![(0, 1)],
    };
    let result = <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::plan::<MaxPlus<f64>>(
        &mut ctx,
        &desc,
        &[&[2], &[3, 3]],
    );
    assert!(matches!(result, Err(Error::InvalidArgument(_))));
}

#[test]
fn tropical_plan_make_contiguous_shape_mismatch_returns_error() {
    use tenferro_device::Error;
    use tenferro_prims::{CpuBackend, CpuContext, PrimDescriptor, TensorPrims};

    let mut ctx = CpuContext::new(1);
    let desc = PrimDescriptor::MakeContiguous;
    let result = <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::plan::<MaxPlus<f64>>(
        &mut ctx,
        &desc,
        &[&[2, 3], &[2, 4]],
    );
    assert!(matches!(result, Err(Error::InvalidArgument(_))));
}

#[test]
fn tropical_plan_trace_empty_paired_returns_error() {
    use tenferro_device::Error;
    use tenferro_prims::{CpuBackend, CpuContext, PrimDescriptor, TensorPrims};

    let mut ctx = CpuContext::new(1);
    let desc = PrimDescriptor::Trace {
        modes_a: vec![0, 1],
        modes_c: vec![0],
        paired: vec![],
    };
    let err = match <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::plan::<
        MaxPlus<f64>,
    >(&mut ctx, &desc, &[&[2, 2], &[2]])
    {
        Ok(_) => panic!("expected InvalidArgument error"),
        Err(e) => e,
    };
    assert!(matches!(err, Error::InvalidArgument(_)));
}

#[test]
fn tropical_plan_antidiag_invalid_pair_anchor_returns_error() {
    use tenferro_device::Error;
    use tenferro_prims::{CpuBackend, CpuContext, PrimDescriptor, TensorPrims};

    let mut ctx = CpuContext::new(1);
    let desc = PrimDescriptor::AntiDiag {
        modes_a: vec![0],
        modes_c: vec![0, 1],
        // first paired label must exist in modes_a, but here it's 1
        paired: vec![(1, 0)],
    };
    let err = match <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::plan::<
        MaxPlus<f64>,
    >(&mut ctx, &desc, &[&[2], &[2, 2]])
    {
        Ok(_) => panic!("expected InvalidArgument error"),
        Err(e) => e,
    };
    assert!(matches!(err, Error::InvalidArgument(_)));
}

#[test]
fn tropical_plan_batched_gemm_shape_mismatch_returns_error() {
    use tenferro_device::Error;
    use tenferro_prims::{CpuBackend, CpuContext, PrimDescriptor, TensorPrims};

    let mut ctx = CpuContext::new(1);
    let desc = PrimDescriptor::BatchedGemm {
        batch_dims: vec![],
        m: 2,
        n: 2,
        k: 3,
    };
    // B shape is wrong (should be [3, 2], here [4, 2])
    let err = match <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::plan::<
        MaxPlus<f64>,
    >(&mut ctx, &desc, &[&[2, 3], &[4, 2], &[2, 2]])
    {
        Ok(_) => panic!("expected InvalidArgument error"),
        Err(e) => e,
    };
    assert!(matches!(err, Error::InvalidArgument(_)));
}

#[test]
fn tropical_execute_wrong_input_arity_returns_error() {
    use tenferro_device::Error;
    use tenferro_prims::{CpuBackend, CpuContext, PrimDescriptor, TensorPrims};

    let mut ctx = CpuContext::new(1);
    let desc = PrimDescriptor::Permute {
        modes_a: vec![0, 1],
        modes_b: vec![1, 0],
    };
    let plan =
        <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::plan::<MaxPlus<f64>>(
            &mut ctx,
            &desc,
            &[&[2, 2], &[2, 2]],
        )
        .unwrap();

    let mut out = [MaxPlus::<f64>::zero(); 4];
    let mut out_view = strided_view::StridedViewMut::new(&mut out, &[2, 2], &[1, 2], 0).unwrap();

    let no_inputs: [&strided_view::StridedView<MaxPlus<f64>>; 0] = [];
    let err = <CpuBackend as TensorPrims<tenferro_tropical::MaxPlusAlgebra>>::execute(
        &mut ctx,
        &plan,
        MaxPlus::one(),
        &no_inputs,
        MaxPlus::zero(),
        &mut out_view,
    )
    .unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(_)));
}
