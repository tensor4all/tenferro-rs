use tenferro_prims::CpuContext;

use super::{cholesky_vec, lu_factor_vec, qr_vec, solve_triangular_vec, solve_vec, thin_svd_vec};

#[test]
fn solve_vec_matches_identity_system() {
    let mut ctx = CpuContext::new(1);
    let a = [1.0_f64, 0.0, 0.0, 1.0];
    let b = [2.0_f64, 3.0];

    let x = solve_vec(&mut ctx, &a, &b, 2, 1).unwrap();

    assert_eq!(x, vec![2.0, 3.0]);
}

#[test]
fn solve_triangular_vec_accepts_vector_rhs() {
    let mut ctx = CpuContext::new(1);
    let a = [1.0_f64, 0.0, 2.0, 3.0];
    let b = [1.0_f64, 5.0];

    let x = solve_triangular_vec(&mut ctx, &a, &b, 2, 1, true).unwrap();

    assert_eq!(x, vec![-7.0 / 3.0, 5.0 / 3.0]);
}

#[test]
fn qr_vec_returns_expected_shapes() {
    let mut ctx = CpuContext::new(1);
    let a = [1.0_f64, 3.0, 2.0, 4.0];

    let (q, r) = qr_vec(&mut ctx, &a, 2, 2).unwrap();

    assert_eq!(q.len(), 4);
    assert_eq!(r.len(), 4);
}

#[test]
fn thin_svd_vec_returns_expected_shapes() {
    let mut ctx = CpuContext::new(1);
    let a = [1.0_f64, 0.0, 0.0, 1.0];

    let (u, s, vt) = thin_svd_vec(&mut ctx, &a, 2, 2).unwrap();

    assert_eq!(u.len(), 4);
    assert_eq!(s.len(), 2);
    assert_eq!(vt.len(), 4);
}

#[test]
fn lu_factor_vec_returns_pivots_and_factors() {
    let mut ctx = CpuContext::new(1);
    let a = [1.0_f64, 3.0, 2.0, 4.0];

    let (pivots, l, u) = lu_factor_vec(&mut ctx, &a, 2, 2).unwrap();

    assert_eq!(pivots.len(), 2);
    assert_eq!(l.len(), 4);
    assert_eq!(u.len(), 4);
}

#[test]
fn cholesky_vec_factorizes_spd_input() {
    let mut ctx = CpuContext::new(1);
    let a = [4.0_f64, 2.0, 2.0, 3.0];

    let l = cholesky_vec(&mut ctx, &a, 2).unwrap();

    assert_eq!(l.len(), 4);
}
