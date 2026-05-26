use tenferro_tensor::{cpu::CpuBackend, TypedTensor};

#[test]
fn typed_svd_f64() {
    let mut ctx = CpuBackend::new();
    let input = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![3.0, 0.0, 0.0, 2.0]);

    let (u, s, vt) = input.svd(&mut ctx).unwrap();

    assert_eq!(u.shape, vec![2, 2]);
    assert_eq!(s.shape, vec![2]);
    assert_eq!(vt.shape, vec![2, 2]);
    assert_eq!(s.as_slice(), &[3.0, 2.0]);
}

#[test]
fn typed_qr_f64() {
    let mut ctx = CpuBackend::new();
    let input = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);

    let (q, r) = input.qr(&mut ctx).unwrap();

    assert_eq!(q.shape, vec![2, 2]);
    assert_eq!(r.shape, vec![2, 2]);
}

#[test]
fn typed_cholesky_f64() {
    let mut ctx = CpuBackend::new();
    let input = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![4.0, 0.0, 0.0, 9.0]);

    let factor = input.cholesky(&mut ctx).unwrap();

    assert_eq!(factor.shape, vec![2, 2]);
    assert_eq!(factor.as_slice(), &[2.0, 0.0, 0.0, 3.0]);
}

#[test]
fn typed_eigh_f64() {
    let mut ctx = CpuBackend::new();
    let input = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![2.0, 0.0, 0.0, 5.0]);

    let (w, v) = input.eigh(&mut ctx).unwrap();

    assert_eq!(w.shape, vec![2]);
    assert_eq!(v.shape, vec![2, 2]);
    assert_eq!(w.as_slice(), &[2.0, 5.0]);
}
