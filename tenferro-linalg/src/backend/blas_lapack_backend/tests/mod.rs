use super::*;

#[test]
fn svd_identity_f64() {
    let mut backend = BlasLapackBackend::new();
    let a = [1.0_f64, 0.0, 0.0, 1.0];
    let mut u = [0.0_f64; 4];
    let mut s = [0.0_f64; 2];
    let mut vt = [0.0_f64; 4];

    backend.thin_svd(&a, 2, 2, &mut u, &mut s, &mut vt).unwrap();
    assert!((s[0] - 1.0).abs() < 1e-10);
    assert!((s[1] - 1.0).abs() < 1e-10);
}

#[test]
fn qr_reconstruction_f64() {
    let mut backend = BlasLapackBackend::new();
    let a = [1.0_f64, 3.0, 2.0, 4.0, 5.0, 6.0]; // 2x3
    let mut q = [0.0_f64; 4]; // 2x2
    let mut r = [0.0_f64; 6]; // 2x3

    backend.qr(&a, 2, 3, &mut q, &mut r).unwrap();

    let mut recon = [0.0_f64; 6];
    for j in 0..3 {
        for i in 0..2 {
            let mut v = 0.0;
            for p in 0..2 {
                v += q[i + p * 2] * r[p + j * 2];
            }
            recon[i + j * 2] = v;
        }
    }
    let err = a
        .iter()
        .zip(recon.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max);
    assert!(err < 1e-9, "qr reconstruction error {err}");
}

#[test]
fn solve_f64_matches_expected() {
    let mut backend = BlasLapackBackend::new();
    // A = [[3,1],[1,2]] in col-major
    let a = [3.0_f64, 1.0, 1.0, 2.0];
    // b = [[9],[8]]
    let b = [9.0_f64, 8.0];
    let mut x = [0.0_f64; 2];

    backend.solve(&a, &b, 2, 1, &mut x).unwrap();
    assert!((x[0] - 2.0).abs() < 1e-10);
    assert!((x[1] - 3.0).abs() < 1e-10);
}

#[test]
fn eig_general_real_returns_expected_eigenvalues() {
    let mut backend = BlasLapackBackend::new();
    // [[0,-1],[1,0]] has eigenvalues +/- i
    let a = [0.0_f64, 1.0, -1.0, 0.0];
    let mut values = [0.0_f64; 4];
    let mut vectors = [0.0_f64; 8];

    backend
        .eig_general(&a, 2, &mut values, &mut vectors)
        .unwrap();

    let lam0 = Complex64::new(values[0], values[1]);
    let lam1 = Complex64::new(values[2], values[3]);
    let ok = (lam0.im.abs() - 1.0).abs() < 1e-8
        && (lam1.im.abs() - 1.0).abs() < 1e-8
        && (lam0.re.abs() < 1e-8)
        && (lam1.re.abs() < 1e-8);
    assert!(ok, "unexpected eigenvalues: {lam0:?}, {lam1:?}");
}
