use super::LapackEigScalar;
use num_complex::{Complex32, Complex64};

#[test]
fn lapack_eig_scalar_f32_round_trip() {
    let (vals, vecs) = f32::eig_buffer_sizes(3);
    assert_eq!(vals, 6);
    assert_eq!(vecs, 18);

    let val_ri = [1.0_f32, 0.5, 2.0, -0.5];
    let vec_ri = [1.0_f32, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
    let mut vals_out = [Complex32::new(0.0, 0.0); 2];
    let mut vecs_out = [Complex32::new(0.0, 0.0); 4];
    f32::eig_ri_to_complex(2, &val_ri, &vec_ri, &mut vals_out, &mut vecs_out);
    assert!((vals_out[0].re - 1.0).abs() < 1e-6);
    assert!((vals_out[0].im - 0.5).abs() < 1e-6);
}

#[test]
fn lapack_eig_scalar_f64_round_trip() {
    let (vals, vecs) = f64::eig_buffer_sizes(2);
    assert_eq!(vals, 4);
    assert_eq!(vecs, 8);

    let mut vals_out = [Complex64::new(0.0, 0.0); 2];
    let mut vecs_out = [Complex64::new(0.0, 0.0); 4];
    f64::eig_ri_to_complex(
        2,
        &[1.0, 0.5, -2.0, 1.25],
        &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        &mut vals_out,
        &mut vecs_out,
    );
    assert_eq!(
        vals_out,
        [Complex64::new(1.0, 0.5), Complex64::new(-2.0, 1.25)]
    );
}

#[test]
fn lapack_eig_scalar_complex64_passthrough() {
    let (vals, vecs) = Complex64::eig_buffer_sizes(2);
    assert_eq!(vals, 2);
    assert_eq!(vecs, 4);

    let c = |re, im| Complex64::new(re, im);
    let val_ri = [c(1.0, 0.5), c(2.0, -0.5)];
    let vec_ri = [c(1.0, 0.0), c(0.0, 0.0), c(0.0, 1.0), c(1.0, 0.0)];
    let mut vals_out = [Complex64::new(0.0, 0.0); 2];
    let mut vecs_out = [Complex64::new(0.0, 0.0); 4];
    Complex64::eig_ri_to_complex(2, &val_ri, &vec_ri, &mut vals_out, &mut vecs_out);
    assert_eq!(vals_out, val_ri);
    assert_eq!(vecs_out, vec_ri);
}

#[test]
fn lapack_eig_scalar_complex32_passthrough() {
    let (vals, vecs) = Complex32::eig_buffer_sizes(2);
    assert_eq!(vals, 2);
    assert_eq!(vecs, 4);

    let c = |re, im| Complex32::new(re, im);
    let val_ri = [c(1.0, 0.5), c(2.0, -0.5)];
    let vec_ri = [c(1.0, 0.0), c(0.0, 0.0), c(0.0, 1.0), c(1.0, 0.0)];
    let mut vals_out = [Complex32::new(0.0, 0.0); 2];
    let mut vecs_out = [Complex32::new(0.0, 0.0); 4];
    Complex32::eig_ri_to_complex(2, &val_ri, &vec_ri, &mut vals_out, &mut vecs_out);
    assert_eq!(vals_out, val_ri);
    assert_eq!(vecs_out, vec_ri);
}
