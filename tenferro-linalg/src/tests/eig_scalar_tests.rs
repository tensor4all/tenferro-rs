use super::*;
use num_complex::{Complex32, Complex64};

#[test]
fn eig_buffer_sizes_f32() {
    let (vals, vecs) = f32::eig_buffer_sizes(3);
    assert_eq!(vals, 6);
    assert_eq!(vecs, 18);
}

#[test]
fn eig_ri_to_complex_f32() {
    let val_ri = [1.0_f32, 0.5, 2.0, -0.5];
    let vec_ri = [1.0_f32, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
    let mut vals = [Complex32::new(0.0, 0.0); 2];
    let mut vecs = [Complex32::new(0.0, 0.0); 4];
    f32::eig_ri_to_complex(2, &val_ri, &vec_ri, &mut vals, &mut vecs);
    assert!((vals[0].re - 1.0).abs() < 1e-6);
    assert!((vals[0].im - 0.5).abs() < 1e-6);
}

#[test]
fn eig_buffer_sizes_complex64() {
    let (vals, vecs) = Complex64::eig_buffer_sizes(3);
    assert_eq!(vals, 3);
    assert_eq!(vecs, 9);
}

#[test]
fn eig_ri_to_complex_complex64() {
    let c = |re, im| Complex64::new(re, im);
    let val_ri = [c(1.0, 0.5), c(2.0, -0.5)];
    let vec_ri = [c(1.0, 0.0), c(0.0, 0.0), c(0.0, 1.0), c(1.0, 0.0)];
    let mut vals = [Complex64::new(0.0, 0.0); 2];
    let mut vecs = [Complex64::new(0.0, 0.0); 4];
    Complex64::eig_ri_to_complex(2, &val_ri, &vec_ri, &mut vals, &mut vecs);
    assert!((vals[0].re - 1.0).abs() < 1e-12);
    assert!((vals[1].im + 0.5).abs() < 1e-12);
}

#[test]
fn eig_buffer_sizes_complex32() {
    let (vals, vecs) = Complex32::eig_buffer_sizes(2);
    assert_eq!(vals, 2);
    assert_eq!(vecs, 4);
}

#[test]
fn eig_ri_to_complex_complex32() {
    let c = |re, im| Complex32::new(re, im);
    let val_ri = [c(1.0, 0.5), c(2.0, -0.5)];
    let vec_ri = [c(1.0, 0.0), c(0.0, 0.0), c(0.0, 1.0), c(1.0, 0.0)];
    let mut vals = [Complex32::new(0.0, 0.0); 2];
    let mut vecs = [Complex32::new(0.0, 0.0); 4];
    Complex32::eig_ri_to_complex(2, &val_ri, &vec_ri, &mut vals, &mut vecs);
    assert!((vals[0].re - 1.0).abs() < 1e-6);
}
