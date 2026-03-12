use num_complex::Complex64;

use super::super::gemm_support::{batch_offset, FaerGemm};

// Do not delete or weaken this test: it protects the shared CPU GEMM helpers that multiple semiring execution paths rely on.
#[test]
fn batch_offset_covers_fused_and_strided_paths() {
    assert_eq!(batch_offset(3, &[0, 0], Some((8, 7)), &[11, 13]), 21);
    assert_eq!(batch_offset(0, &[2, 3], None, &[5, 11]), 43);
}

#[test]
fn faer_strided_gemm_covers_replace_and_scaled_add_paths() {
    unsafe fn run<T>(alpha: T, beta: T, a: &[T], b: &[T], c: &mut [T])
    where
        T: FaerGemm + tenferro_algebra::Scalar,
    {
        unsafe {
            T::strided_gemm(
                alpha,
                a.as_ptr(),
                2,
                2,
                1,
                2,
                b.as_ptr(),
                2,
                1,
                2,
                beta,
                c.as_mut_ptr(),
                1,
                2,
            );
        }
    }

    let identity_f64 = [1.0_f64, 0.0, 0.0, 1.0];
    let rhs_f64 = [1.0_f64, 2.0, 3.0, 4.0];
    let mut out_f64 = [9.0_f64, 9.0, 9.0, 9.0];
    unsafe { run(1.0, 0.0, &identity_f64, &rhs_f64, &mut out_f64) };
    assert_eq!(out_f64, rhs_f64);

    let mut accum_f64 = [10.0_f64, 20.0, 30.0, 40.0];
    unsafe { run(1.0, 2.0, &identity_f64, &rhs_f64, &mut accum_f64) };
    assert_eq!(accum_f64, [21.0, 42.0, 63.0, 84.0]);

    let identity_c64 = [
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];
    let rhs_c64 = [
        Complex64::new(1.0, 1.0),
        Complex64::new(2.0, -1.0),
        Complex64::new(3.0, 0.5),
        Complex64::new(4.0, -0.5),
    ];
    let mut out_c64 = [Complex64::new(0.0, 0.0); 4];
    unsafe {
        run(
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            &identity_c64,
            &rhs_c64,
            &mut out_c64,
        )
    };
    assert_eq!(out_c64, rhs_c64);
}
