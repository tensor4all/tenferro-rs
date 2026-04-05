#[cfg(feature = "cpu-faer")]
use super::faer_gemm::FaerGemm;

#[cfg(feature = "cpu-faer")]
#[test]
fn faer_strided_gemm_accumulates_with_nontrivial_beta() {
    let a = [1.0, 0.0, 0.0, 1.0];
    let b = [10.0, 20.0, 30.0, 40.0];
    let mut c = [1.0, 2.0, 3.0, 4.0];

    unsafe {
        <f64 as FaerGemm>::strided_gemm(
            1.0,
            a.as_ptr(),
            2,
            2,
            1,
            2,
            b.as_ptr(),
            2,
            1,
            2,
            2.0,
            c.as_mut_ptr(),
            1,
            2,
        );
    }

    assert_eq!(c, [12.0, 24.0, 36.0, 48.0]);
}

#[cfg(feature = "cpu-faer")]
#[test]
fn faer_strided_gemm_accumulates_with_unit_beta_without_prescaling() {
    let a = [1.0, 0.0, 0.0, 1.0];
    let b = [10.0, 20.0, 30.0, 40.0];
    let mut c = [1.0, 2.0, 3.0, 4.0];

    unsafe {
        <f64 as FaerGemm>::strided_gemm(
            1.0,
            a.as_ptr(),
            2,
            2,
            1,
            2,
            b.as_ptr(),
            2,
            1,
            2,
            1.0,
            c.as_mut_ptr(),
            1,
            2,
        );
    }

    assert_eq!(c, [11.0, 22.0, 33.0, 44.0]);
}
