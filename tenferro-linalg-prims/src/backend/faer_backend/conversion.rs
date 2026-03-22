use num_complex::{Complex32, Complex64};

pub(crate) fn to_faer_c64(src: &[Complex64]) -> Vec<faer::c64> {
    src.iter().map(|c| faer::c64::new(c.re, c.im)).collect()
}

pub(crate) fn from_faer_c64_mat(
    mat: faer::MatRef<'_, faer::c64>,
    out: &mut [Complex64],
    rows: usize,
    cols: usize,
) {
    for j in 0..cols {
        for i in 0..rows {
            let v = mat[(i, j)];
            out[i + j * rows] = Complex64::new(v.re, v.im);
        }
    }
}

pub(crate) fn to_faer_c32(src: &[Complex32]) -> Vec<faer::c32> {
    src.iter().map(|c| faer::c32::new(c.re, c.im)).collect()
}

pub(crate) fn from_faer_c32_mat(
    mat: faer::MatRef<'_, faer::c32>,
    out: &mut [Complex32],
    rows: usize,
    cols: usize,
) {
    for j in 0..cols {
        for i in 0..rows {
            let v = mat[(i, j)];
            out[i + j * rows] = Complex32::new(v.re, v.im);
        }
    }
}
