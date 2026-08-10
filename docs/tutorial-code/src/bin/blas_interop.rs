//! Downstream BLAS/LAPACK interoperability (issue #1602).
//!
//! A downstream application can keep its tensors in tenferro and call
//! cblas-sys / lapack entry points directly on tenferro-owned host storage
//! when tenferro's operation APIs do not cover the needed routine. This
//! binary is the executable source of truth for the BLAS/LAPACK section of
//! `docs/guides/external-linalg-interop.md`.
//!
//! Provider contract:
//! - Enable `cpu-blas` plus exactly one provider feature (`blas-openblas`,
//!   `blas-accelerate`, or `blas-mkl`) on tenferro crates, or link a native
//!   provider yourself (CI links the system OpenBLAS through `RUSTFLAGS`).
//! - cblas-sys and lapack use the LP64 convention: dimensions and leading
//!   dimensions are 32-bit (`c_int`). They are column-major like tenferro, so
//!   the leading dimension of an `m x n` matrix is `m`.
//! - Provider threads are process-wide environment variables
//!   (`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, ...); direct calls do not
//!   join tenferro's CPU execution domain.

use cblas_sys::{CBLAS_LAYOUT, CBLAS_TRANSPOSE};
use tenferro_tensor::TypedTensor;

// The from-source OpenBLAS provider builds the native library through its
// build script. The provider rlib is intentionally empty, so it must be
// referenced for cargo to keep it in the final link and forward the native
// library metadata; tenferro-cpu does the same for blas-src/lapack-src.
#[cfg(feature = "blas-openblas")]
#[allow(unused_extern_crates)]
extern crate openblas_src;

fn assert_slice_close(actual: &[f64], expected: &[f64], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: length mismatch");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual - expected).abs();
        assert!(
            error < 1.0e-12,
            "{context}[{index}]: actual={actual}, expected={expected}, error={error}"
        );
    }
}

/// Convert a shape extent to the LP64 32-bit dimension used by BLAS/LAPACK.
fn dim_i32(value: usize, what: &str) -> i32 {
    i32::try_from(value)
        .unwrap_or_else(|_| panic!("{what}={value} exceeds the LP64 32-bit dimension limit"))
}

// snippet-start:blas-dgemm
fn blas_dgemm_on_host_storage() -> tenferro_tensor::Result<()> {
    // A (2x3) and B (3x2) are compact column-major host tensors. For
    // CblasColMajor, the leading dimension of an m x n matrix is m.
    let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0])?;
    let b =
        TypedTensor::<f64>::from_vec_col_major(vec![3, 2], vec![7.0, 9.0, 11.0, 8.0, 10.0, 12.0])?;
    let mut c = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![0.0; 4])?;
    assert!(a.is_col_major_contiguous()? && b.is_col_major_contiguous()?);
    assert!(c.is_col_major_contiguous()?);

    let m = dim_i32(a.shape()[0], "m");
    let k = dim_i32(a.shape()[1], "k");
    let n = dim_i32(b.shape()[1], "n");
    let lda = m; // A is m x k, column-major
    let ldb = k; // B is k x n, column-major
    let ldc = m; // C is m x n, column-major

    // Snapshot the inputs so we can prove the call does not mutate them.
    let a_before = a.as_slice()?.to_vec();
    let b_before = b.as_slice()?.to_vec();

    // SAFETY: a, b, c are live for the duration of the call, their slices are
    // at least m*k, k*n, m*n elements, and the leading dimensions match the
    // column-major layouts verified above. C is the only mutated buffer.
    unsafe {
        cblas_sys::cblas_dgemm(
            CBLAS_LAYOUT::CblasColMajor,
            CBLAS_TRANSPOSE::CblasNoTrans,
            CBLAS_TRANSPOSE::CblasNoTrans,
            m,
            n,
            k,
            1.0,
            a.as_slice()?.as_ptr(),
            lda,
            b.as_slice()?.as_ptr(),
            ldb,
            0.0,
            c.host_data_mut()?.as_mut_ptr(),
            ldc,
        );
    }

    // Inputs are untouched; the output holds A * B in tenferro's storage.
    assert_slice_close(a.as_slice()?, &a_before, "A unchanged");
    assert_slice_close(b.as_slice()?, &b_before, "B unchanged");
    // C = [[58,64],[139,154]] stored column-major.
    assert_slice_close(c.as_slice()?, &[58.0, 139.0, 64.0, 154.0], "C=A*B");
    Ok(())
}
// snippet-end:blas-dgemm

// snippet-start:blas-potrf
fn lapack_potrf_on_host_storage() -> tenferro_tensor::Result<()> {
    // A is a 2x2 SPD matrix, column-major [3,1,1,2]. dpotrf factors in place:
    // the lower triangle of the buffer is replaced by L with A = L L^T.
    let mut a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![3.0, 1.0, 1.0, 2.0])?;
    let original = a.as_slice()?.to_vec();
    let n = dim_i32(a.shape()[0], "n");
    let lda = n;
    let mut info: i32 = 0;

    // SAFETY: a holds a live n*n column-major buffer and info is a live output.
    unsafe {
        lapack::dpotrf(b'L', n, a.host_data_mut()?, lda, &mut info);
    }
    assert_eq!(info, 0, "dpotrf failed with info={info}");

    // Reconstruct A = L L^T from the factor (L is unit-free, stored in the
    // lower triangle; the upper triangle is left untouched).
    let data = a.as_slice()?;
    let (l00, l10, l11) = (data[0], data[1], data[3]);
    assert_slice_close(
        &[l00 * l00, l00 * l10, l00 * l10, l10 * l10 + l11 * l11],
        &original,
        "L L^T",
    );
    Ok(())
}
// snippet-end:blas-potrf

fn main() -> Result<(), Box<dyn std::error::Error>> {
    blas_dgemm_on_host_storage()?;
    lapack_potrf_on_host_storage()?;
    println!("blas_interop: all checks passed");
    Ok(())
}
