//! Case-detection unit tests for the cuBLAS batched-GEMM fast path.
//!
//! These run without a device: they only exercise the shape/config predicate
//! that decides between one `gemmStridedBatchedEx` and the general cuTENSOR
//! contraction. Device parity between the paths lives in
//! `crate::cubecl::tests::gemm_tests`.

use cudarc::cublas::sys as cublas;

use super::BatchedGemmLaunch;
use crate::config::DotGeneralConfig;

fn config(
    lhs_contracting: &[usize],
    rhs_contracting: &[usize],
    batch: &[usize],
) -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: lhs_contracting.into(),
        rhs_contracting_dims: rhs_contracting.into(),
        lhs_batch_dims: batch.into(),
        rhs_batch_dims: batch.into(),
    }
}

fn detect(
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    lhs_contracting: &[usize],
    rhs_contracting: &[usize],
    batch: &[usize],
) -> Option<BatchedGemmLaunch> {
    BatchedGemmLaunch::detect(
        lhs_shape,
        rhs_shape,
        &config(lhs_contracting, rhs_contracting, batch),
    )
}

fn transposed(launch: &BatchedGemmLaunch) -> (bool, bool) {
    (
        launch.trans_a == cublas::cublasOperation_t::CUBLAS_OP_T,
        launch.trans_b == cublas::cublasOperation_t::CUBLAS_OP_T,
    )
}

#[test]
fn the_benchmark_shape_is_one_strided_batched_gemm() {
    // 16 batches of 32x32, the shape of `dense_batched_matmul_f64_b16_32`.
    let launch = detect(&[32, 32, 16], &[32, 32, 16], &[1], &[0], &[2])
        .expect("standard batched matmul must take the cuBLAS path");
    assert_eq!((launch.m, launch.n, launch.k), (32, 32, 32));
    assert_eq!(launch.batch_count, 16);
    assert_eq!((launch.lda, launch.ldb, launch.ldc), (32, 32, 32));
    assert_eq!(
        (launch.stride_a, launch.stride_b, launch.stride_c),
        (32 * 32, 32 * 32, 32 * 32)
    );
    assert_eq!(transposed(&launch), (false, false));
    assert!(launch.use_cublaslt());
}

#[test]
fn the_small_batched_threshold_selects_cublaslt() {
    // Every dimension at the bound stays on cuBLASLt.
    let at_bound = detect(&[64, 64, 16], &[64, 64, 16], &[1], &[0], &[2]).expect("64 cube");
    assert!(at_bound.use_cublaslt());
    // One extent above the bound keeps the cuBLAS strided-batched path.
    let above = detect(&[65, 64, 16], &[64, 64, 16], &[1], &[0], &[2]).expect("65 rows");
    assert!(!above.use_cublaslt());
    let above_k = detect(&[64, 65, 16], &[65, 64, 16], &[1], &[0], &[2]).expect("65 inner");
    assert!(!above_k.use_cublaslt());
}

#[test]
fn rectangular_slices_keep_the_column_major_leading_dimensions() {
    // lhs 5x3x3, rhs 3x7x3: lda/ldb are the stored row counts (5 and 3) and
    // the batch strides are the per-slice element counts.
    let launch = detect(&[5, 3, 3], &[3, 7, 3], &[1], &[0], &[2]).expect("rectangular batched");
    assert_eq!((launch.m, launch.n, launch.k), (5, 7, 3));
    assert_eq!((launch.lda, launch.ldb, launch.ldc), (5, 3, 5));
    assert_eq!(
        (launch.stride_a, launch.stride_b, launch.stride_c),
        (15, 21, 35)
    );
    assert_eq!(transposed(&launch), (false, false));
}

#[test]
fn both_transpose_combinations_are_expressed_with_cublas_flags() {
    // lhs stored as k x m and contracted on axis 0.
    let lhs_transposed = detect(&[3, 5, 3], &[3, 7, 3], &[0], &[0], &[2]).expect("opA = T");
    assert_eq!(
        (lhs_transposed.m, lhs_transposed.n, lhs_transposed.k),
        (5, 7, 3)
    );
    assert_eq!(lhs_transposed.lda, 3, "stored matrix is k x m, so lda = k");
    assert_eq!(transposed(&lhs_transposed), (true, false));

    // rhs stored as n x k and contracted on axis 1.
    let rhs_transposed = detect(&[5, 3, 3], &[7, 3, 3], &[1], &[1], &[2]).expect("opB = T");
    assert_eq!(rhs_transposed.ldb, 7, "stored matrix is n x k, so ldb = n");
    assert_eq!(transposed(&rhs_transposed), (false, true));

    let both = detect(&[3, 5, 3], &[7, 3, 3], &[0], &[1], &[2]).expect("opA = T, opB = T");
    assert_eq!(transposed(&both), (true, true));
}

#[test]
fn several_batch_axes_flatten_into_one_batch_stride() {
    // Column-major flattening of trailing axes makes the batch offset linear.
    let launch = detect(&[4, 3, 2, 3], &[3, 5, 2, 3], &[1], &[0], &[2, 3]).expect("two batch axes");
    assert_eq!(launch.batch_count, 6);
    assert_eq!(
        (launch.stride_a, launch.stride_b, launch.stride_c),
        (12, 15, 20)
    );
}

#[test]
fn a_single_matrix_keeps_the_cutensor_path() {
    assert!(detect(&[32, 32], &[32, 32], &[1], &[0], &[]).is_none());
    assert!(detect(&[32, 32, 1], &[32, 32, 1], &[1], &[0], &[2]).is_none());
}

#[test]
fn non_standard_forms_fall_back() {
    // Two contraction axes.
    assert!(detect(&[4, 3, 2], &[3, 4, 2], &[1, 2], &[0, 1], &[]).is_none());
    // A batch axis inside the matrix axes: the batch index interleaves into the
    // slice, which a fixed batch stride cannot address.
    assert!(detect(&[6, 4, 3], &[4, 5, 3], &[1], &[0], &[0]).is_none());
    // Batch axes not trailing in ascending order.
    assert!(detect(&[4, 3, 2, 3], &[3, 5, 2, 3], &[1], &[0], &[2, 0]).is_none());
    // Batch axes paired in a different order on the two sides.
    let crossed = DotGeneralConfig {
        lhs_contracting_dims: [1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: [2].as_slice().into(),
        rhs_batch_dims: [3].as_slice().into(),
    };
    assert!(BatchedGemmLaunch::detect(&[4, 3, 2, 3], &[3, 5, 2, 3], &crossed).is_none());
    // Rank mismatch between the operands.
    assert!(detect(&[4, 3, 2], &[3, 5], &[1], &[0], &[2]).is_none());
    // An operand with fewer than two axes has no matrix pair at all.
    assert!(detect(&[3], &[3], &[0], &[0], &[]).is_none());
    // Empty contraction.
    assert!(detect(&[4, 0, 2], &[0, 5, 2], &[1], &[0], &[2]).is_none());
}

#[test]
fn sizes_outside_the_portable_cublas_parameters_fall_back() {
    // `m` does not fit the `c_int` the entry takes, so the case keeps cuTENSOR
    // instead of reporting a size error of its own.
    let too_wide = i32::MAX as usize + 2;
    assert!(detect(&[2, too_wide, 2], &[2, 3, 2], &[0], &[0], &[2]).is_none());
}
