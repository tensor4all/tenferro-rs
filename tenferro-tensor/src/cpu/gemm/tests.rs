use super::{
    analyse_gemm_cached, canonical_gemm_layout, checked_product, try_fuse_dims, GemmAnalysisCache,
    GemmAnalysisCacheKind,
};

#[cfg(feature = "cpu-blas")]
use super::blas_gemm::BlasGemm;
#[cfg(feature = "cpu-blas")]
use super::dot_general;
#[cfg(feature = "cpu-faer")]
use super::faer_gemm::FaerGemm;
#[cfg(feature = "cpu-blas")]
use crate::buffer_pool::BufferPool;
use crate::config::DotGeneralConfig;
#[cfg(feature = "cpu-faer")]
use crate::cpu::CpuContext;
use crate::types::TypedTensor;
#[cfg(feature = "cpu-blas")]
use num_complex::Complex64;

#[test]
fn try_fuse_dims_reversed_strides() {
    assert_eq!(try_fuse_dims(&[3, 4], &[4, 1]), Some((12, 1)));
}

#[test]
fn try_fuse_dims_sorted_strides_unchanged() {
    assert_eq!(try_fuse_dims(&[3, 4], &[1, 3]), Some((12, 1)));
}

#[test]
fn try_fuse_dims_non_adjacent_fails() {
    assert_eq!(try_fuse_dims(&[3, 2], &[1, 6]), None);
}

#[test]
fn try_fuse_dims_single_dim() {
    assert_eq!(try_fuse_dims(&[5], &[3]), Some((5, 3)));
}

#[test]
fn try_fuse_dims_empty() {
    assert_eq!(try_fuse_dims(&[], &[]), Some((1, 0)));
}

#[test]
fn try_fuse_dims_rejects_extent_that_does_not_fit_isize() {
    let too_large = (isize::MAX as usize).saturating_add(1);

    assert_eq!(try_fuse_dims(&[too_large], &[1]), None);
}

#[test]
fn try_fuse_dims_rejects_fused_stride_overflow() {
    assert_eq!(
        try_fuse_dims(&[isize::MAX as usize, 2], &[1, isize::MAX]),
        None
    );
}

#[test]
fn checked_product_rejects_product_overflow() {
    assert_eq!(checked_product(&[usize::MAX, 2]), None);
}

#[test]
fn gemm_analysis_cache_keeps_direct_and_canonical_candidates_separate() {
    let lhs = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![0.0; 6]);
    let rhs = TypedTensor::<f64>::from_vec_col_major(vec![3, 2], vec![0.0; 6]);
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![0, 1],
        rhs_contracting_dims: vec![1, 0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let mut cache = GemmAnalysisCache::default();

    let direct = analyse_gemm_cached(
        &mut cache,
        Some(7),
        GemmAnalysisCacheKind::Direct,
        &lhs,
        &rhs,
        &config,
    )
    .expect("direct analysis should validate");
    assert!(direct.is_none());

    let (_lhs_perm, rhs_perm, canonical_config) =
        canonical_gemm_layout(&config, lhs.shape.len(), rhs.shape.len());
    assert_eq!(rhs_perm.as_slice(), &[1, 0]);
    let rhs_canonical = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![0.0; 6]);
    let canonical = analyse_gemm_cached(
        &mut cache,
        Some(7),
        GemmAnalysisCacheKind::Canonical,
        &lhs,
        &rhs_canonical,
        &canonical_config,
    )
    .expect("canonical analysis should validate");
    assert!(canonical.is_some());

    let slot = &cache.slots[7];
    assert!(slot.direct.as_ref().is_some_and(|plan| plan.dims.is_none()));
    assert!(slot
        .canonical
        .as_ref()
        .is_some_and(|plan| plan.dims.is_some()));
}

#[cfg(feature = "cpu-blas")]
#[test]
fn blas_dot_general_contract_trailing_rhs_dim() {
    let lhs = TypedTensor::from_vec_col_major(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let rhs = TypedTensor::from_vec_col_major(vec![2, 3], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![1],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let mut buffers = BufferPool::new();
    let mut cache = GemmAnalysisCache::default();
    let out = dot_general(&mut buffers, &mut cache, &lhs, &rhs, &config)
        .expect("dot_general should succeed");

    assert_eq!(out.shape, vec![2, 2]);
    assert_eq!(out.host_data(), &[89.0, 116.0, 98.0, 128.0]);
}

#[cfg(feature = "cpu-blas")]
#[test]
fn blas_complex_conj_trans_executes_without_materializing_transposed_operand() {
    let a = [
        Complex64::new(1.0, 2.0),
        Complex64::new(-2.0, 0.5),
        Complex64::new(0.25, -3.0),
        Complex64::new(4.0, -1.0),
        Complex64::new(-0.5, 2.5),
        Complex64::new(3.0, 0.75),
    ];
    let b = [
        Complex64::new(2.0, -1.0),
        Complex64::new(0.5, 3.0),
        Complex64::new(-1.0, 0.25),
        Complex64::new(1.5, 0.5),
        Complex64::new(-2.5, -1.5),
        Complex64::new(0.75, 2.0),
    ];
    let mut c = [Complex64::new(0.0, 0.0); 4];

    let executed = unsafe {
        <Complex64 as BlasGemm>::strided_gemm_with_conj(
            Complex64::new(1.0, 0.0),
            a.as_ptr(),
            2,
            3,
            3,
            1,
            true,
            b.as_ptr(),
            2,
            1,
            3,
            false,
            Complex64::new(0.0, 0.0),
            c.as_mut_ptr(),
            1,
            2,
        )
        .expect("BLAS conj-trans GEMM should succeed")
    };

    assert!(executed);
    let mut expected = [Complex64::new(0.0, 0.0); 4];
    for col in 0..2 {
        for row in 0..2 {
            let mut acc = Complex64::new(0.0, 0.0);
            for p in 0..3 {
                let lhs = a[p + row * 3].conj();
                let rhs = b[p + col * 3];
                acc += lhs * rhs;
            }
            expected[row + col * 2] = acc;
        }
    }
    for (got, want) in c.iter().zip(expected.iter()) {
        assert!((got - want).norm() < 1.0e-12, "got {got}, want {want}");
    }
}

#[cfg(feature = "cpu-blas")]
#[test]
fn blas_complex_conj_no_trans_reports_materialization_needed() {
    let a = [Complex64::new(1.0, 2.0); 6];
    let b = [Complex64::new(3.0, 4.0); 6];
    let mut c = [Complex64::new(0.0, 0.0); 4];

    let executed = unsafe {
        <Complex64 as BlasGemm>::strided_gemm_with_conj(
            Complex64::new(1.0, 0.0),
            a.as_ptr(),
            2,
            3,
            1,
            2,
            true,
            b.as_ptr(),
            2,
            1,
            3,
            false,
            Complex64::new(0.0, 0.0),
            c.as_mut_ptr(),
            1,
            2,
        )
        .expect("layout probe should not fail")
    };

    assert!(!executed);
    assert_eq!(c, [Complex64::new(0.0, 0.0); 4]);
}

#[cfg(feature = "cpu-faer")]
#[test]
fn faer_strided_gemm_accumulates_with_nontrivial_beta() {
    let a = [1.0, 0.0, 0.0, 1.0];
    let b = [10.0, 20.0, 30.0, 40.0];
    let mut c = [1.0, 2.0, 3.0, 4.0];
    let ctx = CpuContext::with_threads(1);

    unsafe {
        <f64 as FaerGemm>::strided_gemm(
            &ctx,
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
    let ctx = CpuContext::with_threads(1);

    unsafe {
        <f64 as FaerGemm>::strided_gemm(
            &ctx,
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

#[cfg(feature = "cpu-faer")]
#[test]
fn faer_singleton_strides_are_normalized_before_raw_gemm() {
    assert_eq!(super::normalize_singleton_stride(0, 1, 4), 4);
    assert_eq!(super::normalize_singleton_stride(0, 3, 4), 0);
}
