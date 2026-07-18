use super::{
    analyse_gemm_cached, canonical_gemm_layout, checked_batch_offset, checked_product,
    try_fuse_dims, GemmAnalysisCache, GemmAnalysisCacheKind,
};

#[cfg(any(feature = "blas-openblas", feature = "blas-mkl"))]
use super::blas_gemm::provider_should_use_gemm_batch;
#[cfg(feature = "cpu-blas")]
use super::blas_gemm::BlasGemm;
#[cfg(any(feature = "blas-openblas", feature = "blas-mkl"))]
use super::blas_gemm::BlasGemmBatch;
#[cfg(feature = "cpu-blas")]
use super::dot_general_blas_cached;
#[cfg(feature = "cpu-faer")]
use super::faer_gemm::FaerGemm;
#[cfg(feature = "cpu-faer")]
use super::{dot_general_faer_read_cached, strided_dot};
#[cfg(any(feature = "cpu-blas", feature = "cpu-faer"))]
use crate::buffer_pool::BufferPool;
#[cfg(feature = "cpu-faer")]
use crate::CpuContext;
#[cfg(feature = "cpu-blas")]
use num_complex::Complex64;
use tenferro_tensor::RuntimeCacheControl;
use tenferro_tensor::{DotGeneralConfig, TypedTensor};
#[cfg(feature = "cpu-faer")]
use tenferro_tensor::{Tensor, TensorRead, TensorView};

#[test]
fn try_fuse_dims_reversed_strides() {
    assert_eq!(try_fuse_dims(&[3, 4], &[4, 1]).unwrap(), Some((12, 1)));
}

#[test]
fn try_fuse_dims_sorted_strides_unchanged() {
    assert_eq!(try_fuse_dims(&[3, 4], &[1, 3]).unwrap(), Some((12, 1)));
}

#[test]
fn try_fuse_dims_non_adjacent_fails() {
    assert_eq!(try_fuse_dims(&[3, 2], &[1, 6]).unwrap(), None);
}

#[test]
fn try_fuse_dims_single_dim() {
    assert_eq!(try_fuse_dims(&[5], &[3]).unwrap(), Some((5, 3)));
}

#[test]
fn try_fuse_dims_empty() {
    assert_eq!(try_fuse_dims(&[], &[]).unwrap(), Some((1, 0)));
}

#[test]
fn try_fuse_dims_rejects_extent_that_does_not_fit_isize() {
    let too_large = (isize::MAX as usize).saturating_add(1);

    let err = try_fuse_dims(&[too_large], &[1]).unwrap_err();
    assert!(
        err.to_string().contains("isize"),
        "expected isize range error, got {err:?}"
    );
}

#[test]
fn try_fuse_dims_rejects_fused_stride_overflow() {
    let err = try_fuse_dims(&[isize::MAX as usize, 2], &[1, isize::MAX]).unwrap_err();
    assert!(
        err.to_string().contains("overflows"),
        "expected stride overflow error, got {err:?}"
    );
}

#[test]
fn checked_batch_offset_reports_batch_conversion_overflow() {
    let too_large = (isize::MAX as usize).saturating_add(1);
    let err = checked_batch_offset(too_large, 1).unwrap_err();
    assert!(
        err.to_string().contains("batch index"),
        "expected batch index range error, got {err:?}"
    );
}

#[test]
fn checked_product_rejects_product_overflow() {
    assert_eq!(checked_product(&[usize::MAX, 2]), None);
}

#[test]
fn strided_dot_into_does_not_acquire_output_buffer() {
    let source = include_str!("strided_dot.rs");
    let start = source
        .find("pub(crate) fn dot_general_strided_with_backend_into")
        .expect("missing strided dot into function");
    let body = &source[start..];

    assert!(
        body.contains("StridedViewMut::new(out_data"),
        "dot_general into should wrap caller-provided output storage"
    );
    assert!(
        !body.contains("pool_acquire"),
        "dot_general into must not acquire a pooled output Vec"
    );
}

#[test]
fn gemm_analysis_cache_keeps_direct_and_canonical_candidates_separate() {
    let lhs = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![0.0; 6]).unwrap();
    let rhs = TypedTensor::<f64>::from_vec_col_major(vec![3, 2], vec![0.0; 6]).unwrap();
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
        canonical_gemm_layout(&config, lhs.shape().len(), rhs.shape().len());
    assert_eq!(rhs_perm.as_slice(), &[1, 0]);
    let rhs_canonical = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![0.0; 6]).unwrap();
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

#[test]
fn canonical_gemm_layout_remains_behind_dot_general_validation() {
    let lhs = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![0.0; 6]).unwrap();
    let rhs = TypedTensor::<f64>::from_vec_col_major(vec![3, 2], vec![0.0; 6]).unwrap();
    let invalid = DotGeneralConfig {
        lhs_contracting_dims: vec![2],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let mut cache = GemmAnalysisCache::default();

    assert!(
        analyse_gemm_cached(
            &mut cache,
            None,
            GemmAnalysisCacheKind::Canonical,
            &lhs,
            &rhs,
            &invalid,
        )
        .is_err(),
        "canonical GEMM analysis must validate configs before canonicalizing layouts"
    );
}

#[test]
fn gemm_analysis_cache_reuses_matching_direct_plan_and_reports_stats() {
    let lhs = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![0.0; 6]).unwrap();
    let rhs = TypedTensor::<f64>::from_vec_col_major(vec![3, 2], vec![0.0; 6]).unwrap();
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let mut cache = GemmAnalysisCache::default();

    let first = analyse_gemm_cached(
        &mut cache,
        Some(3),
        GemmAnalysisCacheKind::Direct,
        &lhs,
        &rhs,
        &config,
    )
    .expect("first analysis should validate")
    .expect("first analysis should be representable");
    assert_eq!((first.m, first.n, first.k, first.batch_total), (2, 2, 3, 1));

    let cached = analyse_gemm_cached(
        &mut cache,
        Some(3),
        GemmAnalysisCacheKind::Direct,
        &lhs,
        &rhs,
        &config,
    )
    .expect("cached analysis should validate")
    .expect("cached analysis should be present");
    assert_eq!(
        (cached.m, cached.n, cached.k, cached.batch_total),
        (2, 2, 3, 1)
    );

    let stats = cache.stats();
    assert_eq!(stats.entries, 1);
    assert!(stats.retained_bytes > 0);
}

#[test]
fn gemm_analysis_cache_matches_view_layouts_before_reusing_a_plan() {
    let lhs = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![0.0; 6]).unwrap();
    let rhs = TypedTensor::<f64>::from_vec_col_major(vec![3, 2], vec![0.0; 6]).unwrap();
    let lhs_view = lhs.as_view();
    let rhs_view = rhs.as_view();
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let mut cache = GemmAnalysisCache::default();

    let first = analyse_gemm_cached(
        &mut cache,
        Some(11),
        GemmAnalysisCacheKind::Direct,
        &lhs_view,
        &rhs_view,
        &config,
    )
    .unwrap()
    .expect("view layout should be representable as GEMM");
    let cached = analyse_gemm_cached(
        &mut cache,
        Some(11),
        GemmAnalysisCacheKind::Direct,
        &lhs_view,
        &rhs_view,
        &config,
    )
    .unwrap()
    .expect("the matching view layout should reuse the cached analysis");

    assert_eq!((first.m, first.n, first.k), (2, 2, 3));
    assert_eq!((cached.m, cached.n, cached.k), (2, 2, 3));
    assert_eq!(cache.stats().entries, 1);
}

#[test]
fn gemm_analysis_cache_shrink_invalidates_entries_instead_of_truncating_by_slot() {
    let lhs = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![0.0; 6]).unwrap();
    let rhs = TypedTensor::<f64>::from_vec_col_major(vec![3, 2], vec![0.0; 6]).unwrap();
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let mut cache = GemmAnalysisCache::with_capacity(8);

    let _ = analyse_gemm_cached(
        &mut cache,
        Some(1),
        GemmAnalysisCacheKind::Direct,
        &lhs,
        &rhs,
        &config,
    )
    .expect("low-slot analysis should validate");
    let _ = analyse_gemm_cached(
        &mut cache,
        Some(7),
        GemmAnalysisCacheKind::Direct,
        &lhs,
        &rhs,
        &config,
    )
    .expect("high-slot analysis should validate");
    assert_eq!(cache.stats().entries, 2);

    cache.set_capacity(2);
    assert_eq!(cache.capacity(), 2);
    assert_eq!(
        cache.stats().entries,
        0,
        "shrinking a direct-indexed cache should not retain arbitrary low-slot entries as a fake LRU"
    );
}

#[test]
fn gemm_analysis_cache_exposes_debug_capacity_and_clear_contract() {
    let mut cache = GemmAnalysisCache::with_capacity(2);

    assert_eq!(cache.capacity(), 2);
    let debug = format!("{cache:?}");
    assert!(debug.contains("GemmAnalysisCache"));
    assert!(debug.contains("max_slots"));

    cache.set_capacity(0);
    assert_eq!(cache.capacity(), 0);
    cache.clear();
    assert_eq!(cache.stats().entries, 0);
}

#[cfg(feature = "cpu-faer")]
#[test]
fn faer_read_transposed_view_uses_strided_dot_without_materializing_input() {
    let lhs_source =
        TypedTensor::<f64>::from_vec_col_major(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
    let lhs_view = lhs_source.as_view().transpose_view([1, 0]).unwrap();
    let rhs = TypedTensor::<f64>::from_vec_col_major(
        vec![3, 2],
        vec![7.0_f64, 8.0, 9.0, 10.0, 11.0, 12.0],
    )
    .unwrap();
    let rhs = Tensor::F64(rhs);
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let mut buffers = BufferPool::new();
    let mut cache = GemmAnalysisCache::default();
    let ctx = CpuContext::with_threads(1).unwrap();

    let dispatch_count_before = strided_dot::test_dispatch_count();
    let out = dot_general_faer_read_cached(
        &mut buffers,
        &mut cache,
        Some(0),
        &ctx,
        &TensorRead::from_view(TensorView::F64(lhs_view)),
        &TensorRead::from_tensor(&rhs),
        &config,
    )
    .unwrap()
    .expect("same-dtype f64 inputs should be handled directly");

    assert!(strided_dot::test_dispatch_count() > dispatch_count_before);
    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(out.as_slice::<f64>().unwrap(), &[50.0, 122.0, 68.0, 167.0]);
}

#[cfg(feature = "cpu-blas")]
#[test]
fn blas_dot_general_contract_trailing_rhs_dim() {
    let lhs =
        TypedTensor::from_vec_col_major(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let rhs =
        TypedTensor::from_vec_col_major(vec![2, 3], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![1],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let mut buffers = BufferPool::new();
    let mut cache = GemmAnalysisCache::default();
    let out = dot_general_blas_cached(&mut buffers, &mut cache, None, &lhs, &rhs, &config)
        .expect("dot_general should succeed");

    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(out.host_data().unwrap(), &[89.0, 116.0, 98.0, 128.0]);
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

#[cfg(any(feature = "blas-openblas", feature = "blas-mkl"))]
#[test]
fn provider_gemm_batch_heuristic_keeps_medium_jobs_on_sequential_path() {
    fn batch(m: usize, n: usize, k: usize) -> BlasGemmBatch<f64> {
        BlasGemmBatch {
            a_ptr: std::ptr::null(),
            b_ptr: std::ptr::null(),
            c_ptr: std::ptr::null_mut(),
            m,
            n,
            k,
            a_rs: 1,
            a_cs: m as isize,
            b_rs: 1,
            b_cs: k as isize,
            c_rs: 1,
            c_cs: m as isize,
        }
    }

    assert!(provider_should_use_gemm_batch(&[
        batch(8, 8, 8),
        batch(8, 8, 8)
    ]));
    assert!(!provider_should_use_gemm_batch(&[batch(8, 8, 8)]));
    assert!(!provider_should_use_gemm_batch(&[
        batch(8, 8, 8),
        batch(32, 32, 32)
    ]));
}

#[cfg(feature = "cpu-faer")]
#[test]
fn faer_strided_gemm_accumulates_with_nontrivial_beta() {
    let a = [1.0, 0.0, 0.0, 1.0];
    let b = [10.0, 20.0, 30.0, 40.0];
    let mut c = [1.0, 2.0, 3.0, 4.0];
    let ctx = CpuContext::with_threads(1).unwrap();

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
    let ctx = CpuContext::with_threads(1).unwrap();

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
