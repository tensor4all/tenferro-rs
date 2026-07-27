use std::sync::Arc;

use tenferro_cpu::provider::CpuGeneralContractionProvider;
use tenferro_cpu::{CpuBackend, CpuBackendKind, CpuProviderBundle};
use tenferro_cpu_tblis::TblisGeneralContractionProvider;
use tenferro_tensor::{DotGeneralConfig, Tensor, TensorDot, TensorElementwise};

fn backend_with_tblis_preferred() -> CpuBackend {
    let bundle = CpuProviderBundle::builder(CpuBackendKind::default_compiled())
        .prefer_general_contraction_provider(Arc::new(TblisGeneralContractionProvider::new()))
        .build()
        .unwrap();
    CpuBackend::new().with_provider_bundle(bundle).unwrap()
}

fn backend_with_tblis_required() -> CpuBackend {
    let bundle = CpuProviderBundle::builder(CpuBackendKind::default_compiled())
        .require_general_contraction_provider(Arc::new(TblisGeneralContractionProvider::new()))
        .build()
        .unwrap();
    CpuBackend::new().with_provider_bundle(bundle).unwrap()
}

#[test]
fn tblis_provider_is_object_safe() {
    let provider: &dyn CpuGeneralContractionProvider = &TblisGeneralContractionProvider::new();
    let capabilities = provider.execution_capabilities();

    assert!(capabilities.worker_local_sequential);
}

#[test]
fn preferred_provider_falls_back_for_scalar_output_inner_product() {
    let mut backend = backend_with_tblis_preferred();
    let lhs = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![3], vec![4.0_f64, 5.0, 6.0]).unwrap();
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![0],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };

    let out = backend.dot_general(&lhs, &rhs, &config).unwrap();

    assert_eq!(out.shape(), &[] as &[usize]);
    assert_eq!(out.as_slice::<f64>().unwrap(), &[32.0]);
}

#[test]
fn preferred_provider_leaves_non_contractions_on_default_backend() {
    let mut backend = backend_with_tblis_preferred();
    let lhs = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap();

    let out = backend.add(&lhs, &rhs).unwrap();

    assert_eq!(out.as_slice::<f64>().unwrap(), &[4.0, 6.0]);
}

#[test]
fn required_provider_reports_unsupported_without_fallback() {
    let mut backend = backend_with_tblis_required();
    let lhs = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![3], vec![4.0_f64, 5.0, 6.0]).unwrap();
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![0],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };

    let error = backend.dot_general(&lhs, &rhs, &config).unwrap_err();

    assert!(error
        .to_string()
        .contains("configured CPU required general-contraction provider reported unsupported"));
}

#[test]
fn tblis_output_view_reachable_range_is_checked_before_ffi() {
    let source = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/lib.rs"))
        .expect("read provider source");

    assert!(
        !source.contains("out_offset > out_storage.len()"),
        "one-past output offsets must not pass the FFI boundary check"
    );
    assert!(
        source.contains("checked_output_base_offset(")
            && source.contains("checked_add")
            && source.contains("checked_mul"),
        "TBLIS output FFI setup must validate the complete reachable shape/stride range"
    );
}
