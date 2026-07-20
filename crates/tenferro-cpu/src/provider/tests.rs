use super::{
    CpuBatchedMatrixLayout, CpuGemmProvider, CpuGemmRequest, CpuGeneralContractionProvider,
    CpuKernelParallelism, CpuLayoutTransformProvider, CpuProviderContext, CpuProviderOutcome,
    CpuProviderUnsupported,
};
use crate::CpuContext;
use tenferro_tensor::{DType, DotGeneralAccumulation, Tensor, TensorRead, TensorWrite};

#[allow(dead_code)]
fn assert_object_safe(
    gemm: &dyn CpuGemmProvider,
    layout: &dyn CpuLayoutTransformProvider,
    general: &dyn CpuGeneralContractionProvider,
) {
    let _ = (gemm, layout, general);
}

#[test]
fn unsupported_is_typed() {
    assert!(matches!(
        CpuProviderOutcome::Unsupported(CpuProviderUnsupported::RuntimeUnavailable),
        CpuProviderOutcome::Unsupported(CpuProviderUnsupported::RuntimeUnavailable),
    ));
}

#[test]
fn provider_context_exposes_only_execution_policy() {
    let context = CpuContext::with_threads(1).unwrap();
    let provider_context = CpuProviderContext::new(&context, CpuKernelParallelism::Sequential);

    assert_eq!(provider_context.thread_budget(), 1);
    assert_eq!(
        provider_context.kernel_parallelism(),
        CpuKernelParallelism::Sequential
    );
}

#[test]
fn gemm_request_borrows_prevalidated_views_and_output() {
    let lhs = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12]).unwrap();
    let mut out = Tensor::from_vec_col_major(vec![2, 4], vec![0.0_f64; 8]).unwrap();
    let lhs = TensorRead::from_tensor(&lhs);
    let rhs = TensorRead::from_tensor(&rhs);
    let mut out = TensorWrite::from_tensor(&mut out);
    let lhs_layout = CpuBatchedMatrixLayout::new(0, 1, 2, 6);
    let rhs_layout = CpuBatchedMatrixLayout::new(0, 1, 3, 12);
    let out_layout = CpuBatchedMatrixLayout::new(0, 1, 2, 8);
    let accumulation = DotGeneralAccumulation::overwrite(DType::F64).unwrap();

    let mut request = CpuGemmRequest::new(
        &lhs,
        &rhs,
        &mut out,
        2,
        4,
        3,
        1,
        lhs_layout,
        rhs_layout,
        out_layout,
        accumulation,
    );

    assert_eq!((request.rows(), request.columns()), (2, 4));
    assert_eq!((request.contracted(), request.batch_count()), (3, 1));
    assert_eq!(request.lhs_layout(), lhs_layout);
    assert_eq!(request.rhs_layout(), rhs_layout);
    assert_eq!(request.output_layout(), out_layout);
    assert_eq!(request.lhs().shape(), &[2, 3]);
    assert_eq!(request.rhs().shape(), &[3, 4]);
    assert_eq!(request.output().shape(), &[2, 4]);
    assert_eq!(request.accumulation(), accumulation);
}
