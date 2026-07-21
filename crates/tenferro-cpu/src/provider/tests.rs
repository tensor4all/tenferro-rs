use super::with_layout_execution_policy;
#[cfg(feature = "cpu-blas")]
use super::CpuOperand;
use super::{
    BlasGemmProvider, CpuBatchedMatrixLayout, CpuGemmProvider, CpuGemmRequest,
    CpuGeneralContractionProvider, CpuKernelParallelism, CpuLayoutTransformProvider,
    CpuProviderContext, CpuProviderOutcome, CpuProviderUnsupported, StridedLayoutTransformProvider,
    TblisGeneralContractionProvider,
};
#[cfg(feature = "cpu-tblis-provider")]
use super::{CpuContractionAxes, CpuDotGeneralRequest};
#[cfg(feature = "cpu-faer")]
use super::{CpuGroupedGemmRequest, FaerGemmProvider};
use crate::CpuContext;
#[cfg(feature = "cpu-faer")]
use num_complex::Complex32;
use num_complex::Complex64;
#[cfg(feature = "cpu-faer")]
use tenferro_tensor::backend::GroupedGemmJob;
#[cfg(feature = "cpu-faer")]
use tenferro_tensor::{
    ContractionScalar, TensorView, TensorViewMut, TypedTensorView, TypedTensorViewMut,
};
use tenferro_tensor::{DType, DotGeneralAccumulation, Tensor, TensorRead, TensorWrite};

use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::ThreadId;
use std::time::Duration;
use strided_kernel::{map_into, StridedArray};

const POLICY_TEST_LEN: usize = 1 << 17;

#[derive(Default)]
struct PolicyParticipants {
    active: AtomicUsize,
    max_active: AtomicUsize,
    thread_ids: Mutex<Vec<ThreadId>>,
    required_concurrency: usize,
    rendezvous_released: AtomicBool,
    rendezvous_lock: Mutex<()>,
    rendezvous: Condvar,
    outside_selected_context: AtomicBool,
}

impl PolicyParticipants {
    fn requiring(required_concurrency: usize) -> Self {
        assert!(required_concurrency >= 2);
        Self {
            required_concurrency,
            ..Self::default()
        }
    }

    fn observe(&self, selected_context: Option<&CpuContext>) {
        if selected_context.is_some_and(|context| !context.owns_current_worker_for_test()) {
            self.outside_selected_context.store(true, Ordering::SeqCst);
        }
        let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_active.fetch_max(active, Ordering::SeqCst);
        {
            let id = std::thread::current().id();
            let mut ids = self.thread_ids.lock().unwrap();
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
        if self.required_concurrency >= 2 && !self.rendezvous_released.load(Ordering::Acquire) {
            let guard = self.rendezvous_lock.lock().unwrap();
            if active >= self.required_concurrency {
                self.rendezvous_released.store(true, Ordering::Release);
                self.rendezvous.notify_all();
            } else if !self.rendezvous_released.load(Ordering::Acquire) {
                let _ = self
                    .rendezvous
                    .wait_timeout_while(guard, Duration::from_secs(2), |_| {
                        !self.rendezvous_released.load(Ordering::Acquire)
                    })
                    .unwrap();
            }
        }
        for _ in 0..32 {
            std::hint::spin_loop();
        }
        self.active.fetch_sub(1, Ordering::SeqCst);
    }

    fn max_active(&self) -> usize {
        self.max_active.load(Ordering::SeqCst)
    }

    fn thread_ids(&self) -> Vec<ThreadId> {
        self.thread_ids.lock().unwrap().clone()
    }
}

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
    assert_eq!(provider_context.nonzero_thread_budget(), NonZeroUsize::MIN);
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

#[test]
#[cfg(feature = "cpu-faer")]
fn faer_provider_executes_into_preallocated_output() {
    let lhs =
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap();
    let rhs =
        Tensor::from_vec_col_major(vec![3, 2], vec![7.0_f64, 9.0, 11.0, 8.0, 10.0, 12.0]).unwrap();
    let mut out = Tensor::from_vec_col_major(vec![2, 2], vec![0.0_f64; 4]).unwrap();
    let lhs = TensorRead::from_tensor(&lhs);
    let rhs = TensorRead::from_tensor(&rhs);
    let mut out_write = TensorWrite::from_tensor(&mut out);
    let accumulation = DotGeneralAccumulation::overwrite(DType::F64).unwrap();
    let context = CpuContext::with_threads(1).unwrap();
    let provider_context = CpuProviderContext::new(&context, CpuKernelParallelism::Sequential);
    let request = CpuGemmRequest::new(
        &lhs,
        &rhs,
        &mut out_write,
        2,
        2,
        3,
        1,
        CpuBatchedMatrixLayout::new(0, 1, 2, 6),
        CpuBatchedMatrixLayout::new(0, 1, 3, 6),
        CpuBatchedMatrixLayout::new(0, 1, 2, 4),
        accumulation,
    );

    assert_eq!(
        FaerGemmProvider.gemm(&provider_context, request).unwrap(),
        CpuProviderOutcome::Executed,
    );
    drop(out_write);
    assert_eq!(out.as_slice::<f64>().unwrap(), &[58.0, 139.0, 64.0, 154.0]);
}

#[test]
#[cfg(feature = "cpu-faer")]
fn faer_provider_covers_f32_c32_and_c64_conjugation() {
    let context = CpuContext::with_threads(1).unwrap();
    let provider_context = CpuProviderContext::new(&context, CpuKernelParallelism::Sequential);

    let lhs = Tensor::from_vec_col_major(vec![1, 1], vec![3.0_f32]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![1, 1], vec![4.0_f32]).unwrap();
    let mut output = Tensor::from_vec_col_major(vec![1, 1], vec![1.0_f32]).unwrap();
    let lhs_read = TensorRead::from_tensor(&lhs);
    let rhs_read = TensorRead::from_tensor(&rhs);
    let mut output_write = TensorWrite::from_tensor(&mut output);
    let request = CpuGemmRequest::new(
        &lhs_read,
        &rhs_read,
        &mut output_write,
        1,
        1,
        1,
        1,
        CpuBatchedMatrixLayout::new(0, 1, 1, 1),
        CpuBatchedMatrixLayout::new(0, 1, 1, 1),
        CpuBatchedMatrixLayout::new(0, 1, 1, 1),
        DotGeneralAccumulation::overwrite(DType::F32).unwrap(),
    );
    assert_eq!(
        FaerGemmProvider.gemm(&provider_context, request).unwrap(),
        CpuProviderOutcome::Executed,
    );
    drop(output_write);
    assert_eq!(output.as_slice::<f32>().unwrap(), &[12.0]);

    let lhs = Tensor::from_vec_col_major(vec![1, 1], vec![Complex32::new(1.0, 1.0)]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![1, 1], vec![Complex32::new(2.0, -1.0)]).unwrap();
    let mut output =
        Tensor::from_vec_col_major(vec![1, 1], vec![Complex32::new(0.0, 0.0)]).unwrap();
    let lhs_read = TensorRead::from_tensor(&lhs);
    let rhs_read = TensorRead::from_tensor(&rhs);
    let mut output_write = TensorWrite::from_tensor(&mut output);
    let request = CpuGemmRequest::new(
        &lhs_read,
        &rhs_read,
        &mut output_write,
        1,
        1,
        1,
        1,
        CpuBatchedMatrixLayout::new(0, 1, 1, 1),
        CpuBatchedMatrixLayout::new(0, 1, 1, 1),
        CpuBatchedMatrixLayout::new(0, 1, 1, 1),
        DotGeneralAccumulation::overwrite(DType::C32).unwrap(),
    );
    assert_eq!(
        FaerGemmProvider.gemm(&provider_context, request).unwrap(),
        CpuProviderOutcome::Executed,
    );
    drop(output_write);
    assert_eq!(
        output.as_slice::<Complex32>().unwrap(),
        &[Complex32::new(3.0, 1.0)]
    );

    let lhs = Tensor::from_vec_col_major(
        vec![1, 2],
        vec![Complex64::new(1.0, 1.0), Complex64::new(2.0, -1.0)],
    )
    .unwrap();
    let rhs = Tensor::from_vec_col_major(
        vec![2, 1],
        vec![Complex64::new(3.0, 2.0), Complex64::new(-1.0, 1.0)],
    )
    .unwrap();
    let mut output =
        Tensor::from_vec_col_major(vec![1, 1], vec![Complex64::new(0.0, 0.0)]).unwrap();
    let lhs_read = TensorRead::from_tensor(&lhs);
    let rhs_read = TensorRead::from_tensor(&rhs);
    let mut output_write = TensorWrite::from_tensor(&mut output);
    let request = CpuGemmRequest::new(
        &lhs_read,
        &rhs_read,
        &mut output_write,
        1,
        1,
        2,
        1,
        CpuBatchedMatrixLayout::new(0, 1, 1, 2),
        CpuBatchedMatrixLayout::new(0, 1, 2, 2),
        CpuBatchedMatrixLayout::new(0, 1, 1, 1),
        DotGeneralAccumulation {
            lhs_conj: true,
            rhs_conj: false,
            alpha: ContractionScalar::C64(Complex64::new(1.0, 0.0)),
            beta: ContractionScalar::C64(Complex64::new(0.0, 0.0)),
        },
    );
    assert_eq!(
        FaerGemmProvider.gemm(&provider_context, request).unwrap(),
        CpuProviderOutcome::Executed,
    );
    drop(output_write);
    assert_eq!(
        output.as_slice::<Complex64>().unwrap(),
        &[Complex64::new(2.0, 0.0)]
    );
}

#[test]
#[cfg(feature = "cpu-faer")]
fn faer_provider_executes_non_unit_strides_and_strided_batches() {
    let context = CpuContext::with_threads(1).unwrap();
    let provider_context = CpuProviderContext::new(&context, CpuKernelParallelism::Inner);
    let mut lhs_storage = vec![0.0_f64; 9];
    lhs_storage[1] = 1.0;
    lhs_storage[3] = 3.0;
    lhs_storage[6] = 2.0;
    lhs_storage[8] = 4.0;
    let mut rhs_storage = vec![0.0_f64; 11];
    rhs_storage[0] = 5.0;
    rhs_storage[3] = 7.0;
    rhs_storage[7] = 6.0;
    rhs_storage[10] = 8.0;
    let lhs_view = TypedTensorView::from_slice([2, 2], [2, 5], 1, &lhs_storage).unwrap();
    let rhs_view = TypedTensorView::from_slice([2, 2], [3, 7], 0, &rhs_storage).unwrap();
    let mut output_storage = vec![-1.0_f64; 10];
    let output_view =
        TypedTensorViewMut::from_slice([2, 2], [2, 6], 1, &mut output_storage).unwrap();
    let lhs_read = TensorRead::from_view(TensorView::F64(lhs_view));
    let rhs_read = TensorRead::from_view(TensorView::F64(rhs_view));
    let mut output_write = TensorWrite::from_view(TensorViewMut::F64(output_view));
    let request = CpuGemmRequest::new(
        &lhs_read,
        &rhs_read,
        &mut output_write,
        2,
        2,
        2,
        1,
        CpuBatchedMatrixLayout::new(1, 2, 5, 0),
        CpuBatchedMatrixLayout::new(0, 3, 7, 0),
        CpuBatchedMatrixLayout::new(1, 2, 6, 0),
        DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
    );
    assert_eq!(
        FaerGemmProvider.gemm(&provider_context, request).unwrap(),
        CpuProviderOutcome::Executed,
    );
    drop(output_write);
    assert_eq!(
        [
            output_storage[1],
            output_storage[3],
            output_storage[7],
            output_storage[9],
        ],
        [19.0, 43.0, 22.0, 50.0]
    );

    let lhs = Tensor::from_vec_col_major(vec![8], vec![1.0_f64, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0])
        .unwrap();
    let rhs = Tensor::from_vec_col_major(vec![8], vec![1.0_f64, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0])
        .unwrap();
    let mut output = Tensor::from_vec_col_major(vec![8], vec![0.0_f64; 8]).unwrap();
    let lhs_read = TensorRead::from_tensor(&lhs);
    let rhs_read = TensorRead::from_tensor(&rhs);
    let mut output_write = TensorWrite::from_tensor(&mut output);
    let request = CpuGemmRequest::new(
        &lhs_read,
        &rhs_read,
        &mut output_write,
        2,
        2,
        2,
        2,
        CpuBatchedMatrixLayout::new(0, 1, 2, 4),
        CpuBatchedMatrixLayout::new(0, 1, 2, 4),
        CpuBatchedMatrixLayout::new(0, 1, 2, 4),
        DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
    );
    assert_eq!(
        FaerGemmProvider
            .strided_batched_gemm(&provider_context, request)
            .unwrap(),
        CpuProviderOutcome::Executed,
    );
    drop(output_write);
    assert_eq!(
        output.as_slice::<f64>().unwrap(),
        &[1.0, 3.0, 2.0, 4.0, 10.0, 21.0, 12.0, 24.0]
    );
}

#[test]
#[cfg(feature = "cpu-faer")]
fn faer_provider_executes_grouped_jobs_without_owning_scheduling() {
    let lhs = Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![2], vec![4.0_f64, 5.0]).unwrap();
    let mut output = Tensor::from_vec_col_major(vec![2], vec![0.0_f64; 2]).unwrap();
    let jobs = [
        GroupedGemmJob::new(0, 0, 0, 1, 1, 1),
        GroupedGemmJob::new(1, 1, 1, 1, 1, 1),
    ];
    let lhs_read = TensorRead::from_tensor(&lhs);
    let rhs_read = TensorRead::from_tensor(&rhs);
    let mut output_write = TensorWrite::from_tensor(&mut output);
    let request = CpuGroupedGemmRequest::new(
        &lhs_read,
        &rhs_read,
        &mut output_write,
        &jobs,
        DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
    );
    let context = CpuContext::with_threads(2).unwrap();
    let provider_context = CpuProviderContext::new(&context, CpuKernelParallelism::Sequential);
    assert_eq!(
        FaerGemmProvider
            .grouped_gemm(&provider_context, request)
            .unwrap(),
        CpuProviderOutcome::Executed,
    );
    drop(output_write);
    assert_eq!(output.as_slice::<f64>().unwrap(), &[8.0, 15.0]);
}

#[test]
#[cfg(feature = "cpu-faer")]
fn faer_unsupported_dtype_preserves_output() {
    let lhs = Tensor::from_vec_col_major(vec![1, 1], vec![2_i32]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![1, 1], vec![3_i32]).unwrap();
    let mut output = Tensor::from_vec_col_major(vec![1, 1], vec![41_i32]).unwrap();
    let before = output.as_slice::<i32>().unwrap().to_vec();
    let lhs_read = TensorRead::from_tensor(&lhs);
    let rhs_read = TensorRead::from_tensor(&rhs);
    let mut output_write = TensorWrite::from_tensor(&mut output);
    let request = CpuGemmRequest::new(
        &lhs_read,
        &rhs_read,
        &mut output_write,
        1,
        1,
        1,
        1,
        CpuBatchedMatrixLayout::new(0, 1, 1, 1),
        CpuBatchedMatrixLayout::new(0, 1, 1, 1),
        CpuBatchedMatrixLayout::new(0, 1, 1, 1),
        DotGeneralAccumulation::overwrite(DType::F32).unwrap(),
    );
    let context = CpuContext::with_threads(1).unwrap();
    let provider_context = CpuProviderContext::new(&context, CpuKernelParallelism::Sequential);
    assert_eq!(
        FaerGemmProvider.gemm(&provider_context, request).unwrap(),
        CpuProviderOutcome::Unsupported(CpuProviderUnsupported::DType(DType::I32)),
    );
    drop(output_write);
    assert_eq!(output.as_slice::<i32>().unwrap(), before.as_slice());
}

#[test]
#[cfg(not(feature = "cpu-blas"))]
fn unavailable_blas_provider_preserves_output() {
    let lhs = Tensor::from_vec_col_major(vec![1, 1], vec![2.0_f64]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![1, 1], vec![3.0_f64]).unwrap();
    let mut output = Tensor::from_vec_col_major(vec![1, 1], vec![41.0_f64]).unwrap();
    let lhs_read = TensorRead::from_tensor(&lhs);
    let rhs_read = TensorRead::from_tensor(&rhs);
    let mut output_write = TensorWrite::from_tensor(&mut output);
    let request = CpuGemmRequest::new(
        &lhs_read,
        &rhs_read,
        &mut output_write,
        1,
        1,
        1,
        1,
        CpuBatchedMatrixLayout::new(0, 1, 1, 1),
        CpuBatchedMatrixLayout::new(0, 1, 1, 1),
        CpuBatchedMatrixLayout::new(0, 1, 1, 1),
        DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
    );
    let context = CpuContext::with_threads(1).unwrap();
    let provider_context = CpuProviderContext::new(&context, CpuKernelParallelism::Sequential);
    assert_eq!(
        BlasGemmProvider.gemm(&provider_context, request).unwrap(),
        CpuProviderOutcome::Unsupported(CpuProviderUnsupported::RuntimeUnavailable),
    );
    drop(output_write);
    assert_eq!(output.as_slice::<f64>().unwrap(), &[41.0]);
}

#[test]
#[cfg(feature = "cpu-blas")]
fn blas_provider_executes_and_rejects_layout_before_mutation() {
    let context = CpuContext::with_threads(1).unwrap();
    let provider_context = CpuProviderContext::new(&context, CpuKernelParallelism::Sequential);
    let lhs = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![2, 2], vec![5.0_f64, 7.0, 6.0, 8.0]).unwrap();
    let mut output = Tensor::from_vec_col_major(vec![2, 2], vec![0.0_f64; 4]).unwrap();
    let lhs_read = TensorRead::from_tensor(&lhs);
    let rhs_read = TensorRead::from_tensor(&rhs);
    let mut output_write = TensorWrite::from_tensor(&mut output);
    let request = CpuGemmRequest::new(
        &lhs_read,
        &rhs_read,
        &mut output_write,
        2,
        2,
        2,
        1,
        CpuBatchedMatrixLayout::new(0, 1, 2, 4),
        CpuBatchedMatrixLayout::new(0, 1, 2, 4),
        CpuBatchedMatrixLayout::new(0, 1, 2, 4),
        DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
    );
    assert_eq!(
        BlasGemmProvider.gemm(&provider_context, request).unwrap(),
        CpuProviderOutcome::Executed,
    );
    drop(output_write);
    assert_eq!(output.as_slice::<f64>().unwrap(), &[19.0, 43.0, 22.0, 50.0]);

    let mut output = Tensor::from_vec_col_major(vec![2, 2], vec![41.0_f64; 4]).unwrap();
    let before = output.as_slice::<f64>().unwrap().to_vec();
    let mut output_write = TensorWrite::from_tensor(&mut output);
    let request = CpuGemmRequest::new(
        &lhs_read,
        &rhs_read,
        &mut output_write,
        2,
        2,
        2,
        1,
        CpuBatchedMatrixLayout::new(0, 2, 5, 0),
        CpuBatchedMatrixLayout::new(0, 1, 2, 4),
        CpuBatchedMatrixLayout::new(0, 1, 2, 4),
        DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
    );
    assert_eq!(
        BlasGemmProvider.gemm(&provider_context, request).unwrap(),
        CpuProviderOutcome::Unsupported(CpuProviderUnsupported::Layout(CpuOperand::Lhs)),
    );
    drop(output_write);
    assert_eq!(output.as_slice::<f64>().unwrap(), before.as_slice());
}

#[test]
fn tblis_provider_is_object_safe_even_when_runtime_is_optional() {
    let provider: &dyn CpuGeneralContractionProvider = &TblisGeneralContractionProvider;
    let _ = provider;
}

#[test]
#[cfg(feature = "cpu-tblis-provider")]
fn tblis_provider_executes_or_preserves_output_when_runtime_is_unavailable() {
    let lhs = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![2, 2], vec![5.0_f64, 7.0, 6.0, 8.0]).unwrap();
    let mut output = Tensor::from_vec_col_major(vec![2, 2], vec![41.0_f64; 4]).unwrap();
    let before = output.as_slice::<f64>().unwrap().to_vec();
    let lhs_read = TensorRead::from_tensor(&lhs);
    let rhs_read = TensorRead::from_tensor(&rhs);
    let mut output_write = TensorWrite::from_tensor(&mut output);
    let request = CpuDotGeneralRequest::new(
        &lhs_read,
        &rhs_read,
        &mut output_write,
        CpuContractionAxes::new(2, 2, &[1], &[0], &[], &[], Some(2), Some(1)),
        DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
    );
    let context = CpuContext::with_threads(1).unwrap();
    let provider_context = CpuProviderContext::new(&context, CpuKernelParallelism::Sequential);
    let outcome = TblisGeneralContractionProvider
        .dot_general(&provider_context, request)
        .unwrap();
    drop(output_write);
    match outcome {
        CpuProviderOutcome::Executed => {
            assert_eq!(output.as_slice::<f64>().unwrap(), &[19.0, 43.0, 22.0, 50.0]);
        }
        CpuProviderOutcome::Unsupported(CpuProviderUnsupported::RuntimeUnavailable) => {
            assert_eq!(output.as_slice::<f64>().unwrap(), before.as_slice());
        }
        other => panic!("unexpected TBLIS provider outcome: {other:?}"),
    }
}

#[test]
fn layout_provider_materializes_into_preallocated_output() {
    use super::{CpuLayoutTransformIntent, CpuLayoutTransformRequest};

    let input = Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]).unwrap();
    let mut output = Tensor::from_vec_col_major(vec![2], vec![0.0_f64; 2]).unwrap();
    let input = TensorRead::from_tensor(&input);
    let mut output_write = TensorWrite::from_tensor(&mut output);
    let context = CpuContext::with_threads(1).unwrap();
    let provider_context = CpuProviderContext::new(&context, CpuKernelParallelism::Sequential);
    let request = CpuLayoutTransformRequest::new(
        &input,
        &mut output_write,
        CpuLayoutTransformIntent::CanonicalColumnMajor,
        false,
    );

    assert_eq!(
        StridedLayoutTransformProvider
            .materialize(&provider_context, request)
            .unwrap(),
        CpuProviderOutcome::Executed,
    );
    drop(output_write);
    assert_eq!(output.as_slice::<f64>().unwrap(), &[2.0, 3.0]);
}

#[test]
fn layout_provider_fuses_conjugation_into_materialization() {
    use super::{CpuLayoutTransformIntent, CpuLayoutTransformRequest};

    let input = Tensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(2.0, 3.0), Complex64::new(-1.0, 4.0)],
    )
    .unwrap();
    let mut output =
        Tensor::from_vec_col_major(vec![2], vec![Complex64::new(41.0, 0.0); 2]).unwrap();
    let input = TensorRead::from_tensor(&input);
    let mut output_write = TensorWrite::from_tensor(&mut output);
    let context = CpuContext::with_threads(1).unwrap();
    let provider_context = CpuProviderContext::new(&context, CpuKernelParallelism::Sequential);
    let request = CpuLayoutTransformRequest::new(
        &input,
        &mut output_write,
        CpuLayoutTransformIntent::CanonicalColumnMajor,
        true,
    );
    assert!(request.conjugate());

    assert_eq!(
        StridedLayoutTransformProvider
            .materialize(&provider_context, request)
            .unwrap(),
        CpuProviderOutcome::Executed,
    );
    drop(output_write);
    assert_eq!(
        output.as_slice::<Complex64>().unwrap(),
        &[Complex64::new(2.0, -3.0), Complex64::new(-1.0, -4.0)],
    );
}

#[test]
fn sequential_layout_policy_uses_one_participant_inside_a_larger_pool() {
    let ambient = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let selected = CpuContext::with_threads(2).unwrap();

    ambient.install(|| {
        let caller = std::thread::current().id();
        let source = StridedArray::<usize>::from_fn_col_major(&[POLICY_TEST_LEN], |index| index[0]);
        let mut destination = StridedArray::<usize>::col_major(&[POLICY_TEST_LEN]);
        let participants = Arc::new(PolicyParticipants::default());
        let observed = Arc::clone(&participants);
        let provider_context = CpuProviderContext::new(&selected, CpuKernelParallelism::Sequential);

        with_layout_execution_policy(&provider_context, || {
            map_into(&mut destination.view_mut(), &source.view(), |value| {
                observed.observe(None);
                value + 1
            })
            .unwrap();
        });

        assert_eq!(participants.max_active(), 1);
        assert_eq!(participants.thread_ids(), vec![caller]);
        assert_eq!(destination.get(&[POLICY_TEST_LEN - 1]), POLICY_TEST_LEN);
    });
}

fn assert_inner_layout_policy_uses_selected_context(selected: &CpuContext) {
    let source = StridedArray::<usize>::from_fn_col_major(&[POLICY_TEST_LEN], |index| index[0]);
    let mut destination = StridedArray::<usize>::col_major(&[POLICY_TEST_LEN]);
    let participants = Arc::new(PolicyParticipants::requiring(2));
    let observed = Arc::clone(&participants);
    let provider_context = CpuProviderContext::new(selected, CpuKernelParallelism::Inner);

    with_layout_execution_policy(&provider_context, || {
        map_into(&mut destination.view_mut(), &source.view(), |value| {
            observed.observe(Some(selected));
            value + 1
        })
        .unwrap();
    });

    assert_eq!(participants.max_active(), 2);
    assert_eq!(participants.thread_ids().len(), 2);
    assert!(!participants.outside_selected_context.load(Ordering::SeqCst));
    assert_eq!(destination.get(&[POLICY_TEST_LEN - 1]), POLICY_TEST_LEN);
}

#[test]
fn inner_layout_policy_uses_selected_context_and_budget_not_ambient_pool() {
    let ambient = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let selected = CpuContext::with_threads(2).unwrap();

    ambient.install(|| {
        assert!(!selected.owns_current_worker_for_test());
        assert_inner_layout_policy_uses_selected_context(&selected);
    });
    selected.install(|| {
        assert!(selected.owns_current_worker_for_test());
        assert_inner_layout_policy_uses_selected_context(&selected);
    });
}
