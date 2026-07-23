use std::error::Error as _;
use std::num::NonZeroUsize;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use tenferro_cpu::linalg_interop::PoolScalar;
use tenferro_cpu::{
    discover_cpu_topology, CpuBackend, CpuDomainExecutor, CpuDomainExecutorCapabilities,
    CpuDomainExecutorError, CpuExecutorAffinity, CpuExecutorReentrancy, CpuExecutorShutdown,
    CpuInnerParallelism, CpuPlacementGuarantee, ExternalCpuDomain, ResolvedCpuPlacement,
    ScopedCpuJob, ScopedCpuJobs,
};
use tenferro_tensor::{
    Buffer, BufferHandle, CpuDomainId, ErrorKind, MemoryKind, Placement,
    SharedTensorAllocationDomain, StridedSliceSpec, Tensor, TensorRead, TensorView, TypedTensor,
};

use super::managed_cholesky::{enter_observed_operation_scope, FakeDomain};
use crate::LinalgBackend;

#[derive(Debug)]
struct CountingNoInnerExecutor {
    installs: Arc<AtomicUsize>,
    submits: Arc<AtomicUsize>,
}

impl CpuDomainExecutor for CountingNoInnerExecutor {
    fn capabilities(&self) -> CpuDomainExecutorCapabilities {
        CpuDomainExecutorCapabilities {
            worker_count: NonZeroUsize::new(1).unwrap(),
            outer_parallelism: false,
            inner_parallelism: CpuInnerParallelism::None,
            reentrancy: CpuExecutorReentrancy::Rejected,
            affinity: CpuExecutorAffinity::CallerDeclaredUnverified,
            shutdown: CpuExecutorShutdown::CallerOwned,
        }
    }

    fn submit(&self, jobs: &dyn ScopedCpuJobs) -> Result<(), CpuDomainExecutorError> {
        self.submits.fetch_add(1, Ordering::Relaxed);
        for index in 0..jobs.len() {
            jobs.run(index)?;
        }
        Ok(())
    }

    fn install(&self, job: &mut dyn ScopedCpuJob) -> Result<(), CpuDomainExecutorError> {
        self.installs.fetch_add(1, Ordering::Relaxed);
        let _scope = enter_observed_operation_scope();
        job.run()
    }
}

fn external_no_inner_backend() -> (CpuBackend, Arc<AtomicUsize>, Arc<AtomicUsize>) {
    let installs = Arc::new(AtomicUsize::new(0));
    let submits = Arc::new(AtomicUsize::new(0));
    let backend = external_backend(Arc::new(CountingNoInnerExecutor {
        installs: Arc::clone(&installs),
        submits: Arc::clone(&submits),
    }));
    (backend, installs, submits)
}

fn external_no_inner_managed_backend(
    domain: &Arc<FakeDomain>,
) -> (CpuBackend, Arc<AtomicUsize>, Arc<AtomicUsize>) {
    let (backend, installs, submits) = external_no_inner_backend();
    let erased: Arc<dyn SharedTensorAllocationDomain> = domain.clone();
    (backend.with_allocation_domain(erased), installs, submits)
}

fn external_backend(executor: Arc<dyn CpuDomainExecutor>) -> CpuBackend {
    let allowed = discover_cpu_topology().unwrap().allowed_cpus().clone();
    let id = CpuDomainId::new(91);
    let domain = ExternalCpuDomain::new(
        id,
        ResolvedCpuPlacement::AllAllowed { cpus: allowed },
        executor,
        NonZeroUsize::new(1).unwrap(),
        CpuPlacementGuarantee::ExactDeclared,
    )
    .unwrap();
    CpuBackend::from_external_managed_domains(id, [domain]).unwrap()
}

#[test]
fn scoped_materialization_reclaims_successful_temporary_for_immediate_reuse() {
    let (mut backend, _, _) = external_no_inner_backend();
    let source = TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let reversed = [StridedSliceSpec::reverse(), StridedSliceSpec::reverse()];

    backend
        .with_linalg_pool(move |context, buffers| {
            let seed = Vec::<f64>::with_capacity(4);
            let seed_ptr = seed.as_ptr() as usize;
            <f64 as PoolScalar>::pool_release(buffers, seed);
            let retained = buffers.stats();

            let view = source.as_view().try_slice(&reversed).unwrap();
            let materialized_ptr = context.with_materialized_tensor_read(
                buffers,
                "scoped_materialization_success",
                TensorRead::from_view(TensorView::F64(view)),
                |input, _| Ok(input.as_slice::<f64>().unwrap().as_ptr() as usize),
            )?;

            assert_eq!(materialized_ptr, seed_ptr);
            assert_eq!(buffers.stats(), retained);
            let reused = buffers.acquire_with_capacity::<f64>(4);
            assert_eq!(reused.as_ptr() as usize, seed_ptr);
            <f64 as PoolScalar>::pool_release(buffers, reused);
            Ok(())
        })
        .unwrap();
}

#[test]
fn scoped_materialization_reclaims_temporary_after_numerical_error() {
    let (mut backend, _, _) = external_no_inner_backend();
    let source = TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let reversed = [StridedSliceSpec::reverse(), StridedSliceSpec::reverse()];

    backend
        .with_linalg_pool(move |context, buffers| {
            let seed = Vec::<f64>::with_capacity(4);
            let seed_ptr = seed.as_ptr() as usize;
            <f64 as PoolScalar>::pool_release(buffers, seed);
            let retained = buffers.stats();

            let view = source.as_view().try_slice(&reversed).unwrap();
            let error = context
                .with_materialized_tensor_read(
                    buffers,
                    "scoped_materialization_error",
                    TensorRead::from_view(TensorView::F64(view)),
                    |_, _| -> tenferro_tensor::Result<()> {
                        Err(crate::error::into_tensor_error(
                            "scoped_materialization_error",
                            crate::Error::NonConvergence {
                                op: "scoped_materialization_error",
                            },
                        ))
                    },
                )
                .unwrap_err();

            assert_eq!(error.kind(), ErrorKind::NumericalFailure);
            assert_eq!(buffers.stats(), retained);
            let reused = buffers.acquire_with_capacity::<f64>(4);
            assert_eq!(reused.as_ptr() as usize, seed_ptr);
            <f64 as PoolScalar>::pool_release(buffers, reused);
            Ok(())
        })
        .unwrap();
}

#[test]
fn nested_materialization_reclaims_first_when_second_materialization_fails() {
    let (mut backend, _, _) = external_no_inner_backend();
    let source = TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let opaque = TypedTensor::from_buffer_col_major(
        vec![2, 2],
        Buffer::Backend(Arc::new(BufferHandle::<f64>::new_with_len(17, 4))),
        Placement {
            memory_kind: MemoryKind::Device,
            device: None,
            cpu_affinity: None,
        },
    )
    .unwrap();
    let reversed = [StridedSliceSpec::reverse(), StridedSliceSpec::reverse()];

    backend
        .with_linalg_pool(move |context, buffers| {
            let seed = Vec::<f64>::with_capacity(4);
            let seed_ptr = seed.as_ptr() as usize;
            <f64 as PoolScalar>::pool_release(buffers, seed);
            let retained = buffers.stats();

            let first = source.as_view().try_slice(&reversed).unwrap();
            let second = opaque.as_view().transpose_view([1, 0]).unwrap();
            let error = context
                .with_materialized_tensor_read(
                    buffers,
                    "nested_materialization",
                    TensorRead::from_view(TensorView::F64(first)),
                    |_, buffers| {
                        context.with_materialized_tensor_read(
                            buffers,
                            "nested_materialization",
                            TensorRead::from_view(TensorView::F64(second)),
                            |_, _| Ok(()),
                        )
                    },
                )
                .unwrap_err();

            assert_eq!(error.kind(), ErrorKind::RuntimeState);
            assert_eq!(buffers.stats(), retained);
            let reused = buffers.acquire_with_capacity::<f64>(4);
            assert_eq!(reused.as_ptr() as usize, seed_ptr);
            <f64 as PoolScalar>::pool_release(buffers, reused);
            Ok(())
        })
        .unwrap();
}

#[test]
fn triangular_solve_read_two_noncompact_views_enters_once() {
    let (mut backend, installs, submits) = external_no_inner_backend();
    let a_base = TypedTensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 3.0]).unwrap();
    let b_base = TypedTensor::from_vec_col_major(vec![1, 2], vec![4.0_f64, 9.0]).unwrap();
    let a = a_base.as_view().transpose_view([1, 0]).unwrap();
    let b = b_base.as_view().transpose_view([1, 0]).unwrap();

    let output = backend
        .triangular_solve_read(
            TensorRead::from_view(TensorView::F64(a)),
            TensorRead::from_view(TensorView::F64(b)),
            true,
            true,
            false,
            false,
        )
        .unwrap();

    assert_eq!(output.as_slice::<f64>().unwrap(), &[2.0, 3.0]);
    assert_eq!(installs.load(Ordering::Relaxed), 1);
    assert_eq!(submits.load(Ordering::Relaxed), 0);
}

#[test]
fn solve_read_two_noncompact_views_enters_once() {
    let (mut backend, installs, submits) = external_no_inner_backend();
    let a_base = TypedTensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 3.0]).unwrap();
    let b_base = TypedTensor::from_vec_col_major(vec![1, 2], vec![4.0_f64, 9.0]).unwrap();
    let a = a_base.as_view().transpose_view([1, 0]).unwrap();
    let b = b_base.as_view().transpose_view([1, 0]).unwrap();

    let output = backend
        .solve_read(
            TensorRead::from_view(TensorView::F64(a)),
            TensorRead::from_view(TensorView::F64(b)),
        )
        .unwrap();

    assert_eq!(output.as_slice::<f64>().unwrap(), &[2.0, 3.0]);
    assert_eq!(installs.load(Ordering::Relaxed), 1);
    assert_eq!(submits.load(Ordering::Relaxed), 0);
}

#[test]
fn solve_read_vector_rhs_reshape_stays_inside_one_entry() {
    let (mut backend, installs, submits) = external_no_inner_backend();
    let a_base = TypedTensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 3.0]).unwrap();
    let b_base = TypedTensor::from_vec_col_major(vec![2], vec![9.0_f64, 4.0]).unwrap();
    let a = a_base.as_view().transpose_view([1, 0]).unwrap();
    let b = b_base
        .as_view()
        .try_slice(&[StridedSliceSpec::reverse()])
        .unwrap();

    let output = backend
        .solve_read(
            TensorRead::from_view(TensorView::F64(a)),
            TensorRead::from_view(TensorView::F64(b)),
        )
        .unwrap();

    assert_eq!(output.shape(), &[2]);
    assert_eq!(output.as_slice::<f64>().unwrap(), &[2.0, 3.0]);
    assert_eq!(installs.load(Ordering::Relaxed), 1);
    assert_eq!(submits.load(Ordering::Relaxed), 0);
}

#[test]
fn managed_backend_handles_noncompact_two_input_reads() {
    let mut backend = CpuBackend::with_threads(1).unwrap();
    let a_base = TypedTensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 3.0]).unwrap();
    let b_base = TypedTensor::from_vec_col_major(vec![1, 2], vec![4.0_f64, 9.0]).unwrap();

    let a = a_base.as_view().transpose_view([1, 0]).unwrap();
    let b = b_base.as_view().transpose_view([1, 0]).unwrap();
    let triangular = backend
        .triangular_solve_read(
            TensorRead::from_view(TensorView::F64(a)),
            TensorRead::from_view(TensorView::F64(b)),
            true,
            true,
            false,
            false,
        )
        .unwrap();
    assert_eq!(triangular.as_slice::<f64>().unwrap(), &[2.0, 3.0]);

    let a = a_base.as_view().transpose_view([1, 0]).unwrap();
    let b = b_base.as_view().transpose_view([1, 0]).unwrap();
    let solved = backend
        .solve_read(
            TensorRead::from_view(TensorView::F64(a)),
            TensorRead::from_view(TensorView::F64(b)),
        )
        .unwrap();
    assert_eq!(solved.as_slice::<f64>().unwrap(), &[2.0, 3.0]);
}

#[test]
fn eig_read_one_input_fallback_enters_once() {
    let (mut backend, installs, submits) = external_no_inner_backend();
    let input = TypedTensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 3.0]).unwrap();
    let input = input.as_view().transpose_view([1, 0]).unwrap();

    let outputs = backend
        .eig_read(TensorRead::from_view(TensorView::F64(input)))
        .unwrap();

    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].shape(), &[2]);
    assert_eq!(outputs[1].shape(), &[2, 2]);
    assert_eq!(installs.load(Ordering::Relaxed), 1);
    assert_eq!(submits.load(Ordering::Relaxed), 0);
}

#[cfg(feature = "cpu-faer")]
#[test]
fn faer_strided_read_fast_path_enters_once() {
    let (mut backend, installs, submits) = external_no_inner_backend();
    let input = TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 5.0]).unwrap();
    let transposed = input.as_view().transpose_view([1, 0]).unwrap();

    let outputs = backend
        .svd_read(TensorRead::from_view(TensorView::F64(transposed)))
        .unwrap();

    assert_eq!(outputs.len(), 3);
    assert_eq!(installs.load(Ordering::Relaxed), 1);
    assert_eq!(submits.load(Ordering::Relaxed), 0);
}

#[cfg(feature = "cpu-faer")]
#[test]
fn faer_full_svd_enters_once() {
    let (mut backend, installs, submits) = external_no_inner_backend();
    let input =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 1], vec![3.0_f64, 4.0]).unwrap());

    let outputs = backend.svd_full(&input).unwrap();

    assert_eq!(outputs.len(), 3);
    assert_eq!(outputs[0].shape(), &[2, 2]);
    assert_eq!(outputs[1].shape(), &[1]);
    assert_eq!(outputs[2].shape(), &[1, 1]);
    assert_eq!(installs.load(Ordering::Relaxed), 1);
    assert_eq!(submits.load(Ordering::Relaxed), 0);
}

fn assert_one_install<R>(
    installs: &AtomicUsize,
    submits: &AtomicUsize,
    operation: impl FnOnce() -> tenferro_tensor::Result<R>,
) -> R {
    let install_start = installs.load(Ordering::Relaxed);
    let submit_start = submits.load(Ordering::Relaxed);
    let output = operation().unwrap();
    assert_eq!(installs.load(Ordering::Relaxed) - install_start, 1);
    assert_eq!(submits.load(Ordering::Relaxed) - submit_start, 0);
    output
}

#[test]
fn every_one_input_read_fallback_enters_once() {
    let (mut backend, installs, submits) = external_no_inner_backend();
    let input = TypedTensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 3.0]).unwrap();
    let reversed = [StridedSliceSpec::reverse(), StridedSliceSpec::reverse()];

    let view = input.as_view().try_slice(&reversed).unwrap();
    assert_one_install(&installs, &submits, || {
        backend.svd_read(TensorRead::from_view(TensorView::F64(view)))
    });

    let view = input.as_view().try_slice(&reversed).unwrap();
    assert_one_install(&installs, &submits, || {
        backend.qr_read(TensorRead::from_view(TensorView::F64(view)))
    });

    let view = input.as_view().try_slice(&reversed).unwrap();
    assert_one_install(&installs, &submits, || {
        backend.eigh_read(TensorRead::from_view(TensorView::F64(view)))
    });

    let view = input.as_view().try_slice(&reversed).unwrap();
    assert_one_install(&installs, &submits, || {
        backend.cholesky_read(TensorRead::from_view(TensorView::F64(view)))
    });

    let view = input.as_view().try_slice(&reversed).unwrap();
    assert_one_install(&installs, &submits, || {
        backend.lu_read(TensorRead::from_view(TensorView::F64(view)))
    });

    let view = input.as_view().try_slice(&reversed).unwrap();
    assert_one_install(&installs, &submits, || {
        backend.full_piv_lu_read(TensorRead::from_view(TensorView::F64(view)))
    });

    let view = input.as_view().try_slice(&reversed).unwrap();
    assert_one_install(&installs, &submits, || {
        backend.eig_read(TensorRead::from_view(TensorView::F64(view)))
    });
}

#[test]
fn managed_cholesky_read_keeps_nonzero_storage_work_inside_one_entry() {
    let domain = FakeDomain::new();
    let (mut backend, installs, submits) = external_no_inner_managed_backend(&domain);
    let input = Tensor::F64(domain.tensor(&[2, 2], vec![4.0_f64, 2.0, 2.0, 3.0]));

    let output = backend
        .cholesky_read(TensorRead::from_tensor(&input))
        .unwrap();

    assert_eq!(installs.load(Ordering::Relaxed), 1);
    assert_eq!(submits.load(Ordering::Relaxed), 0);
    assert_eq!(domain.counts.reads.load(Ordering::Relaxed), 1);
    assert_eq!(domain.counts.writes.load(Ordering::Relaxed), 1);
    assert_eq!(domain.counts.allocations.load(Ordering::Relaxed), 1);
    assert_eq!(domain.counts.outside_entry.load(Ordering::Relaxed), 0);
    let Tensor::F64(output) = output else {
        panic!("managed Cholesky should preserve f64 dtype")
    };
    let Buffer::Backend(buffer) = output.buffer() else {
        panic!("managed Cholesky should preserve backend storage")
    };
    let values = buffer.map_read().unwrap();
    let expected = [2.0_f64, 1.0, 0.0, 2.0_f64.sqrt()];
    for (actual, expected) in values.iter().zip(expected) {
        assert!((actual - expected).abs() <= 1.0e-12);
    }
    drop(values);

    let next = Tensor::from_vec_col_major([1, 1], vec![9.0_f64]).unwrap();
    assert_eq!(
        backend
            .cholesky_read(TensorRead::from_tensor(&next))
            .unwrap()
            .as_slice::<f64>()
            .unwrap(),
        &[3.0]
    );
    assert_eq!(installs.load(Ordering::Relaxed), 2);
    assert_eq!(submits.load(Ordering::Relaxed), 0);
}

#[test]
fn managed_cholesky_read_keeps_zero_size_output_work_inside_one_entry() {
    let domain = FakeDomain::new();
    let (mut backend, installs, submits) = external_no_inner_managed_backend(&domain);
    let input = Tensor::F64(domain.tensor(&[0, 0], Vec::<f64>::new()));

    let output = backend
        .cholesky_read(TensorRead::from_tensor(&input))
        .unwrap();

    assert_eq!(installs.load(Ordering::Relaxed), 1);
    assert_eq!(submits.load(Ordering::Relaxed), 0);
    assert_eq!(domain.counts.reads.load(Ordering::Relaxed), 0);
    assert_eq!(domain.counts.writes.load(Ordering::Relaxed), 1);
    assert_eq!(domain.counts.allocations.load(Ordering::Relaxed), 1);
    assert_eq!(domain.counts.outside_entry.load(Ordering::Relaxed), 0);
    assert_eq!(output.shape(), &[0, 0]);

    let next = Tensor::from_vec_col_major([1, 1], vec![4.0_f64]).unwrap();
    assert_eq!(
        backend
            .cholesky_read(TensorRead::from_tensor(&next))
            .unwrap()
            .as_slice::<f64>()
            .unwrap(),
        &[2.0]
    );
    assert_eq!(installs.load(Ordering::Relaxed), 2);
    assert_eq!(submits.load(Ordering::Relaxed), 0);
}

#[derive(Debug)]
struct RejectOnceExecutor {
    reject_next: AtomicBool,
    installs: Arc<AtomicUsize>,
    submits: Arc<AtomicUsize>,
    job_runs: Arc<AtomicUsize>,
}

impl CpuDomainExecutor for RejectOnceExecutor {
    fn capabilities(&self) -> CpuDomainExecutorCapabilities {
        CpuDomainExecutorCapabilities {
            worker_count: NonZeroUsize::new(1).unwrap(),
            outer_parallelism: false,
            inner_parallelism: CpuInnerParallelism::None,
            reentrancy: CpuExecutorReentrancy::Rejected,
            affinity: CpuExecutorAffinity::CallerDeclaredUnverified,
            shutdown: CpuExecutorShutdown::CallerOwned,
        }
    }

    fn submit(&self, jobs: &dyn ScopedCpuJobs) -> Result<(), CpuDomainExecutorError> {
        self.submits.fetch_add(1, Ordering::Relaxed);
        for index in 0..jobs.len() {
            jobs.run(index)?;
        }
        Ok(())
    }

    fn install(&self, job: &mut dyn ScopedCpuJob) -> Result<(), CpuDomainExecutorError> {
        self.installs.fetch_add(1, Ordering::Relaxed);
        if self.reject_next.swap(false, Ordering::Relaxed) {
            return Err(CpuDomainExecutorError::Admission {
                message: "single-entry test rejection".to_owned(),
            });
        }
        self.job_runs.fetch_add(1, Ordering::Relaxed);
        job.run()
    }
}

#[test]
fn executor_rejection_preserves_input_and_pool_then_recovers() {
    let installs = Arc::new(AtomicUsize::new(0));
    let submits = Arc::new(AtomicUsize::new(0));
    let job_runs = Arc::new(AtomicUsize::new(0));
    let executor = Arc::new(RejectOnceExecutor {
        reject_next: AtomicBool::new(true),
        installs: Arc::clone(&installs),
        submits: Arc::clone(&submits),
        job_runs: Arc::clone(&job_runs),
    });
    let mut backend = external_backend(executor);
    let input = TypedTensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 3.0]).unwrap();
    let input_before = input.host_data().unwrap().to_vec();
    let pool_before = backend.buffer_pool_stats().unwrap();
    let reversed = [StridedSliceSpec::reverse(), StridedSliceSpec::reverse()];
    let view = input.as_view().try_slice(&reversed).unwrap();

    let error = backend
        .svd_read(TensorRead::from_view(TensorView::F64(view)))
        .unwrap_err();

    assert!(matches!(
        error.source().and_then(|source| source.downcast_ref()),
        Some(CpuDomainExecutorError::Admission { message })
            if message == "single-entry test rejection"
    ));
    assert_eq!(installs.load(Ordering::Relaxed), 1);
    assert_eq!(submits.load(Ordering::Relaxed), 0);
    assert_eq!(job_runs.load(Ordering::Relaxed), 0);
    assert_eq!(input.host_data().unwrap(), input_before.as_slice());
    assert_eq!(backend.buffer_pool_stats().unwrap(), pool_before);

    let view = input.as_view().try_slice(&reversed).unwrap();
    let outputs = backend
        .svd_read(TensorRead::from_view(TensorView::F64(view)))
        .unwrap();
    assert_eq!(outputs.len(), 3);
    assert_eq!(installs.load(Ordering::Relaxed), 2);
    assert_eq!(submits.load(Ordering::Relaxed), 0);
    assert_eq!(job_runs.load(Ordering::Relaxed), 1);
}

#[test]
fn provider_independent_cholesky_failure_uses_one_entry_and_recovers() {
    let (mut backend, installs, submits) = external_no_inner_backend();
    let invalid =
        TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 2.0, 1.0]).unwrap();
    let reversed = [StridedSliceSpec::reverse(), StridedSliceSpec::reverse()];
    let view = invalid.as_view().try_slice(&reversed).unwrap();

    let error = backend
        .cholesky_read(TensorRead::from_view(TensorView::F64(view)))
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NumericalFailure);
    assert!(matches!(
        error
            .source()
            .and_then(|source| source.downcast_ref::<crate::Error>()),
        Some(crate::Error::NonConvergence { op: "cholesky" })
    ));
    assert_eq!(installs.load(Ordering::Relaxed), 1);
    assert_eq!(submits.load(Ordering::Relaxed), 0);

    let valid = TypedTensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 0.0, 0.0, 9.0]).unwrap();
    let view = valid.as_view().try_slice(&reversed).unwrap();
    let output = backend
        .cholesky_read(TensorRead::from_view(TensorView::F64(view)))
        .unwrap();
    assert_eq!(output.shape(), &[2, 2]);
    assert_eq!(installs.load(Ordering::Relaxed), 2);
    assert_eq!(submits.load(Ordering::Relaxed), 0);
}

#[test]
fn linalg_provider_panic_allows_next_operation_without_clearing_stats_poison() {
    let (mut backend, installs, submits) = external_no_inner_backend();

    let panic = catch_unwind(AssertUnwindSafe(|| {
        let _ = backend.with_linalg_pool::<()>(|_, _| panic!("single-entry provider panic"));
    }));
    assert!(panic.is_err());
    assert_eq!(installs.load(Ordering::Relaxed), 1);
    assert_eq!(submits.load(Ordering::Relaxed), 0);

    let valid = TypedTensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 3.0]).unwrap();
    let reversed = [StridedSliceSpec::reverse(), StridedSliceSpec::reverse()];
    let view = valid.as_view().try_slice(&reversed).unwrap();
    // This assertion covers operation and pool recovery. The separate public
    // stats API intentionally preserves its documented poisoned-lock error.
    assert_eq!(
        backend
            .svd_read(TensorRead::from_view(TensorView::F64(view)))
            .unwrap()
            .len(),
        3
    );
    assert_eq!(installs.load(Ordering::Relaxed), 2);
    assert_eq!(submits.load(Ordering::Relaxed), 0);
}
