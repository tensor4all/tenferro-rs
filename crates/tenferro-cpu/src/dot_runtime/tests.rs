use super::{
    validate_axis_groups, validate_dot_general, validate_layout_metadata, CpuProviderBundle,
    GroupedJobState, PackedJobStates, GROUPED_INLINE_JOB_CAPACITY, GROUPED_JOBS_PER_STATE_WORD,
};
use crate::buffer_pool::{BufferPool, PoolScalar};
use crate::gemm::GemmAnalysisCache;
use crate::provider::tests::{execution_context_fixture, external_execution_context_fixture};
use crate::provider::{
    CpuExecutionContext, CpuGemmProvider, CpuGemmRequest, CpuGeneralContractionProvider,
    CpuGroupedGemmRequest, CpuLayoutTransformIntent, CpuLayoutTransformProvider,
    CpuLayoutTransformRequest, CpuOperand, CpuProviderOutcome, CpuProviderUnsupported,
    CpuUninitLayoutTransformProvider, ParallelMode, StridedLayoutTransformProvider,
};
use crate::{
    CpuBackendKind, CpuDomainExecutor, CpuDomainExecutorCapabilities, CpuDomainExecutorError,
    CpuExecutorAffinity, CpuExecutorReentrancy, CpuExecutorShutdown, CpuInnerParallelism,
    ScopedCpuJob, ScopedCpuJobs,
};
use num_complex::Complex64;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use tenferro_tensor::backend::{GroupedGemmConfig, GroupedGemmJob};
use tenferro_tensor::{
    ContractionScalar, DType, DotGeneralAccumulation, DotGeneralConfig, ErrorKind, Tensor,
    TensorRead, TensorViewMut, TensorWrite, TypedTensorViewMut, ValidationKind,
};

#[derive(Debug)]
struct CountingExecutor {
    submits: Arc<AtomicUsize>,
    installs: Arc<AtomicUsize>,
}

fn outer_executor_capabilities() -> CpuDomainExecutorCapabilities {
    CpuDomainExecutorCapabilities {
        worker_count: NonZeroUsize::new(4).unwrap(),
        outer_parallelism: true,
        inner_parallelism: CpuInnerParallelism::Rayon,
        reentrancy: CpuExecutorReentrancy::Rejected,
        affinity: CpuExecutorAffinity::CallerDeclaredUnverified,
        shutdown: CpuExecutorShutdown::CallerOwned,
    }
}

impl CpuDomainExecutor for CountingExecutor {
    fn capabilities(&self) -> CpuDomainExecutorCapabilities {
        outer_executor_capabilities()
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
        job.run()
    }
}

#[derive(Debug)]
struct RejectingInstallExecutor {
    submits: Arc<AtomicUsize>,
    installs: Arc<AtomicUsize>,
}

impl CpuDomainExecutor for RejectingInstallExecutor {
    fn capabilities(&self) -> CpuDomainExecutorCapabilities {
        outer_executor_capabilities()
    }

    fn submit(&self, _jobs: &dyn ScopedCpuJobs) -> Result<(), CpuDomainExecutorError> {
        self.submits.fetch_add(1, Ordering::Relaxed);
        Err(CpuDomainExecutorError::Scheduling {
            message: "unexpected submit while testing install failure".to_owned(),
        })
    }

    fn install(&self, _job: &mut dyn ScopedCpuJob) -> Result<(), CpuDomainExecutorError> {
        self.installs.fetch_add(1, Ordering::Relaxed);
        Err(CpuDomainExecutorError::Cancellation {
            message: "intentional install rejection".to_owned(),
        })
    }
}

#[derive(Debug)]
struct ConcurrentDuplicateExecutor;

impl CpuDomainExecutor for ConcurrentDuplicateExecutor {
    fn capabilities(&self) -> CpuDomainExecutorCapabilities {
        outer_executor_capabilities()
    }

    fn submit(&self, jobs: &dyn ScopedCpuJobs) -> Result<(), CpuDomainExecutorError> {
        std::thread::scope(|scope| {
            let first = scope.spawn(|| jobs.run(0));
            let duplicate = scope.spawn(|| jobs.run(0));
            // Deliberately ignore both job results: a safe custom executor can
            // violate the documented contract while still returning success.
            let _ = first.join().unwrap();
            let _ = duplicate.join().unwrap();
        });
        Ok(())
    }

    fn install(&self, job: &mut dyn ScopedCpuJob) -> Result<(), CpuDomainExecutorError> {
        job.run()
    }
}

#[derive(Debug)]
struct MissingJobExecutor;

impl CpuDomainExecutor for MissingJobExecutor {
    fn capabilities(&self) -> CpuDomainExecutorCapabilities {
        outer_executor_capabilities()
    }

    fn submit(&self, jobs: &dyn ScopedCpuJobs) -> Result<(), CpuDomainExecutorError> {
        jobs.run(0)
    }

    fn install(&self, job: &mut dyn ScopedCpuJob) -> Result<(), CpuDomainExecutorError> {
        job.run()
    }
}

#[derive(Debug)]
struct FailingSubmitExecutor;

impl CpuDomainExecutor for FailingSubmitExecutor {
    fn capabilities(&self) -> CpuDomainExecutorCapabilities {
        outer_executor_capabilities()
    }

    fn submit(&self, jobs: &dyn ScopedCpuJobs) -> Result<(), CpuDomainExecutorError> {
        jobs.run(0)?;
        Err(CpuDomainExecutorError::Cancellation {
            message: "authoritative submit failure".to_string(),
        })
    }

    fn install(&self, job: &mut dyn ScopedCpuJob) -> Result<(), CpuDomainExecutorError> {
        job.run()
    }
}

#[derive(Debug)]
struct IgnoredInvalidIndexExecutor {
    requested_index: usize,
}

#[derive(Debug)]
struct InvalidThenFailingSubmitExecutor;

impl CpuDomainExecutor for InvalidThenFailingSubmitExecutor {
    fn capabilities(&self) -> CpuDomainExecutorCapabilities {
        outer_executor_capabilities()
    }

    fn submit(&self, jobs: &dyn ScopedCpuJobs) -> Result<(), CpuDomainExecutorError> {
        for index in 0..jobs.len() {
            jobs.run(index)?;
        }
        let _ = jobs.run(jobs.len());
        Err(CpuDomainExecutorError::Cancellation {
            message: "submit failure after ignored invalid index".to_string(),
        })
    }

    fn install(&self, job: &mut dyn ScopedCpuJob) -> Result<(), CpuDomainExecutorError> {
        job.run()
    }
}

#[derive(Debug)]
struct SequentialDuplicateAllValidExecutor;

impl CpuDomainExecutor for SequentialDuplicateAllValidExecutor {
    fn capabilities(&self) -> CpuDomainExecutorCapabilities {
        outer_executor_capabilities()
    }

    fn submit(&self, jobs: &dyn ScopedCpuJobs) -> Result<(), CpuDomainExecutorError> {
        for index in 0..jobs.len() {
            jobs.run(index)?;
        }
        let _ = jobs.run(0);
        Ok(())
    }

    fn install(&self, job: &mut dyn ScopedCpuJob) -> Result<(), CpuDomainExecutorError> {
        job.run()
    }
}

#[derive(Debug)]
struct CatchProviderPanicExecutor;

impl CpuDomainExecutor for CatchProviderPanicExecutor {
    fn capabilities(&self) -> CpuDomainExecutorCapabilities {
        outer_executor_capabilities()
    }

    fn submit(&self, jobs: &dyn ScopedCpuJobs) -> Result<(), CpuDomainExecutorError> {
        for index in 0..jobs.len() {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| jobs.run(index)));
        }
        Ok(())
    }

    fn install(&self, job: &mut dyn ScopedCpuJob) -> Result<(), CpuDomainExecutorError> {
        job.run()
    }
}

#[derive(Debug)]
struct BarrierDuplicateExecutor {
    provider_entered: Arc<Barrier>,
    release_provider: Arc<Barrier>,
    duplicate_failed: Arc<AtomicBool>,
}

impl CpuDomainExecutor for BarrierDuplicateExecutor {
    fn capabilities(&self) -> CpuDomainExecutorCapabilities {
        outer_executor_capabilities()
    }

    fn submit(&self, jobs: &dyn ScopedCpuJobs) -> Result<(), CpuDomainExecutorError> {
        std::thread::scope(|scope| {
            let first = scope.spawn(|| jobs.run(0));
            self.provider_entered.wait();
            self.duplicate_failed
                .store(jobs.run(0).is_err(), Ordering::Release);
            self.release_provider.wait();
            first
                .join()
                .map_err(|_| CpuDomainExecutorError::Cancellation {
                    message: "barrier-backed grouped job panicked".to_string(),
                })??;
            for index in 1..jobs.len() {
                jobs.run(index)?;
            }
            Ok(())
        })
    }

    fn install(&self, job: &mut dyn ScopedCpuJob) -> Result<(), CpuDomainExecutorError> {
        job.run()
    }
}

impl CpuDomainExecutor for IgnoredInvalidIndexExecutor {
    fn capabilities(&self) -> CpuDomainExecutorCapabilities {
        outer_executor_capabilities()
    }

    fn submit(&self, jobs: &dyn ScopedCpuJobs) -> Result<(), CpuDomainExecutorError> {
        for index in 0..jobs.len() {
            jobs.run(index)?;
        }
        // A safe custom executor may discard the typed error from `run` and
        // still claim that the overall submission succeeded.
        let _ = jobs.run(self.requested_index);
        Ok(())
    }

    fn install(&self, job: &mut dyn ScopedCpuJob) -> Result<(), CpuDomainExecutorError> {
        job.run()
    }
}

fn assert_grouped_scheduling_error(error: &tenferro_tensor::Error, expected: &str) {
    let tenferro_tensor::Error::BackendSource { source, .. } = error else {
        panic!("grouped executor failure must retain a typed source: {error}");
    };
    assert!(matches!(
        source.downcast_ref::<CpuDomainExecutorError>(),
        Some(CpuDomainExecutorError::Scheduling { message }) if message.contains(expected)
    ));
}

fn counting_execution_fixture(
    thread_budget: usize,
) -> (
    Arc<AtomicUsize>,
    Arc<AtomicUsize>,
    crate::provider::tests::CpuExecutionContextFixture,
) {
    let submits = Arc::new(AtomicUsize::new(0));
    let installs = Arc::new(AtomicUsize::new(0));
    let executor: Arc<dyn CpuDomainExecutor> = Arc::new(CountingExecutor {
        submits: Arc::clone(&submits),
        installs: Arc::clone(&installs),
    });
    let fixture =
        external_execution_context_fixture(executor, NonZeroUsize::new(thread_budget).unwrap());
    (submits, installs, fixture)
}

fn config(
    lhs_contracting_dims: &[usize],
    rhs_contracting_dims: &[usize],
    lhs_batch_dims: &[usize],
    rhs_batch_dims: &[usize],
) -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: lhs_contracting_dims.to_vec(),
        rhs_contracting_dims: rhs_contracting_dims.to_vec(),
        lhs_batch_dims: lhs_batch_dims.to_vec(),
        rhs_batch_dims: rhs_batch_dims.to_vec(),
    }
}

#[test]
fn axis_groups_preserve_order_and_find_free_axes() {
    let config = config(&[1], &[0], &[2], &[2]);
    let groups = validate_axis_groups(4, 4, &config).unwrap();

    assert_eq!(groups.contracting_pairs().collect::<Vec<_>>(), vec![(1, 0)]);
    assert_eq!(groups.batch_pairs().collect::<Vec<_>>(), vec![(2, 2)]);
    assert_eq!(groups.lhs_free_axes().collect::<Vec<_>>(), vec![0, 3]);
    assert_eq!(groups.rhs_free_axes().collect::<Vec<_>>(), vec![1, 3]);
}

#[test]
fn axis_groups_match_existing_rank_validation_through_rank_seventy() {
    for rank in [0, 1, 2, 8, 63, 64, 65, 70] {
        let valid = if rank == 0 {
            config(&[], &[], &[], &[])
        } else if rank == 1 {
            config(&[0], &[0], &[], &[])
        } else {
            config(&[rank - 1], &[0], &[rank - 2], &[rank - 1])
        };
        let invalid = [
            config(&[rank], &[0], &[], &[]),
            config(&[0, 0], &[0, 1], &[], &[]),
            config(&[0], &[0], &[0], &[0]),
            config(&[0], &[], &[], &[]),
            config(&[], &[], &[0], &[]),
        ];

        assert_eq!(
            validate_axis_groups(rank, rank, &valid).is_ok(),
            valid.validate_dims_with_ranks(rank, rank).is_ok(),
            "valid parity failed at rank {rank}",
        );
        for candidate in invalid {
            assert_eq!(
                validate_axis_groups(rank, rank, &candidate).is_ok(),
                candidate.validate_dims_with_ranks(rank, rank).is_ok(),
                "invalid parity failed at rank {rank}: {candidate:?}",
            );
        }
    }
}

#[test]
fn axis_group_role_conflict_preserves_ordered_error_parity() {
    let config = config(&[5, 2], &[0, 1], &[2, 5], &[2, 3]);
    let current = config.validate_dims_with_ranks(6, 6).unwrap_err();
    let candidate = validate_axis_groups(6, 6, &config).unwrap_err();

    assert_eq!(candidate.to_string(), current.to_string());
}

#[test]
fn axis_group_competing_errors_match_existing_precedence_through_rank_seventy() {
    for rank in [2, 8, 63, 64, 65, 70] {
        let cases = [
            config(&[0, 0], &[rank], &[], &[]),
            config(&[0, 0], &[0, 0], &[1, 1], &[1, 1]),
            config(&[0], &[], &[0], &[]),
            config(&[rank], &[rank], &[0, 0], &[0, 0]),
        ];
        for candidate in cases {
            let current = candidate.validate_dims_with_ranks(rank, rank).unwrap_err();
            let replacement = validate_axis_groups(rank, rank, &candidate).unwrap_err();
            assert_eq!(
                replacement.to_string(),
                current.to_string(),
                "error precedence diverged at rank {rank} for {candidate:?}",
            );
        }
    }
}

#[test]
fn dot_general_validation_checks_extents_output_and_accumulation() {
    let lhs = Tensor::from_vec_col_major(vec![2, 3, 4], vec![1.0_f64; 24]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![3, 5, 4], vec![1.0_f64; 60]).unwrap();
    let mut output = Tensor::from_vec_col_major(vec![2, 5, 4], vec![0.0_f64; 40]).unwrap();
    let lhs = TensorRead::from_tensor(&lhs);
    let rhs = TensorRead::from_tensor(&rhs);
    let mut output = TensorWrite::from_tensor(&mut output);
    let config = config(&[1], &[0], &[2], &[2]);
    let accumulation = DotGeneralAccumulation::overwrite(DType::F64).unwrap();

    let validated = validate_dot_general(&lhs, &rhs, &output, &config, accumulation).unwrap();
    assert_eq!(validated.output_element_count(), 40);
    assert_eq!(
        validated.axes().lhs_free_axes().collect::<Vec<_>>(),
        vec![0]
    );

    let wrong_accumulation = DotGeneralAccumulation {
        lhs_conj: false,
        rhs_conj: false,
        alpha: ContractionScalar::F32(1.0),
        beta: ContractionScalar::F32(0.0),
    };
    assert!(validate_dot_general(&lhs, &rhs, &output, &config, wrong_accumulation).is_err());

    let mut wrong_shape = Tensor::from_vec_col_major(vec![2, 5], vec![0.0_f64; 10]).unwrap();
    let wrong_shape = TensorWrite::from_tensor(&mut wrong_shape);
    assert!(validate_dot_general(&lhs, &rhs, &wrong_shape, &config, accumulation).is_err());

    let bad_rhs = Tensor::from_vec_col_major(vec![7, 5, 4], vec![1.0_f64; 140]).unwrap();
    let bad_rhs = TensorRead::from_tensor(&bad_rhs);
    assert!(validate_dot_general(&lhs, &bad_rhs, &output, &config, accumulation).is_err());

    let _ = &mut output;
}

#[test]
fn layout_validation_checks_strides_and_reachable_ranges() {
    assert!(validate_layout_metadata("output", &[2, 3], &[1], 0, 6).is_err());
    assert!(validate_layout_metadata("output", &[2], &[-1], 0, 2).is_err());
    assert!(validate_layout_metadata("output", &[2], &[isize::MAX], 1, 2).is_err());
    assert!(validate_layout_metadata("output", &[2, 3], &[1, 2], 0, 5).is_err());
    assert!(validate_layout_metadata("output", &[2, 3], &[-1, 2], 1, 6).is_ok());
}

#[test]
fn dot_general_validation_accepts_checked_negative_stride_output() {
    let lhs = Tensor::from_vec_col_major(vec![2, 3, 4], vec![1.0_f64; 24]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![3, 5, 4], vec![1.0_f64; 60]).unwrap();
    let mut output_storage = vec![0.0_f64; 40];
    let output =
        TypedTensorViewMut::from_slice(vec![2, 5, 4], vec![-1, 2, 10], 1, &mut output_storage)
            .unwrap();
    let output = TensorWrite::from_view(TensorViewMut::F64(output));
    let lhs = TensorRead::from_tensor(&lhs);
    let rhs = TensorRead::from_tensor(&rhs);
    let config = config(&[1], &[0], &[2], &[2]);
    let accumulation = DotGeneralAccumulation::overwrite(DType::F64).unwrap();

    let validated = validate_dot_general(&lhs, &rhs, &output, &config, accumulation).unwrap();
    assert_eq!(validated.output_element_count(), 40);
}

#[derive(Clone, Copy, Debug)]
enum GeneralBehavior {
    Outcome(CpuProviderOutcome),
    Error,
}

#[derive(Debug)]
struct GeneralSpy {
    behavior: GeneralBehavior,
    calls: Arc<Mutex<usize>>,
}

impl CpuGeneralContractionProvider for GeneralSpy {
    fn execution_capabilities(&self) -> crate::CpuProviderExecutionCapabilities {
        crate::provider_capability::engine_worker_capabilities()
    }

    fn dot_general(
        &self,
        _context: &CpuExecutionContext<'_>,
        _request: crate::provider::CpuDotGeneralRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        *self.calls.lock().unwrap() += 1;
        match self.behavior {
            GeneralBehavior::Outcome(outcome) => Ok(outcome),
            GeneralBehavior::Error => Err(tenferro_tensor::Error::runtime_state(
                "dot_general",
                "general spy failure",
            )),
        }
    }
}

#[derive(Debug)]
struct GemmSpy {
    behavior: GemmBehavior,
    capabilities: crate::CpuProviderExecutionCapabilities,
    gemm_calls: Arc<Mutex<usize>>,
    strided_calls: Arc<Mutex<usize>>,
    grouped_calls: Arc<Mutex<usize>>,
    parallelism: Arc<Mutex<Vec<ParallelMode>>>,
    grouped_job_counts: Arc<Mutex<Vec<usize>>>,
    in_selected_pool: Arc<Mutex<Vec<bool>>>,
}

#[derive(Clone, Copy, Debug)]
enum GemmBehavior {
    Outcome(CpuProviderOutcome),
    Error,
}

impl GemmSpy {
    fn new(outcome: CpuProviderOutcome) -> Self {
        Self {
            behavior: GemmBehavior::Outcome(outcome),
            capabilities: crate::provider_capability::engine_worker_capabilities(),
            gemm_calls: Arc::new(Mutex::new(0)),
            strided_calls: Arc::new(Mutex::new(0)),
            grouped_calls: Arc::new(Mutex::new(0)),
            parallelism: Arc::new(Mutex::new(Vec::new())),
            grouped_job_counts: Arc::new(Mutex::new(Vec::new())),
            in_selected_pool: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn error() -> Self {
        let mut spy = Self::new(CpuProviderOutcome::Executed);
        spy.behavior = GemmBehavior::Error;
        spy
    }

    fn with_capabilities(mut self, capabilities: crate::CpuProviderExecutionCapabilities) -> Self {
        self.capabilities = capabilities;
        self
    }

    fn result(&self) -> tenferro_tensor::Result<CpuProviderOutcome> {
        match self.behavior {
            GemmBehavior::Outcome(outcome) => Ok(outcome),
            GemmBehavior::Error => Err(tenferro_tensor::Error::runtime_state(
                "dot_general",
                "GEMM provider spy failure",
            )),
        }
    }
}

impl CpuGemmProvider for GemmSpy {
    fn execution_capabilities(&self) -> crate::CpuProviderExecutionCapabilities {
        self.capabilities
    }

    fn gemm(
        &self,
        context: &CpuExecutionContext<'_>,
        _request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        *self.gemm_calls.lock().unwrap() += 1;
        self.parallelism
            .lock()
            .unwrap()
            .push(context.parallel_mode());
        self.result()
    }

    fn strided_batched_gemm(
        &self,
        context: &CpuExecutionContext<'_>,
        _request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        *self.strided_calls.lock().unwrap() += 1;
        self.parallelism
            .lock()
            .unwrap()
            .push(context.parallel_mode());
        self.result()
    }

    fn grouped_gemm(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuGroupedGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        *self.grouped_calls.lock().unwrap() += 1;
        self.parallelism
            .lock()
            .unwrap()
            .push(context.parallel_mode());
        self.grouped_job_counts
            .lock()
            .unwrap()
            .push(request.jobs().len());
        self.in_selected_pool
            .lock()
            .unwrap()
            .push(rayon::current_thread_index().is_some());
        self.result()
    }
}

#[derive(Clone, Copy, Debug)]
enum CanonicalFallbackBehavior {
    Real,
    ConjugatedComplex,
}

#[derive(Clone, Copy, Debug)]
enum CanonicalFallbackTerminal {
    Unsupported(CpuProviderUnsupported),
    Error,
}

#[derive(Debug)]
struct CanonicalFallbackSpy {
    first_outcome: CpuProviderUnsupported,
    second_terminal: Option<CanonicalFallbackTerminal>,
    behavior: CanonicalFallbackBehavior,
    calls: Arc<Mutex<usize>>,
}

impl CanonicalFallbackSpy {
    fn new(first_outcome: CpuProviderUnsupported, behavior: CanonicalFallbackBehavior) -> Self {
        Self {
            first_outcome,
            second_terminal: None,
            behavior,
            calls: Arc::new(Mutex::new(0)),
        }
    }

    fn rejecting_retry(
        first_outcome: CpuProviderUnsupported,
        second_outcome: CpuProviderUnsupported,
    ) -> Self {
        Self {
            first_outcome,
            second_terminal: Some(CanonicalFallbackTerminal::Unsupported(second_outcome)),
            behavior: CanonicalFallbackBehavior::Real,
            calls: Arc::new(Mutex::new(0)),
        }
    }

    fn erroring_retry(first_outcome: CpuProviderUnsupported) -> Self {
        Self {
            first_outcome,
            second_terminal: Some(CanonicalFallbackTerminal::Error),
            behavior: CanonicalFallbackBehavior::Real,
            calls: Arc::new(Mutex::new(0)),
        }
    }

    fn execute(
        &self,
        request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        let call = {
            let mut calls = self.calls.lock().unwrap();
            *calls += 1;
            *calls
        };
        if call == 1 {
            return Ok(CpuProviderOutcome::Unsupported(self.first_outcome));
        }
        assert_eq!(call, 2, "canonical fallback must retry exactly once");
        if let Some(terminal) = self.second_terminal {
            return match terminal {
                CanonicalFallbackTerminal::Unsupported(reason) => {
                    Ok(CpuProviderOutcome::Unsupported(reason))
                }
                CanonicalFallbackTerminal::Error => Err(tenferro_tensor::Error::runtime_state(
                    "dot_general",
                    "canonical retry spy failure",
                )),
            };
        }

        let parts = request.into_parts();
        assert_eq!(parts.lhs_layout.row_stride(), 1);
        assert_eq!(parts.rhs_layout.row_stride(), 1);
        assert_eq!(parts.output_layout.row_stride(), 1);
        match self.behavior {
            CanonicalFallbackBehavior::Real => {
                assert!(!parts.accumulation.lhs_conj);
                assert!(!parts.accumulation.rhs_conj);
                match &mut *parts.output {
                    TensorWrite::Tensor(Tensor::F64(output)) => {
                        assert_eq!(output.host_data()?, &[41.0; 4]);
                        output
                            .host_data_mut()?
                            .copy_from_slice(&[19.0, 43.0, 22.0, 50.0]);
                    }
                    other => panic!("unexpected fallback output: {other:?}"),
                }
            }
            CanonicalFallbackBehavior::ConjugatedComplex => {
                assert!(!parts.accumulation.lhs_conj);
                assert!(!parts.accumulation.rhs_conj);
                assert_eq!(
                    parts.accumulation.alpha,
                    ContractionScalar::C64(Complex64::new(2.0, 0.0)),
                );
                assert_eq!(
                    parts.accumulation.beta,
                    ContractionScalar::C64(Complex64::new(3.0, 0.0)),
                );
                match parts.lhs {
                    TensorRead::Tensor(Tensor::C64(lhs)) => {
                        assert_eq!(lhs.host_data()?, &[Complex64::new(1.0, -2.0)]);
                    }
                    other => panic!("conjugated lhs was not materialized: {other:?}"),
                }
                match parts.rhs {
                    TensorRead::Tensor(Tensor::C64(rhs)) => {
                        assert_eq!(rhs.host_data()?, &[Complex64::new(3.0, 4.0)]);
                    }
                    other => panic!("rhs was not materialized: {other:?}"),
                }
                match &mut *parts.output {
                    TensorWrite::Tensor(Tensor::C64(output)) => {
                        assert_eq!(output.host_data()?, &[Complex64::new(5.0, 1.0)]);
                        output.host_data_mut()?[0] = Complex64::new(37.0, -1.0);
                    }
                    other => panic!("unexpected fallback output: {other:?}"),
                }
            }
        }
        Ok(CpuProviderOutcome::Executed)
    }
}

impl CpuGemmProvider for CanonicalFallbackSpy {
    fn execution_capabilities(&self) -> crate::CpuProviderExecutionCapabilities {
        crate::provider_capability::engine_worker_capabilities()
    }

    fn gemm(
        &self,
        _context: &CpuExecutionContext<'_>,
        request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        self.execute(request)
    }

    fn strided_batched_gemm(
        &self,
        _context: &CpuExecutionContext<'_>,
        request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        self.execute(request)
    }

    fn grouped_gemm(
        &self,
        _context: &CpuExecutionContext<'_>,
        _request: CpuGroupedGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        panic!("canonical fallback tests do not issue grouped GEMM")
    }
}

#[derive(Debug)]
struct PanicOnceGemmProvider {
    calls: Arc<AtomicUsize>,
}

impl CpuGemmProvider for PanicOnceGemmProvider {
    fn execution_capabilities(&self) -> crate::CpuProviderExecutionCapabilities {
        crate::provider_capability::engine_worker_capabilities()
    }

    fn gemm(
        &self,
        _context: &CpuExecutionContext<'_>,
        _request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        Err(tenferro_tensor::Error::runtime_state(
            "panic_once_test",
            "unexpected scalar GEMM request",
        ))
    }

    fn strided_batched_gemm(
        &self,
        _context: &CpuExecutionContext<'_>,
        _request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        Err(tenferro_tensor::Error::runtime_state(
            "panic_once_test",
            "unexpected strided GEMM request",
        ))
    }

    fn grouped_gemm(
        &self,
        _context: &CpuExecutionContext<'_>,
        _request: CpuGroupedGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        if self.calls.fetch_add(1, Ordering::Relaxed) == 0 {
            panic!("intentional grouped provider panic");
        }
        Ok(CpuProviderOutcome::Executed)
    }
}

#[derive(Debug)]
struct ProviderErrorSpy {
    calls: Arc<AtomicUsize>,
}

impl CpuGemmProvider for ProviderErrorSpy {
    fn execution_capabilities(&self) -> crate::CpuProviderExecutionCapabilities {
        crate::provider_capability::engine_worker_capabilities()
    }

    fn gemm(
        &self,
        _context: &CpuExecutionContext<'_>,
        _request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        Err(tenferro_tensor::Error::runtime_state(
            "grouped_provider_sentinel",
            "unexpected scalar GEMM request",
        ))
    }

    fn strided_batched_gemm(
        &self,
        _context: &CpuExecutionContext<'_>,
        _request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        Err(tenferro_tensor::Error::runtime_state(
            "grouped_provider_sentinel",
            "unexpected strided GEMM request",
        ))
    }

    fn grouped_gemm(
        &self,
        _context: &CpuExecutionContext<'_>,
        _request: CpuGroupedGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Err(tenferro_tensor::Error::RuntimeState {
            op: "grouped_provider_sentinel",
            message: "exact provider failure".to_string(),
        })
    }
}

#[derive(Debug)]
struct BarrierMutationGemmProvider {
    calls: Arc<AtomicUsize>,
    provider_entered: Arc<Barrier>,
    release_provider: Arc<Barrier>,
}

impl CpuGemmProvider for BarrierMutationGemmProvider {
    fn execution_capabilities(&self) -> crate::CpuProviderExecutionCapabilities {
        crate::provider_capability::engine_worker_capabilities()
    }

    fn gemm(
        &self,
        _context: &CpuExecutionContext<'_>,
        _request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        Err(tenferro_tensor::Error::runtime_state(
            "barrier_mutation_test",
            "unexpected scalar GEMM request",
        ))
    }

    fn strided_batched_gemm(
        &self,
        _context: &CpuExecutionContext<'_>,
        _request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        Err(tenferro_tensor::Error::runtime_state(
            "barrier_mutation_test",
            "unexpected strided GEMM request",
        ))
    }

    fn grouped_gemm(
        &self,
        _context: &CpuExecutionContext<'_>,
        mut request: CpuGroupedGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        let call = self.calls.fetch_add(1, Ordering::Relaxed);
        if call == 0 {
            self.provider_entered.wait();
            self.release_provider.wait();
        }
        let TensorWrite::View(TensorViewMut::F64(output)) = request.output() else {
            return Err(tenferro_tensor::Error::runtime_state(
                "barrier_mutation_test",
                "expected an f64 output view",
            ));
        };
        output.host_storage_mut()?[0] += 1.0;
        Ok(CpuProviderOutcome::Executed)
    }
}

#[derive(Debug)]
struct LayoutSpy {
    calls: Arc<Mutex<usize>>,
}

impl CpuLayoutTransformProvider for LayoutSpy {
    fn execution_capabilities(&self) -> crate::CpuProviderExecutionCapabilities {
        StridedLayoutTransformProvider.execution_capabilities()
    }

    fn materialize(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuLayoutTransformRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        *self.calls.lock().unwrap() += 1;
        StridedLayoutTransformProvider.materialize(context, request)
    }
}

fn changing_capabilities(calls: &AtomicUsize) -> crate::CpuProviderExecutionCapabilities {
    if calls.fetch_add(1, Ordering::Relaxed) == 0 {
        crate::provider_capability::engine_worker_capabilities()
    } else {
        crate::CpuProviderExecutionCapabilities::default()
    }
}

#[derive(Debug)]
struct SnapshotGeneralProvider {
    capability_calls: Arc<AtomicUsize>,
    execution_calls: Arc<AtomicUsize>,
}

impl CpuGeneralContractionProvider for SnapshotGeneralProvider {
    fn execution_capabilities(&self) -> crate::CpuProviderExecutionCapabilities {
        changing_capabilities(&self.capability_calls)
    }

    fn dot_general(
        &self,
        _context: &CpuExecutionContext<'_>,
        _request: crate::provider::CpuDotGeneralRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        self.execution_calls.fetch_add(1, Ordering::Relaxed);
        Ok(CpuProviderOutcome::Unsupported(
            CpuProviderUnsupported::RuntimeUnavailable,
        ))
    }
}

#[derive(Debug)]
struct SnapshotGemmProvider {
    capability_calls: Arc<AtomicUsize>,
    execution_calls: Arc<AtomicUsize>,
}

impl CpuGemmProvider for SnapshotGemmProvider {
    fn execution_capabilities(&self) -> crate::CpuProviderExecutionCapabilities {
        changing_capabilities(&self.capability_calls)
    }

    fn gemm(
        &self,
        _context: &CpuExecutionContext<'_>,
        _request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        self.execution_calls.fetch_add(1, Ordering::Relaxed);
        Ok(CpuProviderOutcome::Executed)
    }

    fn strided_batched_gemm(
        &self,
        _context: &CpuExecutionContext<'_>,
        _request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        self.execution_calls.fetch_add(1, Ordering::Relaxed);
        Ok(CpuProviderOutcome::Executed)
    }

    fn grouped_gemm(
        &self,
        _context: &CpuExecutionContext<'_>,
        _request: CpuGroupedGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        self.execution_calls.fetch_add(1, Ordering::Relaxed);
        Ok(CpuProviderOutcome::Executed)
    }
}

#[derive(Debug)]
struct SnapshotLayoutProvider {
    capability_calls: Arc<AtomicUsize>,
}

impl CpuLayoutTransformProvider for SnapshotLayoutProvider {
    fn execution_capabilities(&self) -> crate::CpuProviderExecutionCapabilities {
        changing_capabilities(&self.capability_calls)
    }

    fn materialize(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuLayoutTransformRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        StridedLayoutTransformProvider.materialize(context, request)
    }
}

#[test]
fn provider_capabilities_are_snapshotted_once_when_the_bundle_is_built() {
    let general_capability_calls = Arc::new(AtomicUsize::new(0));
    let gemm_capability_calls = Arc::new(AtomicUsize::new(0));
    let layout_capability_calls = Arc::new(AtomicUsize::new(0));
    let general_execution_calls = Arc::new(AtomicUsize::new(0));
    let gemm_execution_calls = Arc::new(AtomicUsize::new(0));
    let bundle = CpuProviderBundle::custom_builder()
        .gemm_provider(Arc::new(SnapshotGemmProvider {
            capability_calls: Arc::clone(&gemm_capability_calls),
            execution_calls: Arc::clone(&gemm_execution_calls),
        }))
        .layout_transform_provider(Arc::new(SnapshotLayoutProvider {
            capability_calls: Arc::clone(&layout_capability_calls),
        }))
        .prefer_general_contraction_provider(Arc::new(SnapshotGeneralProvider {
            capability_calls: Arc::clone(&general_capability_calls),
            execution_calls: Arc::clone(&general_execution_calls),
        }))
        .build()
        .unwrap();

    assert_eq!(general_capability_calls.load(Ordering::Relaxed), 1);
    assert_eq!(gemm_capability_calls.load(Ordering::Relaxed), 1);
    assert_eq!(layout_capability_calls.load(Ordering::Relaxed), 1);

    let cpus = crate::CpuSet::new([crate::CpuId::new(0), crate::CpuId::new(1)]).unwrap();
    bundle
        .validate_for_domain(
            crate::CpuDomainId::new(17),
            NonZeroUsize::new(2).unwrap(),
            crate::CpuPlacementGuarantee::ExactDeclared,
            &cpus,
            &cpus,
        )
        .unwrap();

    for _ in 0..2 {
        let (lhs, rhs, mut output, config) = route_operands();
        let fixture = execution_context_fixture(2);
        bundle
            .execute_dot_general_into(
                &fixture.entry(),
                &mut BufferPool::new(),
                &mut GemmAnalysisCache::default(),
                None,
                TensorRead::from_tensor(&lhs),
                TensorRead::from_tensor(&rhs),
                &config,
                DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
                TensorWrite::from_tensor(&mut output),
            )
            .unwrap();
    }

    assert_eq!(general_execution_calls.load(Ordering::Relaxed), 2);
    assert_eq!(gemm_execution_calls.load(Ordering::Relaxed), 2);
    assert_eq!(general_capability_calls.load(Ordering::Relaxed), 1);
    assert_eq!(gemm_capability_calls.load(Ordering::Relaxed), 1);
    assert_eq!(layout_capability_calls.load(Ordering::Relaxed), 1);
}

fn route_operands() -> (Tensor, Tensor, Tensor, DotGeneralConfig) {
    (
        Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4]).unwrap(),
        Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4]).unwrap(),
        Tensor::from_vec_col_major(vec![2, 2], vec![0.0_f64; 4]).unwrap(),
        config(&[1], &[0], &[], &[]),
    )
}

fn route_bundle(
    gemm: Arc<dyn CpuGemmProvider>,
    general: Option<(Arc<dyn CpuGeneralContractionProvider>, bool)>,
) -> CpuProviderBundle {
    let builder = CpuProviderBundle::custom_builder()
        .gemm_provider(gemm)
        .engine_outer_grouped_gemm()
        .layout_transform_provider(Arc::new(StridedLayoutTransformProvider));
    match general {
        Some((provider, true)) => builder.require_general_contraction_provider(provider),
        Some((provider, false)) => builder.prefer_general_contraction_provider(provider),
        None => builder,
    }
    .build()
    .unwrap()
}

fn execute_unit_grouped(
    job_count: usize,
    executor: Arc<dyn CpuDomainExecutor>,
    provider: Arc<dyn CpuGemmProvider>,
) -> tenferro_tensor::Result<()> {
    let bundle = route_bundle(provider, None);
    let lhs = Tensor::from_vec_col_major(vec![job_count], vec![2.0_f64; job_count]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![job_count], vec![4.0_f64; job_count]).unwrap();
    let mut output = Tensor::from_vec_col_major(vec![job_count], vec![0.0_f64; job_count]).unwrap();
    let jobs = (0..job_count)
        .map(|index| GroupedGemmJob::new(index, index, index, 1, 1, 1))
        .collect::<Vec<_>>();
    let fixture = external_execution_context_fixture(executor, NonZeroUsize::new(4).unwrap());

    bundle.execute_grouped_gemm(
        &fixture.entry(),
        TensorRead::from_tensor(&lhs),
        TensorRead::from_tensor(&rhs),
        &GroupedGemmConfig::new(
            &jobs,
            DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
        ),
        TensorWrite::from_tensor(&mut output),
    )
}

#[test]
fn packed_grouped_job_states_keep_two_usize_bit_widths_inline_then_spill() {
    assert_eq!(
        GROUPED_INLINE_JOB_CAPACITY,
        2 * usize::BITS as usize,
        "four inline words at two bits per job define the allocation-free bound"
    );
    let cases: [(usize, bool); 5] = [
        (8, false),
        (9, false),
        (GROUPED_INLINE_JOB_CAPACITY, false),
        (GROUPED_INLINE_JOB_CAPACITY + 1, true),
        (GROUPED_INLINE_JOB_CAPACITY * 8, true),
    ];

    for (job_count, expected_spill) in cases {
        let states = PackedJobStates::new(job_count);
        assert_eq!(states.len(), job_count);
        assert_eq!(
            states.word_count(),
            job_count.div_ceil(GROUPED_JOBS_PER_STATE_WORD)
        );
        assert_eq!(
            states.spilled(),
            expected_spill,
            "unexpected SmallVec storage for {job_count} grouped jobs"
        );
        assert!((0..job_count).all(|index| states.state(index) == GroupedJobState::Unclaimed));
    }
}

#[test]
fn packed_grouped_job_states_do_not_clobber_adjacent_or_boundary_jobs() {
    let states = PackedJobStates::new(GROUPED_JOBS_PER_STATE_WORD + 1);

    std::thread::scope(|scope| {
        let left = scope.spawn(|| states.try_claim(0));
        let right = scope.spawn(|| states.try_claim(1));
        assert_eq!(left.join().unwrap(), Ok(()));
        assert_eq!(right.join().unwrap(), Ok(()));
    });
    assert_eq!(states.state(0), GroupedJobState::Running);
    assert_eq!(states.state(1), GroupedJobState::Running);
    assert!(states.complete(0));
    assert_eq!(states.state(0), GroupedJobState::Complete);
    assert_eq!(states.state(1), GroupedJobState::Running);
    assert!(states.complete(1));

    let left_boundary = GROUPED_JOBS_PER_STATE_WORD - 1;
    let right_boundary = GROUPED_JOBS_PER_STATE_WORD;
    assert_eq!(states.try_claim(left_boundary), Ok(()));
    assert_eq!(states.try_claim(right_boundary), Ok(()));
    assert!(states.complete(left_boundary));
    assert_eq!(states.state(right_boundary), GroupedJobState::Running);
    assert!(states.complete(right_boundary));
    assert_eq!(
        states.try_claim(right_boundary),
        Err(GroupedJobState::Complete)
    );
}

#[test]
fn route_general_executed_short_circuits_gemm() {
    let general_calls = Arc::new(Mutex::new(0));
    let general = Arc::new(GeneralSpy {
        behavior: GeneralBehavior::Outcome(CpuProviderOutcome::Executed),
        calls: Arc::clone(&general_calls),
    });
    let gemm = Arc::new(GemmSpy::new(CpuProviderOutcome::Executed));
    let bundle = route_bundle(gemm.clone(), Some((general, false)));
    let (lhs, rhs, mut output, config) = route_operands();
    let fixture = execution_context_fixture(1);
    bundle
        .execute_dot_general_into(
            &fixture.entry(),
            &mut BufferPool::new(),
            &mut GemmAnalysisCache::default(),
            None,
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &config,
            DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
            TensorWrite::from_tensor(&mut output),
        )
        .unwrap();
    assert_eq!(*general_calls.lock().unwrap(), 1);
    assert_eq!(*gemm.gemm_calls.lock().unwrap(), 0);
}

#[test]
fn route_general_unsupported_falls_back_only_when_preferred() {
    let general = Arc::new(GeneralSpy {
        behavior: GeneralBehavior::Outcome(CpuProviderOutcome::Unsupported(
            CpuProviderUnsupported::Layout(crate::provider::CpuOperand::Lhs),
        )),
        calls: Arc::new(Mutex::new(0)),
    });
    let gemm = Arc::new(GemmSpy::new(CpuProviderOutcome::Executed));
    let bundle = route_bundle(gemm.clone(), Some((general, false)));
    let (lhs, rhs, mut output, config) = route_operands();
    let fixture = execution_context_fixture(1);
    bundle
        .execute_dot_general_into(
            &fixture.entry(),
            &mut BufferPool::new(),
            &mut GemmAnalysisCache::default(),
            None,
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &config,
            DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
            TensorWrite::from_tensor(&mut output),
        )
        .unwrap();
    assert_eq!(*gemm.gemm_calls.lock().unwrap(), 1);
}

#[test]
fn route_general_error_is_terminal() {
    let gemm = Arc::new(GemmSpy::new(CpuProviderOutcome::Executed));
    let general = Arc::new(GeneralSpy {
        behavior: GeneralBehavior::Error,
        calls: Arc::new(Mutex::new(0)),
    });
    let bundle = route_bundle(gemm.clone(), Some((general, false)));
    let (lhs, rhs, mut output, config) = route_operands();
    let fixture = execution_context_fixture(1);
    let error = bundle
        .execute_dot_general_into(
            &fixture.entry(),
            &mut BufferPool::new(),
            &mut GemmAnalysisCache::default(),
            None,
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &config,
            DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
            TensorWrite::from_tensor(&mut output),
        )
        .unwrap_err();
    assert!(error.to_string().contains("general spy failure"));
    assert_eq!(*gemm.gemm_calls.lock().unwrap(), 0);
}

#[test]
fn route_required_general_unsupported_is_typed_and_terminal() {
    let gemm = Arc::new(GemmSpy::new(CpuProviderOutcome::Executed));
    let general = Arc::new(GeneralSpy {
        behavior: GeneralBehavior::Outcome(CpuProviderOutcome::Unsupported(
            CpuProviderUnsupported::RuntimeUnavailable,
        )),
        calls: Arc::new(Mutex::new(0)),
    });
    let bundle = route_bundle(gemm.clone(), Some((general, true)));
    let (lhs, rhs, mut output, config) = route_operands();
    let fixture = execution_context_fixture(1);
    let error = bundle
        .execute_dot_general_into(
            &fixture.entry(),
            &mut BufferPool::new(),
            &mut GemmAnalysisCache::default(),
            None,
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &config,
            DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
            TensorWrite::from_tensor(&mut output),
        )
        .unwrap_err();
    assert_eq!(error.kind(), tenferro_tensor::ErrorKind::Unsupported);
    assert_eq!(*gemm.gemm_calls.lock().unwrap(), 0);
}

#[test]
fn route_gemm_unsupported_is_terminal() {
    let gemm = Arc::new(GemmSpy::new(CpuProviderOutcome::Unsupported(
        CpuProviderUnsupported::DType(DType::F64),
    )));
    let bundle = route_bundle(gemm.clone(), None);
    let (lhs, rhs, mut output, config) = route_operands();
    let fixture = execution_context_fixture(1);
    let error = bundle
        .execute_dot_general_into(
            &fixture.entry(),
            &mut BufferPool::new(),
            &mut GemmAnalysisCache::default(),
            None,
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &config,
            DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
            TensorWrite::from_tensor(&mut output),
        )
        .unwrap_err();
    assert_eq!(error.kind(), tenferro_tensor::ErrorKind::Unsupported);
    assert_eq!(*gemm.gemm_calls.lock().unwrap(), 1);
    assert_eq!(output.as_slice::<f64>().unwrap(), &[0.0; 4]);
}

#[test]
fn route_gemm_output_layout_unsupported_is_terminal_without_mutation() {
    let gemm = Arc::new(GemmSpy::new(CpuProviderOutcome::Unsupported(
        CpuProviderUnsupported::Layout(CpuOperand::Output),
    )));
    let bundle = route_bundle(gemm.clone(), None);
    let (lhs, rhs, mut output, config) = route_operands();
    let fixture = execution_context_fixture(1);

    let error = bundle
        .execute_dot_general_into(
            &fixture.entry(),
            &mut BufferPool::new(),
            &mut GemmAnalysisCache::default(),
            None,
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &config,
            DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
            TensorWrite::from_tensor(&mut output),
        )
        .unwrap_err();

    assert_eq!(error.kind(), tenferro_tensor::ErrorKind::Unsupported);
    assert_eq!(*gemm.gemm_calls.lock().unwrap(), 1);
    assert_eq!(output.as_slice::<f64>().unwrap(), &[0.0; 4]);
}

#[test]
fn route_gemm_provider_error_is_terminal_without_mutation() {
    let gemm = Arc::new(GemmSpy::error());
    let bundle = route_bundle(gemm.clone(), None);
    let (lhs, rhs, mut output, config) = route_operands();
    let fixture = execution_context_fixture(1);

    let error = bundle
        .execute_dot_general_into(
            &fixture.entry(),
            &mut BufferPool::new(),
            &mut GemmAnalysisCache::default(),
            None,
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &config,
            DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
            TensorWrite::from_tensor(&mut output),
        )
        .unwrap_err();

    assert_eq!(error.kind(), tenferro_tensor::ErrorKind::RuntimeState);
    assert!(error.to_string().contains("GEMM provider spy failure"));
    assert_eq!(*gemm.gemm_calls.lock().unwrap(), 1);
    assert_eq!(output.as_slice::<f64>().unwrap(), &[0.0; 4]);
}

#[test]
fn route_gemm_canonical_retry_unsupported_is_terminal_without_mutation() {
    for second_outcome in [
        CpuProviderUnsupported::Layout(CpuOperand::Lhs),
        CpuProviderUnsupported::Conjugation,
        CpuProviderUnsupported::DType(DType::F64),
    ] {
        let gemm = Arc::new(CanonicalFallbackSpy::rejecting_retry(
            CpuProviderUnsupported::Layout(CpuOperand::Rhs),
            second_outcome,
        ));
        let bundle = route_bundle(gemm.clone(), None);
        let lhs = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4]).unwrap();
        let rhs = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4]).unwrap();
        let mut output = Tensor::from_vec_col_major(vec![2, 2], vec![41.0_f64; 4]).unwrap();
        let mut buffers = BufferPool::new();
        <f64 as PoolScalar>::pool_release(&mut buffers, vec![0.0; 4]);
        <f64 as PoolScalar>::pool_release(&mut buffers, vec![0.0; 4]);
        let seed_stats = buffers.stats();
        let fixture = execution_context_fixture(1);

        let error = bundle
            .execute_dot_general_into(
                &fixture.entry(),
                &mut buffers,
                &mut GemmAnalysisCache::default(),
                None,
                TensorRead::from_tensor(&lhs),
                TensorRead::from_tensor(&rhs),
                &config(&[1], &[0], &[], &[]),
                DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
                TensorWrite::from_tensor(&mut output),
            )
            .unwrap_err();

        assert_eq!(error.kind(), tenferro_tensor::ErrorKind::Unsupported);
        assert_eq!(*gemm.calls.lock().unwrap(), 2);
        assert_eq!(output.as_slice::<f64>().unwrap(), &[41.0; 4]);
        assert_eq!(buffers.stats(), seed_stats);
    }
}

#[test]
fn route_gemm_canonical_retry_error_reclaims_both_materializations() {
    let gemm = Arc::new(CanonicalFallbackSpy::erroring_retry(
        CpuProviderUnsupported::Layout(CpuOperand::Rhs),
    ));
    let bundle = route_bundle(gemm.clone(), None);
    let lhs = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4]).unwrap();
    let mut output = Tensor::from_vec_col_major(vec![2, 2], vec![41.0_f64; 4]).unwrap();
    let mut buffers = BufferPool::new();
    <f64 as PoolScalar>::pool_release(&mut buffers, vec![0.0; 4]);
    <f64 as PoolScalar>::pool_release(&mut buffers, vec![0.0; 4]);
    let seed_stats = buffers.stats();
    let fixture = execution_context_fixture(1);

    let error = bundle
        .execute_dot_general_into(
            &fixture.entry(),
            &mut buffers,
            &mut GemmAnalysisCache::default(),
            None,
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &config(&[1], &[0], &[], &[]),
            DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
            TensorWrite::from_tensor(&mut output),
        )
        .unwrap_err();

    assert_eq!(error.kind(), tenferro_tensor::ErrorKind::RuntimeState);
    assert!(error.to_string().contains("canonical retry spy failure"));
    assert_eq!(*gemm.calls.lock().unwrap(), 2);
    assert_eq!(output.as_slice::<f64>().unwrap(), &[41.0; 4]);
    assert_eq!(buffers.stats(), seed_stats);
}

fn assert_layout_unsupported_materializes_and_retries(operand: CpuOperand) {
    let gemm = Arc::new(CanonicalFallbackSpy::new(
        CpuProviderUnsupported::Layout(operand),
        CanonicalFallbackBehavior::Real,
    ));
    let bundle = route_bundle(gemm.clone(), None);
    let lhs = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![2, 2], vec![5.0_f64, 7.0, 6.0, 8.0]).unwrap();
    let mut output = Tensor::from_vec_col_major(vec![2, 2], vec![41.0_f64; 4]).unwrap();
    let mut buffers = BufferPool::new();
    <f64 as PoolScalar>::pool_release(&mut buffers, vec![0.0; 4]);
    <f64 as PoolScalar>::pool_release(&mut buffers, vec![0.0; 4]);
    let seed_stats = buffers.stats();
    let fixture = execution_context_fixture(1);

    bundle
        .execute_dot_general_into(
            &fixture.entry(),
            &mut buffers,
            &mut GemmAnalysisCache::default(),
            None,
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &config(&[1], &[0], &[], &[]),
            DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
            TensorWrite::from_tensor(&mut output),
        )
        .unwrap();

    assert_eq!(*gemm.calls.lock().unwrap(), 2);
    assert_eq!(output.as_slice::<f64>().unwrap(), &[19.0, 43.0, 22.0, 50.0]);
    assert_eq!(buffers.stats(), seed_stats);
}

#[test]
fn route_gemm_lhs_layout_unsupported_materializes_and_retries() {
    assert_layout_unsupported_materializes_and_retries(CpuOperand::Lhs);
}

#[test]
fn route_gemm_rhs_layout_unsupported_materializes_and_retries() {
    assert_layout_unsupported_materializes_and_retries(CpuOperand::Rhs);
}

#[test]
fn route_gemm_conjugation_unsupported_materializes_conjugate_and_retries() {
    let gemm = Arc::new(CanonicalFallbackSpy::new(
        CpuProviderUnsupported::Conjugation,
        CanonicalFallbackBehavior::ConjugatedComplex,
    ));
    let bundle = route_bundle(gemm.clone(), None);
    let lhs = Tensor::from_vec_col_major(vec![1, 1], vec![Complex64::new(1.0, 2.0)]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![1, 1], vec![Complex64::new(3.0, 4.0)]).unwrap();
    let mut output =
        Tensor::from_vec_col_major(vec![1, 1], vec![Complex64::new(5.0, 1.0)]).unwrap();
    let mut accumulation = DotGeneralAccumulation::scaled(
        ContractionScalar::C64(Complex64::new(2.0, 0.0)),
        ContractionScalar::C64(Complex64::new(3.0, 0.0)),
    )
    .unwrap();
    accumulation.lhs_conj = true;
    let fixture = execution_context_fixture(1);

    bundle
        .execute_dot_general_into(
            &fixture.entry(),
            &mut BufferPool::new(),
            &mut GemmAnalysisCache::default(),
            None,
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &config(&[1], &[0], &[], &[]),
            accumulation,
            TensorWrite::from_tensor(&mut output),
        )
        .unwrap();

    assert_eq!(*gemm.calls.lock().unwrap(), 2);
    assert_eq!(
        output.as_slice::<Complex64>().unwrap(),
        &[Complex64::new(37.0, -1.0)],
    );
}

#[test]
fn route_install_failure_precedes_provider_and_output_mutation() {
    let gemm = Arc::new(GemmSpy::new(CpuProviderOutcome::Executed));
    let bundle = route_bundle(gemm.clone(), None);
    let (lhs, rhs, _, config) = route_operands();
    let mut output = Tensor::from_vec_col_major(vec![2, 2], vec![41.0_f64; 4]).unwrap();
    let before = output.as_slice::<f64>().unwrap().to_vec();
    let submits = Arc::new(AtomicUsize::new(0));
    let installs = Arc::new(AtomicUsize::new(0));
    let fixture = external_execution_context_fixture(
        Arc::new(RejectingInstallExecutor {
            submits: Arc::clone(&submits),
            installs: Arc::clone(&installs),
        }),
        NonZeroUsize::new(4).unwrap(),
    );

    let error = bundle
        .execute_dot_general_into(
            &fixture.entry(),
            &mut BufferPool::new(),
            &mut GemmAnalysisCache::default(),
            None,
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &config,
            DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
            TensorWrite::from_tensor(&mut output),
        )
        .unwrap_err();

    let tenferro_tensor::Error::BackendSource { source, .. } = error else {
        panic!("executor failure must retain its typed source: {error}");
    };
    assert!(matches!(
        source.downcast_ref::<CpuDomainExecutorError>(),
        Some(CpuDomainExecutorError::Cancellation { message })
            if message == "intentional install rejection"
    ));
    assert_eq!(installs.load(Ordering::Relaxed), 1);
    assert_eq!(submits.load(Ordering::Relaxed), 0);
    assert_eq!(*gemm.gemm_calls.lock().unwrap(), 0);
    assert_eq!(*gemm.strided_calls.lock().unwrap(), 0);
    assert_eq!(output.as_slice::<f64>().unwrap(), before.as_slice());
}

#[test]
fn route_budget_one_rejects_non_inline_provider_before_output_mutation() {
    let gemm = Arc::new(
        GemmSpy::new(CpuProviderOutcome::Executed)
            .with_capabilities(crate::CpuProviderExecutionCapabilities::default()),
    );
    let bundle = route_bundle(gemm.clone(), None);
    let (lhs, rhs, _, config) = route_operands();
    let mut output = Tensor::from_vec_col_major(vec![2, 2], vec![41.0_f64; 4]).unwrap();
    let before = output.as_slice::<f64>().unwrap().to_vec();
    let fixture = execution_context_fixture(1);

    let error = bundle
        .execute_dot_general_into(
            &fixture.entry(),
            &mut BufferPool::new(),
            &mut GemmAnalysisCache::default(),
            None,
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &config,
            DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
            TensorWrite::from_tensor(&mut output),
        )
        .unwrap_err();

    let tenferro_tensor::Error::BackendSource { source, .. } = error else {
        panic!("provider-mode failure must retain its typed source: {error}");
    };
    assert!(matches!(
        source.downcast_ref::<crate::CpuProviderDomainError>(),
        Some(crate::CpuProviderDomainError::ThreadCountNotEnforceable {
            thread_budget: 1,
            ..
        })
    ));
    assert_eq!(*gemm.gemm_calls.lock().unwrap(), 0);
    assert_eq!(*gemm.strided_calls.lock().unwrap(), 0);
    assert_eq!(output.as_slice::<f64>().unwrap(), before.as_slice());
}

#[test]
fn route_budget_one_enters_controlled_external_provider_sequentially() {
    let capabilities = crate::CpuProviderExecutionCapabilities {
        thread_count: crate::CpuThreadCountControl::PerCallUpperBound,
        placement: crate::CpuPlacementControl::ExternalWorkers,
        worker_local_sequential: true,
        accepts_sequential: true,
        accepts_outer: true,
        accepts_inner: true,
    };
    let gemm = Arc::new(GemmSpy::new(CpuProviderOutcome::Executed).with_capabilities(capabilities));
    let bundle = route_bundle(gemm.clone(), None);
    let (lhs, rhs, mut output, config) = route_operands();
    let fixture = execution_context_fixture(1);

    bundle
        .execute_dot_general_into(
            &fixture.entry(),
            &mut BufferPool::new(),
            &mut GemmAnalysisCache::default(),
            None,
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &config,
            DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
            TensorWrite::from_tensor(&mut output),
        )
        .unwrap();

    assert_eq!(
        gemm.parallelism.lock().unwrap().as_slice(),
        &[ParallelMode::Sequential]
    );
}

#[test]
fn route_provider_default_compatibility_preserves_legacy_entry_modes() {
    for (threads, expected_mode) in [(1, ParallelMode::Sequential), (2, ParallelMode::Inner)] {
        let gemm = Arc::new(
            GemmSpy::new(CpuProviderOutcome::Executed)
                .with_capabilities(crate::CpuProviderExecutionCapabilities::default()),
        );
        let bundle = CpuProviderBundle::custom_builder()
            .gemm_provider(gemm.clone())
            .layout_transform_provider(Arc::new(StridedLayoutTransformProvider))
            .provider_default_compatibility()
            .build()
            .unwrap();
        let (lhs, rhs, mut output, config) = route_operands();
        let fixture = execution_context_fixture(threads);

        bundle
            .execute_dot_general_into(
                &fixture.entry(),
                &mut BufferPool::new(),
                &mut GemmAnalysisCache::default(),
                None,
                TensorRead::from_tensor(&lhs),
                TensorRead::from_tensor(&rhs),
                &config,
                DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
                TensorWrite::from_tensor(&mut output),
            )
            .unwrap();

        assert_eq!(
            gemm.parallelism.lock().unwrap().as_slice(),
            &[expected_mode]
        );
    }
}

#[test]
fn route_canonical_fallback_uses_layout_slot_before_gemm() {
    let layout_calls = Arc::new(Mutex::new(0));
    let gemm = Arc::new(GemmSpy::new(CpuProviderOutcome::Executed));
    let bundle = CpuProviderBundle::custom_builder()
        .gemm_provider(gemm.clone())
        .layout_transform_provider(Arc::new(LayoutSpy {
            calls: Arc::clone(&layout_calls),
        }))
        .build()
        .unwrap();
    let lhs = Tensor::from_vec_col_major(vec![2, 2, 2, 2], vec![1.0_f64; 16]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![2, 2, 2, 2], vec![1.0_f64; 16]).unwrap();
    let mut output = Tensor::from_vec_col_major(vec![2, 2, 2, 2], vec![0.0_f64; 16]).unwrap();
    let fixture = execution_context_fixture(1);

    bundle
        .execute_dot_general_into(
            &fixture.entry(),
            &mut BufferPool::new(),
            &mut GemmAnalysisCache::default(),
            None,
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &config(&[1, 3], &[2, 1], &[], &[]),
            DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
            TensorWrite::from_tensor(&mut output),
        )
        .unwrap();

    assert_eq!(*layout_calls.lock().unwrap(), 2);
    assert_eq!(*gemm.gemm_calls.lock().unwrap(), 1);
}

#[test]
fn route_strided_batch_allows_inner_parallelism() {
    let gemm = Arc::new(GemmSpy::new(CpuProviderOutcome::Executed));
    let bundle = route_bundle(gemm.clone(), None);
    let lhs = Tensor::from_vec_col_major(vec![2, 2, 2], vec![1.0_f64; 8]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![2, 2, 2], vec![1.0_f64; 8]).unwrap();
    let mut output = Tensor::from_vec_col_major(vec![2, 2, 2], vec![0.0_f64; 8]).unwrap();
    let fixture = execution_context_fixture(2);
    bundle
        .execute_dot_general_into(
            &fixture.entry(),
            &mut BufferPool::new(),
            &mut GemmAnalysisCache::default(),
            None,
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &config(&[1], &[0], &[2], &[2]),
            DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
            TensorWrite::from_tensor(&mut output),
        )
        .unwrap();
    assert_eq!(*gemm.strided_calls.lock().unwrap(), 1);
    assert_eq!(
        gemm.parallelism.lock().unwrap().as_slice(),
        &[ParallelMode::Inner]
    );
}

#[test]
fn route_engine_outer_grouped_submits_once_with_sequential_children() {
    let gemm = Arc::new(GemmSpy::new(CpuProviderOutcome::Executed));
    let bundle = route_bundle(gemm.clone(), None);
    let lhs = Tensor::from_vec_col_major(vec![8], vec![2.0_f64; 8]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![8], vec![4.0_f64; 8]).unwrap();
    let mut output = Tensor::from_vec_col_major(vec![8], vec![0.0_f64; 8]).unwrap();
    let jobs =
        std::array::from_fn::<_, 8, _>(|index| GroupedGemmJob::new(index, index, index, 1, 1, 1));
    let (submits, installs, fixture) = counting_execution_fixture(4);
    bundle
        .execute_grouped_gemm(
            &fixture.entry(),
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &GroupedGemmConfig::new(
                &jobs,
                DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
            ),
            TensorWrite::from_tensor(&mut output),
        )
        .unwrap();
    assert_eq!(*gemm.grouped_calls.lock().unwrap(), 8);
    assert_eq!(gemm.grouped_job_counts.lock().unwrap().as_slice(), &[1; 8]);
    assert_eq!(
        gemm.parallelism.lock().unwrap().as_slice(),
        &[ParallelMode::Sequential; 8]
    );
    assert_eq!(submits.load(Ordering::Relaxed), 1);
    assert_eq!(installs.load(Ordering::Relaxed), 0);
}

#[test]
fn route_engine_outer_rejects_ignored_index_equal_to_len_after_valid_jobs() {
    let gemm = Arc::new(GemmSpy::new(CpuProviderOutcome::Executed));
    let error = execute_unit_grouped(
        2,
        Arc::new(IgnoredInvalidIndexExecutor { requested_index: 2 }),
        gemm.clone(),
    )
    .unwrap_err();

    assert_grouped_scheduling_error(&error, "index 2");
    assert_eq!(*gemm.grouped_calls.lock().unwrap(), 2);
}

#[test]
fn route_engine_outer_rejects_ignored_usize_max_index_after_valid_jobs() {
    let gemm = Arc::new(GemmSpy::new(CpuProviderOutcome::Executed));
    let error = execute_unit_grouped(
        2,
        Arc::new(IgnoredInvalidIndexExecutor {
            requested_index: usize::MAX,
        }),
        gemm.clone(),
    )
    .unwrap_err();

    assert_grouped_scheduling_error(&error, &format!("index {}", usize::MAX));
    assert_eq!(*gemm.grouped_calls.lock().unwrap(), 2);
}

#[test]
fn route_engine_outer_submit_error_overrides_ignored_invalid_index_audit() {
    let gemm = Arc::new(GemmSpy::new(CpuProviderOutcome::Executed));
    let error = execute_unit_grouped(2, Arc::new(InvalidThenFailingSubmitExecutor), gemm.clone())
        .unwrap_err();

    let tenferro_tensor::Error::BackendSource { source, .. } = error else {
        panic!("submit failure must retain a typed source");
    };
    assert!(matches!(
        source.downcast_ref::<CpuDomainExecutorError>(),
        Some(CpuDomainExecutorError::Cancellation { message })
            if message == "submit failure after ignored invalid index"
    ));
    assert_eq!(*gemm.grouped_calls.lock().unwrap(), 2);
}

#[test]
fn route_engine_outer_rejects_sequential_duplicate_after_all_valid_jobs() {
    let gemm = Arc::new(GemmSpy::new(CpuProviderOutcome::Executed));
    let error = execute_unit_grouped(
        2,
        Arc::new(SequentialDuplicateAllValidExecutor),
        gemm.clone(),
    )
    .unwrap_err();

    assert_grouped_scheduling_error(&error, "duplicate index 0");
    assert_eq!(
        *gemm.grouped_calls.lock().unwrap(),
        2,
        "the duplicate invocation must not reach provider mutation"
    );
}

#[test]
fn route_engine_outer_reports_running_job_when_executor_catches_provider_panic() {
    let calls = Arc::new(AtomicUsize::new(0));
    let provider: Arc<dyn CpuGemmProvider> = Arc::new(PanicOnceGemmProvider {
        calls: Arc::clone(&calls),
    });
    let error =
        execute_unit_grouped(2, Arc::new(CatchProviderPanicExecutor), provider).unwrap_err();

    assert_grouped_scheduling_error(&error, "did not complete grouped-GEMM index 0");
    assert_eq!(calls.load(Ordering::Relaxed), 2);
}

#[test]
fn route_engine_outer_preserves_exact_provider_error_after_successful_submit() {
    let calls = Arc::new(AtomicUsize::new(0));
    let provider: Arc<dyn CpuGemmProvider> = Arc::new(ProviderErrorSpy {
        calls: Arc::clone(&calls),
    });
    let executor: Arc<dyn CpuDomainExecutor> = Arc::new(CountingExecutor {
        submits: Arc::new(AtomicUsize::new(0)),
        installs: Arc::new(AtomicUsize::new(0)),
    });
    let error = execute_unit_grouped(2, executor, provider).unwrap_err();

    assert!(matches!(
        error,
        tenferro_tensor::Error::RuntimeState { op, message }
            if op == "grouped_provider_sentinel" && message == "exact provider failure"
    ));
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

#[test]
fn route_engine_outer_submit_error_overrides_prior_provider_error() {
    let calls = Arc::new(AtomicUsize::new(0));
    let provider: Arc<dyn CpuGemmProvider> = Arc::new(ProviderErrorSpy {
        calls: Arc::clone(&calls),
    });
    let error = execute_unit_grouped(2, Arc::new(FailingSubmitExecutor), provider).unwrap_err();

    let tenferro_tensor::Error::BackendSource { source, .. } = error else {
        panic!("submit failure must retain a typed source");
    };
    assert!(matches!(
        source.downcast_ref::<CpuDomainExecutorError>(),
        Some(CpuDomainExecutorError::Cancellation { message })
            if message == "authoritative submit failure"
    ));
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

#[test]
fn route_engine_outer_rejects_duplicate_while_first_provider_call_is_running() {
    let provider_entered = Arc::new(Barrier::new(2));
    let release_provider = Arc::new(Barrier::new(2));
    let duplicate_failed = Arc::new(AtomicBool::new(false));
    let calls = Arc::new(AtomicUsize::new(0));
    let provider: Arc<dyn CpuGemmProvider> = Arc::new(BarrierMutationGemmProvider {
        calls: Arc::clone(&calls),
        provider_entered: Arc::clone(&provider_entered),
        release_provider: Arc::clone(&release_provider),
    });
    let executor: Arc<dyn CpuDomainExecutor> = Arc::new(BarrierDuplicateExecutor {
        provider_entered,
        release_provider,
        duplicate_failed: Arc::clone(&duplicate_failed),
    });
    let bundle = route_bundle(provider, None);
    let lhs = Tensor::from_vec_col_major(vec![2], vec![2.0_f64; 2]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![2], vec![4.0_f64; 2]).unwrap();
    let mut output = Tensor::from_vec_col_major(vec![2], vec![0.0_f64; 2]).unwrap();
    let jobs = [
        GroupedGemmJob::new(0, 0, 0, 1, 1, 1),
        GroupedGemmJob::new(1, 1, 1, 1, 1, 1),
    ];
    let fixture = external_execution_context_fixture(executor, NonZeroUsize::new(4).unwrap());

    let error = bundle
        .execute_grouped_gemm(
            &fixture.entry(),
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &GroupedGemmConfig::new(
                &jobs,
                DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
            ),
            TensorWrite::from_tensor(&mut output),
        )
        .unwrap_err();

    assert_grouped_scheduling_error(&error, "duplicate index 0");
    assert!(duplicate_failed.load(Ordering::Acquire));
    assert_eq!(calls.load(Ordering::Relaxed), 2);
    assert_eq!(output.as_slice::<f64>().unwrap(), &[1.0, 1.0]);
}

#[test]
fn route_engine_outer_rejects_concurrent_duplicate_job_before_provider_mutation() {
    let gemm = Arc::new(GemmSpy::new(CpuProviderOutcome::Executed));
    let bundle = route_bundle(gemm.clone(), None);
    let lhs = Tensor::from_vec_col_major(vec![2], vec![2.0_f64; 2]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![2], vec![4.0_f64; 2]).unwrap();
    let mut output = Tensor::from_vec_col_major(vec![2], vec![0.0_f64; 2]).unwrap();
    let jobs = [
        GroupedGemmJob::new(0, 0, 0, 1, 1, 1),
        GroupedGemmJob::new(1, 1, 1, 1, 1, 1),
    ];
    let fixture = external_execution_context_fixture(
        Arc::new(ConcurrentDuplicateExecutor),
        NonZeroUsize::new(4).unwrap(),
    );

    let error = bundle
        .execute_grouped_gemm(
            &fixture.entry(),
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &GroupedGemmConfig::new(
                &jobs,
                DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
            ),
            TensorWrite::from_tensor(&mut output),
        )
        .unwrap_err();

    assert_grouped_scheduling_error(&error, "duplicate index 0");
    assert_eq!(
        *gemm.grouped_calls.lock().unwrap(),
        1,
        "a duplicate safe-executor invocation must not reach provider mutation"
    );
}

#[test]
fn route_engine_outer_rejects_successful_submit_that_omits_a_job() {
    let gemm = Arc::new(GemmSpy::new(CpuProviderOutcome::Executed));
    let bundle = route_bundle(gemm.clone(), None);
    let lhs = Tensor::from_vec_col_major(vec![2], vec![2.0_f64; 2]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![2], vec![4.0_f64; 2]).unwrap();
    let mut output = Tensor::from_vec_col_major(vec![2], vec![0.0_f64; 2]).unwrap();
    let jobs = [
        GroupedGemmJob::new(0, 0, 0, 1, 1, 1),
        GroupedGemmJob::new(1, 1, 1, 1, 1, 1),
    ];
    let fixture = external_execution_context_fixture(
        Arc::new(MissingJobExecutor),
        NonZeroUsize::new(4).unwrap(),
    );

    let error = bundle
        .execute_grouped_gemm(
            &fixture.entry(),
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &GroupedGemmConfig::new(
                &jobs,
                DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
            ),
            TensorWrite::from_tensor(&mut output),
        )
        .unwrap_err();

    assert_grouped_scheduling_error(&error, "missing index 1");
    assert_eq!(*gemm.grouped_calls.lock().unwrap(), 1);
}

#[test]
fn route_engine_outer_preserves_submit_error_before_missing_job_audit() {
    let gemm = Arc::new(GemmSpy::new(CpuProviderOutcome::Executed));
    let bundle = route_bundle(gemm.clone(), None);
    let lhs = Tensor::from_vec_col_major(vec![2], vec![2.0_f64; 2]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![2], vec![4.0_f64; 2]).unwrap();
    let mut output = Tensor::from_vec_col_major(vec![2], vec![0.0_f64; 2]).unwrap();
    let jobs = [
        GroupedGemmJob::new(0, 0, 0, 1, 1, 1),
        GroupedGemmJob::new(1, 1, 1, 1, 1, 1),
    ];
    let fixture = external_execution_context_fixture(
        Arc::new(FailingSubmitExecutor),
        NonZeroUsize::new(4).unwrap(),
    );

    let error = bundle
        .execute_grouped_gemm(
            &fixture.entry(),
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &GroupedGemmConfig::new(
                &jobs,
                DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
            ),
            TensorWrite::from_tensor(&mut output),
        )
        .unwrap_err();

    let tenferro_tensor::Error::BackendSource { source, .. } = error else {
        panic!("submit failure must retain a typed source");
    };
    assert!(matches!(
        source.downcast_ref::<CpuDomainExecutorError>(),
        Some(CpuDomainExecutorError::Cancellation { message })
            if message == "authoritative submit failure"
    ));
    assert_eq!(*gemm.grouped_calls.lock().unwrap(), 1);
}

#[test]
fn route_engine_outer_rejects_output_boundary_before_submission() {
    let gemm = Arc::new(GemmSpy::new(CpuProviderOutcome::Executed));
    let bundle = route_bundle(gemm.clone(), None);
    let lhs = Tensor::from_vec_col_major(vec![3], vec![2.0_f64; 3]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![3], vec![4.0_f64; 3]).unwrap();
    let mut output = Tensor::from_vec_col_major(vec![2], vec![-7.0_f64; 2]).unwrap();
    let jobs = [
        GroupedGemmJob::new(0, 0, 0, 1, 1, 1),
        GroupedGemmJob::new(2, 2, 2, 1, 1, 1),
    ];
    let (submits, installs, fixture) = counting_execution_fixture(4);

    let error = bundle
        .execute_grouped_gemm(
            &fixture.entry(),
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &GroupedGemmConfig::new(
                &jobs,
                DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
            ),
            TensorWrite::from_tensor(&mut output),
        )
        .unwrap_err();

    assert_eq!(
        error.kind(),
        ErrorKind::Validation(ValidationKind::InvalidArgument)
    );
    assert_eq!(submits.load(Ordering::Relaxed), 0);
    assert_eq!(installs.load(Ordering::Relaxed), 0);
    assert_eq!(*gemm.grouped_calls.lock().unwrap(), 0);
    assert_eq!(output.as_slice::<f64>().unwrap(), &[-7.0, -7.0]);
}

#[test]
fn route_provider_owned_grouped_installs_once_with_inner_parallelism() {
    let gemm = Arc::new(GemmSpy::new(CpuProviderOutcome::Executed));
    let bundle = CpuProviderBundle::builder(CpuBackendKind::Faer)
        .gemm_provider(gemm.clone())
        .build()
        .unwrap();
    let lhs = Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![2], vec![4.0_f64, 5.0]).unwrap();
    let mut output = Tensor::from_vec_col_major(vec![2], vec![0.0_f64; 2]).unwrap();
    let jobs = [
        GroupedGemmJob::new(0, 0, 0, 1, 1, 1),
        GroupedGemmJob::new(1, 1, 1, 1, 1, 1),
    ];
    let (submits, installs, fixture) = counting_execution_fixture(2);
    bundle
        .execute_grouped_gemm(
            &fixture.entry(),
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &GroupedGemmConfig::new(
                &jobs,
                DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
            ),
            TensorWrite::from_tensor(&mut output),
        )
        .unwrap();

    assert_eq!(*gemm.grouped_calls.lock().unwrap(), 1);
    assert_eq!(gemm.grouped_job_counts.lock().unwrap().as_slice(), &[2]);
    assert_eq!(
        gemm.parallelism.lock().unwrap().as_slice(),
        &[ParallelMode::Inner]
    );
    assert_eq!(submits.load(Ordering::Relaxed), 0);
    assert_eq!(installs.load(Ordering::Relaxed), 1);
}

#[cfg(feature = "cpu-faer")]
#[test]
fn engine_outer_real_faer_writes_only_nonzero_base_output_view_jobs() {
    let bundle = CpuProviderBundle::builder(CpuBackendKind::Faer)
        .build()
        .unwrap();
    let lhs = Tensor::from_vec_col_major(vec![8], vec![1.0_f64, 3.0, 2.0, 4.0, 2.0, 0.0, 0.0, 3.0])
        .unwrap();
    let rhs = Tensor::from_vec_col_major(vec![8], vec![5.0_f64, 7.0, 6.0, 8.0, 4.0, 0.0, 0.0, 5.0])
        .unwrap();
    let mut guarded = vec![-99.0_f64; 10];
    let output = TypedTensorViewMut::from_slice([8], [1], 1, &mut guarded).unwrap();
    let jobs = [
        GroupedGemmJob::new(0, 0, 0, 2, 2, 2),
        GroupedGemmJob::new(4, 4, 4, 2, 2, 2),
    ];
    let (submits, installs, fixture) = counting_execution_fixture(4);

    bundle
        .execute_grouped_gemm(
            &fixture.entry(),
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &GroupedGemmConfig::new(
                &jobs,
                DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
            ),
            TensorWrite::from_view(TensorViewMut::F64(output)),
        )
        .unwrap();

    assert_eq!(guarded[0], -99.0);
    assert_eq!(guarded[9], -99.0);
    assert_eq!(
        &guarded[1..9],
        &[19.0, 43.0, 22.0, 50.0, 8.0, 0.0, 0.0, 15.0]
    );
    assert_eq!(submits.load(Ordering::Relaxed), 1);
    assert_eq!(installs.load(Ordering::Relaxed), 0);
}

/// Layout provider without the uninitialized-output witness (default opt-out).
#[derive(Debug)]
struct OptOutLayoutProvider;

impl CpuLayoutTransformProvider for OptOutLayoutProvider {
    fn execution_capabilities(&self) -> crate::CpuProviderExecutionCapabilities {
        crate::provider_capability::engine_worker_capabilities()
    }

    fn materialize(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuLayoutTransformRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        StridedLayoutTransformProvider.materialize(context, request)
    }
}

/// Layout provider that opts into the uninitialized-output contract but always
/// reports [`CpuProviderOutcome::Unsupported`], forcing the caller's
/// discard-and-reallocate fallback onto the zeroed path.
#[derive(Debug)]
struct UnsupportedUninitLayoutProvider {
    uninit_calls: Arc<Mutex<usize>>,
}

impl CpuLayoutTransformProvider for UnsupportedUninitLayoutProvider {
    fn execution_capabilities(&self) -> crate::CpuProviderExecutionCapabilities {
        crate::provider_capability::engine_worker_capabilities()
    }

    fn materialize(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuLayoutTransformRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        StridedLayoutTransformProvider.materialize(context, request)
    }

    fn uninit_provider(&self) -> Option<&dyn CpuUninitLayoutTransformProvider> {
        Some(self)
    }
}

// SAFETY: this test provider asserts the unsafe trait only to exercise the
// caller's `Unsupported` fallback; it never writes the destination, so it must
// always return `Unsupported` (never `Executed`).
unsafe impl CpuUninitLayoutTransformProvider for UnsupportedUninitLayoutProvider {
    unsafe fn materialize_into_uninit(
        &self,
        _context: &CpuExecutionContext<'_>,
        _input: &TensorRead<'_>,
        _intent: CpuLayoutTransformIntent,
        _conjugate: bool,
        _output_bytes: &mut [std::mem::MaybeUninit<u8>],
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        *self.uninit_calls.lock().unwrap() += 1;
        Ok(CpuProviderOutcome::Unsupported(
            CpuProviderUnsupported::DType(DType::F64),
        ))
    }
}

fn assert_canonical_operand_materialization_with_layout(
    layout: Arc<dyn CpuLayoutTransformProvider>,
) {
    let gemm = Arc::new(CanonicalFallbackSpy::new(
        CpuProviderUnsupported::Conjugation,
        CanonicalFallbackBehavior::ConjugatedComplex,
    ));
    let bundle = CpuProviderBundle::custom_builder()
        .gemm_provider(gemm.clone())
        .engine_outer_grouped_gemm()
        .layout_transform_provider(layout)
        .build()
        .unwrap();
    let lhs = Tensor::from_vec_col_major(vec![1, 1], vec![Complex64::new(1.0, 2.0)]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![1, 1], vec![Complex64::new(3.0, 4.0)]).unwrap();
    let mut output =
        Tensor::from_vec_col_major(vec![1, 1], vec![Complex64::new(5.0, 1.0)]).unwrap();
    let mut accumulation = DotGeneralAccumulation::scaled(
        ContractionScalar::C64(Complex64::new(2.0, 0.0)),
        ContractionScalar::C64(Complex64::new(3.0, 0.0)),
    )
    .unwrap();
    accumulation.lhs_conj = true;
    let fixture = execution_context_fixture(1);

    bundle
        .execute_dot_general_into(
            &fixture.entry(),
            &mut BufferPool::new(),
            &mut GemmAnalysisCache::default(),
            None,
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &config(&[1], &[0], &[], &[]),
            accumulation,
            TensorWrite::from_tensor(&mut output),
        )
        .unwrap();

    assert_eq!(*gemm.calls.lock().unwrap(), 2);
    assert_eq!(
        output.as_slice::<Complex64>().unwrap(),
        &[Complex64::new(37.0, -1.0)],
    );
}

#[test]
fn opted_out_layout_provider_keeps_zeroed_canonical_operand_values() {
    assert_canonical_operand_materialization_with_layout(Arc::new(OptOutLayoutProvider));
}

#[test]
fn opted_in_layout_provider_unsupported_falls_back_to_zeroed_materialization() {
    let layout = Arc::new(UnsupportedUninitLayoutProvider {
        uninit_calls: Arc::new(Mutex::new(0)),
    });
    let uninit_calls = Arc::clone(&layout.uninit_calls);
    assert_canonical_operand_materialization_with_layout(layout);
    // Both canonical operands (lhs and rhs) attempted the uninit path before
    // falling back to the zeroed materialization.
    assert_eq!(*uninit_calls.lock().unwrap(), 2);
}
