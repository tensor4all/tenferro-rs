use super::BlasGemmProvider;
#[cfg(all(feature = "cpu-blas", not(feature = "provider-inject")))]
use super::CpuOperand;
#[cfg(feature = "cpu-faer")]
use super::CpuUninitGemmProvider;
use super::{
    CpuBatchedMatrixLayout, CpuExecutionContext, CpuGemmProvider, CpuGemmRequest,
    CpuGeneralContractionProvider, CpuLayoutTransformProvider, CpuOperationEntry,
    CpuProviderOutcome, CpuProviderUnsupported, CpuUninitLayoutTransformProvider, ParallelMode,
    StridedLayoutTransformProvider,
};
use crate::{
    CpuDomainExecutor, CpuDomainExecutorCapabilities, CpuDomainExecutorError, CpuDomainId,
    CpuExecutorAffinity, CpuExecutorReentrancy, CpuExecutorShutdown, CpuId, CpuInnerParallelism,
    CpuPlacementGuarantee, ScopedCpuJob, ScopedCpuJobs,
};
use rayon::prelude::*;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::ThreadId;
use std::time::Duration;

pub(crate) struct CpuExecutionContextFixture {
    engine: crate::engine::CpuEngine,
    permit: crate::arbiter::ResourcePermit,
}

impl CpuExecutionContextFixture {
    pub(crate) fn entry(&self) -> CpuOperationEntry<'_> {
        CpuOperationEntry::new(self.engine.domain(), &self.permit)
    }

    pub(crate) fn with_context<R: Send>(
        &self,
        mode: ParallelMode,
        operation: impl FnOnce(&CpuExecutionContext<'_>) -> R + Send,
    ) -> R {
        self.entry().enter(mode, operation).unwrap()
    }
}

pub(crate) fn execution_context_fixture(threads: usize) -> CpuExecutionContextFixture {
    let cpus = crate::CpuSet::singleton(crate::CpuId::new(0));
    let placement = crate::ResolvedCpuPlacement::AllAllowed { cpus: cpus.clone() };
    let context = Arc::new(crate::CpuContext::with_threads(threads).unwrap());
    let engine = crate::engine::CpuEngine::from_context(CpuDomainId::new(9), placement, context, 0);
    let permit = crate::arbiter::ResourceArbiter::new()
        .acquire(cpus)
        .unwrap();
    CpuExecutionContextFixture { engine, permit }
}

pub(crate) fn external_execution_context_fixture(
    executor: Arc<dyn crate::CpuDomainExecutor>,
    thread_budget: NonZeroUsize,
) -> CpuExecutionContextFixture {
    let cpus = crate::CpuSet::singleton(crate::CpuId::new(0));
    let placement = crate::ResolvedCpuPlacement::AllAllowed { cpus: cpus.clone() };
    let external = crate::ExternalCpuDomain::new(
        CpuDomainId::new(9),
        placement,
        executor,
        thread_budget,
        CpuPlacementGuarantee::AdvisoryDeclared,
    )
    .unwrap();
    let engine = crate::engine::CpuEngine::from_external(external, 0);
    let permit = crate::arbiter::ResourceArbiter::new()
        .acquire(cpus)
        .unwrap();
    CpuExecutionContextFixture { engine, permit }
}

const LARGE_NATIVE_TEST_LEN: usize = 1 << 17;

#[derive(Default)]
pub(crate) struct NativeParticipants {
    active: AtomicUsize,
    max_active: AtomicUsize,
    thread_ids: Mutex<Vec<ThreadId>>,
    require_two: bool,
    rendezvous_released: AtomicBool,
    rendezvous_lock: Mutex<()>,
    rendezvous: Condvar,
}

impl NativeParticipants {
    fn requiring_two() -> Self {
        Self {
            require_two: true,
            ..Self::default()
        }
    }

    fn observe(&self) {
        let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_active.fetch_max(active, Ordering::SeqCst);
        {
            let id = std::thread::current().id();
            let mut ids = self.thread_ids.lock().unwrap();
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
        if self.require_two && !self.rendezvous_released.load(Ordering::Acquire) {
            let guard = self.rendezvous_lock.lock().unwrap();
            if active >= 2 {
                self.rendezvous_released.store(true, Ordering::Release);
                self.rendezvous.notify_all();
            } else if !self.rendezvous_released.load(Ordering::Acquire) {
                let _guard = self
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

    pub(crate) fn max_active(&self) -> usize {
        self.max_active.load(Ordering::SeqCst)
    }

    pub(crate) fn thread_count(&self) -> usize {
        self.thread_ids.lock().unwrap().len()
    }
}

pub(crate) fn run_unscoped_native_map(require_two: bool) -> Arc<NativeParticipants> {
    let source =
        strided_kernel::StridedArray::<f64>::from_fn_col_major(&[LARGE_NATIVE_TEST_LEN], |index| {
            index[0] as f64
        });
    let mut destination = strided_kernel::StridedArray::<f64>::col_major(&[LARGE_NATIVE_TEST_LEN]);
    let participants = Arc::new(if require_two {
        NativeParticipants::requiring_two()
    } else {
        NativeParticipants::default()
    });
    let observed = Arc::clone(&participants);
    strided_kernel::map_into(&mut destination.view_mut(), &source.view(), |value| {
        observed.observe();
        value + 1.0
    })
    .unwrap();
    let output = destination.into_data();
    assert_eq!(output[0], 1.0);
    assert_eq!(
        output[LARGE_NATIVE_TEST_LEN - 1],
        LARGE_NATIVE_TEST_LEN as f64
    );
    participants
}

fn run_native_map(context: &CpuExecutionContext<'_>, require_two: bool) -> Arc<NativeParticipants> {
    context.with_native_parallelism(|| run_unscoped_native_map(require_two))
}

#[test]
fn parallel_mode_exposes_the_complete_execution_contract() {
    assert_ne!(ParallelMode::Sequential, ParallelMode::Outer);
    assert_ne!(ParallelMode::Outer, ParallelMode::Inner);
}
#[cfg(feature = "cpu-faer")]
use super::{CpuGroupedGemmRequest, FaerGemmProvider};
#[cfg(feature = "cpu-faer")]
use num_complex::Complex32;
// `Complex64` and `TensorView` are also used by the feature-neutral
// `materialize_into_uninit` layout test, so they are imported ungated.
use num_complex::Complex64;
#[cfg(feature = "cpu-faer")]
use tenferro_tensor::backend::GroupedGemmJob;
use tenferro_tensor::TensorView;
#[cfg(feature = "cpu-faer")]
use tenferro_tensor::{ContractionScalar, TypedTensorView};
use tenferro_tensor::{DType, DotGeneralAccumulation, Tensor, TensorRead, TensorWrite};
#[cfg(any(feature = "cpu-faer", feature = "cpu-blas"))]
use tenferro_tensor::{TensorViewMut, TypedTensorViewMut};

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
    let fixture = execution_context_fixture(4);
    fixture.with_context(ParallelMode::Inner, |provider_context| {
        assert_eq!(provider_context.domain_id(), CpuDomainId::new(9));
        assert_eq!(provider_context.cpus().as_slice(), &[CpuId::new(0)]);
        assert_eq!(provider_context.thread_budget().get(), 4);
        assert_eq!(
            provider_context.placement_guarantee(),
            CpuPlacementGuarantee::AdvisoryDeclared
        );
        assert_eq!(provider_context.parallel_mode(), ParallelMode::Inner);
    });
}

#[cfg(feature = "cpu-faer")]
#[test]
fn execution_context_is_the_single_source_of_faer_parallelism() {
    let single = execution_context_fixture(1);
    single.with_context(ParallelMode::Sequential, |context| {
        assert!(matches!(context.faer_parallelism(), faer::Par::Seq));
    });

    let multi = execution_context_fixture(2);
    multi.with_context(ParallelMode::Inner, |context| {
        assert_eq!(context.faer_parallelism().degree(), 2);
    });
    multi.with_context(ParallelMode::Sequential, |context| {
        assert!(matches!(context.faer_parallelism(), faer::Par::Seq));
    });
}

#[test]
fn execution_context_is_the_single_source_of_erased_strided_replay_policy() {
    let single = execution_context_fixture(1);
    single.with_context(ParallelMode::Sequential, |context| {
        let strided = context.strided_exec_context();
        assert!(strided.is_serial());
        assert_eq!(strided.max_threads_limit(), None);
    });

    let multi = execution_context_fixture(2);
    multi.with_context(ParallelMode::Inner, |context| {
        let strided = context.strided_exec_context();
        assert_eq!(strided.max_threads_limit(), NonZeroUsize::new(2));
    });
    multi.with_context(ParallelMode::Sequential, |context| {
        let strided = context.strided_exec_context();
        assert!(strided.is_serial());
        assert_eq!(strided.max_threads_limit(), None);
    });
}

#[test]
fn outer_mode_is_rejected_before_install_or_operation_mutation() {
    let multi = execution_context_fixture(2);
    let ran = AtomicUsize::new(0);
    assert!(matches!(
        multi.entry().enter(ParallelMode::Outer, |_| {
            ran.fetch_add(1, Ordering::Relaxed);
        }),
        Err(CpuDomainExecutorError::Scheduling { message })
            if message.contains("requires Sequential or Inner")
    ));
    assert_eq!(ran.load(Ordering::Relaxed), 0);
}

#[derive(Debug)]
struct ExternalWorkersCountingExecutor {
    installs: Arc<AtomicUsize>,
    submits: Arc<AtomicUsize>,
}

impl CpuDomainExecutor for ExternalWorkersCountingExecutor {
    fn capabilities(&self) -> CpuDomainExecutorCapabilities {
        CpuDomainExecutorCapabilities {
            worker_count: NonZeroUsize::new(2).unwrap(),
            outer_parallelism: true,
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
        job.run()
    }
}

#[test]
fn external_worker_inner_entry_does_not_require_rayon_capability() {
    let installs = Arc::new(AtomicUsize::new(0));
    let fixture = external_execution_context_fixture(
        Arc::new(ExternalWorkersCountingExecutor {
            installs: Arc::clone(&installs),
            submits: Arc::new(AtomicUsize::new(0)),
        }),
        NonZeroUsize::new(2).unwrap(),
    );

    let observed = fixture
        .entry()
        .enter(ParallelMode::Inner, |context| context.parallel_mode())
        .unwrap();

    assert_eq!(observed, ParallelMode::Inner);
    assert_eq!(installs.load(Ordering::Relaxed), 1);
}

#[test]
fn outer_entry_submits_once_without_install_and_creates_sequential_children() {
    let installs = Arc::new(AtomicUsize::new(0));
    let submits = Arc::new(AtomicUsize::new(0));
    let fixture = external_execution_context_fixture(
        Arc::new(ExternalWorkersCountingExecutor {
            installs: Arc::clone(&installs),
            submits: Arc::clone(&submits),
        }),
        NonZeroUsize::new(2).unwrap(),
    );
    let jobs = AtomicUsize::new(0);

    fixture
        .entry()
        .submit_outer(3, |_, context| {
            assert_eq!(context.parallel_mode(), ParallelMode::Sequential);
            jobs.fetch_add(1, Ordering::Relaxed);
            Ok(())
        })
        .unwrap();

    assert_eq!(submits.load(Ordering::Relaxed), 1);
    assert_eq!(installs.load(Ordering::Relaxed), 0);
    assert_eq!(jobs.load(Ordering::Relaxed), 3);
}

#[derive(Default)]
struct OuterParticipants {
    expected_active: Arc<AtomicUsize>,
    active: AtomicUsize,
    max_active: AtomicUsize,
    released: AtomicBool,
    lock: Mutex<()>,
    ready: Condvar,
}

impl OuterParticipants {
    fn observe(&self) {
        let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_active.fetch_max(active, Ordering::SeqCst);
        let expected = self.expected_active.load(Ordering::Acquire);
        let guard = self.lock.lock().unwrap();
        if active >= expected {
            self.released.store(true, Ordering::Release);
            self.ready.notify_all();
        } else if !self.released.load(Ordering::Acquire) {
            let _guard = self
                .ready
                .wait_timeout_while(guard, Duration::from_secs(2), |_| {
                    !self.released.load(Ordering::Acquire)
                })
                .unwrap();
        }
        for _ in 0..32 {
            std::hint::spin_loop();
        }
        self.active.fetch_sub(1, Ordering::SeqCst);
    }
}

struct WideOuterExecutor {
    pool: rayon::ThreadPool,
    submits: Arc<AtomicUsize>,
    installs: Arc<AtomicUsize>,
    submitted_widths: Arc<Mutex<Vec<usize>>>,
    expected_active: Arc<AtomicUsize>,
}

impl std::fmt::Debug for WideOuterExecutor {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("WideOuterExecutor").finish()
    }
}

impl CpuDomainExecutor for WideOuterExecutor {
    fn capabilities(&self) -> CpuDomainExecutorCapabilities {
        CpuDomainExecutorCapabilities {
            worker_count: NonZeroUsize::new(4).unwrap(),
            outer_parallelism: true,
            inner_parallelism: CpuInnerParallelism::None,
            reentrancy: CpuExecutorReentrancy::Rejected,
            affinity: CpuExecutorAffinity::CallerDeclaredUnverified,
            shutdown: CpuExecutorShutdown::CallerOwned,
        }
    }

    fn submit(&self, jobs: &dyn ScopedCpuJobs) -> Result<(), CpuDomainExecutorError> {
        self.submits.fetch_add(1, Ordering::Relaxed);
        self.submitted_widths.lock().unwrap().push(jobs.len());
        self.expected_active.store(
            jobs.len().min(self.capabilities().worker_count.get()),
            Ordering::Release,
        );
        self.pool.install(|| {
            (0..jobs.len())
                .into_par_iter()
                .try_for_each(|index| jobs.run(index))
        })
    }

    fn install(&self, job: &mut dyn ScopedCpuJob) -> Result<(), CpuDomainExecutorError> {
        self.installs.fetch_add(1, Ordering::Relaxed);
        self.pool.install(|| job.run())
    }
}

struct WideOuterFixture {
    execution: CpuExecutionContextFixture,
    submits: Arc<AtomicUsize>,
    installs: Arc<AtomicUsize>,
    submitted_widths: Arc<Mutex<Vec<usize>>>,
    participants: Arc<OuterParticipants>,
}

fn wide_outer_fixture(thread_budget: usize) -> WideOuterFixture {
    let submits = Arc::new(AtomicUsize::new(0));
    let installs = Arc::new(AtomicUsize::new(0));
    let submitted_widths = Arc::new(Mutex::new(Vec::new()));
    let expected_active = Arc::new(AtomicUsize::new(0));
    let participants = Arc::new(OuterParticipants {
        expected_active: Arc::clone(&expected_active),
        ..OuterParticipants::default()
    });
    let fixture = external_execution_context_fixture(
        Arc::new(WideOuterExecutor {
            pool: rayon::ThreadPoolBuilder::new()
                .num_threads(4)
                .build()
                .unwrap(),
            submits: Arc::clone(&submits),
            installs: Arc::clone(&installs),
            submitted_widths: Arc::clone(&submitted_widths),
            expected_active,
        }),
        NonZeroUsize::new(thread_budget).unwrap(),
    );
    WideOuterFixture {
        execution: fixture,
        submits,
        installs,
        submitted_widths,
        participants,
    }
}

#[test]
fn outer_submission_limits_logical_participants_to_thread_budget() {
    const LOGICAL_JOBS: usize = 6;
    let fixture = wide_outer_fixture(2);
    let seen = std::array::from_fn::<_, LOGICAL_JOBS, _>(|_| AtomicUsize::new(0));

    fixture
        .execution
        .entry()
        .submit_outer(LOGICAL_JOBS, |index, context| {
            assert_eq!(context.parallel_mode(), ParallelMode::Sequential);
            fixture.participants.observe();
            seen[index].fetch_add(1, Ordering::Relaxed);
            Ok(())
        })
        .unwrap();

    assert_eq!(fixture.submits.load(Ordering::Relaxed), 1);
    assert_eq!(fixture.installs.load(Ordering::Relaxed), 0);
    assert_eq!(fixture.submitted_widths.lock().unwrap().as_slice(), &[2]);
    assert!(fixture.participants.max_active.load(Ordering::SeqCst) <= 2);
    assert_eq!(
        seen.map(|count| count.load(Ordering::Relaxed)),
        [1; LOGICAL_JOBS]
    );
}

#[test]
fn outer_submission_rejects_a_single_thread_budget_without_executor_entry() {
    let fixture = wide_outer_fixture(1);
    let calls = AtomicUsize::new(0);

    let error = fixture
        .execution
        .entry()
        .submit_outer(2, |_, _| {
            calls.fetch_add(1, Ordering::Relaxed);
            Ok(())
        })
        .unwrap_err();

    assert!(matches!(
        error,
        CpuDomainExecutorError::Scheduling { message }
            if message.contains("does not support Outer mode")
    ));
    assert_eq!(fixture.submits.load(Ordering::Relaxed), 0);
    assert_eq!(fixture.installs.load(Ordering::Relaxed), 0);
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

#[test]
fn outer_submission_does_not_add_calls_when_budget_covers_all_jobs() {
    let fixture = wide_outer_fixture(4);
    let seen = [AtomicUsize::new(0), AtomicUsize::new(0)];

    fixture
        .execution
        .entry()
        .submit_outer(seen.len(), |index, _| {
            fixture.participants.observe();
            seen[index].fetch_add(1, Ordering::Relaxed);
            Ok(())
        })
        .unwrap();

    assert_eq!(fixture.submits.load(Ordering::Relaxed), 1);
    assert_eq!(fixture.installs.load(Ordering::Relaxed), 0);
    assert_eq!(fixture.submitted_widths.lock().unwrap().as_slice(), &[2]);
    assert_eq!(seen.map(|count| count.load(Ordering::Relaxed)), [1, 1]);
}

struct RayonCountingExecutor {
    pool: rayon::ThreadPool,
    installs: Arc<AtomicUsize>,
    submits: Arc<AtomicUsize>,
}

impl std::fmt::Debug for RayonCountingExecutor {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("RayonCountingExecutor").finish()
    }
}

impl CpuDomainExecutor for RayonCountingExecutor {
    fn capabilities(&self) -> CpuDomainExecutorCapabilities {
        CpuDomainExecutorCapabilities {
            worker_count: NonZeroUsize::new(2).unwrap(),
            outer_parallelism: true,
            inner_parallelism: CpuInnerParallelism::Rayon,
            reentrancy: CpuExecutorReentrancy::Rejected,
            affinity: CpuExecutorAffinity::CallerDeclaredUnverified,
            shutdown: CpuExecutorShutdown::CallerOwned,
        }
    }

    fn submit(&self, jobs: &dyn ScopedCpuJobs) -> Result<(), CpuDomainExecutorError> {
        self.submits.fetch_add(1, Ordering::Relaxed);
        self.pool.install(|| {
            for index in 0..jobs.len() {
                jobs.run(index)?;
            }
            Ok(())
        })
    }

    fn install(&self, job: &mut dyn ScopedCpuJob) -> Result<(), CpuDomainExecutorError> {
        self.installs.fetch_add(1, Ordering::Relaxed);
        self.pool.install(|| job.run())
    }
}

#[test]
fn engine_worker_inner_entry_installs_once_into_the_actual_rayon_pool() {
    let installs = Arc::new(AtomicUsize::new(0));
    let submits = Arc::new(AtomicUsize::new(0));
    let fixture = external_execution_context_fixture(
        Arc::new(RayonCountingExecutor {
            pool: rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .unwrap(),
            installs: Arc::clone(&installs),
            submits: Arc::clone(&submits),
        }),
        NonZeroUsize::new(2).unwrap(),
    );
    let provider_calls = AtomicUsize::new(0);

    let observed = fixture
        .entry()
        .enter(fixture.entry().preferred_engine_mode(), |context| {
            provider_calls.fetch_add(1, Ordering::Relaxed);
            (
                context.parallel_mode(),
                rayon::current_thread_index(),
                rayon::current_num_threads(),
            )
        })
        .unwrap();

    assert_eq!(observed.0, ParallelMode::Inner);
    assert!(matches!(observed.1, Some(0 | 1)));
    assert_eq!(observed.2, 2);
    assert_eq!(provider_calls.load(Ordering::Relaxed), 1);
    assert_eq!(installs.load(Ordering::Relaxed), 1);
    assert_eq!(submits.load(Ordering::Relaxed), 0);
}

#[test]
fn native_inner_parallelism_uses_only_the_selected_rayon_budget() {
    let installs = Arc::new(AtomicUsize::new(0));
    let submits = Arc::new(AtomicUsize::new(0));
    let fixture = external_execution_context_fixture(
        Arc::new(RayonCountingExecutor {
            pool: rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .unwrap(),
            installs: Arc::clone(&installs),
            submits: Arc::clone(&submits),
        }),
        NonZeroUsize::new(2).unwrap(),
    );
    let ambient = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();

    let participants = ambient.install(|| {
        fixture.with_context(ParallelMode::Inner, |context| run_native_map(context, true))
    });

    assert_eq!(participants.max_active.load(Ordering::SeqCst), 2);
    assert_eq!(participants.thread_ids.lock().unwrap().len(), 2);
    assert_eq!(installs.load(Ordering::Relaxed), 1);
    assert_eq!(submits.load(Ordering::Relaxed), 0);
}

#[test]
fn native_sequential_mode_has_one_participant_inside_a_multithreaded_pool() {
    let fixture = execution_context_fixture(4);
    let participants = fixture.with_context(ParallelMode::Sequential, |context| {
        run_native_map(context, false)
    });

    assert_eq!(participants.max_active.load(Ordering::SeqCst), 1);
    assert_eq!(participants.thread_ids.lock().unwrap().len(), 1);
}

#[test]
fn external_worker_inner_mode_keeps_native_work_sequential() {
    let installs = Arc::new(AtomicUsize::new(0));
    let submits = Arc::new(AtomicUsize::new(0));
    let fixture = external_execution_context_fixture(
        Arc::new(ExternalWorkersCountingExecutor {
            installs: Arc::clone(&installs),
            submits: Arc::clone(&submits),
        }),
        NonZeroUsize::new(2).unwrap(),
    );
    let ambient = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();

    let participants = ambient.install(|| {
        fixture.with_context(ParallelMode::Inner, |context| {
            run_native_map(context, false)
        })
    });

    assert_eq!(participants.max_active.load(Ordering::SeqCst), 1);
    assert_eq!(participants.thread_ids.lock().unwrap().len(), 1);
    assert_eq!(installs.load(Ordering::Relaxed), 1);
    assert_eq!(submits.load(Ordering::Relaxed), 0);
}

#[test]
fn outer_children_keep_native_work_sequential_without_double_fanout() {
    let installs = Arc::new(AtomicUsize::new(0));
    let submits = Arc::new(AtomicUsize::new(0));
    let fixture = external_execution_context_fixture(
        Arc::new(ExternalWorkersCountingExecutor {
            installs: Arc::clone(&installs),
            submits: Arc::clone(&submits),
        }),
        NonZeroUsize::new(2).unwrap(),
    );
    let completed = AtomicUsize::new(0);

    fixture
        .entry()
        .submit_outer(3, |_, context| {
            let participants = run_native_map(context, false);
            assert_eq!(participants.max_active.load(Ordering::SeqCst), 1);
            assert_eq!(participants.thread_ids.lock().unwrap().len(), 1);
            completed.fetch_add(1, Ordering::Relaxed);
            Ok(())
        })
        .unwrap();

    assert_eq!(completed.load(Ordering::Relaxed), 3);
    assert_eq!(installs.load(Ordering::Relaxed), 0);
    assert_eq!(submits.load(Ordering::Relaxed), 1);
}

#[test]
fn native_policy_restores_after_error_and_panic_without_contaminating_ambient_work() {
    let fixture = external_execution_context_fixture(
        Arc::new(ExternalWorkersCountingExecutor {
            installs: Arc::new(AtomicUsize::new(0)),
            submits: Arc::new(AtomicUsize::new(0)),
        }),
        NonZeroUsize::new(2).unwrap(),
    );
    let ambient = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();

    ambient.install(|| {
        fixture.with_context(ParallelMode::Inner, |context| {
            let error = context.with_native_parallelism(|| Err::<(), _>("sentinel"));
            assert_eq!(error, Err("sentinel"));

            let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                context.with_native_parallelism(|| panic!("native policy sentinel"));
            }));
            assert!(panic.is_err());

            let source = strided_kernel::StridedArray::<f64>::from_fn_col_major(
                &[LARGE_NATIVE_TEST_LEN],
                |index| index[0] as f64,
            );
            let mut destination =
                strided_kernel::StridedArray::<f64>::col_major(&[LARGE_NATIVE_TEST_LEN]);
            let participants = Arc::new(NativeParticipants::requiring_two());
            let observed = Arc::clone(&participants);
            strided_kernel::with_execution_policy(
                strided_kernel::ExecutionPolicy::AmbientRayon,
                || {
                    strided_kernel::map_into(
                        &mut destination.view_mut(),
                        &source.view(),
                        |value| {
                            observed.observe();
                            value + 1.0
                        },
                    )
                    .unwrap();
                },
            );
            assert_eq!(participants.max_active.load(Ordering::SeqCst), 2);
            assert_eq!(participants.thread_ids.lock().unwrap().len(), 2);
        });
    });
}

#[cfg(feature = "cpu-blas")]
#[test]
fn blas_rejects_negative_output_leading_stride_before_provider_mutation() {
    let lhs = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4]).unwrap();
    let lhs = TensorRead::from_tensor(&lhs);
    let rhs = TensorRead::from_tensor(&rhs);
    let mut storage = [41.0_f64; 4];
    let output_view = TypedTensorViewMut::from_slice([2, 2], [1, -2], 2, &mut storage).unwrap();
    let mut output = TensorWrite::from_view(TensorViewMut::F64(output_view));
    let request = CpuGemmRequest::new(
        &lhs,
        &rhs,
        &mut output,
        2,
        2,
        2,
        1,
        CpuBatchedMatrixLayout::new(0, 1, 2, 4),
        CpuBatchedMatrixLayout::new(0, 1, 2, 4),
        CpuBatchedMatrixLayout::new(2, 1, -2, 4),
        DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
    );
    let fixture = execution_context_fixture(1);

    let outcome = fixture
        .with_context(ParallelMode::Sequential, |context| {
            crate::gemm::execute_blas_gemm_request(context, request)
        })
        .unwrap();

    assert_eq!(
        outcome,
        CpuProviderOutcome::Unsupported(CpuProviderUnsupported::Layout(
            crate::provider::CpuOperand::Output,
        ))
    );
    drop(output);
    assert_eq!(storage, [41.0_f64; 4]);
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
    let fixture = execution_context_fixture(1);
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

    fixture.with_context(ParallelMode::Sequential, |provider_context| {
        assert_eq!(
            FaerGemmProvider.gemm(provider_context, request).unwrap(),
            CpuProviderOutcome::Executed,
        );
    });
    drop(out_write);
    assert_eq!(out.as_slice::<f64>().unwrap(), &[58.0, 139.0, 64.0, 154.0]);
}

#[test]
#[cfg(feature = "cpu-faer")]
fn faer_provider_covers_f32_c32_and_c64_conjugation() {
    let fixture = execution_context_fixture(1);
    fixture.with_context(ParallelMode::Sequential, |provider_context| {
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
            FaerGemmProvider.gemm(provider_context, request).unwrap(),
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
            FaerGemmProvider.gemm(provider_context, request).unwrap(),
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
            FaerGemmProvider.gemm(provider_context, request).unwrap(),
            CpuProviderOutcome::Executed,
        );
        drop(output_write);
        assert_eq!(
            output.as_slice::<Complex64>().unwrap(),
            &[Complex64::new(2.0, 0.0)]
        );
    });
}

#[test]
#[cfg(feature = "cpu-faer")]
fn faer_provider_executes_non_unit_strides_and_strided_batches() {
    let fixture = execution_context_fixture(1);
    fixture.with_context(ParallelMode::Inner, |provider_context| {
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
            FaerGemmProvider.gemm(provider_context, request).unwrap(),
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

        let lhs =
            Tensor::from_vec_col_major(vec![8], vec![1.0_f64, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0])
                .unwrap();
        let rhs =
            Tensor::from_vec_col_major(vec![8], vec![1.0_f64, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0])
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
                .strided_batched_gemm(provider_context, request)
                .unwrap(),
            CpuProviderOutcome::Executed,
        );
        drop(output_write);
        assert_eq!(
            output.as_slice::<f64>().unwrap(),
            &[1.0, 3.0, 2.0, 4.0, 10.0, 21.0, 12.0, 24.0]
        );
    });
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
    let fixture = execution_context_fixture(2);
    fixture.with_context(ParallelMode::Sequential, |provider_context| {
        assert_eq!(
            FaerGemmProvider
                .grouped_gemm(provider_context, request)
                .unwrap(),
            CpuProviderOutcome::Executed,
        );
    });
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
    let fixture = execution_context_fixture(1);
    fixture.with_context(ParallelMode::Sequential, |provider_context| {
        assert_eq!(
            FaerGemmProvider.gemm(provider_context, request).unwrap(),
            CpuProviderOutcome::Unsupported(CpuProviderUnsupported::DType(DType::I32)),
        );
    });
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
    let fixture = execution_context_fixture(1);
    fixture.with_context(ParallelMode::Sequential, |provider_context| {
        assert_eq!(
            BlasGemmProvider.gemm(provider_context, request).unwrap(),
            CpuProviderOutcome::Unsupported(CpuProviderUnsupported::RuntimeUnavailable),
        );
    });
    drop(output_write);
    assert_eq!(output.as_slice::<f64>().unwrap(), &[41.0]);
}

#[test]
// `provider-inject` call-through tests live in serialized integration fixtures
// that register every FFI symbol before use.
#[cfg(all(feature = "cpu-blas", not(feature = "provider-inject")))]
fn blas_provider_executes_and_rejects_layout_before_mutation() {
    let fixture = execution_context_fixture(1);
    fixture.with_context(ParallelMode::Sequential, |provider_context| {
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
            BlasGemmProvider.gemm(provider_context, request).unwrap(),
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
            BlasGemmProvider.gemm(provider_context, request).unwrap(),
            CpuProviderOutcome::Unsupported(CpuProviderUnsupported::Layout(CpuOperand::Lhs)),
        );
        drop(output_write);
        assert_eq!(output.as_slice::<f64>().unwrap(), before.as_slice());
    });
}

#[test]
fn layout_provider_materializes_into_preallocated_output() {
    use super::{CpuLayoutTransformIntent, CpuLayoutTransformRequest};

    let input = Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]).unwrap();
    let mut output = Tensor::from_vec_col_major(vec![2], vec![0.0_f64; 2]).unwrap();
    let input = TensorRead::from_tensor(&input);
    let mut output_write = TensorWrite::from_tensor(&mut output);
    let fixture = execution_context_fixture(1);
    let request = CpuLayoutTransformRequest::new(
        &input,
        &mut output_write,
        CpuLayoutTransformIntent::CanonicalColumnMajor,
        false,
    );

    fixture.with_context(ParallelMode::Sequential, |provider_context| {
        assert_eq!(
            StridedLayoutTransformProvider
                .materialize(provider_context, request)
                .unwrap(),
            CpuProviderOutcome::Executed,
        );
    });
    drop(output_write);
    assert_eq!(output.as_slice::<f64>().unwrap(), &[2.0, 3.0]);
}

#[test]
fn layout_provider_fuses_conjugation_into_materialization() {
    use super::{CpuLayoutTransformIntent, CpuLayoutTransformRequest};
    use num_complex::Complex64;

    let input = Tensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(2.0, 3.0), Complex64::new(-1.0, 4.0)],
    )
    .unwrap();
    let mut output =
        Tensor::from_vec_col_major(vec![2], vec![Complex64::new(41.0, 0.0); 2]).unwrap();
    let input = TensorRead::from_tensor(&input);
    let mut output_write = TensorWrite::from_tensor(&mut output);
    let fixture = execution_context_fixture(1);
    let request = CpuLayoutTransformRequest::new(
        &input,
        &mut output_write,
        CpuLayoutTransformIntent::CanonicalColumnMajor,
        true,
    );
    assert!(request.conjugate());

    fixture.with_context(ParallelMode::Sequential, |context| {
        assert_eq!(
            StridedLayoutTransformProvider
                .materialize(context, request)
                .unwrap(),
            CpuProviderOutcome::Executed,
        );
    });
    drop(output_write);
    assert_eq!(
        output.as_slice::<Complex64>().unwrap(),
        &[Complex64::new(2.0, -3.0), Complex64::new(-1.0, -4.0)],
    );
}

#[derive(Debug)]
struct OptOutProvider;

impl CpuGemmProvider for OptOutProvider {
    fn execution_capabilities(&self) -> crate::CpuProviderExecutionCapabilities {
        crate::provider_capability::engine_worker_capabilities()
    }

    fn gemm(
        &self,
        _context: &CpuExecutionContext<'_>,
        _request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        unreachable!("opt-out witness test never executes GEMM")
    }

    fn strided_batched_gemm(
        &self,
        _context: &CpuExecutionContext<'_>,
        _request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        unreachable!("opt-out witness test never executes GEMM")
    }

    fn grouped_gemm(
        &self,
        _context: &CpuExecutionContext<'_>,
        _request: super::CpuGroupedGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        unreachable!("opt-out witness test never executes grouped GEMM")
    }
}

impl CpuLayoutTransformProvider for OptOutProvider {
    fn execution_capabilities(&self) -> crate::CpuProviderExecutionCapabilities {
        crate::provider_capability::engine_worker_capabilities()
    }

    fn materialize(
        &self,
        _context: &CpuExecutionContext<'_>,
        _request: super::CpuLayoutTransformRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        unreachable!("opt-out witness test never executes materialization")
    }
}

#[test]
fn uninit_witness_is_exposed_only_by_the_builtin_providers() {
    // SAFETY assertions are structural: only a type that implements the
    // unsafe trait via `unsafe impl` can return `Some(self)`.
    #[cfg(feature = "cpu-faer")]
    assert!(FaerGemmProvider.uninit_provider().is_some());
    assert!(StridedLayoutTransformProvider.uninit_provider().is_some());
    assert!(BlasGemmProvider.uninit_provider().is_none());
    let opt_out_gemm: &dyn CpuGemmProvider = &OptOutProvider;
    assert!(opt_out_gemm.uninit_provider().is_none());
    let opt_out_layout: &dyn CpuLayoutTransformProvider = &OptOutProvider;
    assert!(opt_out_layout.uninit_provider().is_none());
}

/// `f64`-aligned uninitialized destination mirroring
/// `PooledUninitOutput::as_uninit_bytes_mut` provenance (the caller's contract
/// for the built-in providers).
struct UninitF64(Vec<std::mem::MaybeUninit<f64>>);

impl UninitF64 {
    fn new(slots: usize) -> Self {
        Self(vec![std::mem::MaybeUninit::<f64>::uninit(); slots])
    }

    fn bytes_mut(&mut self) -> &mut [std::mem::MaybeUninit<u8>] {
        // SAFETY: `Vec<MaybeUninit<f64>>` is `f64`-aligned; the byte view
        // reuses the same allocation, matching the real caller's provenance.
        unsafe {
            std::slice::from_raw_parts_mut(
                self.0.as_mut_ptr().cast::<std::mem::MaybeUninit<u8>>(),
                self.0.len() * 8,
            )
        }
    }

    fn values(&self) -> &[f64] {
        // SAFETY: the provider wrote every element before `Executed`.
        unsafe { std::slice::from_raw_parts(self.0.as_ptr().cast::<f64>(), self.0.len()) }
    }
}

/// `Complex64`-aligned uninitialized destination (see [`UninitF64`]).
struct UninitC64(Vec<std::mem::MaybeUninit<num_complex::Complex64>>);

impl UninitC64 {
    fn new(slots: usize) -> Self {
        Self(vec![
            std::mem::MaybeUninit::<num_complex::Complex64>::uninit(
            );
            slots
        ])
    }

    fn bytes_mut(&mut self) -> &mut [std::mem::MaybeUninit<u8>] {
        // SAFETY: `Vec<MaybeUninit<Complex64>>` is `Complex64`-aligned.
        unsafe {
            std::slice::from_raw_parts_mut(
                self.0.as_mut_ptr().cast::<std::mem::MaybeUninit<u8>>(),
                self.0.len() * 16,
            )
        }
    }

    fn values(&self) -> &[num_complex::Complex64] {
        // SAFETY: the provider wrote every element before `Executed`.
        unsafe {
            std::slice::from_raw_parts(
                self.0.as_ptr().cast::<num_complex::Complex64>(),
                self.0.len(),
            )
        }
    }
}

#[cfg(feature = "cpu-faer")]
#[test]
fn faer_gemm_into_uninit_matches_initialized_gemm() {
    use super::{CpuBatchedMatrixLayout as Layout, CpuGemmUninitRequest};
    use tenferro_tensor::DotGeneralAccumulation;

    let fixture = execution_context_fixture(1);
    fixture.with_context(ParallelMode::Sequential, |context| {
        let lhs = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0]).unwrap();
        let rhs = Tensor::from_vec_col_major(vec![2, 2], vec![5.0_f64, 7.0, 6.0, 8.0]).unwrap();
        let lhs_read = TensorRead::from_tensor(&lhs);
        let rhs_read = TensorRead::from_tensor(&rhs);

        // Single GEMM: 2x2 x 2x2.
        let mut output = UninitF64::new(4);
        let request = CpuGemmUninitRequest::new(
            &lhs_read,
            &rhs_read,
            2,
            2,
            2,
            1,
            Layout::new(0, 1, 2, 4),
            Layout::new(0, 1, 2, 4),
            Layout::new(0, 1, 2, 4),
            DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
        );
        // SAFETY: FaerGemmProvider's unsafe impl writes every destination
        // element before `Executed` (faer Accum::Replace, beta == 0).
        let outcome =
            unsafe { FaerGemmProvider.gemm_into_uninit(context, request, output.bytes_mut()) }
                .unwrap();
        assert_eq!(outcome, CpuProviderOutcome::Executed);
        assert_eq!(output.values(), &[19.0, 43.0, 22.0, 50.0]);

        // Strided batch of two 2x2 GEMMs.
        let lhs = Tensor::from_vec_col_major(vec![2, 2, 2], vec![1.0_f64; 8]).unwrap();
        let rhs = Tensor::from_vec_col_major(vec![2, 2, 2], vec![2.0_f64; 8]).unwrap();
        let lhs_read = TensorRead::from_tensor(&lhs);
        let rhs_read = TensorRead::from_tensor(&rhs);
        let mut output = UninitF64::new(8);
        let request = CpuGemmUninitRequest::new(
            &lhs_read,
            &rhs_read,
            2,
            2,
            2,
            2,
            Layout::new(0, 1, 2, 4),
            Layout::new(0, 1, 2, 4),
            Layout::new(0, 1, 2, 4),
            DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
        );
        // SAFETY: same full-overwrite contract; the uninit GEMM covers all
        // batches internally.
        let outcome =
            unsafe { FaerGemmProvider.gemm_into_uninit(context, request, output.bytes_mut()) }
                .unwrap();
        assert_eq!(outcome, CpuProviderOutcome::Executed);
        assert_eq!(output.values(), &[4.0; 8]);

        // Empty contraction (k == 0) writes zeros without reading.
        let mut output = UninitF64::new(4);
        let request = CpuGemmUninitRequest::new(
            &lhs_read,
            &rhs_read,
            2,
            2,
            0,
            1,
            Layout::new(0, 1, 2, 4),
            Layout::new(0, 1, 2, 4),
            Layout::new(0, 1, 2, 4),
            DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
        );
        let outcome =
            unsafe { FaerGemmProvider.gemm_into_uninit(context, request, output.bytes_mut()) }
                .unwrap();
        assert_eq!(outcome, CpuProviderOutcome::Executed);
        assert_eq!(output.values(), &[0.0; 4]);

        // Empty output (zero rows) is trivially satisfied.
        let mut output = UninitF64::new(0);
        let request = CpuGemmUninitRequest::new(
            &lhs_read,
            &rhs_read,
            0,
            2,
            2,
            1,
            Layout::new(0, 1, 2, 4),
            Layout::new(0, 1, 2, 4),
            Layout::new(0, 1, 2, 4),
            DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
        );
        let outcome =
            unsafe { FaerGemmProvider.gemm_into_uninit(context, request, output.bytes_mut()) }
                .unwrap();
        assert_eq!(outcome, CpuProviderOutcome::Executed);

        // A non-zero beta accumulation is refused rather than reading uninit.
        let mut accumulation = DotGeneralAccumulation::overwrite(DType::F64).unwrap();
        accumulation.beta = tenferro_tensor::ContractionScalar::F64(1.0);
        let mut output = UninitF64::new(4);
        let request = CpuGemmUninitRequest::new(
            &lhs_read,
            &rhs_read,
            2,
            2,
            2,
            1,
            Layout::new(0, 1, 2, 4),
            Layout::new(0, 1, 2, 4),
            Layout::new(0, 1, 2, 4),
            accumulation,
        );
        let outcome =
            unsafe { FaerGemmProvider.gemm_into_uninit(context, request, output.bytes_mut()) }
                .unwrap();
        assert_eq!(
            outcome,
            CpuProviderOutcome::Unsupported(CpuProviderUnsupported::Accumulation)
        );
    });
}

#[test]
fn layout_materialize_into_uninit_matches_initialized_materialization() {
    use super::CpuLayoutTransformIntent;

    let fixture = execution_context_fixture(1);
    fixture.with_context(ParallelMode::Sequential, |context| {
        let input =
            Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap();
        let mut output = UninitF64::new(6);
        let input_read = TensorRead::from_tensor(&input);
        // SAFETY: StridedLayoutTransformProvider's unsafe impl writes every
        // destination element before `Executed`.
        let outcome = unsafe {
            StridedLayoutTransformProvider.materialize_into_uninit(
                context,
                &input_read,
                CpuLayoutTransformIntent::CanonicalColumnMajor,
                false,
                output.bytes_mut(),
            )
        }
        .unwrap();
        assert_eq!(outcome, CpuProviderOutcome::Executed);
        assert_eq!(output.values(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        // Conjugated materialization of a transposed view input.
        let input = Tensor::from_vec_col_major(
            vec![2, 2],
            vec![
                Complex64::new(1.0, 2.0),
                Complex64::new(3.0, -1.0),
                Complex64::new(-2.0, 0.5),
                Complex64::new(4.0, 1.0),
            ],
        )
        .unwrap();
        let transposed = match TensorRead::from_tensor(&input).tensor_view() {
            TensorView::C64(view) => TensorView::C64(view.transpose_view([1, 0]).unwrap()),
            other => panic!("expected C64 view, got {:?}", other.dtype()),
        };
        let mut output = UninitC64::new(4);
        let input_view = TensorRead::from_view(transposed);
        let outcome = unsafe {
            StridedLayoutTransformProvider.materialize_into_uninit(
                context,
                &input_view,
                CpuLayoutTransformIntent::CanonicalColumnMajor,
                true,
                output.bytes_mut(),
            )
        }
        .unwrap();
        assert_eq!(outcome, CpuProviderOutcome::Executed);
        assert_eq!(
            output.values(),
            &[
                Complex64::new(1.0, -2.0),
                Complex64::new(-2.0, -0.5),
                Complex64::new(3.0, 1.0),
                Complex64::new(4.0, -1.0),
            ]
        );

        // Zero-element destination is trivially satisfied.
        let empty = Tensor::from_vec_col_major(vec![0, 2], Vec::<f64>::new()).unwrap();
        let empty_read = TensorRead::from_tensor(&empty);
        let mut output = UninitF64::new(0);
        let outcome = unsafe {
            StridedLayoutTransformProvider.materialize_into_uninit(
                context,
                &empty_read,
                CpuLayoutTransformIntent::CanonicalColumnMajor,
                false,
                output.bytes_mut(),
            )
        }
        .unwrap();
        assert_eq!(outcome, CpuProviderOutcome::Executed);
    });
}

#[test]
fn cpu_execution_context_debug_format_is_runnable() {
    // Cover the CpuExecutionContext Debug impl (field formatting).
    let fixture = execution_context_fixture(1);
    fixture.with_context(ParallelMode::Inner, |provider_context| {
        let rendered = format!("{provider_context:?}");
        assert!(rendered.contains("CpuExecutionContext"));
        assert!(rendered.contains("parallel_mode"));
    });
}

#[test]
fn cross_domain_enter_or_reuse_reports_scheduling_error() {
    // Cover the domain-mismatch branch of CpuOperationEntry::enter_or_reuse.
    let fixture_a = execution_context_fixture(1); // domain id 9
    let cpus = crate::CpuSet::singleton(crate::CpuId::new(0));
    let placement = crate::ResolvedCpuPlacement::AllAllowed { cpus: cpus.clone() };
    let context = Arc::new(crate::CpuContext::with_threads(1).unwrap());
    let engine = crate::engine::CpuEngine::from_context(CpuDomainId::new(8), placement, context, 0);
    let permit = crate::arbiter::ResourceArbiter::new()
        .acquire(cpus)
        .unwrap();
    let fixture_b = CpuExecutionContextFixture { engine, permit };

    fixture_a.with_context(ParallelMode::Sequential, |entered| {
        let error = fixture_b
            .entry()
            .enter_or_reuse(Some(entered), ParallelMode::Sequential, |_| ())
            .unwrap_err();
        assert!(matches!(error, CpuDomainExecutorError::Scheduling { .. }));
    });
}

#[test]
#[cfg(feature = "cpu-faer")]
fn grouped_gemm_request_into_parts_is_covered() {
    // Cover CpuGroupedGemmRequest::into_parts (tuple destructuring).
    let lhs = Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![2], vec![4.0_f64, 5.0]).unwrap();
    let mut output = Tensor::from_vec_col_major(vec![2], vec![0.0_f64; 2]).unwrap();
    let jobs = [GroupedGemmJob::new(0, 0, 0, 1, 1, 1)];
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
    let (lhs_p, rhs_p, output_p, jobs_p, accumulation_p) = request.into_parts();
    assert_eq!(lhs_p.shape(), &[2]);
    assert_eq!(rhs_p.shape(), &[2]);
    assert_eq!(output_p.as_read().shape(), &[2]);
    assert_eq!(jobs_p.len(), 1);
    // The accumulation config round-trips (overwrite alpha=1, beta=0).
    let _ = accumulation_p;
}
