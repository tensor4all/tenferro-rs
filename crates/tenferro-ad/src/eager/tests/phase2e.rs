use std::fmt::Write as _;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Barrier, Mutex,
};

use tenferro_cpu::provider::{
    CpuExecutionContext, CpuGemmProvider, CpuGemmRequest, CpuGroupedGemmRequest,
    CpuProviderOutcome, FaerGemmProvider,
};
use tenferro_cpu::{
    process_cpu_affinity, CpuBackend, CpuBackendKind, CpuDomainExecutor,
    CpuDomainExecutorCapabilities, CpuDomainExecutorError, CpuExecutorAffinity,
    CpuExecutorReentrancy, CpuExecutorShutdown, CpuInnerParallelism, CpuPlacementGuarantee,
    CpuProviderBundle, CpuSet, ExternalCpuDomain, NumaNodeId, ResolvedCpuPlacement, ScopedCpuJob,
    ScopedCpuJobs,
};
use tenferro_tensor::{CpuDomainId, DotGeneralConfig, SliceConfig};

use crate::eager_backend::EagerBackend;

use super::super::{EagerRuntime, EagerTensor};
use super::Tensor;

fn current_cpu() -> Option<usize> {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        unsafe extern "C" {
            fn sched_getcpu() -> std::ffi::c_int;
        }
        // SAFETY: `sched_getcpu` takes no pointers and has no preconditions.
        let cpu = unsafe { sched_getcpu() };
        (cpu >= 0).then_some(cpu as usize)
    }
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    {
        None
    }
}

fn audit_context(context: &tenferro_cpu::CpuContext, budget: usize) -> Vec<[usize; 2]> {
    let barrier = Arc::new(Barrier::new(budget));
    context.install(|| {
        rayon::broadcast(|worker| {
            barrier.wait();
            [
                worker.index(),
                current_cpu().expect("Phase 2E requires current-CPU observation"),
            ]
        })
    })
}

#[derive(Debug)]
struct RecordingExecutor {
    inner: tenferro_cpu::CpuContext,
    capabilities: CpuDomainExecutorCapabilities,
    installs: AtomicUsize,
    submits: AtomicUsize,
    observed_cpus: Mutex<Vec<usize>>,
}

impl RecordingExecutor {
    fn new(budget: usize, exact: bool) -> Arc<Self> {
        let allowed = process_cpu_affinity().expect("Phase 2E needs process affinity");
        let selected = CpuSet::new(allowed.as_slice().iter().copied().take(budget)).unwrap();
        let declared = selected.as_usize_vec();
        let inner = tenferro_cpu::CpuContext::with_pinned_cpus(selected, budget).unwrap();
        let executor = Arc::new(Self {
            inner,
            capabilities: CpuDomainExecutorCapabilities {
                worker_count: NonZeroUsize::new(budget).unwrap(),
                outer_parallelism: budget > 1,
                inner_parallelism: if budget > 1 {
                    CpuInnerParallelism::Rayon
                } else {
                    CpuInnerParallelism::None
                },
                reentrancy: CpuExecutorReentrancy::Rejected,
                affinity: if exact {
                    CpuExecutorAffinity::CallerDeclaredUnverified
                } else {
                    CpuExecutorAffinity::None
                },
                shutdown: CpuExecutorShutdown::CallerOwned,
            },
            installs: AtomicUsize::new(0),
            submits: AtomicUsize::new(0),
            observed_cpus: Mutex::new(Vec::new()),
        });
        if exact {
            let audit = audit_context(&executor.inner, budget);
            assert_eq!(audit.len(), budget);
            assert_eq!(
                audit
                    .iter()
                    .map(|item| item[0])
                    .collect::<std::collections::BTreeSet<_>>(),
                (0..budget).collect()
            );
            assert!(audit.iter().all(|item| declared.contains(&item[1])));
        }
        executor
    }

    fn observe_cpu(&self) {
        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            if let Some(cpu) = current_cpu() {
                let mut observed = self.observed_cpus.lock().unwrap();
                if !observed.contains(&cpu) {
                    observed.push(cpu);
                }
            }
        }
    }

    fn reset(&self) {
        self.installs.store(0, Ordering::SeqCst);
        self.submits.store(0, Ordering::SeqCst);
        self.observed_cpus.lock().unwrap().clear();
    }
}

struct ObservedJob<'a>(&'a RecordingExecutor, &'a mut dyn ScopedCpuJob);

impl ScopedCpuJob for ObservedJob<'_> {
    fn run(&mut self) -> Result<(), CpuDomainExecutorError> {
        self.0.observe_cpu();
        self.1.run()
    }
}

struct ObservedJobs<'a>(&'a RecordingExecutor, &'a dyn ScopedCpuJobs);

impl ScopedCpuJobs for ObservedJobs<'_> {
    fn len(&self) -> usize {
        self.1.len()
    }

    fn run(&self, index: usize) -> Result<(), CpuDomainExecutorError> {
        self.0.observe_cpu();
        self.1.run(index)
    }
}

impl CpuDomainExecutor for RecordingExecutor {
    fn capabilities(&self) -> CpuDomainExecutorCapabilities {
        self.capabilities
    }

    fn submit(&self, jobs: &dyn ScopedCpuJobs) -> Result<(), CpuDomainExecutorError> {
        self.submits.fetch_add(1, Ordering::SeqCst);
        CpuDomainExecutor::submit(&self.inner, &ObservedJobs(self, jobs))
    }

    fn install(&self, job: &mut dyn ScopedCpuJob) -> Result<(), CpuDomainExecutorError> {
        self.installs.fetch_add(1, Ordering::SeqCst);
        CpuDomainExecutor::install(&self.inner, &mut ObservedJob(self, job))
    }
}

#[derive(Debug, Default)]
struct RecordingGemm {
    calls: AtomicUsize,
    observed_cpus: Mutex<Vec<usize>>,
}

impl RecordingGemm {
    fn observe_cpu(&self) {
        if let Some(cpu) = current_cpu() {
            let mut observed = self.observed_cpus.lock().unwrap();
            if !observed.contains(&cpu) {
                observed.push(cpu);
            }
        }
    }
}

impl CpuGemmProvider for RecordingGemm {
    fn execution_capabilities(&self) -> tenferro_cpu::CpuProviderExecutionCapabilities {
        FaerGemmProvider.execution_capabilities()
    }

    fn gemm(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.observe_cpu();
        FaerGemmProvider.gemm(context, request)
    }

    fn strided_batched_gemm(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.observe_cpu();
        FaerGemmProvider.strided_batched_gemm(context, request)
    }

    fn grouped_gemm(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuGroupedGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.observe_cpu();
        FaerGemmProvider.grouped_gemm(context, request)
    }
}

struct CpuFixture {
    backend: CpuBackend,
    executor: Option<Arc<RecordingExecutor>>,
    provider: Arc<RecordingGemm>,
    declared_cpus: Vec<usize>,
    placement_audit: Vec<[usize; 2]>,
}

fn external_cpu_fixture(
    executor: Arc<RecordingExecutor>,
    provider: Arc<RecordingGemm>,
    bundle: CpuProviderBundle,
    exact: bool,
) -> CpuFixture {
    let budget = executor.capabilities.worker_count.get();
    let allowed = process_cpu_affinity().unwrap();
    let selected = CpuSet::new(allowed.as_slice().iter().copied().take(budget)).unwrap();
    let declared_cpus = if exact {
        selected.as_usize_vec()
    } else {
        Vec::new()
    };
    let placement_audit = audit_context(&executor.inner, budget);
    let id = CpuDomainId::new(0x2ead + budget as u64 + u64::from(u8::from(exact)));
    let domain = ExternalCpuDomain::new(
        id,
        if exact {
            ResolvedCpuPlacement::NumaNode {
                id: NumaNodeId::new(0x2e),
                cpus: selected,
            }
        } else {
            ResolvedCpuPlacement::AllAllowed { cpus: allowed }
        },
        Arc::clone(&executor) as Arc<dyn CpuDomainExecutor>,
        NonZeroUsize::new(budget).unwrap(),
        if exact {
            CpuPlacementGuarantee::ExactDeclared
        } else {
            CpuPlacementGuarantee::AdvisoryDeclared
        },
    )
    .unwrap();
    let backend =
        CpuBackend::from_external_managed_domains_with_provider_bundle(id, [domain], bundle)
            .unwrap();
    CpuFixture {
        backend,
        executor: Some(executor),
        provider,
        declared_cpus,
        placement_audit,
    }
}

fn cpu_fixture(ownership: &str, budget: usize) -> CpuFixture {
    let provider = Arc::new(RecordingGemm::default());
    let bundle = CpuProviderBundle::builder(CpuBackendKind::Faer)
        .gemm_provider(Arc::clone(&provider) as Arc<dyn CpuGemmProvider>)
        .build()
        .unwrap();
    if ownership == "managed-exact" {
        let coordinator = CpuBackend::with_threads_and_kind(budget, CpuBackendKind::Faer)
            .unwrap()
            .with_provider_bundle(bundle)
            .unwrap();
        let backend = coordinator
            .topology()
            .nodes()
            .first()
            .map(|node| {
                coordinator
                    .for_placement(tenferro_cpu::CpuPlacement::NumaNode(node.id()))
                    .unwrap()
            })
            .unwrap_or(coordinator);
        let barrier = Arc::new(Barrier::new(budget));
        let placement_audit = backend.install(|| {
            rayon::broadcast(|worker| {
                barrier.wait();
                [
                    worker.index(),
                    current_cpu().expect("Phase 2E requires current CPU"),
                ]
            })
        });
        return CpuFixture {
            declared_cpus: backend.resolved_placement().unwrap().cpus().as_usize_vec(),
            backend,
            executor: None,
            provider,
            placement_audit,
        };
    }
    let exact = ownership == "external-exact";
    let executor = RecordingExecutor::new(budget, exact);
    external_cpu_fixture(executor, provider, bundle, exact)
}

#[derive(Debug)]
struct Row {
    key: String,
    surface: &'static str,
    budget: usize,
    session_entry: usize,
    session_entry_cpus: Vec<usize>,
    placement_audit: Vec<[usize; 2]>,
    declared_cpus: Vec<usize>,
    downstream_vector: &'static str,
    actual_install: Option<usize>,
    actual_submit: Option<usize>,
    actual_provider: usize,
    operation_workers: Vec<[usize; 2]>,
    observed_cpus: Vec<usize>,
    numerical_passed: bool,
    typed_error_recovered: bool,
    unwind_recovered: bool,
    post_recovery_passed: bool,
    recovery: RecoveryEvidence,
}

#[derive(Debug)]
struct RecoveryEvidence {
    fresh_reset: bool,
    session_entry: usize,
    actual_install: Option<usize>,
    actual_submit: Option<usize>,
    actual_provider: usize,
    operation_workers: Vec<[usize; 2]>,
    observed_cpus: Vec<usize>,
    numerical_passed: bool,
    subset_passed: bool,
}

fn observe_actual_operation<R>(key: &str, operation: impl FnOnce() -> R) -> (R, Vec<[usize; 2]>) {
    #[cfg(tenferro_phase2e_operation_observe)]
    {
        let root = std::env::var_os("TENFERRO_PHASE2E_EVIDENCE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        let directory = root.join("dispatch-gates").join("operation-workers");
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join(format!("{}.csv", key.replace('/', "__")));
        let _ = std::fs::remove_file(&path);
        // SAFETY: the Phase 2E evidence test is one serialized test process and
        // restores the variable before proceeding to another row.
        unsafe { std::env::set_var("TENFERRO_PHASE2E_OPERATION_OBSERVATION_FILE", &path) };
        let result = operation();
        // SAFETY: see the serialized scope above.
        unsafe { std::env::remove_var("TENFERRO_PHASE2E_OPERATION_OBSERVATION_FILE") };
        let observations = std::fs::read_to_string(&path)
            .unwrap_or_default()
            .lines()
            .map(|line| {
                let (lane, cpu) = line.split_once(',').expect("valid operation observation");
                [lane.parse().unwrap(), cpu.parse().unwrap()]
            })
            .collect();
        let _ = std::fs::remove_file(&path);
        (result, observations)
    }
    #[cfg(not(tenferro_phase2e_operation_observe))]
    {
        let _ = key;
        (operation(), Vec::new())
    }
}

impl Row {
    fn gating_passed(&self) -> bool {
        self.session_entry == 1
            && self.numerical_passed
            && self.typed_error_recovered
            && self.unwind_recovered
            && self.post_recovery_passed
    }
}

struct Evidence {
    session_entries: Vec<usize>,
    characterization: Vec<Row>,
}

fn tensor(shape: Vec<usize>, salt: usize) -> Tensor {
    let len = shape.iter().product();
    Tensor::from_vec_col_major(
        shape,
        (0..len)
            .map(|index| ((index * 19 + salt * 11) % 103) as f64 / 41.0 - 1.0)
            .collect(),
    )
    .unwrap()
}

fn session_count_for(
    operation: impl FnOnce(&EagerTensor, &EagerTensor) -> crate::Result<EagerTensor>,
) -> usize {
    let sessions = Arc::new(AtomicUsize::new(0));
    let backend = EagerBackend::recording_cpu_with_sessions(
        Arc::new(AtomicUsize::new(0)),
        Arc::clone(&sessions),
    );
    let runtime = Arc::new(EagerRuntime::from_backend(backend));
    let lhs = EagerTensor::from_tensor_in(tensor(vec![4, 4], 1), Arc::clone(&runtime)).unwrap();
    let rhs = EagerTensor::from_tensor_in(tensor(vec![4, 4], 2), runtime).unwrap();
    let output = operation(&lhs, &rhs).unwrap();
    assert!(output.materialized().is_ok());
    sessions.load(Ordering::Relaxed)
}

fn prove_session_entries() -> Vec<usize> {
    vec![
        session_count_for(|lhs, _| lhs.neg()),
        session_count_for(|lhs, rhs| lhs.add(rhs)),
        session_count_for(|lhs, _| lhs.reduce_sum(Some(&[0]))),
        session_count_for(|lhs, _| {
            lhs.slice(SliceConfig {
                starts: vec![0, 0],
                limits: vec![4, 4],
                strides: vec![1, 1],
            })
        }),
        session_count_for(|lhs, rhs| {
            lhs.dot_general(
                rhs,
                DotGeneralConfig {
                    lhs_contracting_dims: vec![1],
                    rhs_contracting_dims: vec![0],
                    lhs_batch_dims: vec![],
                    rhs_batch_dims: vec![],
                },
            )
        }),
    ]
}

fn reset_counters(executor: Option<&Arc<RecordingExecutor>>, provider: &RecordingGemm) {
    provider.calls.store(0, Ordering::SeqCst);
    provider.observed_cpus.lock().unwrap().clear();
    if let Some(executor) = executor {
        executor.reset();
    }
}

fn eager_surface(
    placed: &mut super::super::CpuPlacementBoundEager,
    surface: &str,
    native_lhs: &Tensor,
    native_rhs: &Tensor,
    dot_lhs: &Tensor,
    dot_rhs: &Tensor,
    dot_config: &DotGeneralConfig,
) -> crate::Result<Tensor> {
    placed.with_eager_session(|session| match surface {
        "E-N" => session
            .add(native_lhs, native_rhs)
            .map_err(crate::Error::from),
        "E-D" => session
            .dot_general(dot_lhs, dot_rhs, dot_config)
            .map_err(crate::Error::from),
        _ => unreachable!(),
    })
}

#[allow(clippy::too_many_arguments)]
fn recovery_proof(
    placed: &mut super::super::CpuPlacementBoundEager,
    surface: &str,
    native_lhs: &Tensor,
    native_rhs: &Tensor,
    dot_lhs: &Tensor,
    dot_rhs: &Tensor,
    dot_config: &DotGeneralConfig,
    executor: Option<&Arc<RecordingExecutor>>,
    provider: &RecordingGemm,
    key: &str,
    declared_cpus: &[usize],
) -> (bool, bool, RecoveryEvidence) {
    reset_counters(executor, provider);
    let typed_error = if surface == "E-N" {
        let wrong = tensor(vec![native_lhs.shape()[0] - 1], 21);
        matches!(
            placed
                .with_eager_session(|session| {
                    session.add(native_lhs, &wrong).map_err(crate::Error::from)
                })
                .unwrap_err(),
            crate::Error::TensorRuntime(tenferro_tensor::Error::Validation {
                op: "add",
                source: tenferro_tensor::ValidationError::ShapeMismatch(_),
            })
        )
    } else {
        let invalid = DotGeneralConfig {
            lhs_contracting_dims: vec![2],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
        };
        matches!(
            placed
                .with_eager_session(|session| {
                    session
                        .dot_general(dot_lhs, dot_rhs, &invalid)
                        .map_err(crate::Error::from)
                })
                .unwrap_err(),
            crate::Error::TensorRuntime(tenferro_tensor::Error::Validation {
                op: "dot_general",
                source: tenferro_tensor::ValidationError::AxisOutOfBounds { .. },
            })
        )
    };
    reset_counters(executor, provider);
    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _: crate::Result<()> = placed.with_eager_session(|session| {
            if surface == "E-N" {
                let _ = session.add(native_lhs, native_rhs).unwrap();
            } else {
                let _ = session.dot_general(dot_lhs, dot_rhs, dot_config).unwrap();
            }
            panic!("phase2e eager same-workload unwind")
        });
    }))
    .is_err();
    reset_counters(executor, provider);
    let recorder = Arc::new(super::super::phase2e_eager_events::SessionRecorder::default());
    let recovery_key = format!("{key}__recovery");
    let (post_recovery, operation_workers) = observe_actual_operation(&recovery_key, || {
        super::super::phase2e_eager_events::with_recorder(&recorder, || {
            eager_surface(
                placed, surface, native_lhs, native_rhs, dot_lhs, dot_rhs, dot_config,
            )
        })
    });
    let numerical_passed = post_recovery
        .map(|output| {
            if surface == "E-N" {
                output.shape() == native_lhs.shape()
            } else {
                relative_dot_error(dot_lhs, dot_rhs, &output, 128) <= 1.0e-12
            }
        })
        .unwrap_or(false);
    let (session_entry, _) = recorder.snapshot();
    let actual_provider = provider.calls.load(Ordering::SeqCst);
    let mut observed_cpus = provider.observed_cpus.lock().unwrap().clone();
    let (actual_install, actual_submit) = executor
        .map(|executor| {
            let install = executor.installs.load(Ordering::SeqCst);
            let submit = executor.submits.load(Ordering::SeqCst);
            let cpus = executor.observed_cpus.lock().unwrap().clone();
            for cpu in cpus {
                if !observed_cpus.contains(&cpu) {
                    observed_cpus.push(cpu);
                }
            }
            (Some(install), Some(submit))
        })
        .unwrap_or((None, None));
    for [_, cpu] in &operation_workers {
        if !observed_cpus.contains(cpu) {
            observed_cpus.push(*cpu);
        }
    }
    let subset_passed =
        declared_cpus.is_empty() || observed_cpus.iter().all(|cpu| declared_cpus.contains(cpu));
    (
        typed_error,
        unwind,
        RecoveryEvidence {
            fresh_reset: true,
            session_entry,
            actual_install,
            actual_submit,
            actual_provider,
            operation_workers,
            observed_cpus,
            numerical_passed,
            subset_passed,
        },
    )
}

fn relative_dot_error(lhs: &Tensor, rhs: &Tensor, actual: &Tensor, size: usize) -> f64 {
    let lhs = lhs.as_slice::<f64>().unwrap();
    let rhs = rhs.as_slice::<f64>().unwrap();
    let actual = actual.as_slice::<f64>().unwrap();
    let mut error_squared = 0.0;
    let mut reference_squared = 0.0;
    for column in 0..size {
        for row in 0..size {
            let mut expected = 0.0;
            for contracted in 0..size {
                expected += lhs[row + contracted * size] * rhs[contracted + column * size];
            }
            let delta = actual[row + column * size] - expected;
            error_squared += delta * delta;
            reference_squared += expected * expected;
        }
    }
    error_squared.sqrt() / reference_squared.sqrt().max(f64::MIN_POSITIVE)
}

fn run_ad_owned_rows() -> Evidence {
    let native_lhs = tensor(vec![65_536], 3);
    let native_rhs = tensor(vec![65_536], 4);
    let dot_lhs = tensor(vec![128, 128], 5);
    let dot_rhs = tensor(vec![128, 128], 6);
    let dot_config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let mut rows = Vec::with_capacity(18);
    for ownership in ["managed-exact", "external-exact", "external-advisory"] {
        for budget in [1, 2, 4] {
            let fixture = cpu_fixture(ownership, budget);
            let executor = fixture.executor.as_ref().map(Arc::clone);
            let provider = Arc::clone(&fixture.provider);
            let declared_cpus = fixture.declared_cpus.clone();
            let placement_audit = fixture.placement_audit.clone();
            assert_eq!(placement_audit.len(), budget);
            assert_eq!(
                placement_audit
                    .iter()
                    .map(|item| item[0])
                    .collect::<std::collections::BTreeSet<_>>(),
                (0..budget).collect()
            );
            if ownership != "external-advisory" {
                assert!(placement_audit
                    .iter()
                    .all(|item| declared_cpus.contains(&item[1])));
            }
            let placement = fixture.backend.placement();
            let runtime = EagerRuntime::with_cpu_backend(fixture.backend);
            let mut placed = runtime.on_cpu(placement).unwrap();
            for surface in ["E-N", "E-D"] {
                let key = format!("{ownership}/budget-{budget}/{surface}");
                reset_counters(executor.as_ref(), &provider);
                let session_recorder =
                    Arc::new(super::super::phase2e_eager_events::SessionRecorder::default());
                let (output, operation_workers) = observe_actual_operation(&key, || {
                    super::super::phase2e_eager_events::with_recorder(&session_recorder, || {
                        eager_surface(
                            &mut placed,
                            surface,
                            &native_lhs,
                            &native_rhs,
                            &dot_lhs,
                            &dot_rhs,
                            &dot_config,
                        )
                    })
                    .unwrap()
                });
                let (session_entry, session_entry_cpus) = session_recorder.snapshot();
                if ownership != "external-advisory" {
                    assert!(operation_workers
                        .iter()
                        .all(|item| declared_cpus.contains(&item[1])));
                }
                let numerical_passed = if surface == "E-N" {
                    let expected: Vec<_> = native_lhs
                        .as_slice::<f64>()
                        .unwrap()
                        .iter()
                        .zip(native_rhs.as_slice::<f64>().unwrap())
                        .map(|(lhs, rhs)| lhs + rhs)
                        .collect();
                    output.as_slice::<f64>().unwrap() == expected
                } else {
                    relative_dot_error(&dot_lhs, &dot_rhs, &output, 128) <= 1.0e-12
                };
                let actual_provider = provider.calls.load(Ordering::SeqCst);
                let provider_cpus = provider.observed_cpus.lock().unwrap().clone();
                let (actual_install, actual_submit, observed_cpus) = executor
                    .as_ref()
                    .map(|executor| {
                        (
                            Some(executor.installs.load(Ordering::SeqCst)),
                            Some(executor.submits.load(Ordering::SeqCst)),
                            executor.observed_cpus.lock().unwrap().clone(),
                        )
                    })
                    .unwrap_or((None, None, Vec::new()));
                if ownership == "external-exact" {
                    assert!(observed_cpus.iter().all(|cpu| declared_cpus.contains(cpu)));
                }
                let (typed_error_recovered, unwind_recovered, recovery) = recovery_proof(
                    &mut placed,
                    surface,
                    &native_lhs,
                    &native_rhs,
                    &dot_lhs,
                    &dot_rhs,
                    &dot_config,
                    executor.as_ref(),
                    &provider,
                    &key,
                    &declared_cpus,
                );
                let expected_provider = usize::from(surface == "E-D");
                let operation_evidence_passed = !cfg!(tenferro_phase2e_operation_observe)
                    || (!recovery.observed_cpus.is_empty()
                        && (surface != "E-N" || !recovery.operation_workers.is_empty()));
                let post_recovery_passed = recovery.session_entry == 1
                    && recovery.actual_provider == expected_provider
                    && recovery.actual_install.is_none_or(|value| value == 1)
                    && recovery.actual_submit.is_none_or(|value| value == 0)
                    && recovery.numerical_passed
                    && recovery.subset_passed
                    && operation_evidence_passed;
                rows.push(Row {
                    key,
                    surface,
                    budget,
                    session_entry,
                    session_entry_cpus,
                    placement_audit: placement_audit.clone(),
                    declared_cpus: declared_cpus.clone(),
                    downstream_vector: if surface == "E-N" {
                        "borrowed-add"
                    } else {
                        "borrowed-dot"
                    },
                    actual_install,
                    actual_submit,
                    actual_provider,
                    operation_workers: operation_workers.clone(),
                    observed_cpus: {
                        let mut cpus = observed_cpus;
                        for cpu in provider_cpus {
                            if !cpus.contains(&cpu) {
                                cpus.push(cpu);
                            }
                        }
                        for [_, cpu] in operation_workers {
                            if !cpus.contains(&cpu) {
                                cpus.push(cpu);
                            }
                        }
                        cpus
                    },
                    numerical_passed,
                    typed_error_recovered,
                    unwind_recovered,
                    post_recovery_passed,
                    recovery,
                });
            }
        }
    }
    Evidence {
        session_entries: prove_session_entries(),
        characterization: rows,
    }
}

fn quoted(value: &str) -> String {
    format!("{value:?}")
}

fn write_evidence(evidence: &Evidence) -> std::io::Result<()> {
    let Some(root) = std::env::var_os("TENFERRO_PHASE2E_EVIDENCE_DIR") else {
        return Ok(());
    };
    let directory = PathBuf::from(root).join("dispatch-gates");
    std::fs::create_dir_all(&directory)?;
    let mut output = format!(
        "{{\"owner\":\"ad\",\"session_entries\":{:?},\"characterization\":[",
        evidence.session_entries
    );
    for (index, row) in evidence.characterization.iter().enumerate() {
        if index != 0 {
            output.push(',');
        }
        write!(
            output,
            "{{\"key\":{},\"owner\":\"ad\",\"surface\":{},\"budget\":{},\"session_entry\":{},\"session_entry_cpus\":{:?},\"placement_audit\":{:?},\"declared_cpus\":{:?},\"downstream_vector\":{},\"actual_install\":{},\"actual_submit\":{},\"actual_provider\":{},\"operation_workers\":{:?},\"numerical_passed\":{},\"typed_error_recovered\":{},\"unwind_recovered\":{},\"post_recovery_passed\":{},\"observed_cpus\":{:?},\"recovery\":{{\"fresh_reset\":{},\"session_entry\":{},\"actual_install\":{},\"actual_submit\":{},\"actual_provider\":{},\"operation_workers\":{:?},\"observed_cpus\":{:?},\"numerical_passed\":{},\"subset_passed\":{}}},\"hardware_skip\":null}}",
            quoted(&row.key), quoted(row.surface), row.budget, row.session_entry,
            row.session_entry_cpus, row.placement_audit, row.declared_cpus,
            quoted(row.downstream_vector),
            row.actual_install.map_or_else(|| "null".into(), |value| value.to_string()),
            row.actual_submit.map_or_else(|| "null".into(), |value| value.to_string()),
            row.actual_provider,
            row.operation_workers,
            row.numerical_passed, row.typed_error_recovered, row.unwind_recovered,
            row.post_recovery_passed, row.observed_cpus,
            row.recovery.fresh_reset, row.recovery.session_entry,
            row.recovery.actual_install.map_or_else(|| "null".into(), |value| value.to_string()),
            row.recovery.actual_submit.map_or_else(|| "null".into(), |value| value.to_string()),
            row.recovery.actual_provider, row.recovery.operation_workers,
            row.recovery.observed_cpus, row.recovery.numerical_passed, row.recovery.subset_passed,
        ).unwrap();
    }
    output.push_str("]}\n");
    let temporary = directory.join("ad-evidence.json.partial");
    std::fs::write(&temporary, output)?;
    std::fs::rename(temporary, directory.join("ad-evidence.json"))
}

#[test]
fn phase2e_eager_characterization_evidence() {
    let evidence = run_ad_owned_rows();
    assert_eq!(evidence.session_entries.len(), 5);
    assert!(evidence.session_entries.iter().all(|count| *count == 1));
    assert_eq!(evidence.characterization.len(), 18);
    assert!(evidence.characterization.iter().all(Row::gating_passed));
    write_evidence(&evidence).unwrap();
}
