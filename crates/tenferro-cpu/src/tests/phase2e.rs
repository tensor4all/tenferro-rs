use std::fmt::Write as _;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex};

use tenferro_tensor::{
    BackendSessionHost, DotGeneralConfig, SliceConfig, TensorDot, TensorElementwise,
    TensorReduction,
};

use super::*;
use crate::backend::phase2e_test_events::{with_recorder, EventCounts, EventRecorder};
use crate::provider::{
    CpuExecutionContext, CpuGemmProvider, CpuGemmRequest, CpuGroupedGemmRequest,
    CpuProviderOutcome, FaerGemmProvider,
};
use crate::{
    process_cpu_affinity, CpuBackendKind, CpuDomainExecutor, CpuDomainExecutorCapabilities,
    CpuDomainExecutorError, CpuDomainId, CpuExecutorAffinity, CpuExecutorReentrancy,
    CpuExecutorShutdown, CpuInnerParallelism, CpuPlacementGuarantee, CpuProviderBundle,
    ExternalCpuDomain, NumaNodeId, ResolvedCpuPlacement, ScopedCpuJob, ScopedCpuJobs,
};

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

fn audit_pool_workers(
    budget: usize,
    run: impl FnOnce(Box<dyn FnOnce() -> Vec<(usize, usize)> + Send>) -> Vec<(usize, usize)>,
) -> Vec<(usize, usize)> {
    let barrier = Arc::new(Barrier::new(budget));
    run(Box::new(move || {
        rayon::broadcast(|worker| {
            barrier.wait();
            (
                worker.index(),
                current_cpu().expect("Phase 2E requires current-CPU observation"),
            )
        })
    }))
}

#[derive(Debug)]
struct RecordingExecutor {
    inner: CpuContext,
    capabilities: CpuDomainExecutorCapabilities,
    installs: AtomicUsize,
    submits: AtomicUsize,
    observed_cpus: Mutex<Vec<usize>>,
}

impl RecordingExecutor {
    fn new(budget: usize, exact: bool) -> Arc<Self> {
        Self::new_with_parallelism(
            budget,
            exact,
            budget > 1,
            if budget > 1 {
                CpuInnerParallelism::Rayon
            } else {
                CpuInnerParallelism::None
            },
        )
    }

    fn new_with_parallelism(
        budget: usize,
        exact: bool,
        outer_parallelism: bool,
        inner_parallelism: CpuInnerParallelism,
    ) -> Arc<Self> {
        let allowed = process_cpu_affinity().expect("Phase 2E needs a process CPU set");
        let selected = crate::CpuSet::new(allowed.as_slice().iter().copied().take(budget)).unwrap();
        let declared = selected.as_usize_vec();
        let inner = CpuContext::with_pinned_cpus(selected, budget).unwrap();
        let executor = Arc::new(Self {
            inner,
            capabilities: CpuDomainExecutorCapabilities {
                worker_count: NonZeroUsize::new(budget).unwrap(),
                outer_parallelism,
                inner_parallelism,
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
            let audit = executor.affinity_audit();
            assert_eq!(audit.len(), budget);
            assert_eq!(
                audit
                    .iter()
                    .map(|(worker, _)| *worker)
                    .collect::<std::collections::BTreeSet<_>>(),
                (0..budget).collect()
            );
            assert!(audit.iter().all(|(_, cpu)| declared.contains(cpu)));
        }
        executor
    }

    fn observe_cpu(&self) {
        if let Some(cpu) = current_cpu() {
            let mut observed = self.observed_cpus.lock().unwrap();
            if !observed.contains(&cpu) {
                observed.push(cpu);
            }
        }
    }

    fn affinity_audit(&self) -> Vec<(usize, usize)> {
        audit_pool_workers(self.capabilities.worker_count.get(), |operation| {
            self.inner.install(operation)
        })
    }

    fn snapshot(&self) -> (usize, usize, Vec<usize>) {
        (
            self.installs.load(Ordering::SeqCst),
            self.submits.load(Ordering::SeqCst),
            self.observed_cpus.lock().unwrap().clone(),
        )
    }

    fn reset(&self) {
        self.installs.store(0, Ordering::SeqCst);
        self.submits.store(0, Ordering::SeqCst);
        self.observed_cpus.lock().unwrap().clear();
    }
}

struct ObservedJob<'a> {
    owner: &'a RecordingExecutor,
    inner: &'a mut dyn ScopedCpuJob,
}

impl ScopedCpuJob for ObservedJob<'_> {
    fn run(&mut self) -> Result<(), CpuDomainExecutorError> {
        self.owner.observe_cpu();
        self.inner.run()
    }
}

struct ObservedJobs<'a> {
    owner: &'a RecordingExecutor,
    inner: &'a dyn ScopedCpuJobs,
}

impl ScopedCpuJobs for ObservedJobs<'_> {
    fn len(&self) -> usize {
        self.inner.len()
    }

    fn run(&self, index: usize) -> Result<(), CpuDomainExecutorError> {
        self.owner.observe_cpu();
        self.inner.run(index)
    }
}

impl CpuDomainExecutor for RecordingExecutor {
    fn capabilities(&self) -> CpuDomainExecutorCapabilities {
        self.capabilities
    }

    fn submit(&self, jobs: &dyn ScopedCpuJobs) -> Result<(), CpuDomainExecutorError> {
        self.submits.fetch_add(1, Ordering::SeqCst);
        CpuDomainExecutor::submit(
            &self.inner,
            &ObservedJobs {
                owner: self,
                inner: jobs,
            },
        )
    }

    fn install(&self, job: &mut dyn ScopedCpuJob) -> Result<(), CpuDomainExecutorError> {
        self.installs.fetch_add(1, Ordering::SeqCst);
        CpuDomainExecutor::install(
            &self.inner,
            &mut ObservedJob {
                owner: self,
                inner: job,
            },
        )
    }
}

#[derive(Debug)]
struct RecordingGemm {
    calls: AtomicUsize,
}

impl CpuGemmProvider for RecordingGemm {
    fn execution_capabilities(&self) -> crate::CpuProviderExecutionCapabilities {
        FaerGemmProvider.execution_capabilities()
    }

    fn gemm(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuGemmRequest<'_, '_, '_>,
    ) -> crate::Result<CpuProviderOutcome> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        crate::backend::phase2e_test_events::record(
            crate::backend::phase2e_test_events::Event::Provider,
        );
        FaerGemmProvider.gemm(context, request)
    }

    fn strided_batched_gemm(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuGemmRequest<'_, '_, '_>,
    ) -> crate::Result<CpuProviderOutcome> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        FaerGemmProvider.strided_batched_gemm(context, request)
    }

    fn grouped_gemm(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuGroupedGemmRequest<'_, '_, '_>,
    ) -> crate::Result<CpuProviderOutcome> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        crate::backend::phase2e_test_events::record(
            crate::backend::phase2e_test_events::Event::Provider,
        );
        FaerGemmProvider.grouped_gemm(context, request)
    }
}

struct Fixture {
    backend: CpuBackend,
    executor: Option<Arc<RecordingExecutor>>,
    provider: Arc<RecordingGemm>,
    declared_cpus: Vec<usize>,
}

fn fixture(ownership: &str, budget: usize) -> Fixture {
    let provider = Arc::new(RecordingGemm {
        calls: AtomicUsize::new(0),
    });
    let bundle = CpuProviderBundle::builder(CpuBackendKind::Faer)
        .gemm_provider(Arc::clone(&provider) as Arc<dyn CpuGemmProvider>)
        .engine_outer_grouped_gemm()
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
                    .for_placement(crate::CpuPlacement::NumaNode(node.id()))
                    .unwrap()
            })
            .unwrap_or(coordinator);
        let declared_cpus = backend.resolved_placement().unwrap().cpus().as_usize_vec();
        let audit = audit_pool_workers(budget, |operation| backend.install(operation));
        assert_eq!(audit.len(), budget);
        assert_eq!(
            audit
                .iter()
                .map(|(worker, _)| *worker)
                .collect::<std::collections::BTreeSet<_>>(),
            (0..budget).collect()
        );
        assert!(audit.iter().all(|(_, cpu)| declared_cpus.contains(cpu)));
        return Fixture {
            declared_cpus,
            backend,
            executor: None,
            provider,
        };
    }
    let exact = ownership == "external-exact";
    let executor = RecordingExecutor::new(budget, exact);
    external_fixture(executor, provider, bundle, exact)
}

fn external_fixture(
    executor: Arc<RecordingExecutor>,
    provider: Arc<RecordingGemm>,
    bundle: CpuProviderBundle,
    exact: bool,
) -> Fixture {
    let budget = executor.capabilities.worker_count.get();
    let allowed = process_cpu_affinity().expect("Phase 2E needs process affinity");
    let selected = crate::CpuSet::new(allowed.as_slice().iter().copied().take(budget)).unwrap();
    let declared_cpus = if exact {
        selected.as_usize_vec()
    } else {
        Vec::new()
    };
    let domain_id = CpuDomainId::new(0x2e00 + budget as u64 + u64::from(u8::from(exact)));
    let domain = ExternalCpuDomain::new(
        domain_id,
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
        CpuBackend::from_external_managed_domains_with_provider_bundle(domain_id, [domain], bundle)
            .unwrap();
    Fixture {
        backend,
        executor: Some(executor),
        provider,
        declared_cpus,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Counts([usize; 6]);

impl Counts {
    fn measured(snapshot: &crate::backend::phase2e_test_events::EventSnapshot) -> Self {
        let counts = snapshot.counts;
        Self([
            counts.session,
            counts.scope,
            counts.permit,
            counts.install,
            counts.submit,
            counts.provider,
        ])
    }
}

fn measured_mode(modes: &[crate::ParallelMode]) -> &'static str {
    if modes.contains(&crate::ParallelMode::Outer) {
        "Outer"
    } else if modes.contains(&crate::ParallelMode::Inner) {
        "Inner"
    } else if modes.contains(&crate::ParallelMode::Sequential) {
        "Sequential"
    } else {
        "UnsupportedOuter"
    }
}

#[derive(Debug)]
struct Row {
    key: String,
    owner: &'static str,
    surface: &'static str,
    budget: usize,
    mode: &'static str,
    counts: Counts,
    observed_cpus: Vec<usize>,
    numerical_passed: bool,
    typed_error_recovered: bool,
    typed_error_kind: &'static str,
    typed_error_source: String,
    unwind_recovered: bool,
    post_recovery_passed: bool,
    recovery: RecoveryEvidence,
    hardware_skip: Option<&'static str>,
}

#[derive(Debug)]
struct RecoveryEvidence {
    fresh_reset: bool,
    counts: Counts,
    mode: &'static str,
    observed_cpus: Vec<usize>,
    numerical_passed: bool,
    subset_passed: bool,
}

impl Row {
    fn gating_passed(&self) -> bool {
        self.numerical_passed
            && self.typed_error_recovered
            && self.unwind_recovered
            && self.post_recovery_passed
    }
}

struct Evidence {
    canonical_vectors: Vec<Counts>,
    characterization: Vec<Row>,
    cross_socket_locality: CrossSocketEvidence,
}

#[derive(Debug)]
struct CrossSocketProbe {
    node: usize,
    declared_cpus: Vec<usize>,
    observed_cpus: Vec<usize>,
    numerical_passed: bool,
    subset_passed: bool,
}

#[derive(Debug)]
struct CrossSocketEvidence {
    usable_numa_nodes: usize,
    probes: Vec<CrossSocketProbe>,
}

fn cross_socket_node_probe(
    mut backend: CpuBackend,
    node: usize,
    declared_cpus: Vec<usize>,
    barrier: Arc<Barrier>,
    salt: usize,
) -> CrossSocketProbe {
    // First-touch both inputs on the node-owned executor before the concurrent
    // operation. This makes the locality claim about executed work, not only
    // about a topology declaration.
    let lhs = backend.install(move || deterministic_vector(65_536, salt));
    let rhs = backend.install(move || deterministic_vector(65_536, salt + 1));
    barrier.wait();
    let recorder = Arc::new(EventRecorder::default());
    let output = with_recorder(&recorder, || backend.add(&lhs, &rhs)).unwrap();
    let numerical_passed = output
        .as_slice::<f64>()
        .unwrap()
        .iter()
        .zip(lhs.as_slice::<f64>().unwrap())
        .zip(rhs.as_slice::<f64>().unwrap())
        .all(|((actual, left), right)| actual == &(left + right));
    let observed_cpus = recorder.snapshot().observed_cpus;
    let subset_passed =
        !observed_cpus.is_empty() && observed_cpus.iter().all(|cpu| declared_cpus.contains(cpu));
    CrossSocketProbe {
        node,
        declared_cpus,
        observed_cpus,
        numerical_passed,
        subset_passed,
    }
}

fn cross_socket_locality_evidence() -> CrossSocketEvidence {
    let coordinator = CpuBackend::with_threads_and_kind(1, CpuBackendKind::Faer).unwrap();
    let nodes: Vec<_> = coordinator
        .topology()
        .nodes()
        .iter()
        .map(|node| (node.id(), node.cpus().as_usize_vec()))
        .collect();
    if nodes.len() < 2 {
        return CrossSocketEvidence {
            usable_numa_nodes: nodes.len(),
            probes: Vec::new(),
        };
    }
    let first = coordinator
        .for_placement(crate::CpuPlacement::NumaNode(nodes[0].0))
        .unwrap();
    let second = coordinator
        .for_placement(crate::CpuPlacement::NumaNode(nodes[1].0))
        .unwrap();
    let barrier = Arc::new(Barrier::new(2));
    let probes = std::thread::scope(|scope| {
        let first_barrier = Arc::clone(&barrier);
        let first_cpus = nodes[0].1.clone();
        let first_node = nodes[0].0.as_usize();
        let first = scope.spawn(move || {
            cross_socket_node_probe(first, first_node, first_cpus, first_barrier, 71)
        });
        let second_cpus = nodes[1].1.clone();
        let second_node = nodes[1].0.as_usize();
        let second = scope
            .spawn(move || cross_socket_node_probe(second, second_node, second_cpus, barrier, 73));
        vec![first.join().unwrap(), second.join().unwrap()]
    });
    assert_ne!(probes[0].node, probes[1].node);
    assert!(probes
        .iter()
        .all(|probe| probe.numerical_passed && probe.subset_passed));
    CrossSocketEvidence {
        usable_numa_nodes: nodes.len(),
        probes,
    }
}

fn deterministic_vector(len: usize, offset: usize) -> Tensor {
    Tensor::from_vec_col_major(
        vec![len],
        (0..len)
            .map(|index| ((index * 17 + offset * 13) % 101) as f64 / 37.0 - 1.0)
            .collect(),
    )
    .unwrap()
}

#[test]
fn phase2e_recorder_measures_one_direct_native_operation() {
    let recorder = Arc::new(EventRecorder::default());
    let lhs = deterministic_vector(64, 41);
    let rhs = deterministic_vector(64, 42);
    let mut backend = CpuBackend::with_threads(2).unwrap();
    let output = with_recorder(&recorder, || backend.add(&lhs, &rhs)).unwrap();
    assert_eq!(output.shape(), [64]);
    assert_eq!(
        recorder.snapshot().counts,
        EventCounts {
            session: 0,
            scope: 1,
            permit: 1,
            install: 1,
            submit: 0,
            provider: 0,
        }
    );
}

fn prove_direct_and_borrowed_vectors() -> Vec<Counts> {
    let input = deterministic_vector(64, 1);
    let rhs = deterministic_vector(64, 2);
    let matrix_lhs = Tensor::from_vec_col_major(
        vec![8, 8],
        (0..64).map(|i| (i * 7 % 29) as f64 / 11.0).collect(),
    )
    .unwrap();
    let matrix_rhs = Tensor::from_vec_col_major(
        vec![8, 8],
        (0..64).map(|i| (i * 5 % 31) as f64 / 13.0).collect(),
    )
    .unwrap();
    let dot = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let slice = SliceConfig {
        starts: vec![1],
        limits: vec![63],
        strides: vec![2],
    };
    let mut fixture = fixture("managed-exact", 2);
    let mut vectors = Vec::with_capacity(10);
    macro_rules! direct {
        ($operation:expr, $shape:expr) => {{
            let recorder = Arc::new(EventRecorder::default());
            let output = with_recorder(&recorder, || $operation).unwrap();
            assert_eq!(output.shape(), $shape);
            vectors.push(Counts::measured(&recorder.snapshot()));
        }};
    }
    direct!(fixture.backend.neg(&input), &[64]);
    direct!(fixture.backend.add(&input, &rhs), &[64]);
    direct!(fixture.backend.reduce_sum(&input, &[0]), &[] as &[usize]);
    direct!(fixture.backend.slice(&input, &slice), &[31]);
    direct!(
        fixture.backend.dot_general(&matrix_lhs, &matrix_rhs, &dot),
        &[8, 8]
    );

    for operation in 0..5 {
        let recorder = Arc::new(EventRecorder::default());
        with_recorder(&recorder, || {
            fixture
                .backend
                .with_backend_session(|session| match operation {
                    0 => session
                        .neg(&input)
                        .map(|tensor| assert_eq!(tensor.shape(), &[64])),
                    1 => session
                        .add(&input, &rhs)
                        .map(|tensor| assert_eq!(tensor.shape(), &[64])),
                    2 => session
                        .reduce_sum(&input, &[0])
                        .map(|tensor| assert_eq!(tensor.shape(), &[] as &[usize])),
                    3 => session
                        .slice(&input, &slice)
                        .map(|tensor| assert_eq!(tensor.shape(), &[31])),
                    4 => session
                        .dot_general(&matrix_lhs, &matrix_rhs, &dot)
                        .map(|tensor| assert_eq!(tensor.shape(), &[8, 8])),
                    _ => unreachable!(),
                })
        })
        .unwrap();
        vectors.push(Counts::measured(&recorder.snapshot()));
    }
    vectors
}

fn recovery_proof() {
    let mut backend = CpuBackend::new();
    let lhs = deterministic_vector(8, 3);
    let wrong = deterministic_vector(7, 4);
    assert!(backend.add(&lhs, &wrong).is_err());
    assert!(
        std::panic::catch_unwind(AssertUnwindSafe(|| backend.install(|| panic!("phase2e"))))
            .is_err()
    );
    let output = backend.add(&lhs, &lhs).unwrap();
    assert_eq!(output.shape(), &[8]);
}

fn reset_fixture(fixture: &Fixture) {
    fixture.provider.calls.store(0, Ordering::SeqCst);
    if let Some(executor) = &fixture.executor {
        executor.reset();
    }
}

fn run_native_row(fixture: &mut Fixture) -> bool {
    let lhs = deterministic_vector(65_536, 9);
    let rhs = deterministic_vector(65_536, 10);
    let expected: Vec<_> = lhs
        .as_slice::<f64>()
        .unwrap()
        .iter()
        .zip(rhs.as_slice::<f64>().unwrap())
        .map(|(lhs, rhs)| lhs + rhs)
        .collect();
    fixture
        .backend
        .add(&lhs, &rhs)
        .map(|actual| actual.as_slice::<f64>().unwrap() == expected)
        .unwrap_or(false)
}

fn run_surface(fixture: &mut Fixture, surface: &str, budget: usize) -> bool {
    match surface {
        "D-N" | "E-N" => run_native_row(fixture),
        "D-D" | "E-D" | "U-I" => run_dot_row(fixture),
        "G-O" => run_grouped_row(fixture, budget),
        _ => unreachable!("unknown Phase 2E surface {surface}"),
    }
}

fn typed_surface_error(fixture: &mut Fixture, surface: &str, budget: usize) -> bool {
    match surface {
        "D-N" | "E-N" => {
            let lhs = deterministic_vector(8, 51);
            let rhs = deterministic_vector(7, 52);
            matches!(
                fixture.backend.add(&lhs, &rhs).unwrap_err(),
                crate::Error::Validation {
                    op: "add",
                    source: tenferro_tensor::ValidationError::ShapeMismatch(_),
                }
            )
        }
        "D-D" | "E-D" | "U-I" => {
            let lhs = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4]).unwrap();
            let rhs = lhs.clone();
            let invalid = DotGeneralConfig {
                lhs_contracting_dims: vec![2],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: vec![],
                rhs_batch_dims: vec![],
            };
            matches!(
                fixture
                    .backend
                    .dot_general(&lhs, &rhs, &invalid)
                    .unwrap_err(),
                crate::Error::Validation {
                    op: "dot_general",
                    source: tenferro_tensor::ValidationError::AxisOutOfBounds { .. },
                }
            )
        }
        "G-O" => {
            let size = 2;
            let jobs_len = 2 * budget + 1;
            let lhs = deterministic_vector(jobs_len * size * size, 53);
            let rhs = deterministic_vector(jobs_len * size * size, 54);
            let mut output = Tensor::from_vec_col_major(
                vec![jobs_len * size * size],
                vec![0.0_f64; jobs_len * size * size],
            )
            .unwrap();
            let jobs = [
                GroupedGemmJob::new(0, 0, 0, size, size, size),
                GroupedGemmJob::new(0, size * size, size * size, size, size, size),
            ];
            let config = GroupedGemmConfig::new(
                &jobs,
                DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
            );
            let mut cache = <CpuBackend as BackendRuntimeCache>::RuntimeCache::default();
            matches!(
                BackendCachedDot::grouped_gemm_cached(
                    &mut fixture.backend,
                    &mut cache,
                    None,
                    TensorRead::from_tensor(&lhs),
                    TensorRead::from_tensor(&rhs),
                    &config,
                    TensorWrite::from_tensor(&mut output),
                )
                .unwrap_err(),
                crate::Error::Validation {
                    op: "grouped_gemm",
                    source: tenferro_tensor::ValidationError::InvalidArgument { .. },
                }
            )
        }
        _ => unreachable!("unknown Phase 2E surface {surface}"),
    }
}

fn row_recovery_proof(
    fixture: &mut Fixture,
    surface: &str,
    budget: usize,
    primary: Counts,
) -> (bool, bool, RecoveryEvidence) {
    reset_fixture(fixture);
    let typed_error = typed_surface_error(fixture, surface, budget);
    reset_fixture(fixture);

    let panic_recorder = Arc::new(EventRecorder::default());
    panic_recorder.panic_on_next_worker();
    let unwind = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let _ = with_recorder(&panic_recorder, || run_surface(fixture, surface, budget));
    }))
    .is_err();
    reset_fixture(fixture);

    let recovery_recorder = Arc::new(EventRecorder::default());
    let numerical = with_recorder(&recovery_recorder, || run_surface(fixture, surface, budget));
    let recovery = recovery_recorder.snapshot();
    let counts_match = Counts::measured(&recovery) == primary;
    let recovery_counts = Counts::measured(&recovery);
    let recovery_mode = measured_mode(&recovery.modes);
    let observed_cpus = recovery.observed_cpus;
    let subset_passed = fixture.declared_cpus.is_empty()
        || observed_cpus
            .iter()
            .all(|cpu| fixture.declared_cpus.contains(cpu));
    let evidence = RecoveryEvidence {
        fresh_reset: true,
        counts: recovery_counts,
        mode: recovery_mode,
        observed_cpus,
        numerical_passed: numerical,
        subset_passed,
    };
    reset_fixture(fixture);
    assert!(counts_match, "recovery rerun count vector changed");
    (typed_error, unwind, evidence)
}

fn run_dot_row(fixture: &mut Fixture) -> bool {
    let size = 128;
    let lhs = Tensor::from_vec_col_major(
        vec![size, size],
        deterministic_vector(size * size, 11)
            .as_slice::<f64>()
            .unwrap()
            .to_vec(),
    )
    .unwrap();
    let rhs = Tensor::from_vec_col_major(
        vec![size, size],
        deterministic_vector(size * size, 12)
            .as_slice::<f64>()
            .unwrap()
            .to_vec(),
    )
    .unwrap();
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let actual = fixture.backend.dot_general(&lhs, &rhs, &config).unwrap();
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
    error_squared.sqrt() / reference_squared.sqrt().max(f64::MIN_POSITIVE) <= 1.0e-12
}

fn run_grouped_row(fixture: &mut Fixture, budget: usize) -> bool {
    let size = 64;
    let jobs_len = 2 * budget + 1;
    let matrix_len = size * size;
    let lhs = deterministic_vector(jobs_len * matrix_len, 13);
    let rhs = deterministic_vector(jobs_len * matrix_len, 14);
    let mut output = Tensor::from_vec_col_major(
        vec![jobs_len * matrix_len],
        vec![0.0_f64; jobs_len * matrix_len],
    )
    .unwrap();
    let jobs: Vec<_> = (0..jobs_len)
        .map(|index| {
            let offset = index * matrix_len;
            GroupedGemmJob::new(offset, offset, offset, size, size, size)
        })
        .collect();
    let config = GroupedGemmConfig::new(
        &jobs,
        DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
    );
    let mut cache = <CpuBackend as BackendRuntimeCache>::RuntimeCache::default();
    BackendCachedDot::grouped_gemm_cached(
        &mut fixture.backend,
        &mut cache,
        None,
        TensorRead::from_tensor(&lhs),
        TensorRead::from_tensor(&rhs),
        &config,
        TensorWrite::from_tensor(&mut output),
    )
    .unwrap();
    let lhs = lhs.as_slice::<f64>().unwrap();
    let rhs = rhs.as_slice::<f64>().unwrap();
    let actual = output.as_slice::<f64>().unwrap();
    let mut error_squared = 0.0;
    let mut reference_squared = 0.0;
    for job in 0..jobs_len {
        let base = job * matrix_len;
        for column in 0..size {
            for row in 0..size {
                let mut expected = 0.0;
                for contracted in 0..size {
                    expected += lhs[base + row + contracted * size]
                        * rhs[base + contracted + column * size];
                }
                let delta = actual[base + row + column * size] - expected;
                error_squared += delta * delta;
                reference_squared += expected * expected;
            }
        }
    }
    error_squared.sqrt() / reference_squared.sqrt().max(f64::MIN_POSITIVE) <= 1.0e-12
}

fn unsupported_outer_row() -> Row {
    let executor =
        RecordingExecutor::new_with_parallelism(2, false, false, CpuInnerParallelism::Rayon);
    let context_fixture = crate::provider::tests::external_execution_context_fixture(
        Arc::clone(&executor) as Arc<dyn CpuDomainExecutor>,
        NonZeroUsize::new(2).unwrap(),
    );
    let provider_calls = AtomicUsize::new(0);
    let recorder = Arc::new(EventRecorder::default());
    let error = with_recorder(&recorder, || {
        context_fixture.entry().submit_outer(5, |_, _| {
            provider_calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })
    })
    .unwrap_err();
    assert!(matches!(
        &error,
        CpuDomainExecutorError::Scheduling { message }
            if message == "CPU domain CpuDomainId(9) does not support Outer mode"
    ));
    assert_eq!(executor.snapshot(), (0, 0, vec![]));
    assert_eq!(provider_calls.load(Ordering::SeqCst), 0);
    let snapshot = recorder.snapshot();
    let unwind_recovered = std::panic::catch_unwind(AssertUnwindSafe(|| {
        context_fixture
            .entry()
            .enter(crate::ParallelMode::Sequential, |_| panic!("u-o unwind"))
            .unwrap()
    }))
    .is_err();
    let recovery_recorder = Arc::new(EventRecorder::default());
    let post_recovery_passed = with_recorder(&recovery_recorder, || {
        context_fixture
            .entry()
            .enter(crate::ParallelMode::Sequential, |_| 7)
            .unwrap()
            == 7
    });
    let recovery_snapshot = recovery_recorder.snapshot();
    Row {
        key: "external-no-outer/budget-2/U-O".into(),
        owner: "cpu",
        surface: "U-O",
        budget: 2,
        mode: "UnsupportedOuter",
        counts: Counts::measured(&snapshot),
        observed_cpus: snapshot.observed_cpus,
        numerical_passed: true,
        typed_error_recovered: true,
        typed_error_kind: "Scheduling",
        typed_error_source: error.to_string(),
        unwind_recovered,
        post_recovery_passed,
        recovery: RecoveryEvidence {
            fresh_reset: true,
            counts: Counts::measured(&recovery_snapshot),
            mode: measured_mode(&recovery_snapshot.modes),
            observed_cpus: recovery_snapshot.observed_cpus,
            numerical_passed: post_recovery_passed,
            subset_passed: true,
        },
        hardware_skip: None,
    }
}

fn unsupported_inner_row() -> Row {
    let executor =
        RecordingExecutor::new_with_parallelism(2, false, true, CpuInnerParallelism::None);
    let provider = Arc::new(RecordingGemm {
        calls: AtomicUsize::new(0),
    });
    let bundle = CpuProviderBundle::builder(CpuBackendKind::Faer)
        .gemm_provider(Arc::clone(&provider) as Arc<dyn CpuGemmProvider>)
        .engine_outer_grouped_gemm()
        .build()
        .unwrap();
    let mut fixture = external_fixture(executor, provider, bundle, false);
    let recorder = Arc::new(EventRecorder::default());
    let numerical_passed = with_recorder(&recorder, || run_dot_row(&mut fixture));
    assert_eq!(fixture.executor.as_ref().unwrap().snapshot().0, 1);
    assert_eq!(fixture.provider.calls.load(Ordering::SeqCst), 1);
    let snapshot = recorder.snapshot();
    let counts = Counts::measured(&snapshot);
    let (typed_error_recovered, unwind_recovered, recovery) =
        row_recovery_proof(&mut fixture, "U-I", 2, counts);
    let post_recovery_passed = recovery.numerical_passed && recovery.subset_passed;
    Row {
        key: "external-no-inner/budget-2/U-I".into(),
        owner: "cpu",
        surface: "U-I",
        budget: 2,
        mode: measured_mode(&snapshot.modes),
        counts,
        observed_cpus: snapshot.observed_cpus,
        numerical_passed,
        typed_error_recovered,
        typed_error_kind: "Validation",
        typed_error_source: "dot_general".into(),
        unwind_recovered,
        post_recovery_passed,
        recovery,
        hardware_skip: None,
    }
}

fn run_cpu_owned_rows() -> Evidence {
    recovery_proof();
    let mut rows = Vec::with_capacity(29);
    for ownership in ["managed-exact", "external-exact", "external-advisory"] {
        for budget in [1, 2, 4] {
            let mut fixture = fixture(ownership, budget);
            for surface in ["D-N", "D-D", "G-O"] {
                reset_fixture(&fixture);
                let recorder = Arc::new(EventRecorder::default());
                let numerical_passed = with_recorder(&recorder, || match surface {
                    "D-N" => run_native_row(&mut fixture),
                    "D-D" => run_dot_row(&mut fixture),
                    "G-O" => run_grouped_row(&mut fixture, budget),
                    _ => unreachable!(),
                });
                let snapshot = recorder.snapshot();
                let counts = Counts::measured(&snapshot);
                assert_eq!(
                    fixture.provider.calls.load(Ordering::SeqCst),
                    counts.0[5],
                    "{ownership}/{budget}/{surface} provider count"
                );
                if let Some(executor) = fixture.executor.as_ref() {
                    let (install, submit, _) = executor.snapshot();
                    assert_eq!((install, submit), (counts.0[3], counts.0[4]));
                }
                let observed_cpus = snapshot.observed_cpus;
                if ownership != "external-advisory" {
                    assert!(observed_cpus
                        .iter()
                        .all(|cpu| fixture.declared_cpus.contains(cpu)));
                }
                let (typed_error_recovered, unwind_recovered, recovery) =
                    row_recovery_proof(&mut fixture, surface, budget, counts);
                let post_recovery_passed = recovery.numerical_passed
                    && recovery.subset_passed
                    && recovery.counts == counts
                    && recovery.mode == measured_mode(&snapshot.modes)
                    && !recovery.observed_cpus.is_empty();
                rows.push(Row {
                    key: format!("{ownership}/budget-{budget}/{surface}"),
                    owner: "cpu",
                    surface,
                    budget,
                    mode: measured_mode(&snapshot.modes),
                    counts,
                    observed_cpus,
                    numerical_passed,
                    typed_error_recovered,
                    typed_error_kind: "Validation",
                    typed_error_source: surface.into(),
                    unwind_recovered,
                    post_recovery_passed,
                    recovery,
                    hardware_skip: None,
                });
            }
        }
    }
    rows.push(unsupported_outer_row());
    rows.push(unsupported_inner_row());
    Evidence {
        canonical_vectors: prove_direct_and_borrowed_vectors(),
        characterization: rows,
        cross_socket_locality: cross_socket_locality_evidence(),
    }
}

fn json_string(value: &str) -> String {
    format!("{value:?}")
}

fn write_evidence(evidence: &Evidence) -> std::io::Result<()> {
    let Some(root) = std::env::var_os("TENFERRO_PHASE2E_EVIDENCE_DIR") else {
        return Ok(());
    };
    let directory = PathBuf::from(root).join("dispatch-gates");
    std::fs::create_dir_all(&directory)?;
    let mut output = String::from("{\"owner\":\"cpu\",\"canonical_vectors\":[");
    for (index, counts) in evidence.canonical_vectors.iter().enumerate() {
        if index != 0 {
            output.push(',');
        }
        write!(output, "{:?}", counts.0).unwrap();
    }
    output.push_str("],\"characterization\":[");
    for (index, row) in evidence.characterization.iter().enumerate() {
        if index != 0 {
            output.push(',');
        }
        write!(
            output,
            "{{\"key\":{},\"owner\":{},\"surface\":{},\"budget\":{},\"mode\":{},\"counts\":{:?},\"observed_cpus\":{:?},\"numerical_passed\":{},\"typed_error_recovered\":{},\"typed_error_kind\":{},\"typed_error_source\":{},\"unwind_recovered\":{},\"post_recovery_passed\":{},\"recovery\":{{\"fresh_reset\":{},\"counts\":{:?},\"mode\":{},\"observed_cpus\":{:?},\"numerical_passed\":{},\"subset_passed\":{}}},\"hardware_skip\":{}}}",
            json_string(&row.key), json_string(row.owner), json_string(row.surface), row.budget,
            json_string(row.mode), row.counts.0, row.observed_cpus, row.numerical_passed,
            row.typed_error_recovered, json_string(row.typed_error_kind),
            json_string(&row.typed_error_source), row.unwind_recovered, row.post_recovery_passed,
            row.recovery.fresh_reset, row.recovery.counts.0, json_string(row.recovery.mode), row.recovery.observed_cpus,
            row.recovery.numerical_passed, row.recovery.subset_passed,
            row.hardware_skip.map(json_string).unwrap_or_else(|| "null".into())
        ).unwrap();
    }
    output.push_str("],\"cross_socket_locality\":{");
    write!(
        output,
        "\"usable_numa_nodes\":{},\"hardware_skip\":{},\"probes\":[",
        evidence.cross_socket_locality.usable_numa_nodes,
        if evidence.cross_socket_locality.probes.is_empty() {
            format!(
                "{{\"kind\":\"InsufficientNumaNodes\",\"required\":2,\"available\":{}}}",
                evidence.cross_socket_locality.usable_numa_nodes
            )
        } else {
            "null".into()
        }
    )
    .unwrap();
    for (index, probe) in evidence.cross_socket_locality.probes.iter().enumerate() {
        if index != 0 {
            output.push(',');
        }
        write!(
            output,
            "{{\"node\":{},\"declared_cpus\":{:?},\"observed_cpus\":{:?},\"numerical_passed\":{},\"subset_passed\":{}}}",
            probe.node, probe.declared_cpus, probe.observed_cpus,
            probe.numerical_passed, probe.subset_passed,
        )
        .unwrap();
    }
    output.push_str("]}}\n");
    let temporary = directory.join("cpu-evidence.json.partial");
    std::fs::write(&temporary, output)?;
    std::fs::rename(temporary, directory.join("cpu-evidence.json"))
}

#[test]
fn phase2e_characterization_evidence() {
    let evidence = run_cpu_owned_rows();
    assert_eq!(evidence.canonical_vectors.len(), 10);
    assert_eq!(evidence.characterization.len(), 29);
    assert!(evidence.characterization.iter().all(Row::gating_passed));
    let no_inner = evidence
        .characterization
        .iter()
        .find(|row| row.surface == "U-I")
        .unwrap();
    assert_eq!(no_inner.mode, "Sequential");
    write_evidence(&evidence).unwrap();
}
