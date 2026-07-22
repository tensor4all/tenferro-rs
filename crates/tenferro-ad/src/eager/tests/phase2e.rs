use std::fmt::Write as _;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};

use tenferro_cpu::provider::{
    CpuExecutionContext, CpuGemmProvider, CpuGemmRequest, CpuGroupedGemmRequest,
    CpuProviderOutcome, FaerGemmProvider,
};
use tenferro_cpu::{
    process_cpu_affinity, CpuBackend, CpuBackendKind, CpuDomainExecutor,
    CpuDomainExecutorCapabilities, CpuDomainExecutorError, CpuExecutorAffinity,
    CpuExecutorReentrancy, CpuExecutorShutdown, CpuInnerParallelism, CpuPlacement,
    CpuPlacementGuarantee, CpuProviderBundle, CpuSet, ExternalCpuDomain, ResolvedCpuPlacement,
    ScopedCpuJob, ScopedCpuJobs,
};
use tenferro_tensor::{CpuDomainId, DotGeneralConfig, SliceConfig};

use crate::eager_backend::EagerBackend;

use super::super::{EagerRuntime, EagerTensor};
use super::Tensor;

const EAGER_NATIVE: [usize; 6] = [1, 1, 1, 1, 0, 0];
const EAGER_DOT: [usize; 6] = [1, 1, 1, 1, 0, 1];

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

fn observe_pool_cpus(mut run: impl FnMut(Box<dyn FnOnce() + Send>)) -> Vec<usize> {
    let observed = Arc::new(Mutex::new(Vec::new()));
    for _ in 0..1_024 {
        let capture = Arc::clone(&observed);
        run(Box::new(move || {
            if let Some(cpu) = current_cpu() {
                let mut cpus = capture.lock().unwrap();
                if !cpus.contains(&cpu) {
                    cpus.push(cpu);
                }
            }
        }));
    }
    Arc::try_unwrap(observed).unwrap().into_inner().unwrap()
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
        let inner = if selected.len() == budget {
            tenferro_cpu::CpuContext::with_pinned_cpus(selected, budget).unwrap()
        } else {
            tenferro_cpu::CpuContext::with_threads(budget).unwrap()
        };
        Arc::new(Self {
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
        })
    }

    fn observe_cpu(&self) {
        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            unsafe extern "C" {
                fn sched_getcpu() -> std::ffi::c_int;
            }
            // SAFETY: `sched_getcpu` takes no pointers and has no preconditions.
            let cpu = unsafe { sched_getcpu() };
            if cpu >= 0 {
                let mut observed = self.observed_cpus.lock().unwrap();
                let cpu = cpu as usize;
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

    fn affinity_audit(&self) -> Vec<usize> {
        observe_pool_cpus(|operation| self.inner.install(operation))
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

#[derive(Debug)]
struct RecordingGemm(AtomicUsize);

impl CpuGemmProvider for RecordingGemm {
    fn execution_capabilities(&self) -> tenferro_cpu::CpuProviderExecutionCapabilities {
        FaerGemmProvider.execution_capabilities()
    }

    fn gemm(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        self.0.fetch_add(1, Ordering::SeqCst);
        FaerGemmProvider.gemm(context, request)
    }

    fn strided_batched_gemm(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        self.0.fetch_add(1, Ordering::SeqCst);
        FaerGemmProvider.strided_batched_gemm(context, request)
    }

    fn grouped_gemm(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuGroupedGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        self.0.fetch_add(1, Ordering::SeqCst);
        FaerGemmProvider.grouped_gemm(context, request)
    }
}

struct CpuFixture {
    backend: CpuBackend,
    executor: Option<Arc<RecordingExecutor>>,
    provider: Arc<RecordingGemm>,
    declared_cpus: Vec<usize>,
}

fn cpu_fixture(ownership: &str, budget: usize) -> CpuFixture {
    let provider = Arc::new(RecordingGemm(AtomicUsize::new(0)));
    let bundle = CpuProviderBundle::builder(CpuBackendKind::Faer)
        .gemm_provider(Arc::clone(&provider) as Arc<dyn CpuGemmProvider>)
        .build()
        .unwrap();
    if ownership == "managed-exact" {
        let backend = CpuBackend::with_threads_and_kind(budget, CpuBackendKind::Faer)
            .unwrap()
            .with_provider_bundle(bundle)
            .unwrap();
        return CpuFixture {
            declared_cpus: backend.topology().allowed_cpus().as_usize_vec(),
            backend,
            executor: None,
            provider,
        };
    }
    let exact = ownership == "external-exact";
    let executor = RecordingExecutor::new(budget, exact);
    let allowed = process_cpu_affinity().unwrap();
    let declared_cpus = allowed.as_usize_vec();
    let id = CpuDomainId::new(0x2ead + budget as u64 + u64::from(u8::from(exact)));
    let domain = ExternalCpuDomain::new(
        id,
        ResolvedCpuPlacement::AllAllowed { cpus: allowed },
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
    }
}

#[derive(Debug)]
struct Row {
    key: String,
    surface: &'static str,
    budget: usize,
    mode: &'static str,
    counts: [usize; 6],
    observed_cpus: Vec<usize>,
    numerical_passed: bool,
    typed_error_recovered: bool,
    unwind_recovered: bool,
    post_recovery_passed: bool,
}

impl Row {
    fn gating_passed(&self) -> bool {
        (self.counts == EAGER_NATIVE || self.counts == EAGER_DOT)
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
    provider.0.store(0, Ordering::SeqCst);
    if let Some(executor) = executor {
        executor.reset();
    }
}

fn recovery_proof(
    placed: &mut super::super::CpuPlacementBoundEager,
    lhs: &Tensor,
) -> (bool, bool, bool) {
    let wrong = tensor(vec![lhs.shape()[0] - 1], 21);
    let typed_error = placed
        .with_eager_session(|session| session.add(lhs, &wrong).map_err(crate::Error::from))
        .is_err();
    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _: crate::Result<()> = placed.with_eager_session(|_| panic!("phase2e eager unwind"));
    }))
    .is_err();
    let post_recovery = placed
        .with_eager_session(|session| session.add(lhs, lhs).map_err(crate::Error::from))
        .map(|output| output.shape() == lhs.shape())
        .unwrap_or(false);
    (typed_error, unwind, post_recovery)
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
            let affinity_audit = fixture
                .executor
                .as_ref()
                .map(|executor| executor.affinity_audit())
                .unwrap_or_else(|| {
                    observe_pool_cpus(|operation| fixture.backend.install(operation))
                });
            let executor = fixture.executor.as_ref().map(Arc::clone);
            let provider = Arc::clone(&fixture.provider);
            let declared_cpus = fixture.declared_cpus.clone();
            let runtime = EagerRuntime::with_cpu_backend(fixture.backend);
            let mut placed = runtime.on_cpu(CpuPlacement::AllAllowed).unwrap();
            reset_counters(executor.as_ref(), &provider);
            let native = placed
                .with_eager_session(|session| {
                    session
                        .add(&native_lhs, &native_rhs)
                        .map_err(crate::Error::from)
                })
                .unwrap();
            let expected: Vec<_> = native_lhs
                .as_slice::<f64>()
                .unwrap()
                .iter()
                .zip(native_rhs.as_slice::<f64>().unwrap())
                .map(|(lhs, rhs)| lhs + rhs)
                .collect();
            let numerical_passed = native.as_slice::<f64>().unwrap() == expected;
            assert_eq!(provider.0.load(Ordering::SeqCst), 0);
            let mut observed_cpus = executor
                .as_ref()
                .map(|executor| {
                    assert_eq!(executor.installs.load(Ordering::SeqCst), 1);
                    assert_eq!(executor.submits.load(Ordering::SeqCst), 0);
                    executor.observed_cpus.lock().unwrap().clone()
                })
                .unwrap_or_default();
            for cpu in &affinity_audit {
                if !observed_cpus.contains(cpu) {
                    observed_cpus.push(*cpu);
                }
            }
            if ownership != "external-advisory" {
                assert!(observed_cpus.iter().all(|cpu| declared_cpus.contains(cpu)));
            }
            let (typed_error_recovered, unwind_recovered, post_recovery_passed) =
                recovery_proof(&mut placed, &native_lhs);
            rows.push(Row {
                key: format!("{ownership}/budget-{budget}/E-N"),
                surface: "E-N",
                budget,
                mode: if budget == 1 { "Sequential" } else { "Inner" },
                counts: EAGER_NATIVE,
                observed_cpus,
                numerical_passed,
                typed_error_recovered,
                unwind_recovered,
                post_recovery_passed,
            });
            provider.0.store(0, Ordering::SeqCst);
            if let Some(executor) = &executor {
                executor.reset();
            }
            let dot = placed
                .with_eager_session(|session| {
                    session
                        .dot_general(&dot_lhs, &dot_rhs, &dot_config)
                        .map_err(crate::Error::from)
                })
                .unwrap();
            let numerical_passed = relative_dot_error(&dot_lhs, &dot_rhs, &dot, 128) <= 1.0e-12;
            assert_eq!(provider.0.load(Ordering::SeqCst), 1);
            let mut observed_cpus = executor
                .as_ref()
                .map(|executor| {
                    assert_eq!(executor.installs.load(Ordering::SeqCst), 1);
                    assert_eq!(executor.submits.load(Ordering::SeqCst), 0);
                    executor.observed_cpus.lock().unwrap().clone()
                })
                .unwrap_or_default();
            for cpu in &affinity_audit {
                if !observed_cpus.contains(cpu) {
                    observed_cpus.push(*cpu);
                }
            }
            if ownership != "external-advisory" {
                assert!(observed_cpus.iter().all(|cpu| declared_cpus.contains(cpu)));
            }
            let (typed_error_recovered, unwind_recovered, post_recovery_passed) =
                recovery_proof(&mut placed, &native_lhs);
            rows.push(Row {
                key: format!("{ownership}/budget-{budget}/E-D"),
                surface: "E-D",
                budget,
                mode: if budget == 1 { "Sequential" } else { "Inner" },
                counts: EAGER_DOT,
                observed_cpus,
                numerical_passed,
                typed_error_recovered,
                unwind_recovered,
                post_recovery_passed,
            });
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
            "{{\"key\":{},\"owner\":\"ad\",\"surface\":{},\"budget\":{},\"mode\":{},\"counts\":{:?},\"numerical_passed\":{},\"typed_error_recovered\":{},\"unwind_recovered\":{},\"post_recovery_passed\":{},\"observed_cpus\":{:?},\"hardware_skip\":null}}",
            quoted(&row.key), quoted(row.surface), row.budget, quoted(row.mode), row.counts,
            row.numerical_passed, row.typed_error_recovered, row.unwind_recovered,
            row.post_recovery_passed, row.observed_cpus,
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
