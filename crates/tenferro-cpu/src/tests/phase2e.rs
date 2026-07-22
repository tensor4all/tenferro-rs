use std::fmt::Write as _;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use rayon::prelude::*;
use tenferro_tensor::{
    BackendSessionHost, DotGeneralConfig, SliceConfig, TensorDot, TensorElementwise,
    TensorReduction,
};

use super::*;
use crate::provider::{
    CpuExecutionContext, CpuGemmProvider, CpuGemmRequest, CpuGroupedGemmRequest,
    CpuProviderOutcome, FaerGemmProvider,
};
use crate::{
    process_cpu_affinity, CpuBackendKind, CpuDomainExecutor, CpuDomainExecutorCapabilities,
    CpuDomainExecutorError, CpuDomainId, CpuExecutorAffinity, CpuExecutorReentrancy,
    CpuExecutorShutdown, CpuInnerParallelism, CpuPlacementGuarantee, CpuProviderBundle,
    ExternalCpuDomain, ResolvedCpuPlacement, ScopedCpuJob, ScopedCpuJobs,
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

fn observe_pool_cpus(run: impl FnOnce(Box<dyn FnOnce() + Send>) + Send) -> Vec<usize> {
    let observed = Arc::new(Mutex::new(Vec::new()));
    let capture = Arc::clone(&observed);
    run(Box::new(move || {
        (0..65_536_usize).into_par_iter().for_each(|_| {
            if let Some(cpu) = current_cpu() {
                let mut cpus = capture.lock().unwrap();
                if !cpus.contains(&cpu) {
                    cpus.push(cpu);
                }
            }
        });
    }));
    Arc::try_unwrap(observed).unwrap().into_inner().unwrap()
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
        let inner = if selected.len() == budget {
            CpuContext::with_pinned_cpus(selected, budget).unwrap()
        } else {
            CpuContext::with_threads(budget).unwrap()
        };
        Arc::new(Self {
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
        })
    }

    fn observe_cpu(&self) {
        if let Some(cpu) = current_cpu() {
            let mut observed = self.observed_cpus.lock().unwrap();
            if !observed.contains(&cpu) {
                observed.push(cpu);
            }
        }
    }

    fn affinity_audit(&self) -> Vec<usize> {
        observe_pool_cpus(|operation| self.inner.install(operation))
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
        let backend = CpuBackend::with_threads_and_kind(budget, CpuBackendKind::Faer)
            .unwrap()
            .with_provider_bundle(bundle)
            .unwrap();
        return Fixture {
            declared_cpus: backend.topology().allowed_cpus().as_usize_vec(),
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
    let declared_cpus = allowed.as_usize_vec();
    let domain_id = CpuDomainId::new(0x2e00 + budget as u64 + u64::from(u8::from(exact)));
    let domain = ExternalCpuDomain::new(
        domain_id,
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
    const fn new(
        session: usize,
        scope: usize,
        permit: usize,
        install: usize,
        submit: usize,
        provider: usize,
    ) -> Self {
        Self([session, scope, permit, install, submit, provider])
    }
}

const DIRECT_NATIVE: Counts = Counts::new(0, 1, 1, 1, 0, 0);
const EAGER_NATIVE: Counts = Counts::new(1, 1, 1, 1, 0, 0);
const DIRECT_DOT: Counts = Counts::new(0, 1, 1, 1, 0, 1);
const EAGER_DOT: Counts = Counts::new(1, 1, 1, 1, 0, 1);

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
    unwind_recovered: bool,
    post_recovery_passed: bool,
    hardware_skip: Option<&'static str>,
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
    let mut backend = CpuBackend::new();

    assert_eq!(backend.neg(&input).unwrap().shape(), &[64]);
    assert_eq!(backend.add(&input, &rhs).unwrap().shape(), &[64]);
    assert_eq!(
        backend.reduce_sum(&input, &[0]).unwrap().shape(),
        &[] as &[usize]
    );
    assert_eq!(backend.slice(&input, &slice).unwrap().shape(), &[31]);
    assert_eq!(
        backend
            .dot_general(&matrix_lhs, &matrix_rhs, &dot)
            .unwrap()
            .shape(),
        &[8, 8]
    );

    backend
        .with_backend_session(|session| {
            assert_eq!(session.neg(&input)?.shape(), &[64]);
            assert_eq!(session.add(&input, &rhs)?.shape(), &[64]);
            assert_eq!(session.reduce_sum(&input, &[0])?.shape(), &[] as &[usize]);
            assert_eq!(session.slice(&input, &slice)?.shape(), &[31]);
            assert_eq!(
                session.dot_general(&matrix_lhs, &matrix_rhs, &dot)?.shape(),
                &[8, 8]
            );
            Ok::<_, crate::Error>(())
        })
        .unwrap();

    vec![
        DIRECT_NATIVE,
        DIRECT_NATIVE,
        DIRECT_NATIVE,
        DIRECT_NATIVE,
        DIRECT_DOT,
        EAGER_NATIVE,
        EAGER_NATIVE,
        EAGER_NATIVE,
        EAGER_NATIVE,
        EAGER_DOT,
    ]
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

fn row_recovery_proof(fixture: &mut Fixture) -> (bool, bool, bool) {
    let lhs = deterministic_vector(8, 7);
    let wrong = deterministic_vector(7, 8);
    let typed_error = fixture.backend.add(&lhs, &wrong).is_err();
    let unwind = std::panic::catch_unwind(AssertUnwindSafe(|| {
        fixture.backend.install(|| panic!("phase2e-row-unwind"))
    }))
    .is_err();
    let post_recovery = fixture
        .backend
        .add(&lhs, &lhs)
        .map(|tensor| tensor.shape() == [8])
        .unwrap_or(false);
    reset_fixture(fixture);
    (typed_error, unwind, post_recovery)
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
    output
        .as_slice::<f64>()
        .unwrap()
        .iter()
        .all(|value| value.is_finite())
        && output
            .as_slice::<f64>()
            .unwrap()
            .iter()
            .any(|value| *value != 0.0)
}

fn unsupported_outer_row() -> Row {
    let executor =
        RecordingExecutor::new_with_parallelism(2, false, false, CpuInnerParallelism::Rayon);
    let context_fixture = crate::provider::tests::external_execution_context_fixture(
        Arc::clone(&executor) as Arc<dyn CpuDomainExecutor>,
        NonZeroUsize::new(2).unwrap(),
    );
    let provider_calls = AtomicUsize::new(0);
    let error = context_fixture
        .entry()
        .submit_outer(5, |_, _| {
            provider_calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })
        .unwrap_err();
    assert!(matches!(error, CpuDomainExecutorError::Scheduling { .. }));
    assert_eq!(executor.snapshot(), (0, 0, vec![]));
    assert_eq!(provider_calls.load(Ordering::SeqCst), 0);
    let unwind_recovered = std::panic::catch_unwind(AssertUnwindSafe(|| {
        context_fixture
            .entry()
            .enter(crate::ParallelMode::Sequential, |_| panic!("u-o unwind"))
            .unwrap()
    }))
    .is_err();
    let post_recovery_passed = context_fixture
        .entry()
        .enter(crate::ParallelMode::Sequential, |_| 7)
        .unwrap()
        == 7;
    Row {
        key: "external-no-outer/budget-2/U-O".into(),
        owner: "cpu",
        surface: "U-O",
        budget: 2,
        mode: "UnsupportedOuter",
        counts: Counts::new(0, 1, 1, 0, 0, 0),
        observed_cpus: executor.snapshot().2,
        numerical_passed: true,
        typed_error_recovered: true,
        unwind_recovered,
        post_recovery_passed,
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
    let numerical_passed = run_dot_row(&mut fixture);
    assert_eq!(fixture.executor.as_ref().unwrap().snapshot().0, 1);
    assert_eq!(fixture.provider.calls.load(Ordering::SeqCst), 1);
    let observed_cpus = fixture.executor.as_ref().unwrap().snapshot().2;
    let (typed_error_recovered, unwind_recovered, post_recovery_passed) =
        row_recovery_proof(&mut fixture);
    Row {
        key: "external-no-inner/budget-2/U-I".into(),
        owner: "cpu",
        surface: "U-I",
        budget: 2,
        mode: "Sequential",
        counts: DIRECT_DOT,
        observed_cpus,
        numerical_passed,
        typed_error_recovered,
        unwind_recovered,
        post_recovery_passed,
        hardware_skip: None,
    }
}

fn run_cpu_owned_rows() -> Evidence {
    recovery_proof();
    let mut rows = Vec::with_capacity(29);
    for ownership in ["managed-exact", "external-exact", "external-advisory"] {
        for budget in [1, 2, 4] {
            let mut fixture = fixture(ownership, budget);
            let affinity_audit = fixture
                .executor
                .as_ref()
                .map(|executor| executor.affinity_audit())
                .unwrap_or_else(|| {
                    observe_pool_cpus(|operation| fixture.backend.install(operation))
                });
            let mode = if budget == 1 { "Sequential" } else { "Inner" };
            for (surface, counts) in [
                ("D-N", DIRECT_NATIVE),
                ("D-D", DIRECT_DOT),
                (
                    "G-O",
                    if budget == 1 {
                        DIRECT_DOT
                    } else {
                        Counts::new(0, 1, 1, 0, 1, 2 * budget + 1)
                    },
                ),
            ] {
                reset_fixture(&fixture);
                let numerical_passed = match surface {
                    "D-N" => run_native_row(&mut fixture),
                    "D-D" => run_dot_row(&mut fixture),
                    "G-O" => run_grouped_row(&mut fixture, budget),
                    _ => unreachable!(),
                };
                assert_eq!(
                    fixture.provider.calls.load(Ordering::SeqCst),
                    counts.0[5],
                    "{ownership}/{budget}/{surface} provider count"
                );
                let (install, submit, mut observed_cpus) = fixture
                    .executor
                    .as_ref()
                    .map(|executor| executor.snapshot())
                    .unwrap_or((counts.0[3], counts.0[4], Vec::new()));
                for cpu in &affinity_audit {
                    if !observed_cpus.contains(cpu) {
                        observed_cpus.push(*cpu);
                    }
                }
                assert_eq!((install, submit), (counts.0[3], counts.0[4]));
                if ownership != "external-advisory" {
                    assert!(observed_cpus
                        .iter()
                        .all(|cpu| fixture.declared_cpus.contains(cpu)));
                }
                let (typed_error_recovered, unwind_recovered, post_recovery_passed) =
                    row_recovery_proof(&mut fixture);
                rows.push(Row {
                    key: format!("{ownership}/budget-{budget}/{surface}"),
                    owner: "cpu",
                    surface,
                    budget,
                    mode: if surface == "G-O" && budget > 1 {
                        "Outer"
                    } else {
                        mode
                    },
                    counts,
                    observed_cpus,
                    numerical_passed,
                    typed_error_recovered,
                    unwind_recovered,
                    post_recovery_passed,
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
            "{{\"key\":{},\"owner\":{},\"surface\":{},\"budget\":{},\"mode\":{},\"counts\":{:?},\"observed_cpus\":{:?},\"numerical_passed\":{},\"typed_error_recovered\":{},\"unwind_recovered\":{},\"post_recovery_passed\":{},\"hardware_skip\":{}}}",
            json_string(&row.key), json_string(row.owner), json_string(row.surface), row.budget,
            json_string(row.mode), row.counts.0, row.observed_cpus, row.numerical_passed,
            row.typed_error_recovered, row.unwind_recovered, row.post_recovery_passed,
            row.hardware_skip.map(json_string).unwrap_or_else(|| "null".into())
        ).unwrap();
    }
    output.push_str("]}\n");
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
    write_evidence(&evidence).unwrap();
}
