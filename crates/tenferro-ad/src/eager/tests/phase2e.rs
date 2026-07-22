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
    CpuExecutorReentrancy, CpuExecutorShutdown, CpuInnerParallelism, CpuPlacementGuarantee,
    CpuProviderBundle, CpuSet, ExternalCpuDomain, NumaNodeId, ResolvedCpuPlacement, ScopedCpuJob,
    ScopedCpuJobs,
};
use tenferro_tensor::{CpuDomainId, DotGeneralConfig, SliceConfig};

use crate::eager_backend::EagerBackend;

use super::super::{EagerRuntime, EagerTensor};
use super::Tensor;

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
        return CpuFixture {
            declared_cpus: backend.resolved_placement().unwrap().cpus().as_usize_vec(),
            backend,
            executor: None,
            provider,
        };
    }
    let exact = ownership == "external-exact";
    let executor = RecordingExecutor::new(budget, exact);
    let allowed = process_cpu_affinity().unwrap();
    let selected = CpuSet::new(allowed.as_slice().iter().copied().take(budget)).unwrap();
    let declared_cpus = if exact {
        selected.as_usize_vec()
    } else {
        Vec::new()
    };
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
    }
}

#[derive(Debug)]
struct Row {
    key: String,
    surface: &'static str,
    budget: usize,
    session_entry: usize,
    downstream_vector: &'static str,
    actual_install: Option<usize>,
    actual_submit: Option<usize>,
    actual_provider: usize,
    observed_cpus: Vec<usize>,
    numerical_passed: bool,
    typed_error_recovered: bool,
    unwind_recovered: bool,
    post_recovery_passed: bool,
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
    provider.0.store(0, Ordering::SeqCst);
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
    sessions: &AtomicUsize,
) -> crate::Result<Tensor> {
    sessions.fetch_add(1, Ordering::SeqCst);
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
) -> (bool, bool, bool) {
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
    let sessions = AtomicUsize::new(0);
    let post_recovery = eager_surface(
        placed, surface, native_lhs, native_rhs, dot_lhs, dot_rhs, dot_config, &sessions,
    )
    .map(|output| {
        if surface == "E-N" {
            output.shape() == native_lhs.shape()
        } else {
            relative_dot_error(dot_lhs, dot_rhs, &output, 128) <= 1.0e-12
        }
    })
    .unwrap_or(false)
        && sessions.load(Ordering::SeqCst) == 1;
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
            let executor = fixture.executor.as_ref().map(Arc::clone);
            let provider = Arc::clone(&fixture.provider);
            let declared_cpus = fixture.declared_cpus.clone();
            let placement = fixture.backend.placement();
            let runtime = EagerRuntime::with_cpu_backend(fixture.backend);
            let mut placed = runtime.on_cpu(placement).unwrap();
            for surface in ["E-N", "E-D"] {
                reset_counters(executor.as_ref(), &provider);
                let sessions = AtomicUsize::new(0);
                let output = eager_surface(
                    &mut placed,
                    surface,
                    &native_lhs,
                    &native_rhs,
                    &dot_lhs,
                    &dot_rhs,
                    &dot_config,
                    &sessions,
                )
                .unwrap();
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
                let actual_provider = provider.0.load(Ordering::SeqCst);
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
                let (typed_error_recovered, unwind_recovered, post_recovery_passed) =
                    recovery_proof(
                        &mut placed,
                        surface,
                        &native_lhs,
                        &native_rhs,
                        &dot_lhs,
                        &dot_rhs,
                        &dot_config,
                    );
                rows.push(Row {
                    key: format!("{ownership}/budget-{budget}/{surface}"),
                    surface,
                    budget,
                    session_entry: sessions.load(Ordering::SeqCst),
                    downstream_vector: if surface == "E-N" {
                        "borrowed-add"
                    } else {
                        "borrowed-dot"
                    },
                    actual_install,
                    actual_submit,
                    actual_provider,
                    observed_cpus,
                    numerical_passed,
                    typed_error_recovered,
                    unwind_recovered,
                    post_recovery_passed,
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
            "{{\"key\":{},\"owner\":\"ad\",\"surface\":{},\"budget\":{},\"session_entry\":{},\"downstream_vector\":{},\"actual_install\":{},\"actual_submit\":{},\"actual_provider\":{},\"numerical_passed\":{},\"typed_error_recovered\":{},\"unwind_recovered\":{},\"post_recovery_passed\":{},\"observed_cpus\":{:?},\"hardware_skip\":null}}",
            quoted(&row.key), quoted(row.surface), row.budget, row.session_entry,
            quoted(row.downstream_vector),
            row.actual_install.map_or_else(|| "null".into(), |value| value.to_string()),
            row.actual_submit.map_or_else(|| "null".into(), |value| value.to_string()),
            row.actual_provider,
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
