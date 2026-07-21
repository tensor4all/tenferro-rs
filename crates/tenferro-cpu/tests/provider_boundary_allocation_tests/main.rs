use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::sync::Mutex;

use tenferro_cpu::provider::{
    CpuDotGeneralRequest, CpuExecutionContext, CpuGeneralContractionProvider, CpuProviderOutcome,
};
use tenferro_cpu::{
    discover_cpu_topology, CpuBackend, CpuBackendKind, CpuDomainExecutor,
    CpuDomainExecutorCapabilities, CpuDomainExecutorError, CpuExecutorAffinity,
    CpuExecutorReentrancy, CpuExecutorShutdown, CpuInnerParallelism, CpuPlacementControl,
    CpuPlacementGuarantee, CpuProviderBundle, CpuProviderExecutionCapabilities,
    CpuThreadCountControl, ExternalCpuDomain, ResolvedCpuPlacement, ScopedCpuJob, ScopedCpuJobs,
};
use tenferro_tensor::{
    BackendSessionHost, CpuDomainId, DType, DotGeneralAccumulation, DotGeneralConfig, SliceConfig,
    Tensor, TensorBuffer, TensorDot, TensorElementwise, TensorIndexing, TensorRead,
    TensorReduction, TensorWrite,
};

struct CountingAllocator;

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
static PROBE_LOCK: Mutex<()> = Mutex::new(());

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        // SAFETY: this allocator forwards the unchanged layout to System.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: ptr and layout came from the corresponding System allocation.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(new_size, Ordering::Relaxed);
        }
        // SAFETY: ptr and layout came from System, and new_size is forwarded unchanged.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AllocationCount {
    allocations: usize,
    bytes: usize,
}

fn count_repeated(mut op: impl FnMut(), iterations: usize) -> AllocationCount {
    ALLOCATIONS.store(0, Ordering::Relaxed);
    BYTES.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::SeqCst);
    for _ in 0..iterations {
        op();
    }
    COUNTING.store(false, Ordering::SeqCst);
    AllocationCount {
        allocations: ALLOCATIONS.load(Ordering::Relaxed),
        bytes: BYTES.load(Ordering::Relaxed),
    }
}

fn assert_not_above_baseline(case: &str, actual: AllocationCount, baseline: AllocationCount) {
    assert!(
        actual.allocations <= baseline.allocations && actual.bytes <= baseline.bytes,
        "{case} allocation regression: actual={actual:?}, fixed-main baseline={baseline:?}"
    );
}

#[derive(Debug)]
struct ValueWritingGeneralProvider {
    calls: Arc<AtomicUsize>,
}

#[derive(Debug)]
struct InlineExecutor;

impl CpuDomainExecutor for InlineExecutor {
    fn capabilities(&self) -> CpuDomainExecutorCapabilities {
        CpuDomainExecutorCapabilities {
            worker_count: NonZeroUsize::new(1).unwrap(),
            outer_parallelism: false,
            inner_parallelism: CpuInnerParallelism::None,
            reentrancy: CpuExecutorReentrancy::Rejected,
            affinity: CpuExecutorAffinity::None,
            shutdown: CpuExecutorShutdown::CallerOwned,
        }
    }

    fn submit(&self, jobs: &dyn ScopedCpuJobs) -> Result<(), CpuDomainExecutorError> {
        for index in 0..jobs.len() {
            jobs.run(index)?;
        }
        Ok(())
    }

    fn install(&self, job: &mut dyn ScopedCpuJob) -> Result<(), CpuDomainExecutorError> {
        job.run()
    }
}

impl CpuGeneralContractionProvider for ValueWritingGeneralProvider {
    fn execution_capabilities(&self) -> CpuProviderExecutionCapabilities {
        CpuProviderExecutionCapabilities {
            thread_count: CpuThreadCountControl::Sequential,
            placement: CpuPlacementControl::CallingThread,
            worker_local_sequential: true,
            accepts_sequential: true,
            accepts_outer: true,
            accepts_inner: true,
        }
    }

    fn dot_general(
        &self,
        _context: &CpuExecutionContext<'_>,
        mut request: CpuDotGeneralRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        let lhs = request
            .lhs()
            .as_tensor()
            .ok_or_else(|| {
                tenferro_tensor::Error::runtime_state(
                    "allocation_probe",
                    "expected an owned lhs tensor",
                )
            })?
            .as_slice::<f64>()?[0];
        let rhs = request
            .rhs()
            .as_tensor()
            .ok_or_else(|| {
                tenferro_tensor::Error::runtime_state(
                    "allocation_probe",
                    "expected an owned rhs tensor",
                )
            })?
            .as_slice::<f64>()?[0];
        let TensorWrite::Tensor(output) = request.output() else {
            return Err(tenferro_tensor::Error::runtime_state(
                "allocation_probe",
                "expected an owned output tensor",
            ));
        };
        output.as_slice_mut::<f64>()?[0] = lhs * rhs;
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(CpuProviderOutcome::Executed)
    }
}

#[test]
fn warmed_public_session_request_provider_dispatch_does_not_allocate() {
    let _probe = PROBE_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    const WARMUP: usize = 32;
    const ITERATIONS: usize = 10_000;
    let calls = Arc::new(AtomicUsize::new(0));
    let providers = CpuProviderBundle::builder(CpuBackendKind::default_compiled())
        .require_general_contraction_provider(Arc::new(ValueWritingGeneralProvider {
            calls: Arc::clone(&calls),
        }))
        .build()
        .unwrap();
    let domain_id = CpuDomainId::new(1);
    let allowed_cpus = discover_cpu_topology().unwrap().allowed_cpus().clone();
    let domain = ExternalCpuDomain::new(
        domain_id,
        ResolvedCpuPlacement::AllAllowed { cpus: allowed_cpus },
        Arc::new(InlineExecutor),
        NonZeroUsize::new(1).unwrap(),
        CpuPlacementGuarantee::ExactDeclared,
    )
    .unwrap();
    let mut backend = CpuBackend::from_external_managed_domains(domain_id, [domain])
        .unwrap()
        .with_provider_bundle(providers)
        .unwrap();
    let lhs = Tensor::from_vec_col_major(vec![1, 1], vec![black_box(2.0_f64)]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![1, 1], vec![black_box(3.0_f64)]).unwrap();
    let mut output = Tensor::from_vec_col_major(vec![1, 1], vec![0.0_f64]).unwrap();
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let accumulation = DotGeneralAccumulation::overwrite(DType::F64).unwrap();

    let count = backend.with_backend_session(|session| {
        let mut dispatch = || {
            session
                .dot_general_read_into_accum(
                    TensorRead::from_tensor(black_box(&lhs)),
                    TensorRead::from_tensor(black_box(&rhs)),
                    black_box(&config),
                    black_box(accumulation),
                    TensorWrite::from_tensor(black_box(&mut output)),
                )
                .unwrap();
            black_box(output.as_slice::<f64>().unwrap()[0]);
        };
        for _ in 0..WARMUP {
            dispatch();
        }
        count_repeated(&mut dispatch, ITERATIONS)
    });

    assert_eq!(count.allocations, 0);
    assert_eq!(count.bytes, 0);
    assert_eq!(calls.load(Ordering::Relaxed), WARMUP + ITERATIONS);
    assert_eq!(black_box(output.as_slice::<f64>().unwrap()[0]), 6.0);
}

#[test]
fn warmed_tiny_cpu_backend_cases_do_not_exceed_fixed_main_allocations() {
    // This direct CpuBackend probe isolates the steady-state allocation boundary
    // changed by the provider routing. Full AD eager non-inferiority is measured
    // separately by the fixed three-pair Criterion campaign.
    //
    // Baselines were measured with this identical setup at immutable commit
    // 85855e272b1495611deb601a9ee06f3546772c3c using the default cpu-faer
    // feature set. Setup and 32 warm-up iterations are outside counted loops.
    let _probe = PROBE_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    const ITERATIONS: usize = 100;
    let mut backend = CpuBackend::with_threads(1).unwrap();
    let matrix = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![2, 2], vec![5.0_f64, 6.0, 7.0, 8.0]).unwrap();
    let slice = SliceConfig {
        starts: vec![0, 0],
        limits: vec![2, 2],
        strides: vec![1, 1],
    };
    let dot = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };

    for _ in 0..32 {
        let output = backend.add(&matrix, &rhs).unwrap();
        backend.reclaim_buffer(output);
        let output = backend.reduce_sum(&matrix, &[0]).unwrap();
        backend.reclaim_buffer(output);
        let output = backend.slice(&matrix, &slice).unwrap();
        backend.reclaim_buffer(output);
        let output = backend.dot_general(&matrix, &rhs, &dot).unwrap();
        backend.reclaim_buffer(output);
    }

    let elementwise = count_repeated(
        || {
            let output = backend.add(&matrix, &rhs).unwrap();
            backend.reclaim_buffer(output);
        },
        ITERATIONS,
    );
    let reduction = count_repeated(
        || {
            let output = backend.reduce_sum(&matrix, &[0]).unwrap();
            backend.reclaim_buffer(output);
        },
        ITERATIONS,
    );
    let slice_count = count_repeated(
        || {
            let output = backend.slice(&matrix, &slice).unwrap();
            backend.reclaim_buffer(output);
        },
        ITERATIONS,
    );
    let dot_count = count_repeated(
        || {
            let output = backend.dot_general(&matrix, &rhs, &dot).unwrap();
            backend.reclaim_buffer(output);
        },
        ITERATIONS,
    );

    eprintln!(
        "candidate allocation probe: elementwise={elementwise:?} reduction={reduction:?} slice={slice_count:?} dot={dot_count:?}"
    );
    assert_not_above_baseline(
        "elementwise",
        elementwise,
        AllocationCount {
            allocations: 1_201,
            bytes: 55_920,
        },
    );
    assert_not_above_baseline(
        "reduction",
        reduction,
        AllocationCount {
            allocations: 5_005,
            bytes: 112_592,
        },
    );
    assert_not_above_baseline(
        "slice",
        slice_count,
        AllocationCount {
            allocations: 601,
            bytes: 38_320,
        },
    );
    assert_not_above_baseline(
        "dot_general",
        dot_count,
        AllocationCount {
            allocations: 3_802,
            bytes: 112_440,
        },
    );
}
