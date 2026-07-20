use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use smallvec::SmallVec;
use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashMap;
use std::mem::size_of;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use tenferro_cpu::CpuBackend;
use tenferro_tensor::{DotGeneralConfig, Tensor, TensorDot};

const RANKS: &[usize] = &[2, 3, 4, 8, 9, 16, 32];
const ALLOCATION_PROBE_ITERS: usize = 10_000;

struct CountingAllocator;

static COUNT_ALLOCATIONS: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let result = unsafe { System.alloc(layout) };
        if COUNT_ALLOCATIONS.load(Ordering::Relaxed) && !result.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        result
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let result = unsafe { System.alloc_zeroed(layout) };
        if COUNT_ALLOCATIONS.load(Ordering::Relaxed) && !result.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        result
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let result = unsafe { System.realloc(ptr, layout, new_size) };
        if COUNT_ALLOCATIONS.load(Ordering::Relaxed) && !result.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        }
        result
    }
}

#[derive(Clone, Copy)]
struct AllocationStats {
    allocations: u64,
    bytes: u64,
}

fn count_allocations(mut operation: impl FnMut() -> usize) -> AllocationStats {
    ALLOCATIONS.store(0, Ordering::Relaxed);
    ALLOCATED_BYTES.store(0, Ordering::Relaxed);
    COUNT_ALLOCATIONS.store(true, Ordering::Relaxed);
    for _ in 0..ALLOCATION_PROBE_ITERS {
        black_box(operation());
    }
    COUNT_ALLOCATIONS.store(false, Ordering::Relaxed);
    AllocationStats {
        allocations: ALLOCATIONS.load(Ordering::Relaxed),
        bytes: ALLOCATED_BYTES.load(Ordering::Relaxed),
    }
}

#[derive(Clone, Copy)]
struct BorrowedRequest<'a> {
    lhs_contracting: &'a [usize],
    rhs_contracting: &'a [usize],
    lhs_batch: &'a [usize],
    rhs_batch: &'a [usize],
    lhs_rank: usize,
    rhs_rank: usize,
}

impl<'a> BorrowedRequest<'a> {
    fn from_config(config: &'a DotGeneralConfig, rank: usize) -> Self {
        Self {
            lhs_contracting: &config.lhs_contracting_dims,
            rhs_contracting: &config.rhs_contracting_dims,
            lhs_batch: &config.lhs_batch_dims,
            rhs_batch: &config.rhs_batch_dims,
            lhs_rank: rank,
            rhs_rank: rank,
        }
    }
}

struct SmallVecRequest {
    lhs_contracting: SmallVec<[usize; 8]>,
    rhs_contracting: SmallVec<[usize; 8]>,
    lhs_batch: SmallVec<[usize; 8]>,
    rhs_batch: SmallVec<[usize; 8]>,
    lhs_rank: usize,
    rhs_rank: usize,
}

impl SmallVecRequest {
    fn from_borrowed(request: BorrowedRequest<'_>) -> Self {
        Self {
            lhs_contracting: SmallVec::from_slice(request.lhs_contracting),
            rhs_contracting: SmallVec::from_slice(request.rhs_contracting),
            lhs_batch: SmallVec::from_slice(request.lhs_batch),
            rhs_batch: SmallVec::from_slice(request.rhs_batch),
            lhs_rank: request.lhs_rank,
            rhs_rank: request.rhs_rank,
        }
    }

    fn as_borrowed(&self) -> BorrowedRequest<'_> {
        BorrowedRequest {
            lhs_contracting: &self.lhs_contracting,
            rhs_contracting: &self.rhs_contracting,
            lhs_batch: &self.lhs_batch,
            rhs_batch: &self.rhs_batch,
            lhs_rank: self.lhs_rank,
            rhs_rank: self.rhs_rank,
        }
    }
}

#[derive(Clone)]
struct FixedAxes {
    values: [usize; 32],
    len: usize,
}

impl FixedAxes {
    fn from_slice(values: &[usize]) -> Self {
        assert!(values.len() <= 32);
        let mut result = Self {
            values: [0; 32],
            len: values.len(),
        };
        result.values[..values.len()].copy_from_slice(values);
        result
    }

    fn as_slice(&self) -> &[usize] {
        &self.values[..self.len]
    }
}

struct FixedRequest {
    lhs_contracting: FixedAxes,
    rhs_contracting: FixedAxes,
    lhs_batch: FixedAxes,
    rhs_batch: FixedAxes,
    lhs_rank: usize,
    rhs_rank: usize,
}

impl FixedRequest {
    fn from_borrowed(request: BorrowedRequest<'_>) -> Self {
        Self {
            lhs_contracting: FixedAxes::from_slice(request.lhs_contracting),
            rhs_contracting: FixedAxes::from_slice(request.rhs_contracting),
            lhs_batch: FixedAxes::from_slice(request.lhs_batch),
            rhs_batch: FixedAxes::from_slice(request.rhs_batch),
            lhs_rank: request.lhs_rank,
            rhs_rank: request.rhs_rank,
        }
    }

    fn as_borrowed(&self) -> BorrowedRequest<'_> {
        BorrowedRequest {
            lhs_contracting: self.lhs_contracting.as_slice(),
            rhs_contracting: self.rhs_contracting.as_slice(),
            lhs_batch: self.lhs_batch.as_slice(),
            rhs_batch: self.rhs_batch.as_slice(),
            lhs_rank: self.lhs_rank,
            rhs_rank: self.rhs_rank,
        }
    }
}

fn representative_config(rank: usize) -> DotGeneralConfig {
    let split = rank.div_ceil(2);
    DotGeneralConfig {
        lhs_contracting_dims: (0..split).collect(),
        rhs_contracting_dims: (0..split).collect(),
        lhs_batch_dims: (split..rank).collect(),
        rhs_batch_dims: (split..rank).collect(),
    }
}

#[inline(never)]
fn request_checksum(request: BorrowedRequest<'_>) -> usize {
    request.lhs_rank
        ^ request.rhs_rank.rotate_left(3)
        ^ request.lhs_contracting.len().rotate_left(6)
        ^ request
            .rhs_contracting
            .first()
            .copied()
            .unwrap_or(0)
            .rotate_left(9)
        ^ request.lhs_batch.len().rotate_left(12)
        ^ request
            .rhs_batch
            .first()
            .copied()
            .unwrap_or(0)
            .rotate_left(15)
}

fn validate_linear(request: BorrowedRequest<'_>) -> bool {
    fn valid_role(values: &[usize], rank: usize) -> bool {
        values
            .iter()
            .enumerate()
            .all(|(index, &axis)| axis < rank && !values[..index].contains(&axis))
    }

    valid_role(request.lhs_contracting, request.lhs_rank)
        && valid_role(request.rhs_contracting, request.rhs_rank)
        && valid_role(request.lhs_batch, request.lhs_rank)
        && valid_role(request.rhs_batch, request.rhs_rank)
        && !request
            .lhs_contracting
            .iter()
            .any(|axis| request.lhs_batch.contains(axis))
        && !request
            .rhs_contracting
            .iter()
            .any(|axis| request.rhs_batch.contains(axis))
        && request.lhs_contracting.len() == request.rhs_contracting.len()
        && request.lhs_batch.len() == request.rhs_batch.len()
}

fn role_mask(values: &[usize], rank: usize) -> Option<u64> {
    let mut mask = 0_u64;
    for &axis in values {
        if axis >= rank || axis >= 64 {
            return None;
        }
        let bit = 1_u64 << axis;
        if mask & bit != 0 {
            return None;
        }
        mask |= bit;
    }
    Some(mask)
}

fn validate_bitset(request: BorrowedRequest<'_>) -> bool {
    let Some(lhs_contracting) = role_mask(request.lhs_contracting, request.lhs_rank) else {
        return false;
    };
    let Some(rhs_contracting) = role_mask(request.rhs_contracting, request.rhs_rank) else {
        return false;
    };
    let Some(lhs_batch) = role_mask(request.lhs_batch, request.lhs_rank) else {
        return false;
    };
    let Some(rhs_batch) = role_mask(request.rhs_batch, request.rhs_rank) else {
        return false;
    };
    lhs_contracting & lhs_batch == 0
        && rhs_contracting & rhs_batch == 0
        && request.lhs_contracting.len() == request.rhs_contracting.len()
        && request.lhs_batch.len() == request.rhs_batch.len()
}

trait Provider {
    fn execute(&self, request: BorrowedRequest<'_>) -> usize;
}

struct NoopProvider;

impl Provider for NoopProvider {
    #[inline(never)]
    fn execute(&self, request: BorrowedRequest<'_>) -> usize {
        request_checksum(request)
    }
}

trait TryProvider {
    fn try_execute(&self, request: BorrowedRequest<'_>) -> Option<usize>;
}

struct AlwaysHandle;
struct AlwaysDelegate;

impl TryProvider for AlwaysHandle {
    #[inline(never)]
    fn try_execute(&self, request: BorrowedRequest<'_>) -> Option<usize> {
        Some(request_checksum(request))
    }
}

impl TryProvider for AlwaysDelegate {
    #[inline(never)]
    fn try_execute(&self, _request: BorrowedRequest<'_>) -> Option<usize> {
        None
    }
}

struct DecoratingProvider<'a> {
    primary: &'a dyn TryProvider,
    fallback: &'a dyn Provider,
}

impl Provider for DecoratingProvider<'_> {
    #[inline(never)]
    fn execute(&self, request: BorrowedRequest<'_>) -> usize {
        self.primary
            .try_execute(request)
            .unwrap_or_else(|| self.fallback.execute(request))
    }
}

struct RuntimeScope {
    thread_budget: AtomicUsize,
}

impl RuntimeScope {
    #[inline(never)]
    fn execute(&self, provider: &dyn Provider, request: BorrowedRequest<'_>) -> usize {
        provider.execute(request) ^ self.thread_budget.load(Ordering::Relaxed)
    }
}

fn allocation_report() {
    eprintln!("provider-boundary prototype layout (bytes):");
    eprintln!("  BorrowedRequest={}", size_of::<BorrowedRequest<'_>>());
    eprintln!("  SmallVecRequest={}", size_of::<SmallVecRequest>());
    eprintln!("  FixedRequest={}", size_of::<FixedRequest>());
    eprintln!("allocation probe ({ALLOCATION_PROBE_ITERS} calls):");
    for &rank in RANKS {
        let config = representative_config(rank);
        let borrowed = BorrowedRequest::from_config(&config, rank);
        let borrowed_stats = count_allocations(|| request_checksum(borrowed));
        let smallvec_stats = count_allocations(|| {
            request_checksum(SmallVecRequest::from_borrowed(borrowed).as_borrowed())
        });
        let prepared = SmallVecRequest::from_borrowed(borrowed);
        let prepared_stats = count_allocations(|| request_checksum(prepared.as_borrowed()));
        let fixed_stats = count_allocations(|| {
            request_checksum(FixedRequest::from_borrowed(borrowed).as_borrowed())
        });
        let current_validation_stats =
            count_allocations(|| usize::from(config.validate_dims_with_ranks(rank, rank).is_ok()));
        eprintln!(
            "  rank={rank:>2}: borrowed={}/{}B smallvec={}/{}B prepared={}/{}B fixed={}/{}B current_validation={}/{}B",
            borrowed_stats.allocations,
            borrowed_stats.bytes,
            smallvec_stats.allocations,
            smallvec_stats.bytes,
            prepared_stats.allocations,
            prepared_stats.bytes,
            fixed_stats.allocations,
            fixed_stats.bytes,
            current_validation_stats.allocations,
            current_validation_stats.bytes,
        );
    }

    eprintln!("full eager dot_general allocation probe ({ALLOCATION_PROBE_ITERS} calls):");
    for size in [1, 2, 8] {
        let lhs = matrix(size, 0.25);
        let rhs = matrix(size, 0.75);
        let config = DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
        };
        let mut backend = CpuBackend::with_threads(1)
            .expect("one-thread prototype backend construction must succeed");
        let stats = count_allocations(|| {
            let output = backend
                .dot_general(&lhs, &rhs, &config)
                .expect("prototype dot_general must succeed");
            output.shape().iter().product()
        });
        eprintln!(
            "  {size}x{size}: {}/{}B total; {:.1} allocations/{:.1}B per call",
            stats.allocations,
            stats.bytes,
            stats.allocations as f64 / ALLOCATION_PROBE_ITERS as f64,
            stats.bytes as f64 / ALLOCATION_PROBE_ITERS as f64,
        );
    }
}

fn bench_request_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("provider_boundary/request");
    for &rank in RANKS {
        let config = representative_config(rank);
        let borrowed = BorrowedRequest::from_config(&config, rank);
        let prepared = SmallVecRequest::from_borrowed(borrowed);

        group.bench_function(BenchmarkId::new("borrow_existing_vec", rank), |b| {
            b.iter(|| request_checksum(black_box(BorrowedRequest::from_config(&config, rank))))
        });
        group.bench_function(BenchmarkId::new("smallvec_per_call", rank), |b| {
            b.iter(|| {
                let request = SmallVecRequest::from_borrowed(black_box(borrowed));
                request_checksum(black_box(request.as_borrowed()))
            })
        });
        group.bench_function(BenchmarkId::new("prepared_smallvec_borrow", rank), |b| {
            b.iter(|| request_checksum(black_box(prepared.as_borrowed())))
        });
        group.bench_function(BenchmarkId::new("fixed_inline_per_call", rank), |b| {
            b.iter(|| {
                let request = FixedRequest::from_borrowed(black_box(borrowed));
                request_checksum(black_box(request.as_borrowed()))
            })
        });
    }
    group.finish();
}

fn bench_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("provider_boundary/validation");
    for &rank in RANKS {
        let config = representative_config(rank);
        let request = BorrowedRequest::from_config(&config, rank);
        assert!(config.validate_dims_with_ranks(rank, rank).is_ok());
        assert!(validate_linear(request));
        assert!(validate_bitset(request));

        group.bench_function(BenchmarkId::new("current_hashset", rank), |b| {
            b.iter(|| black_box(&config).validate_dims_with_ranks(rank, rank))
        });
        group.bench_function(BenchmarkId::new("linear_scan", rank), |b| {
            b.iter(|| validate_linear(black_box(request)))
        });
        group.bench_function(BenchmarkId::new("u64_bitset", rank), |b| {
            b.iter(|| validate_bitset(black_box(request)))
        });
    }
    group.finish();
}

fn bench_dispatch(c: &mut Criterion) {
    let config = representative_config(4);
    let request = BorrowedRequest::from_config(&config, 4);
    let provider = NoopProvider;
    let dyn_provider: &dyn Provider = black_box(&provider);
    let handle = AlwaysHandle;
    let delegate = AlwaysDelegate;
    let handled = DecoratingProvider {
        primary: &handle,
        fallback: dyn_provider,
    };
    let delegated = DecoratingProvider {
        primary: &delegate,
        fallback: dyn_provider,
    };
    let scope = RuntimeScope {
        thread_budget: AtomicUsize::new(1),
    };
    let mut registry: HashMap<&'static str, &dyn Provider> = HashMap::new();
    registry.insert("tenferro.dot_general.v1", dyn_provider);
    registry.insert("tenferro.linalg.qr.v1", dyn_provider);
    registry.insert("tenferro.linalg.svd.v1", dyn_provider);
    let slots = [dyn_provider, dyn_provider, dyn_provider];

    let mut group = c.benchmark_group("provider_boundary/dispatch/rank_4");
    group.bench_function("direct_function", |b| {
        b.iter(|| request_checksum(black_box(request)))
    });
    group.bench_function("concrete_provider", |b| {
        b.iter(|| provider.execute(black_box(request)))
    });
    group.bench_function("dyn_provider", |b| {
        b.iter(|| dyn_provider.execute(black_box(request)))
    });
    group.bench_function("decorator_handled", |b| {
        b.iter(|| handled.execute(black_box(request)))
    });
    group.bench_function("decorator_delegated", |b| {
        b.iter(|| delegated.execute(black_box(request)))
    });
    group.bench_function("runtime_scope_atomic_budget", |b| {
        b.iter(|| scope.execute(dyn_provider, black_box(request)))
    });
    group.bench_function("string_hashmap_lookup", |b| {
        b.iter(|| {
            registry
                .get(black_box("tenferro.dot_general.v1"))
                .expect("prototype registry entry must exist")
                .execute(black_box(request))
        })
    });
    group.bench_function("resolved_slot_lookup", |b| {
        b.iter(|| slots[black_box(0)].execute(black_box(request)))
    });
    group.finish();
}

fn matrix(size: usize, seed: f64) -> Tensor {
    let values = (0..size * size)
        .map(|index| seed + index as f64 / (size * size) as f64)
        .collect();
    Tensor::from_vec_col_major(vec![size, size], values)
        .expect("prototype matrix shape and data length must match")
}

fn bench_tiny_dot_general(c: &mut Criterion) {
    let mut group = c.benchmark_group("provider_boundary/full_dot_general/one_thread");
    for size in [1, 2, 8] {
        let lhs = matrix(size, 0.25);
        let rhs = matrix(size, 0.75);
        let config = DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
        };
        let mut backend = CpuBackend::with_threads(1)
            .expect("one-thread prototype backend construction must succeed");
        group.bench_function(BenchmarkId::new("eager", format!("{size}x{size}")), |b| {
            b.iter(|| {
                black_box(&mut backend).dot_general(
                    black_box(&lhs),
                    black_box(&rhs),
                    black_box(&config),
                )
            })
        });
    }
    group.finish();
}

fn benches(c: &mut Criterion) {
    allocation_report();
    bench_request_construction(c);
    bench_validation(c);
    bench_dispatch(c);
    bench_tiny_dot_general(c);
}

criterion_group!(provider_boundary_benches, benches);
criterion_main!(provider_boundary_benches);
