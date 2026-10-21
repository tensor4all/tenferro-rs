//! Prepared-program dispatch overhead on one caller-held CPU engine (#2186).
//!
//! The tenferro-benchmark `cpu/einsum` `small/grad_sum_*` rows measure a traced
//! program whose cost is flat in matrix size: the time is per-command dispatch
//! and per-run bookkeeping, not arithmetic. This harness reproduces that shape
//! locally without the benchmark repository: it builds the same traced graphs,
//! compiles and prepares them once, and times repeated `Runtime::run` calls
//! against the same operations executed eagerly on the same engine.
//!
//! Each arm also reports the prepared program's command census
//! (`Prepared::execution_command_count`) and its elementwise region counters, so
//! "one command per instruction" can be separated from "one fused command per
//! region", and — with `TENFERRO_PROBE_ALLOCS=1` — the heap allocations one run
//! performs.
//!
//! ```text
//! TENFERRO_PROBE_THREADS=1,4 cargo bench -p tenferro-linalg --features autodiff \
//!   --bench traced_small_ad_overhead
//! ```
//!
//! Environment: `TENFERRO_PROBE_THREADS` (comma-separated engine widths),
//! `TENFERRO_PROBE_RUNS` (sampled batches, default 15), `TENFERRO_PROBE_ITERATIONS`
//! (runs per batch, default 50), `TENFERRO_PROBE_ALLOCS=1` (allocation counters).
//! Pin the process (`taskset -c <core>`) and check the core is idle: the results
//! are microseconds, so an unpinned host noise floor can hide the difference.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use tenferro_ad::AdContext;
use tenferro_cpu::CpuEngine;
use tenferro_linalg::TracedTensorLinalgExt;
use tenferro_runtime::{GraphCompiler, Prepared, Runtime, TensorSessionOpsExt, TracedTensor};
use tenferro_tensor::{Tensor, TypedTensor};

struct CountingAllocator;

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);

// SAFETY: both hooks forward to the system allocator with the same layout, so
// the allocator contract is unchanged; the counters are plain atomics.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Heap allocations and bytes performed by `f`.
fn count_allocations<R>(f: impl FnOnce() -> R) -> (R, u64, u64) {
    COUNTING.store(true, Ordering::Relaxed);
    ALLOCATIONS.store(0, Ordering::Relaxed);
    ALLOCATED_BYTES.store(0, Ordering::Relaxed);
    let value = f();
    COUNTING.store(false, Ordering::Relaxed);
    (
        value,
        ALLOCATIONS.load(Ordering::Relaxed),
        ALLOCATED_BYTES.load(Ordering::Relaxed),
    )
}

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::from_typed::<f64>(TypedTensor::from_vec_col_major(shape, data).unwrap())
}

fn traced(shape: &[usize], data: Vec<f64>) -> TracedTensor {
    TracedTensor::from_tensor_concrete_shape(f64_tensor(shape.to_vec(), data)).unwrap()
}

fn ad_context() -> AdContext {
    AdContext::builder()
        .with_semantic_extension_rules(tenferro_linalg::semantic_ad_rules().unwrap())
        .unwrap()
        .build()
        .unwrap()
}

/// The diagonal-dominant matrix the benchmark's `grad_sum_*` rows use.
fn well_conditioned_matrix(n: usize, seed: u64) -> Vec<f64> {
    let mut data = vec![0.0; n * n];
    for col in 0..n {
        for row in 0..n {
            let mixed = (row as u64)
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add((col as u64).wrapping_mul(1_442_695_040_888_963_407))
                .wrapping_add(seed);
            let mut value = (mixed % 2048) as f64 / 2048.0 - 0.5;
            value *= 0.05;
            if row == col {
                value += 2.0 + row as f64 / n as f64;
            }
            data[row + n * col] = value;
        }
    }
    data
}

fn scalar_one() -> TracedTensor {
    traced(&[], vec![1.0])
}

fn probe_threads() -> Vec<usize> {
    std::env::var("TENFERRO_PROBE_THREADS")
        .ok()
        .map(|value| {
            value
                .split(',')
                .filter_map(|part| part.trim().parse::<usize>().ok())
                .filter(|&threads| threads > 0)
                .collect()
        })
        .filter(|list: &Vec<usize>| !list.is_empty())
        .unwrap_or_else(|| vec![1, 4])
}

fn probe_runs() -> usize {
    std::env::var("TENFERRO_PROBE_RUNS")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|&runs| runs > 0)
        .unwrap_or(15)
}

fn probe_iterations() -> usize {
    std::env::var("TENFERRO_PROBE_ITERATIONS")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|&iterations| iterations > 0)
        .unwrap_or(50)
}

fn probe_allocations() -> bool {
    std::env::var_os("TENFERRO_PROBE_ALLOCS").is_some()
}

/// One engine of the requested width; width one is the pool-free serial engine.
fn engine(threads: usize) -> CpuEngine {
    if threads == 1 {
        return CpuEngine::serial();
    }
    assert_eq!(
        std::env::var("RAYON_NUM_THREADS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok()),
        None,
        "an ambient RAYON_NUM_THREADS would size a hidden pool; unset it"
    );
    CpuEngine::new(
        tenferro_cpu::thread_pool_builder()
            .num_threads(threads)
            .build()
            .unwrap(),
    )
}

fn runtime_for(engine: &CpuEngine) -> Runtime {
    let mut builder = Runtime::builder();
    builder
        .register_engine(tenferro_cpu::runtime_engine_registration(engine).unwrap())
        .unwrap();
    builder
        .install_extension_module(
            tenferro_linalg::extension_module::<CpuEngine>(
                tenferro_cpu::runtime_engine_id().unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
    builder.build().unwrap()
}

fn median(samples: &mut [Duration]) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

/// One batch of `iterations` prepared runs, averaged per run.
fn run_batch(
    runtime: &Runtime,
    engine: &mut CpuEngine,
    prepared: &Prepared,
    iterations: usize,
) -> Duration {
    let start = Instant::now();
    for _ in 0..iterations {
        let outputs = runtime
            .run(&mut [&mut *engine], prepared, &[])
            .expect("prepared run");
        black_box(&outputs);
    }
    start.elapsed() / iterations as u32
}

/// Time `runs` batches of `iterations` prepared runs.
fn time_run(
    runtime: &Runtime,
    engine: &mut CpuEngine,
    prepared: &Prepared,
    runs: usize,
    iterations: usize,
    allocations: bool,
) -> (Duration, Duration, Option<(f64, f64)>) {
    for _ in 0..runs.min(3) {
        let _ = run_batch(runtime, engine, prepared, iterations);
    }
    let counts = allocations.then(|| {
        let (_, allocations, bytes) =
            count_allocations(|| run_batch(runtime, engine, prepared, iterations));
        (
            allocations as f64 / iterations as f64,
            bytes as f64 / iterations as f64,
        )
    });
    let mut samples: Vec<Duration> = (0..runs)
        .map(|_| run_batch(runtime, engine, prepared, iterations))
        .collect();
    let best = *samples.iter().min().unwrap();
    let median = median(&mut samples);
    (median, best, counts)
}

struct Report<'a> {
    name: &'a str,
    threads: usize,
    commands: usize,
    regions: (usize, usize),
    allocations: Option<(f64, f64)>,
    median: Duration,
    best: Duration,
}

fn report(report: Report<'_>) {
    let Report {
        name,
        threads,
        commands,
        regions,
        allocations,
        median,
        best,
    } = report;
    let allocations = allocations.map_or_else(String::new, |(count, bytes)| {
        format!(" allocs/run={count:.1} bytes/run={bytes:.0}")
    });
    println!(
        "{name:<26} t={threads} commands={commands:<4} fused={} fallback={}{allocations} median={:>8.3}us best={:>8.3}us per_command={:>6.3}us",
        regions.0,
        regions.1,
        median.as_secs_f64() * 1e6,
        best.as_secs_f64() * 1e6,
        best.as_secs_f64() * 1e6 / commands.max(1) as f64,
    );
}

fn prepare(engine: &CpuEngine, runtime: &Runtime, outputs: &[&TracedTensor]) -> Prepared {
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_many(outputs).unwrap();
    runtime
        .prepare(
            &program,
            &[],
            &[tenferro_runtime::ExecutionEngine::target(engine)],
        )
        .unwrap()
}

fn main() {
    let runs = probe_runs();
    let iterations = probe_iterations();
    let allocations = probe_allocations();
    let ad = ad_context();

    for threads in probe_threads() {
        let mut engine = engine(threads);
        assert_eq!(engine.num_threads(), threads.max(1));
        let runtime = runtime_for(&engine);

        // The benchmark's `small/grad_sum_qr_vjp` arm: qr, a summed loss over q
        // and r, and the VJP with respect to the matrix.
        for n in [2usize, 4, 8] {
            let a = traced(&[n, n], well_conditioned_matrix(n, 4));
            let (q, r) = a.qr().unwrap();
            let loss = (&q.reduce_sum(Some(&[0, 1])).unwrap()
                + &r.reduce_sum(Some(&[0, 1])).unwrap())
                .unwrap();
            let derivative = ad.vjp(&loss, &a, &scalar_one()).unwrap();
            let prepared = prepare(&engine, &runtime, &[&loss, &derivative]);
            let commands = prepared.execution_command_count();
            let (median, best, allocations) = time_run(
                &runtime,
                &mut engine,
                &prepared,
                runs,
                iterations,
                allocations,
            );
            let regions = prepared.elementwise_region_execution_counts();
            report(Report {
                name: &format!("grad_sum_qr_vjp/{n}x{n}"),
                threads,
                commands,
                regions,
                allocations,
                median,
                best,
            });
        }

        // An elementwise chain: per-command cost with no vendor extension, and
        // the arm where the planner offers a fused region that the CPU declines
        // below its element floor.
        for length in [1usize, 10, 50, 200] {
            let mut value = traced(&[8], vec![0.5; 8]);
            for _ in 0..length {
                value = value.exp().unwrap();
            }
            let prepared = prepare(&engine, &runtime, &[&value]);
            let commands = prepared.execution_command_count();
            let summary = prepared.elementwise_region_summary();
            let (median, best, allocations) = time_run(
                &runtime,
                &mut engine,
                &prepared,
                runs,
                iterations,
                allocations,
            );
            let regions = prepared.elementwise_region_execution_counts();
            report(Report {
                name: &format!("exp_chain/{length}[{}r/{}i]", summary.0, summary.1),
                threads,
                commands,
                regions,
                allocations,
                median,
                best,
            });
        }

        // The same elementwise work executed eagerly per operation on the same
        // engine: the reference the prepared path is compared against.
        for length in [1usize, 10, 50, 200] {
            let mut value =
                Tensor::from_vec_col_major(vec![8], vec![0.5; 8]).expect("benchmark tensor");
            let mut batch = |iterations: usize| {
                let start = Instant::now();
                for _ in 0..iterations {
                    for _ in 0..length {
                        value = value.exp(&mut engine).expect("eager exp");
                    }
                    black_box(&value);
                }
                start.elapsed() / iterations as u32
            };
            for _ in 0..runs.min(3) {
                let _ = batch(iterations);
            }
            let mut samples: Vec<Duration> = (0..runs).map(|_| batch(iterations)).collect();
            let best = *samples.iter().min().unwrap();
            let median = median(&mut samples);
            report(Report {
                name: &format!("eager_exp_chain/{length}"),
                threads,
                commands: length,
                regions: (0, 0),
                allocations: None,
                median,
                best,
            });
        }
    }
}
