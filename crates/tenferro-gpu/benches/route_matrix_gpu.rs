//! GPU execution-route matrix: the coexisting CUDA entry mechanisms on one
//! logical contraction, reported separately for enqueue and synchronized
//! completion.
//!
//! Issue #1926 / umbrella #1929, reshaped by #2206: the engine owns its stream,
//! so the operation route is `dot_general_read` called on the `&mut CudaEngine`
//! itself; the before-only `oneshot` numbers stay in
//! `docs/testing/session-route-baseline.json`.
//!
//! Timing modes. The umbrella requires GPU enqueue cost and synchronized
//! completion cost to be reported distinctly, because a per-call "round trip"
//! number mixes launch cost with `synchronize` cost:
//!
//! * `round_trip` — one operation, then `synchronize`, per iteration.
//! * `enqueue_batch16` — 16 operations enqueued, one `synchronize`, timed as a
//!   whole and divided by 16. The batch bounds the outstanding launch queue so
//!   the measurement cannot queue unbounded work.
//!
//! `sync_empty` measures `synchronize` with nothing outstanding, so a reader can
//! subtract it from `round_trip` to recover enqueue-plus-completion latency.
//!
//! Every arm uploads its operands before the timed region and validates the
//! result outside it.

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tenferro_gpu::cuda::{download_tensor, gpu_available, upload_tensor, CudaDeviceId, CudaEngine};
use tenferro_runtime::{
    DType, ExecutionEngine, GraphCompiler, Output, Prepared, Runtime, TracedTensor,
};
use tenferro_tensor::{DotGeneralConfig, Tensor, TensorDot, TensorRead, TensorScalar, TensorWrite};

const SIZES: &[usize] = &[8, 64, 256];
const BATCH: usize = 16;

fn matmul_config() -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: [1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: [].as_slice().into(),
        rhs_batch_dims: [].as_slice().into(),
    }
}

fn matrix<T>(rows: usize, cols: usize) -> Tensor
where
    T: TensorScalar + From<f32>,
{
    let len = rows * cols;
    let data = (0..len)
        .map(|index| T::from(((index % 97) as f32 + 1.0) * 0.001))
        .collect();
    Tensor::from_vec_col_major(vec![rows, cols], data).expect("benchmark matrix")
}

/// Engine route: the kernel issued on the engine's stream.
fn session(backend: &mut CudaEngine, lhs: &Tensor, rhs: &Tensor, config: &DotGeneralConfig) {
    backend
        .dot_general_read(
            TensorRead::from_tensor(lhs),
            TensorRead::from_tensor(rhs),
            config,
        )
        .expect("engine dot_general should succeed");
}

/// Owned-destination session route: the same contraction written into a
/// caller-owned device tensor.
///
/// This measures the raw cuTENSOR write path whose destination resolves
/// through `write_device_ptr` (#1949), alongside the allocating read route
/// above; the two also differ in whether the destination allocation is reused
/// across iterations and how the result is validated.
fn session_into(
    backend: &mut CudaEngine,
    lhs: &Tensor,
    rhs: &Tensor,
    config: &DotGeneralConfig,
    out: &mut Tensor,
) {
    backend
        .dot_general_read_into(
            TensorRead::from_tensor(lhs),
            TensorRead::from_tensor(rhs),
            config,
            TensorWrite::from_tensor(out),
        )
        .expect("engine dot_general_read_into should succeed");
}

fn bench_size<T>(c: &mut Criterion, label: &str, size: usize)
where
    T: TensorScalar + From<f32> + PartialEq + std::fmt::Debug + 'static,
{
    let mut backend = CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("CUDA backend");
    let lhs = upload_tensor(backend.runtime(), &matrix::<T>(size, size)).expect("upload lhs");
    let rhs = upload_tensor(backend.runtime(), &matrix::<T>(size, size)).expect("upload rhs");
    let config = matmul_config();

    // Validation outside the timed region, on both routes.
    let device_out = backend
        .dot_general_read(
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &config,
        )
        .expect("validation dot_general");
    backend.synchronize().expect("validation sync");
    assert_eq!(device_out.shape(), &[size, size], "{label}: output shape");
    session(&mut backend, &lhs, &rhs, &config);
    backend.synchronize().expect("validation sync");

    let mut group = c.benchmark_group(format!("route_matrix_gpu/dot_general_{label}"));
    group.bench_with_input(
        BenchmarkId::new("session/round_trip", size),
        &size,
        |bench, _| {
            bench.iter(|| {
                session(&mut backend, black_box(&lhs), black_box(&rhs), &config);
                backend.synchronize().expect("sync");
            });
        },
    );
    group.bench_with_input(
        BenchmarkId::new("session/enqueue_batch16", size),
        &size,
        |bench, _| {
            bench.iter(|| {
                for _ in 0..BATCH {
                    session(&mut backend, black_box(&lhs), black_box(&rhs), &config);
                }
                backend.synchronize().expect("sync");
            });
        },
    );
    group.finish();

    // Owned-destination route on the same operands. Validate its output
    // against the allocating route before timing.
    let expected = download_tensor(backend.runtime(), &device_out).expect("download reference");
    let mut out = device_out;
    session_into(&mut backend, &lhs, &rhs, &config, &mut out);
    backend.synchronize().expect("validation sync");
    let actual = download_tensor(backend.runtime(), &out).expect("download owned output");
    assert_eq!(
        actual.shape(),
        expected.shape(),
        "{label}: owned output shape"
    );
    assert_eq!(
        actual.as_slice::<T>().unwrap(),
        expected.as_slice::<T>().unwrap(),
        "{label}: owned-destination result"
    );
    let mut group = c.benchmark_group(format!("route_matrix_gpu/dot_general_into_{label}"));
    group.bench_with_input(
        BenchmarkId::new("session/round_trip", size),
        &size,
        |bench, _| {
            bench.iter(|| {
                session_into(
                    &mut backend,
                    black_box(&lhs),
                    black_box(&rhs),
                    &config,
                    black_box(&mut out),
                );
                backend.synchronize().expect("sync");
            });
        },
    );
    group.bench_with_input(
        BenchmarkId::new("session/enqueue_batch16", size),
        &size,
        |bench, _| {
            bench.iter(|| {
                for _ in 0..BATCH {
                    session_into(
                        &mut backend,
                        black_box(&lhs),
                        black_box(&rhs),
                        &config,
                        black_box(&mut out),
                    );
                }
                backend.synchronize().expect("sync");
            });
        },
    );
    group.finish();
}

fn bench_sync_empty(c: &mut Criterion) {
    let mut backend = CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("CUDA backend");
    let mut group = c.benchmark_group("route_matrix_gpu/sync_empty");
    group.bench_function("synchronize", |bench| {
        bench.iter(|| {
            backend.synchronize().expect("sync");
        });
    });
    group.finish();
}

/// Prepared compiled-graph execution on one engine (#2182, #2206).
///
/// `prepared_enqueue` measures one `Runtime::run` per iteration: the engine keeps
/// the run pending until its bound (or a synchronize) retires it, so this is the
/// per-call admission and launch cost. `prepared_synchronized` adds the engine
/// synchronize per call, the completed round trip.
fn bench_prepared(
    c: &mut Criterion,
    group_name: &str,
    backend: &mut CudaEngine,
    runtime: &Runtime,
    prepared: &Prepared,
    input: &Tensor,
) {
    let mut group = c.benchmark_group(group_name);
    group.bench_function("prepared_enqueue", |bench| {
        bench.iter(|| {
            let outputs = runtime
                .run(
                    &mut [&mut *backend],
                    black_box(prepared),
                    &[TensorRead::from_tensor(black_box(input))],
                )
                .expect("prepared run");
            black_box(outputs);
        });
    });
    group.bench_function("prepared_synchronized", |bench| {
        bench.iter(|| {
            let outputs = runtime
                .run(
                    &mut [&mut *backend],
                    black_box(prepared),
                    &[TensorRead::from_tensor(black_box(input))],
                )
                .expect("prepared run");
            black_box(outputs);
            backend.synchronize().expect("sync");
        });
    });
    group.finish();
}

fn first_output<'o>(outputs: &'o [Output<'_>]) -> &'o Tensor {
    match outputs.first().expect("one output") {
        Output::Owned(tensor) => tensor,
        Output::Borrowed(read) => read.as_tensor().expect("dense output"),
    }
}

fn engine_runtime(backend: &CudaEngine) -> Runtime {
    let mut builder = Runtime::builder();
    builder
        .register_engine(
            tenferro_gpu::cuda::cuda_runtime_engine_registration(backend)
                .expect("cuda registration"),
        )
        .expect("register engine");
    builder.build().expect("runtime")
}

/// A small fused elementwise chain: the cost is admission and launch, not kernel work.
fn prepared_routes(c: &mut Criterion) {
    let mut backend = CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("CUDA backend");
    let runtime = engine_runtime(&backend);

    let x = TracedTensor::input_symbolic_shape(DType::F64, 1).expect("input");
    let squared = (&x * &x).expect("square");
    let y = (&squared + &x).expect("shift");
    let program = GraphCompiler::new()
        .compile_with_input_specs(&y, &[(&x, DType::F64, &[256])])
        .expect("compile");
    let host = Tensor::from_vec_col_major(
        vec![256],
        (0..256).map(|index| (index as f64) * 0.01).collect(),
    )
    .expect("host input");
    let input = upload_tensor(backend.runtime(), &host).expect("upload");
    let prepared = runtime
        .prepare(
            &program,
            &[TensorRead::from_tensor(&input)],
            &[backend.target()],
        )
        .expect("prepare");

    // Validate outside the timed region.
    let outputs = runtime
        .run(
            &mut [&mut backend],
            &prepared,
            &[TensorRead::from_tensor(&input)],
        )
        .expect("validation run");
    let output = first_output(&outputs);
    let values = download_tensor(backend.runtime(), output).expect("download");
    drop(outputs);
    let expected: Vec<f64> = host
        .as_slice::<f64>()
        .expect("host values")
        .iter()
        .map(|value| value * value + value)
        .collect();
    // The fused kernel may contract to an FMA, so compare within rounding.
    for (actual, expected) in values
        .as_slice::<f64>()
        .expect("host values")
        .iter()
        .zip(&expected)
    {
        assert!(
            (actual - expected).abs() <= 1e-12,
            "prepared run: {actual} vs {expected}"
        );
    }

    bench_prepared(
        c,
        "route_matrix_gpu/prepared_fused",
        &mut backend,
        &runtime,
        &prepared,
        &input,
    );

    prepared_matmul_chain(c);
}

/// Prepared compiled-graph execution of an unfused eight-node matmul chain.
///
/// The fused elementwise row above runs a whole region as one scheduled node, so
/// it cannot show a per-node cost. This chain schedules eight `dot_general`
/// nodes (#2182 B3).
fn prepared_matmul_chain(c: &mut Criterion) {
    const SIZE: usize = 16;
    const MATMULS: usize = 8;

    let mut backend = CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("CUDA backend");
    let runtime = engine_runtime(&backend);

    let x = TracedTensor::input_symbolic_shape(DType::F64, 2).expect("input");
    let mut y = x.matmul(&x).expect("first matmul");
    for _ in 1..MATMULS {
        y = y.matmul(&x).expect("chained matmul");
    }
    let program = GraphCompiler::new()
        .compile_with_input_specs(&y, &[(&x, DType::F64, &[SIZE, SIZE])])
        .expect("compile");
    let host = Tensor::from_vec_col_major(
        vec![SIZE, SIZE],
        (0..SIZE * SIZE)
            .map(|index| ((index % 17) as f64 + 1.0) * 0.01)
            .collect(),
    )
    .expect("host input");
    let input = upload_tensor(backend.runtime(), &host).expect("upload");
    let prepared = runtime
        .prepare(
            &program,
            &[TensorRead::from_tensor(&input)],
            &[backend.target()],
        )
        .expect("prepare");

    let outputs = runtime
        .run(
            &mut [&mut backend],
            &prepared,
            &[TensorRead::from_tensor(&input)],
        )
        .expect("validation run");
    let output = first_output(&outputs);
    assert_eq!(output.shape(), &[SIZE, SIZE]);
    drop(outputs);

    bench_prepared(
        c,
        "route_matrix_gpu/prepared_matmul_chain",
        &mut backend,
        &runtime,
        &prepared,
        &input,
    );
}

fn route_matrix_gpu(c: &mut Criterion) {
    if !gpu_available() {
        eprintln!("skipping GPU route matrix: no CUDA device available");
        return;
    }
    for &size in SIZES {
        bench_size::<f32>(c, "f32", size);
        bench_size::<f64>(c, "f64", size);
    }
    prepared_routes(c);
    bench_sync_empty(c);
}

fn criterion_config() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(5))
        .sample_size(100)
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = route_matrix_gpu
}
criterion_main!(benches);
