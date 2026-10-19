//! Whole-arm timing tool for the prepared, caller-held-session path (#2087).
//!
//! This measures the host-visible cost of the call shape behind
//! `dense_batched_matmul_f64_b16_32`: a zero-input prepared graph (two bound GPU
//! constants, one `dot_general`) run through `Runtime::run_prepared_in` inside
//! one caller-held CUDA session. Every arm below runs in one process, so a
//! reader compares arms under the same host conditions instead of across
//! sessions:
//!
//! * the prepared call;
//! * the same contraction through the session with no graph machinery, for owned
//!   operands (cuBLAS strided-batched) and for view operands (cuTENSOR, which is
//!   what a prepared graph's bound constants reach);
//! * the enqueue arm the batched-GEMM fast path uses
//!   (`batched_gemm::run_allocating`: it resolves three operands and reads the
//!   per-stream cuBLAS handle, so it is a whole-arm floor, not a raw cuBLAS API
//!   measurement), with and without the runtime's synchronize;
//! * one empty kernel launch plus the same synchronize;
//! * the two host/device-thread hand-offs a prepared call can pay: a CubeCL
//!   `flush` and a raw `cuStreamSynchronize`.
//!
//! What it does *not* do: it does not attribute time to internal phases. The
//! differences printed at the end are whole-arm deltas; a per-phase census needs
//! instrumentation inside the runtime and the backend, which is a separate
//! experiment and is not reproduced here. The test asserts correctness only (all
//! arms against a CPU f64 reference, with a residual) — never timing, because a
//! shared A100 host is not a stable clock.
//!
//! It stays in the tree as the reproducible instrument for that case. Run with
//! `cargo test -p tenferro-gpu --features cuda --lib -- --ignored host_path_phases
//! --nocapture`.

use std::time::{Duration, Instant};

use tenferro_runtime::{EngineId, GraphCompiler, Runtime, TracedTensor};
use tenferro_tensor::backend::BackendSessionHost as _;
use tenferro_tensor::{DotGeneralConfig, Tensor, TensorRead, TensorWrite};

use super::super::interop::fill_zero_write;
use super::super::{batched_gemm, dispatch, CudaBackend, CudaDeviceId};
use super::{cpu_backend, tensor_f64};

const CALLS: usize = 400;
const WARMUPS: usize = 3;
const BATCH: usize = 16;
const EXTENT: usize = 32;
/// Cross-provider comparison tolerance for the f64 reference below: a 32-term
/// dot product of order-one values, so a different reduction order may move the
/// last bits. Comparing bit patterns would fail on another library version.
const RTOL: f64 = 1e-12;
const ATOL: f64 = 1e-12;

fn batched_matmul_config() -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: [1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: [2].as_slice().into(),
        rhs_batch_dims: [2].as_slice().into(),
    }
}

fn normal_data(len: usize, seed: f64) -> Vec<f64> {
    // Deterministic spread without a dependency: distinct magnitudes keep a
    // wrong contraction order from cancelling to the same value.
    (0..len)
        .map(|index| (index as f64 * 0.618_033_988_75 + seed).sin() * 0.5)
        .collect()
}

fn cpu_dot_general(lhs: &Tensor, rhs: &Tensor, config: &DotGeneralConfig) -> Tensor {
    let mut cpu = cpu_backend();
    cpu.with_backend_session(|session| {
        session.dot_general_read(
            TensorRead::from_tensor(lhs),
            TensorRead::from_tensor(rhs),
            config,
        )
    })
    .expect("CPU session")
    .expect("CPU dot_general")
}

/// Largest absolute difference, the quantity the tolerance is checked against.
fn max_abs_diff(actual: &Tensor, expected: &Tensor) -> f64 {
    assert_eq!(actual.shape(), expected.shape());
    actual
        .as_slice::<f64>()
        .expect("f64 arm")
        .iter()
        .zip(expected.as_slice::<f64>().expect("f64 reference").iter())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f64, f64::max)
}

fn assert_matches_reference(name: &str, actual: &Tensor, expected: &Tensor) {
    let tolerance = ATOL
        + RTOL
            * expected
                .as_slice::<f64>()
                .expect("f64")
                .iter()
                .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let residual = max_abs_diff(actual, expected);
    assert!(
        residual <= tolerance,
        "{name}: max abs residual {residual:e} exceeds tolerance {tolerance:e}"
    );
}

fn summarize(name: &str, mut samples: Vec<Duration>) -> (String, f64) {
    samples.sort_unstable();
    let median = samples[samples.len() / 2].as_secs_f64() * 1e6;
    let min = samples[0].as_secs_f64() * 1e6;
    let max = samples[samples.len() - 1].as_secs_f64() * 1e6;
    let mean = samples.iter().map(|d| d.as_secs_f64()).sum::<f64>() / samples.len() as f64 * 1e6;
    (
        format!("{name:<40} median {median:8.3} us  min {min:8.3}  mean {mean:8.3}  max {max:8.3}"),
        median,
    )
}

fn measure(mut call: impl FnMut()) -> Vec<Duration> {
    for _ in 0..WARMUPS {
        call();
    }
    (0..CALLS)
        .map(|_| {
            let start = Instant::now();
            call();
            start.elapsed()
        })
        .collect()
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn host_path_phases() {
    let mut backend: CudaBackend =
        CudaBackend::new(CudaDeviceId::from_ordinal(0)).expect("CUDA backend");
    let mut builder = Runtime::builder();
    builder
        .register_engine(
            crate::cuda::cuda_runtime_engine_registration(
                &backend,
                EngineId::new("tenferro-cuda.host-path-phases.v1").expect("engine id"),
            )
            .expect("cuda registration"),
        )
        .expect("register engine");
    let runtime = builder.build().expect("runtime");
    let rt = backend.runtime().clone();

    // The benchmark's trace arm: both operands are GPU-resident constants, so
    // the prepared graph takes no run-time inputs.
    let lhs_host = tensor_f64(
        vec![EXTENT, EXTENT, BATCH],
        normal_data(EXTENT * EXTENT * BATCH, 0.5),
    );
    let rhs_host = tensor_f64(
        vec![EXTENT, EXTENT, BATCH],
        normal_data(EXTENT * EXTENT * BATCH, 1.5),
    );
    let config = batched_matmul_config();
    let reference = cpu_dot_general(&lhs_host, &rhs_host, &config);

    let lhs = super::super::upload_tensor(&rt, &lhs_host).expect("upload lhs");
    let rhs = super::super::upload_tensor(&rt, &rhs_host).expect("upload rhs");
    let lhs_leaf = super::super::upload_tensor(&rt, &lhs_host).expect("upload lhs leaf");
    let rhs_leaf = super::super::upload_tensor(&rt, &rhs_host).expect("upload rhs leaf");
    let lhs_traced = TracedTensor::from_tensor_concrete_shape(lhs_leaf).expect("lhs leaf");
    let rhs_traced = TracedTensor::from_tensor_concrete_shape(rhs_leaf).expect("rhs leaf");
    let out_traced = lhs_traced
        .dot_general(&rhs_traced, config.clone())
        .expect("dot_general");
    let program = GraphCompiler::new()
        .compile_many(&[&out_traced])
        .expect("compile");
    let prepared = runtime.prepare_compiled(&program, &[]).expect("prepare");
    println!(
        "prepared: regions {:?} submissions {}",
        prepared.elementwise_region_summary(),
        prepared.execution_submission_count()
    );

    let mut session = backend.open_session().expect("held session");
    let full = session
        .with_session(|view| runtime.run_prepared_in(view, &prepared, &[]))
        .expect("held callback")
        .expect("prepared run");
    let prepared_out = super::super::download_tensor(&rt, &full[0]).expect("download");
    drop(full);
    assert_matches_reference("prepared run", &prepared_out, &reference);

    let owned_out = session
        .with_session(|view| {
            view.dot_general_read(
                TensorRead::from_tensor(&lhs),
                TensorRead::from_tensor(&rhs),
                &config,
            )
        })
        .expect("held callback")
        .expect("session dot_general");
    let owned_out = super::super::download_tensor(&rt, &owned_out).expect("download");
    assert_matches_reference("session dot_general (owned)", &owned_out, &reference);

    let lhs_view = TensorRead::from_tensor(&lhs).tensor_view();
    let rhs_view = TensorRead::from_tensor(&rhs).tensor_view();
    let view_out = session
        .with_session(|view| {
            view.dot_general_read(
                TensorRead::from_view(lhs_view.clone()),
                TensorRead::from_view(rhs_view.clone()),
                &config,
            )
        })
        .expect("held callback")
        .expect("session dot_general");
    let view_out = super::super::download_tensor(&rt, &view_out).expect("download");
    assert_matches_reference("session dot_general (view)", &view_out, &reference);

    // The cuBLAS enqueue arm: exactly what the batched-GEMM fast path issues for
    // the same algebra, with a preallocated destination and no scheduler.
    let args = batched_gemm::BatchedGemmLaunch::detect(lhs.shape(), rhs.shape(), &config)
        .expect("the 16x32x32 f64 batched matmul must be cuBLAS-addressable");
    let out_tensor = Tensor::from_typed::<f64>(
        dispatch::alloc_output::<f64>(&rt, &[EXTENT, EXTENT, BATCH]).expect("arm output"),
    );
    let lhs_typed = lhs.as_typed::<f64>().expect("lhs f64");
    let rhs_typed = rhs.as_typed::<f64>().expect("rhs f64");
    let out_typed = out_tensor.as_typed::<f64>().expect("out f64");
    let enqueue_gemm = || {
        batched_gemm::run_allocating(&rt, lhs_typed, rhs_typed, out_typed, &args)
            .expect("batched gemm");
    };
    enqueue_gemm();
    rt.synchronize().expect("sync");
    let enqueue_out = super::super::download_tensor(&rt, &out_tensor).expect("download");
    assert_matches_reference("cuBLAS enqueue arm", &enqueue_out, &reference);

    // Empty-kernel arm: one minimal kernel through the CubeCL dispatch path plus
    // the same synchronize, as the launch floor for a queued kernel.
    let mut tiny_tensor =
        Tensor::from_typed::<f64>(dispatch::alloc_output::<f64>(&rt, &[1]).expect("tiny output"));
    let mut empty_kernel = || {
        fill_zero_write(&rt, TensorWrite::from_tensor(&mut tiny_tensor)).expect("empty kernel");
        rt.synchronize().expect("sync");
    };
    empty_kernel();

    // Per-call hand-off census, printed for the record (not asserted): a
    // prepared call should issue one blocking CubeCL dispatch and one stream
    // barrier, and no scheduler events on a coalesced single-domain schedule.
    let raw_stream =
        rt.raw_cuda_stream().expect("raw stream") as usize as cudarc::driver::sys::CUstream;
    let print_census = |label: &str, call: &mut dyn FnMut()| {
        rt.reset_scheduler_counters();
        super::super::runtime::reset_blocking_flush_handoffs();
        call();
        println!(
            "{label:<26} flush_handoffs {} scheduler_fences {} event_records {} retirements {}",
            super::super::runtime::blocking_flush_handoffs(),
            rt.scheduler_fences(),
            rt.scheduler_event_records(),
            rt.scheduler_retirements(),
        );
    };
    print_census("prepared run_prepared_in", &mut || {
        let outputs = session
            .with_session(|view| runtime.run_prepared_in(view, &prepared, &[]))
            .expect("held callback")
            .expect("prepared run");
        std::hint::black_box(outputs);
    });
    print_census("session dot_general (owned)", &mut || {
        let out = session
            .with_session(|view| {
                view.dot_general_read(
                    TensorRead::from_tensor(&lhs),
                    TensorRead::from_tensor(&rhs),
                    &config,
                )
            })
            .expect("held callback")
            .expect("session dot_general");
        std::hint::black_box(out);
    });

    // Timed arms.
    let (line_full, t_full) = summarize(
        "prepared run_prepared_in (harness)",
        measure(|| {
            let outputs = session
                .with_session(|view| runtime.run_prepared_in(view, &prepared, &[]))
                .expect("held callback")
                .expect("prepared run");
            std::hint::black_box(outputs);
        }),
    );
    let (line_entry, t_entry) = summarize(
        "held-session entry only",
        measure(|| {
            session.with_session(|_| ()).expect("held callback");
        }),
    );
    let (line_sync, t_sync) = summarize(
        "runtime.synchronize, empty stream",
        measure(|| rt.synchronize().expect("sync")),
    );
    let (line_owned, t_owned) = summarize(
        "session dot_general (no graph)",
        measure(|| {
            let out = session
                .with_session(|view| {
                    view.dot_general_read(
                        TensorRead::from_tensor(&lhs),
                        TensorRead::from_tensor(&rhs),
                        &config,
                    )
                })
                .expect("held callback")
                .expect("session dot_general");
            std::hint::black_box(out);
        }),
    );
    let (line_owned_sync, t_owned_sync) = summarize(
        "session dot_general + synchronize",
        measure(|| {
            let out = session
                .with_session(|view| {
                    view.dot_general_read(
                        TensorRead::from_tensor(&lhs),
                        TensorRead::from_tensor(&rhs),
                        &config,
                    )
                })
                .expect("held callback")
                .expect("session dot_general");
            std::hint::black_box(out);
            rt.synchronize().expect("sync");
        }),
    );
    let (line_views, t_views) = summarize(
        "session dot_general, view operands",
        measure(|| {
            let out = session
                .with_session(|view| {
                    view.dot_general_read(
                        TensorRead::from_view(lhs_view.clone()),
                        TensorRead::from_view(rhs_view.clone()),
                        &config,
                    )
                })
                .expect("held callback")
                .expect("session dot_general");
            std::hint::black_box(out);
        }),
    );
    let (line_enqueue, t_enqueue) = summarize(
        "cuBLAS enqueue arm, no synchronize",
        measure(|| {
            enqueue_gemm();
        }),
    );
    let (line_enqueue_sync, t_enqueue_sync) = summarize(
        "cuBLAS enqueue arm + synchronize",
        measure(|| {
            enqueue_gemm();
            rt.synchronize().expect("sync");
        }),
    );
    let (line_flush, t_flush) = summarize(
        "cubecl flush (device-thread hand-off)",
        measure(|| rt.flush_cubecl("phase_bench").expect("flush")),
    );
    let (line_raw_sync, t_raw_sync) = summarize(
        "raw cuStreamSynchronize only",
        measure(|| {
            unsafe { cudarc::driver::result::stream::synchronize(raw_stream) }
                .expect("stream sync");
        }),
    );
    let (line_empty, t_empty) = summarize(
        "empty kernel + synchronize",
        measure(|| {
            empty_kernel();
        }),
    );
    let (line_alloc, _) = summarize(
        "output alloc + drop",
        measure(|| {
            let out = dispatch::alloc_output::<f64>(&rt, &[EXTENT, EXTENT, BATCH]).expect("alloc");
            std::hint::black_box(&out);
        }),
    );

    session.close().expect("close session");

    println!("\n=== whole-arm timings, 16 x 32x32 f64 batched matmul, {CALLS} calls ===");
    for line in [
        line_full,
        line_entry,
        line_sync,
        line_owned,
        line_owned_sync,
        line_views,
        line_enqueue,
        line_enqueue_sync,
        line_flush,
        line_raw_sync,
        line_empty,
        line_alloc,
    ] {
        println!("{line}");
    }
    // Whole-arm deltas only. Each arm differs from its neighbour in more than
    // one respect (session entry, allocation, operand resolution, handle lookup,
    // retirement), so none of these is a phase attribution.
    println!("\nwhole-arm deltas (not phase attribution):");
    println!("  held-session entry arm: {t_entry:.3} us");
    println!(
        "  prepared call - (cuBLAS enqueue arm + synchronize): {:.3} us",
        t_full - t_enqueue_sync
    );
    println!(
        "  prepared call - (session contraction arm + synchronize): {:.3} us",
        t_full - t_owned_sync
    );
    println!(
        "  session contraction arm - cuBLAS enqueue arm: {:.3} us",
        t_owned - t_enqueue
    );
    println!(
        "  synchronizing each repetition vs not: {:.3} us (session arm), {:.3} us (cuBLAS arm)",
        t_owned_sync - t_owned,
        t_enqueue_sync - t_enqueue
    );
    println!(
        "  view-operand (cuTENSOR) - owned-operand (cuBLAS) session arm: {:.3} us",
        t_views - t_owned
    );
    println!(
        "  cubecl flush + raw cuStreamSynchronize arms: {:.3} us",
        t_flush + t_raw_sync
    );
    println!(
        "  empty-kernel arm - bare synchronize arm: {:.3} us",
        t_empty - t_sync
    );
}
