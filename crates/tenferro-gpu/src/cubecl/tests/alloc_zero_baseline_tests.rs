//! CUDA allocation / zero-fill measurement baseline for #1887.
//!
//! The issue needs before-numbers for replacing the current internal
//! allocation-plus-memset path with the upstream scheduler-managed
//! allocate/zero command (tensor4all/cubecl#16). This harness does not change
//! production behavior; it only prints a table when explicitly run on the local
//! A100.
//!
//! Run it with a release build (the table is otherwise dominated by debug
//! checks):
//!
//! ```sh
//! CUBECL_DEBUG_LOG=0 \
//! CUDA_PATH=/usr/local/cuda-12.6 \
//! LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:/usr/lib/x86_64-linux-gnu/libcutensor/12:$LD_LIBRARY_PATH \
//! cargo test --release -j 4 -p tenferro-gpu --features cuda \
//!   alloc_zero_baseline -- --ignored --nocapture
//! ```
//!
//! The host-enqueue column times the operation as it is issued, including the
//! `with_cubecl` exit dispatch, and excludes the final device drain. The
//! synchronized column times each operation followed by
//! [`CudaExecSession::synchronize`], so it is the per-operation round trip.
//! Warmups run before either measurement.
//!
//! `flush` counts `CudaRuntime::flush_cubecl` hand-offs (the only interaction
//! counter the code exposes). It does not see the zero fill's internal
//! `resolve_and_flush` server call, which is a direct CubeCL server request, nor
//! the `with_cubecl` exit `Client::flush`. `pool_hit`/`pool_miss` come from the
//! runtime's device-output pool and show whether an allocation reused cached
//! device storage.

use std::time::Instant;

use tenferro_tensor::{CacheStats, Tensor, TensorWrite};

use crate::cubecl::runtime::{blocking_flush_handoffs, reset_blocking_flush_handoffs};
use crate::cubecl::{upload_tensor, CudaExecSession, CudaRuntime};

use super::{gpu_backend, with_cuda_exec};

/// Sizes requested by #1887.
const BYTES: [usize; 5] = [0, 16, 4 * 1024, 256 * 1024, 16 * 1024 * 1024];

/// Allocations per callback for the "many allocations in one callback" row.
fn allocation_batch(bytes: usize) -> usize {
    if bytes >= 16 * 1024 * 1024 {
        4
    } else {
        16
    }
}

/// Enough samples for small sizes without making the 16 MiB cases expensive.
fn iterations(bytes: usize) -> usize {
    match bytes {
        0 | 16 | 4096 => 1_000,
        262_144 => 500,
        _ => 32,
    }
}

fn empty_callback(session: &mut CudaExecSession<'_>) -> crate::Result<()> {
    session.with_cubecl("bench.alloc_zero.empty", |_| Ok(()))
}

fn alloc_batch(
    session: &mut CudaExecSession<'_>,
    shape: &[usize],
    count: usize,
) -> crate::Result<()> {
    session.with_cubecl("bench.alloc_zero.alloc", |cubecl| {
        let mut batch = Vec::with_capacity(count);
        for _ in 0..count {
            batch.push(cubecl.alloc_output::<f32>(shape)?);
        }
        Ok(())
    })
}

fn alloc_zero_batch(
    session: &mut CudaExecSession<'_>,
    shape: &[usize],
    count: usize,
) -> crate::Result<()> {
    session.with_cubecl("bench.alloc_zero.native", |cubecl| {
        let mut batch = Vec::with_capacity(count);
        for _ in 0..count {
            batch.push(cubecl.alloc_zero_output::<f32>(shape)?);
        }
        Ok(())
    })
}

fn fill_zero_reuse_batch(
    session: &mut CudaExecSession<'_>,
    destination: &mut Tensor,
    count: usize,
) -> crate::Result<()> {
    session.with_cubecl("bench.alloc_zero.fill_reuse", |cubecl| {
        for _ in 0..count {
            cubecl.fill_zero_write(TensorWrite::from_tensor(&mut *destination))?;
        }
        Ok(())
    })
}

fn fill_zero_alloc_batch(
    session: &mut CudaExecSession<'_>,
    shape: &[usize],
    count: usize,
) -> crate::Result<()> {
    session.with_cubecl("bench.alloc_zero.fill_alloc", |cubecl| {
        let mut batch = Vec::with_capacity(count);
        for _ in 0..count {
            let mut tensor = Tensor::from_typed::<f32>(cubecl.alloc_output::<f32>(shape)?);
            cubecl.fill_zero_write(TensorWrite::from_tensor(&mut tensor))?;
            batch.push(tensor);
        }
        Ok(())
    })
}

fn upload_zero_batch(
    session: &mut CudaExecSession<'_>,
    runtime: &CudaRuntime,
    host_zero: &Tensor,
    count: usize,
) -> crate::Result<()> {
    session.with_cubecl("bench.alloc_zero.upload", |_| {
        let mut batch = Vec::with_capacity(count);
        for _ in 0..count {
            batch.push(upload_tensor(runtime, host_zero)?);
        }
        Ok(())
    })
}

fn pool_stats(session: &CudaExecSession<'_>) -> CacheStats {
    session
        .runtime()
        .output_pool()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .stats()
}

#[allow(clippy::too_many_arguments)]
fn report(
    session: &mut CudaExecSession<'_>,
    bytes: usize,
    elements: usize,
    operation: &str,
    batch: usize,
    warmup: usize,
    iters: usize,
    mut op: impl FnMut(&mut CudaExecSession<'_>) -> crate::Result<()>,
) {
    for _ in 0..warmup {
        op(session).expect("warmup operation");
    }
    session.synchronize().expect("warmup synchronize");

    let pool_before = pool_stats(session);
    reset_blocking_flush_handoffs();
    let started = Instant::now();
    for _ in 0..iters {
        op(session).expect("enqueue operation");
    }
    let enqueue_us = started.elapsed().as_secs_f64() * 1e6 / (iters * batch) as f64;
    let flush_enqueue = blocking_flush_handoffs() as f64 / (iters * batch) as f64;
    let pool_after = pool_stats(session);
    // Drain outside the enqueue measurement so the synchronized loop starts
    // from an idle stream instead of charging the first operation for the whole
    // enqueue batch.
    session.synchronize().expect("drain after enqueue");

    reset_blocking_flush_handoffs();
    let started = Instant::now();
    for _ in 0..iters {
        op(session).expect("synchronized operation");
        session.synchronize().expect("synchronized completion");
    }
    let sync_us = started.elapsed().as_secs_f64() * 1e6 / (iters * batch) as f64;
    let flush_sync = blocking_flush_handoffs() as f64 / (iters * batch) as f64;

    let ops = (iters * batch) as f64;
    let pool_hits = pool_after.hits.saturating_sub(pool_before.hits) as f64 / ops;
    let pool_misses = pool_after.misses.saturating_sub(pool_before.misses) as f64 / ops;
    eprintln!(
        "{bytes:>9} {elements:>9} {operation:<26} {batch:>5} \
         {enqueue_us:>12.2} {sync_us:>12.2} {flush_enqueue:>10.2} {flush_sync:>10.2} \
         {pool_hits:>10.2} {pool_misses:>10.2}"
    );
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU; local A100 baseline for #1887"]
fn alloc_zero_baseline() {
    assert!(
        crate::cubecl::gpu_available(),
        "CUDA baseline requires an available device"
    );
    let mut backend = gpu_backend();
    with_cuda_exec(&mut backend, |session| {
        let warmup = 2;
        eprintln!(
            "[1887 baseline] profile: debug_assertions={} (run with --release for the table)",
            cfg!(debug_assertions)
        );
        eprintln!(
            "{:>9} {:>9} {:<26} {:>5} {:>12} {:>12} {:>10} {:>10} {:>10} {:>10}",
            "bytes",
            "elements",
            "operation",
            "batch",
            "enqueue_us",
            "sync_us",
            "flush_enq",
            "flush_sync",
            "pool_hit",
            "pool_miss"
        );

        for bytes in BYTES {
            let elements = bytes / std::mem::size_of::<f32>();
            let shape = vec![elements];
            let iters = iterations(bytes);
            let batch = allocation_batch(bytes);

            report(
                session,
                bytes,
                elements,
                "empty callback",
                1,
                warmup,
                iters,
                empty_callback,
            );
            report(
                session,
                bytes,
                elements,
                "alloc only x1",
                1,
                warmup,
                iters,
                |session| alloc_batch(session, &shape, 1),
            );
            report(
                session,
                bytes,
                elements,
                "alloc only xN",
                batch,
                warmup,
                iters,
                |session| alloc_batch(session, &shape, batch),
            );
            report(
                session,
                bytes,
                elements,
                "native zero x1",
                1,
                warmup,
                iters,
                |session| alloc_zero_batch(session, &shape, 1),
            );
            report(
                session,
                bytes,
                elements,
                "native zero xN",
                batch,
                warmup,
                iters,
                |session| alloc_zero_batch(session, &shape, batch),
            );

            let mut reuse_destination = Tensor::from_typed::<f32>(
                session
                    .with_cubecl("bench.alloc_zero.reuse_setup", |cubecl| {
                        cubecl.alloc_output::<f32>(&shape)
                    })
                    .expect("reuse destination allocation"),
            );
            report(
                session,
                bytes,
                elements,
                "fill_zero_write reuse x1",
                1,
                warmup,
                iters,
                |session| fill_zero_reuse_batch(session, &mut reuse_destination, 1),
            );
            report(
                session,
                bytes,
                elements,
                "fill_zero_write reuse xN",
                batch,
                warmup,
                iters,
                |session| fill_zero_reuse_batch(session, &mut reuse_destination, batch),
            );
            drop(reuse_destination);

            report(
                session,
                bytes,
                elements,
                "fill_zero_write alloc+fill",
                1,
                warmup,
                iters,
                |session| fill_zero_alloc_batch(session, &shape, 1),
            );

            let host_zero = Tensor::from_vec_col_major(shape.clone(), vec![0.0_f32; elements])
                .expect("host zero");
            let runtime = session.runtime().clone();
            report(
                session,
                bytes,
                elements,
                "host-zero upload x1",
                1,
                warmup,
                iters,
                |session| upload_zero_batch(session, &runtime, &host_zero, 1),
            );
            report(
                session,
                bytes,
                elements,
                "host-zero upload xN",
                batch,
                warmup,
                iters,
                |session| upload_zero_batch(session, &runtime, &host_zero, batch),
            );
        }
    });
}
