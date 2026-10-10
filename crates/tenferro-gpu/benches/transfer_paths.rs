//! Transfer-path measurements for #1945 U4 / #2009.
//!
//! Compares the existing blocking transfer paths with the pinned pending handoffs added by U4
//! slices 1-2, on one CUDA stream with criterion's default single-threaded harness:
//!
//! Payloads are `f64` element counts; the labels are the resulting byte sizes.
//!
//! | case | host copies | destination memory |
//! | --- | --- | --- |
//! | `download_pageable` (`download_tensor`, large payload) | 1 (device → pageable `Vec`) | pageable |
//! | `download_pinned` (`download_pending` + `wait`) | 1 (device → pinned buffer owned by the handle) | pinned |
//! | `download_small` (`download_tensor`, ≤ `PINNED_SCALAR_BYTES`) | 2 (device → pinned staging, staging → `Vec`) | pageable result |
//! | `upload_staging` (`upload_tensor`) | 1 (borrowed host → CubeCL staging) | — |
//! | `upload_pinned` (`upload_pending` + `wait`) | 0 on the host side (device reads the pinned buffer) | — |
//!
//! The pinned cases reuse one buffer across iterations: `wait` hands it back and the next iteration
//! hands it in again, so the measurement covers the copy and its event, not allocation. The
//! `*_alloc` cases measure allocating a fresh pinned buffer, and `download_pinned_wait` isolates the
//! completion wait (`wait` only, with the enqueue in criterion's untimed setup).
//!
//! These are device/PCIe measurements, not an app-level speedup claim: they do not include the
//! caller's own bookkeeping, and a pending transfer's point is that the caller may wait later.

use std::cell::RefCell;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use tenferro_gpu::cuda::{
    download_pending, download_tensor, gpu_available, upload_pending, upload_tensor, CudaBackend,
    CudaDeviceId, PinnedHostBuffer,
};
use tenferro_tensor::Tensor;

/// Payloads, as `(f64 elements, label)`: the pinned-scalar fast path (two elements, the largest
/// payload that takes the <= 16-byte path), a small vector, a mid-size vector, and a large transfer.
const SIZES: &[(usize, &str)] = &[
    (2, "16B"),
    (8, "64B"),
    (4096, "32KiB"),
    (1024 * 1024, "8MiB"),
];

fn backend() -> CudaBackend {
    assert!(
        gpu_available(),
        "transfer measurements require a CUDA device"
    );
    CudaBackend::new(CudaDeviceId::from_ordinal(0)).expect("CUDA backend")
}

fn host_vector(elements: usize) -> Tensor {
    let data: Vec<f64> = (0..elements).map(|index| index as f64 * 0.5).collect();
    Tensor::from_vec_col_major(vec![elements], data).expect("host vector")
}

fn pinned_source(backend: &CudaBackend, elements: usize) -> PinnedHostBuffer {
    let bytes = elements * size_of::<f64>();
    let mut source = PinnedHostBuffer::new(backend.runtime(), bytes).expect("pinned buffer");
    let data: Vec<f64> = (0..elements).map(|index| index as f64 * 0.25).collect();
    let bytes = data
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<u8>>();
    source.as_mut_slice().copy_from_slice(&bytes);
    source
}

fn bench_download(c: &mut Criterion) {
    let backend = backend();
    let runtime = backend.runtime().clone();
    let mut group = c.benchmark_group("transfer_download");

    for &(elements, label) in SIZES {
        let bytes = elements * size_of::<f64>();
        group.throughput(Throughput::Bytes(bytes as u64));
        let device = upload_tensor(&runtime, &host_vector(elements)).expect("device payload");

        group.bench_with_input(
            BenchmarkId::new("pageable", label),
            &device,
            |bencher, device| {
                black_box(download_tensor(&runtime, device).expect("download"));
                bencher.iter(|| black_box(download_tensor(&runtime, device).expect("download")));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("pinned", label),
            &device,
            |bencher, device| {
                let buffer = RefCell::new(Some(
                    PinnedHostBuffer::new(&runtime, bytes).expect("pinned buffer"),
                ));
                bencher.iter(|| {
                    let taken = buffer.borrow_mut().take().expect("reused pinned buffer");
                    let filled = download_pending(&runtime, device, taken)
                        .expect("pending download")
                        .wait()
                        .expect("wait");
                    black_box(filled.as_slice().len());
                    *buffer.borrow_mut() = Some(filled);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("pinned_wait", label),
            &device,
            |bencher, device| {
                let buffer = RefCell::new(Some(
                    PinnedHostBuffer::new(&runtime, bytes).expect("pinned buffer"),
                ));
                bencher.iter_batched(
                    || {
                        let taken = buffer.borrow_mut().take().expect("reused pinned buffer");
                        download_pending(&runtime, device, taken).expect("pending download")
                    },
                    |pending| {
                        let filled = pending.wait().expect("wait");
                        black_box(filled.as_slice().len());
                        *buffer.borrow_mut() = Some(filled);
                    },
                    // The setup (enqueue) must run once per iteration: the routine hands the
                    // buffer back, so a batched setup would find it already taken.
                    criterion::BatchSize::PerIteration,
                );
            },
        );
    }

    group.bench_function("alloc_8MiB", |bencher| {
        bencher.iter(|| {
            let buffer = PinnedHostBuffer::new(&runtime, 1024 * 1024 * 8).expect("pinned buffer");
            black_box(buffer.len());
        });
    });

    group.finish();
}

fn bench_upload(c: &mut Criterion) {
    let backend = backend();
    let runtime = backend.runtime().clone();
    let mut group = c.benchmark_group("transfer_upload");

    for &(elements, label) in SIZES {
        let bytes = elements * size_of::<f64>();
        group.throughput(Throughput::Bytes(bytes as u64));
        let host = host_vector(elements);

        group.bench_with_input(
            BenchmarkId::new("staging", label),
            &host,
            |bencher, host| {
                black_box(upload_tensor(&runtime, host).expect("upload"));
                bencher.iter(|| black_box(upload_tensor(&runtime, host).expect("upload")));
            },
        );

        group.bench_function(BenchmarkId::new("pinned", label), |bencher| {
            let mut destination = upload_tensor(&runtime, &host).expect("device destination");
            let source = RefCell::new(Some(pinned_source(&backend, elements)));
            bencher.iter(|| {
                let taken = source.borrow_mut().take().expect("reused pinned buffer");
                let returned = upload_pending(&runtime, taken, &mut destination)
                    .expect("pending upload")
                    .wait()
                    .expect("wait");
                black_box(returned.len());
                *source.borrow_mut() = Some(returned);
            });
        });
    }

    group.finish();
}

fn criterion_config() -> Criterion {
    Criterion::default()
        .warm_up_time(std::time::Duration::from_secs(2))
        .measurement_time(std::time::Duration::from_secs(5))
        .sample_size(50)
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = bench_download, bench_upload
}
criterion_main!(benches);
