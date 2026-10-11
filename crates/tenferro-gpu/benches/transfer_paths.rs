//! Transfer-path measurements for #1945 U4 / #2009.
//!
//! Compares the existing blocking transfer paths with the pinned pending handoffs added by U4
//! slices 1-2, on one CUDA stream with criterion's default single-threaded harness:
//!
//! Payloads are `f64` element counts; the labels are the resulting byte sizes.
//!
//! | case | host copies | destination memory |
//! | --- | --- | --- |
//! | `download_pageable` (`download_tensor`, large payload) | 1 (device straight into the result `Vec`) | pageable |
//! | `download_pinned` (`download_pending` + `wait`) | 1 (device → pinned buffer owned by the handle) | pinned |
//! | `download_small` (`download_tensor`, ≤ `PINNED_SCALAR_BYTES`) | 2 (device → pinned staging, staging → `Vec`) | pageable result |
//! | `upload_staging` (`upload_tensor`) | 2 (borrowed host → owned copy; then the write itself) | — |
//! | `upload_pinned` (`upload_pending` + `wait`) | 0 on the host side (device reads the pinned buffer) | — |
//!
//! The pinned cases reuse one buffer across iterations: `wait` hands it back and the next iteration
//! hands it in again, so the measurement covers the copy and its event, not allocation. The
//! `*_alloc` cases measure allocating a fresh pinned buffer, and `download_pinned_wait` isolates the
//! completion wait (`wait` only, with the enqueue in criterion's untimed setup).
//!
//! `transfer_host_copies` attributes the staged upload's cost instead of guessing at it: a pageable
//! ownership copy, a pageable -> pinned copy, the DMA of an already-pinned source, and the staged route
//! end to end. The pieces then show whether a route pays for a copy it does not need. Measured values
//! for the 8 MiB payload are recorded in `docs/design/asynchronous-transfer-1945-u4.md` §3.1.
//!
//! `transfer_large` runs a 256 MiB payload through both directions of each family, so the blocking
//! routes' pageable transport can be compared with the owning routes' pinned transport on the same host,
//! device and payload; §3.2 of the same record holds those numbers.
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

        // The owning route moves the value and its buffer in and takes them back from `wait`, so both
        // circulate between iterations.
        let pinned_value = RefCell::new(Some(
            upload_tensor(&runtime, &host_vector(elements)).expect("device payload"),
        ));
        let pinned_buffer = RefCell::new(Some(
            PinnedHostBuffer::new(&runtime, bytes).expect("pinned buffer"),
        ));
        group.bench_function(BenchmarkId::new("pinned", label), |bencher| {
            bencher.iter(|| {
                let value = pinned_value.borrow_mut().take().expect("reused value");
                let buffer = pinned_buffer.borrow_mut().take().expect("reused buffer");
                let (value, filled) = download_pending(&runtime, value, buffer)
                    .expect("pending download")
                    .wait()
                    .expect("wait");
                black_box(filled.as_slice().len());
                *pinned_value.borrow_mut() = Some(value);
                *pinned_buffer.borrow_mut() = Some(filled);
            });
        });

        group.bench_function(BenchmarkId::new("pinned_wait", label), |bencher| {
            bencher.iter_batched(
                || {
                    let value = pinned_value.borrow_mut().take().expect("reused value");
                    let buffer = pinned_buffer.borrow_mut().take().expect("reused buffer");
                    download_pending(&runtime, value, buffer).expect("pending download")
                },
                |pending| {
                    let (value, filled) = pending.wait().expect("wait");
                    black_box(filled.as_slice().len());
                    *pinned_value.borrow_mut() = Some(value);
                    *pinned_buffer.borrow_mut() = Some(filled);
                },
                // The setup (enqueue) must run once per iteration: the routine hands both back, so a
                // batched setup would find them already taken.
                criterion::BatchSize::PerIteration,
            );
        });
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

        // Owning admission: the destination and the buffer are moved in and come back from `wait`.
        let upload_destination = RefCell::new(Some(
            upload_tensor(&runtime, &host).expect("device destination"),
        ));
        let upload_source = RefCell::new(Some(pinned_source(&backend, elements)));
        group.bench_function(BenchmarkId::new("pinned", label), |bencher| {
            bencher.iter(|| {
                let destination = upload_destination
                    .borrow_mut()
                    .take()
                    .expect("reused destination");
                let source = upload_source.borrow_mut().take().expect("reused buffer");
                let (destination, source) = upload_pending(&runtime, source, destination)
                    .expect("pending upload")
                    .wait()
                    .expect("wait");
                black_box(source.len());
                *upload_destination.borrow_mut() = Some(destination);
                *upload_source.borrow_mut() = Some(source);
            });
        });
    }

    group.finish();
}

/// Attribute the staged upload's cost: which host-side work the pageable route pays for.
///
/// `upload_tensor` on a pageable payload goes through `ComputeClient::create_from_slice`, which owns
/// its input with `slice.to_vec()` and then stages that owned copy into pinned memory before the
/// DMA. This group measures the pieces separately so the route's cost is attributed rather than
/// guessed: a pageable clone, a pageable -> pinned copy, and the DMA of an already-pinned source.
fn bench_host_copies(c: &mut Criterion) {
    let backend = backend();
    let runtime = backend.runtime().clone();
    let mut group = c.benchmark_group("transfer_host_copies");

    let elements = 1024 * 1024;
    let bytes = elements * size_of::<f64>();
    group.throughput(Throughput::Bytes(bytes as u64));
    let host = host_vector(elements);
    let flat: Vec<u8> = (0..elements)
        .flat_map(|index| (index as f64 * 0.5).to_ne_bytes())
        .collect();

    group.bench_function("pageable_clone", |bencher| {
        bencher.iter(|| black_box(flat.to_vec()));
    });

    let mut buffer = PinnedHostBuffer::new(&runtime, bytes).expect("pinned buffer");
    group.bench_function("pageable_to_pinned", |bencher| {
        bencher.iter(|| {
            buffer.as_mut_slice().copy_from_slice(black_box(&flat));
            black_box(buffer.as_slice().len());
        });
    });

    let destination = RefCell::new(Some(
        upload_tensor(&runtime, &host).expect("device destination"),
    ));
    let source = RefCell::new(Some(pinned_source(&backend, elements)));
    group.bench_function("pinned_dma", |bencher| {
        bencher.iter(|| {
            let value = destination.borrow_mut().take().expect("reused destination");
            let buffer = source.borrow_mut().take().expect("reused buffer");
            let (value, buffer) = upload_pending(&runtime, buffer, value)
                .expect("pending upload")
                .wait()
                .expect("wait");
            black_box(buffer.len());
            *destination.borrow_mut() = Some(value);
            *source.borrow_mut() = Some(buffer);
        });
    });

    group.bench_function("staged_total", |bencher| {
        bencher.iter(|| black_box(upload_tensor(&runtime, black_box(&host)).expect("upload")));
    });

    group.finish();
}

/// A payload above the substrate's 100 MB pinned-staging threshold.
///
/// The pending routes take a caller-owned pinned buffer, so this group shows what the blocking routes do
/// when the payload stops fitting the small-transfer policy: whether the staged route's cost per byte
/// stays where the 8 MiB case put it, and what the blocking pinned route costs when its buffer is
/// allocated fresh for a payload this size.
fn bench_large(c: &mut Criterion) {
    let backend = backend();
    let runtime = backend.runtime().clone();
    let mut group = c.benchmark_group("transfer_large");

    let elements = 32 * 1024 * 1024;
    let bytes = elements * size_of::<f64>();
    group.throughput(Throughput::Bytes(bytes as u64));
    let host = host_vector(elements);

    let device = upload_tensor(&runtime, &host).expect("device payload");
    group.bench_with_input(
        BenchmarkId::new("download_pageable", "256MiB"),
        &device,
        |bencher, device| {
            bencher.iter(|| black_box(download_tensor(&runtime, device).expect("download")));
        },
    );

    group.bench_with_input(
        BenchmarkId::new("upload_staging", "256MiB"),
        &host,
        |bencher, host| {
            bencher.iter(|| black_box(upload_tensor(&runtime, host).expect("upload")));
        },
    );

    // The blocking routes are pageable in both directions (the borrowed upload owns a native copy;
    // the download writes straight into its result vector), so this pair shows whether the gap is the
    // pageable DMA itself or the staging work around it.
    let pinned_value = RefCell::new(Some(
        upload_tensor(&runtime, &host).expect("device payload"),
    ));
    let pinned_buffer = RefCell::new(Some(
        PinnedHostBuffer::new(&runtime, bytes).expect("pinned buffer"),
    ));
    group.bench_function(BenchmarkId::new("download_pinned", "256MiB"), |bencher| {
        bencher.iter(|| {
            let value = pinned_value.borrow_mut().take().expect("reused value");
            let buffer = pinned_buffer.borrow_mut().take().expect("reused buffer");
            let (value, filled) = download_pending(&runtime, value, buffer)
                .expect("pending download")
                .wait()
                .expect("wait");
            black_box(filled.as_slice().len());
            *pinned_value.borrow_mut() = Some(value);
            *pinned_buffer.borrow_mut() = Some(filled);
        });
    });

    let destination = RefCell::new(Some(device));
    let source = RefCell::new(Some(pinned_source(&backend, elements)));
    group.bench_function(BenchmarkId::new("upload_pinned", "256MiB"), |bencher| {
        bencher.iter(|| {
            let value = destination.borrow_mut().take().expect("reused destination");
            let buffer = source.borrow_mut().take().expect("reused buffer");
            let (value, buffer) = upload_pending(&runtime, buffer, value)
                .expect("pending upload")
                .wait()
                .expect("wait");
            black_box(buffer.len());
            *destination.borrow_mut() = Some(value);
            *source.borrow_mut() = Some(buffer);
        });
    });

    group.finish();
}

/// The routing boundary: the same payload either side of the staging chunk.
///
/// `STAGING_CHUNK_BYTES` is 8 MiB, so a payload at the chunk takes the previous route and one element
/// more takes the chunked route. Two sizes eight bytes apart make the comparison fair without needing
/// a runtime switch, and they are what decides whether the cutoff should move.
fn bench_chunk_boundary(c: &mut Criterion) {
    let backend = backend();
    let runtime = backend.runtime().clone();
    let mut group = c.benchmark_group("transfer_chunk_boundary");
    let chunk_elements = 8 * 1024 * 1024 / size_of::<f64>();

    for (elements, label) in [(chunk_elements, "8MiB"), (chunk_elements + 1, "8MiB+8B")] {
        let bytes = elements * size_of::<f64>();
        group.throughput(Throughput::Bytes(bytes as u64));
        let host = host_vector(elements);
        group.bench_with_input(
            BenchmarkId::new("upload_staging", label),
            &host,
            |bencher, host| {
                bencher.iter(|| black_box(upload_tensor(&runtime, host).expect("upload")));
            },
        );
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
    targets = bench_download, bench_upload, bench_host_copies, bench_large, bench_chunk_boundary
}
criterion_main!(benches);
