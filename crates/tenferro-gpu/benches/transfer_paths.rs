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

/// Overlap measured with a controlled schedule, not with a value plus a not-ready poll.
///
/// Three arms over the same 8 MiB payload and the same host workload:
///
/// | arm | schedule |
/// | --- | --- |
/// | `copy_only` | enqueue a pending device-to-host copy and wait for it |
/// | `host_only` | the host workload alone |
/// | `overlapped` | enqueue, run the host workload, then wait |
///
/// The overlap claim is the comparison against the **measured** serial arm (copy, wait, then host work), not
/// against a sum of two central estimates. The host workload is compute-bound with a small working set,
/// chosen to minimize host-memory bandwidth demand; its iteration count is opaque inside each call. The copy
/// arms smoke-check three positions per iteration (poisoned first, so a skipped copy fails) and are compared
/// against the full payload once per schedule outside the timed region. This measures host/device overlap for
/// one pending copy; it does not exercise kernel/transfer overlap across two device streams.
fn bench_overlap(c: &mut Criterion) {
    /// Compute-bound host work with a 64-element working set and a fixed iteration count.
    fn host_work(iterations: usize) -> f64 {
        // `black_box` on the iteration count keeps the computation from being folded or hoisted out of a
        // separately optimized closure; the result barrier alone would not.
        let iterations = black_box(iterations);
        let mut state: [f64; 64] = std::array::from_fn(|index| index as f64 * 0.5 + 1.0);
        let mut accumulator = 0.0_f64;
        for iteration in 0..iterations {
            for slot in state.iter_mut() {
                let value = *slot * 1.0000001 + (iteration % 7) as f64;
                *slot = value;
                accumulator += value * 0.5;
            }
        }
        black_box(accumulator)
    }

    /// Tuned so the host workload costs about as much as the 8 MiB copy, making overlap visible instead of
    /// hidden behind one side. `host_only` reports what it actually costs, so the tuning is checkable from
    /// the three arms rather than assumed.
    const HOST_ITERATIONS: usize = 6_000;

    let backend = backend();
    let runtime = backend.runtime().clone();
    let elements = 1024 * 1024;
    let bytes = elements * size_of::<f64>();
    let host = host_vector(elements);
    let device = upload_tensor(&runtime, &host).expect("device payload");
    // A sentinel check rather than a full comparison: a whole-payload `assert_eq!` inside the timed closure
    // would charge the two copy arms several hundred microseconds that `host_only` does not pay.
    let reference = download_tensor(&runtime, &device).expect("reference download");
    let reference_values = reference
        .as_typed::<f64>()
        .expect("f64 tensor")
        .host_data()
        .expect("host data");
    let sentinels = (
        reference_values[0],
        reference_values[reference_values.len() / 2],
        *reference_values.last().expect("non-empty"),
    );
    // The full payload is compared outside the timed region, once per copy schedule, from destinations that
    // do not already hold it: the timed arms only smoke-check three positions.
    let expected_payload: Vec<f64> = reference_values.to_vec();
    {
        for (label, with_host_work) in [("copy_only", false), ("overlapped", true)] {
            let mut destination =
                PinnedHostBuffer::new(&runtime, bytes).expect("verification buffer");
            destination.as_mut_slice().fill(0xAB);
            let value = upload_tensor(&runtime, &host).expect("verification payload");
            let pending = download_pending(&runtime, value, destination).expect("pending download");
            if with_host_work {
                black_box(host_work(HOST_ITERATIONS));
            }
            let (_, filled) = pending.wait().expect("wait");
            assert_full_payload(filled.as_slice(), &expected_payload, label);
        }
    }

    let value = RefCell::new(Some(device));
    let buffer = RefCell::new(Some(
        PinnedHostBuffer::new(&runtime, bytes).expect("pinned buffer"),
    ));

    let mut group = c.benchmark_group("transfer_overlap");
    group.throughput(Throughput::Bytes(bytes as u64));

    group.bench_function(BenchmarkId::new("copy_only", "8MiB"), |bencher| {
        bencher.iter(|| {
            let device_value = value.borrow_mut().take().expect("reused value");
            let mut host_buffer = buffer.borrow_mut().take().expect("reused buffer");
            poison_sentinels(&mut host_buffer, elements);
            let (returned_value, filled) = download_pending(&runtime, device_value, host_buffer)
                .expect("pending download")
                .wait()
                .expect("wait");
            check_sentinels(filled.as_slice(), elements, sentinels);
            black_box(filled.len());
            *value.borrow_mut() = Some(returned_value);
            *buffer.borrow_mut() = Some(filled);
        });
    });

    group.bench_function(BenchmarkId::new("host_only", "8MiB"), |bencher| {
        bencher.iter(|| black_box(host_work(HOST_ITERATIONS)));
    });

    // The measured serial baseline for the same schedule: copy, wait, then the host work. Comparing the
    // overlapped arm against this one avoids resting the overlap claim on a sum of two central estimates.
    group.bench_function(BenchmarkId::new("serial", "8MiB"), |bencher| {
        bencher.iter(|| {
            let device_value = value.borrow_mut().take().expect("reused value");
            let mut host_buffer = buffer.borrow_mut().take().expect("reused buffer");
            poison_sentinels(&mut host_buffer, elements);
            let (returned_value, filled) = download_pending(&runtime, device_value, host_buffer)
                .expect("pending download")
                .wait()
                .expect("wait");
            check_sentinels(filled.as_slice(), elements, sentinels);
            black_box(host_work(HOST_ITERATIONS));
            *value.borrow_mut() = Some(returned_value);
            *buffer.borrow_mut() = Some(filled);
        });
    });

    group.bench_function(BenchmarkId::new("overlapped", "8MiB"), |bencher| {
        bencher.iter(|| {
            let device_value = value.borrow_mut().take().expect("reused value");
            let mut host_buffer = buffer.borrow_mut().take().expect("reused buffer");
            poison_sentinels(&mut host_buffer, elements);
            let pending =
                download_pending(&runtime, device_value, host_buffer).expect("pending download");
            // The controlled schedule: this host work runs while the copy is in flight.
            black_box(host_work(HOST_ITERATIONS));
            let (returned_value, filled) = pending.wait().expect("wait");
            check_sentinels(filled.as_slice(), elements, sentinels);
            black_box(filled.len());
            *value.borrow_mut() = Some(returned_value);
            *buffer.borrow_mut() = Some(filled);
        });
    });

    group.finish();
}

/// Check three positions of a copied payload: enough to catch a wrong or partial copy, and O(1) so the timed
/// closure does not pay for a whole-payload decode.
/// Write a non-matching value into each sampled position.
///
/// Without this, a reused buffer would satisfy the smoke check even if an iteration copied nothing.
fn poison_sentinels(buffer: &mut PinnedHostBuffer, elements: usize) {
    let poison = f64::NAN.to_ne_bytes();
    let byte_len = size_of::<f64>();
    for index in [0, elements / 2, elements - 1] {
        let start = index * byte_len;
        buffer.as_mut_slice()[start..start + byte_len].copy_from_slice(&poison);
    }
}

/// Compare a copied payload against the known host values, element by element.
///
/// Untimed, so a schedule can be validated without charging every iteration a whole-payload decode.
fn assert_full_payload(bytes: &[u8], expected: &[f64], label: &str) {
    assert_eq!(
        bytes.len(),
        std::mem::size_of_val(expected),
        "{label}: length"
    );
    for (index, (chunk, value)) in bytes.as_chunks::<8>().0.iter().zip(expected).enumerate() {
        assert_eq!(
            f64::from_ne_bytes(*chunk),
            *value,
            "{label}: element {index}"
        );
    }
}

/// Smoke-check three positions of a copied payload.
///
/// It is deliberately O(1) so the timed closure does not pay a whole-payload decode, and it can only show
/// that those positions were written with the expected values - not that the whole payload is correct. The
/// untimed `verify_schedules` run below compares every element once per schedule.
fn check_sentinels(bytes: &[u8], elements: usize, expected: (f64, f64, f64)) {
    assert_eq!(bytes.len(), elements * size_of::<f64>(), "payload length");
    let at = |index: usize| {
        let start = index * size_of::<f64>();
        f64::from_ne_bytes(
            bytes[start..start + size_of::<f64>()]
                .try_into()
                .expect("one element"),
        )
    };
    assert_eq!(at(0), expected.0, "first element");
    assert_eq!(at(elements / 2), expected.1, "middle element");
    assert_eq!(at(elements - 1), expected.2, "last element");
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
    targets = bench_download, bench_upload, bench_host_copies, bench_large, bench_chunk_boundary, bench_overlap
}
criterion_main!(benches);
