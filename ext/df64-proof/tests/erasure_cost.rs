//! Measures the cost of carrying a runtime tensor behind an erased payload.
//!
//! The stage 2 decision between a tag plus an erased payload for every member
//! and a fast path for the preset members needs these numbers. The measurement
//! runs on one worker thread, which the repository's rules require for small-work
//! overhead, is host-only, reports the fastest of several rounds, and reports what
//! it measured rather than asserting a threshold. The access and construction
//! loops below do not dispatch work to any other thread, so the 1-worker setting
//! is the measurement's declared configuration rather than a variable.

use std::hint::black_box;
use std::time::Instant;

use tenferro_tensor_core::{ErasedHostTensor, HostTensor};

const ITERATIONS: usize = 200_000;
const ELEMENTS: usize = 64;
const ROUNDS: usize = 9;

fn payload() -> HostTensor<f64> {
    HostTensor::from_vec_col_major(vec![ELEMENTS], vec![1.0_f64; ELEMENTS]).unwrap()
}

fn ns_per_iteration(mut body: impl FnMut()) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..ROUNDS {
        let start = Instant::now();
        body();
        best = best.min(start.elapsed().as_secs_f64());
    }
    best * 1e9 / ITERATIONS as f64
}

#[test]
fn report_erasure_overhead() {
    // (a) Access cost: one value, reused, so allocation is excluded.
    let direct_tensor = payload();
    let erased_tensor = ErasedHostTensor::new(payload());

    let access_direct = ns_per_iteration(|| {
        let mut checksum = 0.0_f64;
        for _ in 0..ITERATIONS {
            checksum += black_box(&direct_tensor).as_slice()[0];
        }
        black_box(checksum);
    });
    let access_erased = ns_per_iteration(|| {
        let mut checksum = 0.0_f64;
        for _ in 0..ITERATIONS {
            checksum += black_box(&erased_tensor)
                .downcast_ref::<f64>()
                .unwrap()
                .as_slice()[0];
        }
        black_box(checksum);
    });

    // (b) Construction and drop cost, including the payload's own allocation.
    let build_direct = ns_per_iteration(|| {
        let mut checksum = 0.0_f64;
        for _ in 0..ITERATIONS {
            let tensor = black_box(payload());
            checksum += tensor.as_slice()[0];
        }
        black_box(checksum);
    });
    let build_erased = ns_per_iteration(|| {
        let mut checksum = 0.0_f64;
        for _ in 0..ITERATIONS {
            let value = black_box(ErasedHostTensor::new(payload()));
            checksum += value.downcast_ref::<f64>().unwrap().as_slice()[0];
        }
        black_box(checksum);
    });

    println!(
        "host payload on 1 worker thread, 9 rounds, fastest of each:\n\
         access   direct {access_direct:.1} ns  erased {access_erased:.1} ns  delta {:+.1} ns\n\
         build    direct {build_direct:.1} ns  erased {build_erased:.1} ns  delta {:+.1} ns",
        access_erased - access_direct,
        build_erased - build_direct
    );
}
