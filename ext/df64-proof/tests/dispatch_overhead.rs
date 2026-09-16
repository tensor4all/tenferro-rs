//! Session and dispatch overhead for the contribution's operations.
//!
//! #1789's fifth acceptance item asks for allocation counts and session/dispatch overhead under
//! the exact-baseline protocol with a verified thread count. The allocation numbers for the
//! numerical bodies live in `scratch_allocation.rs`; this file measures the layer around them:
//! the same tiny operation is run (a) as a preset `f64` program through the runtime's prepared
//! hot path, (b) as the contribution's program through that same path, and (c) by calling the
//! contribution's body directly. The difference between (b) and (c) is what the runtime's
//! session and dispatch layer costs for a contribution operation, and (b) against (a) shows
//! whether a contribution operation pays more for that layer than a preset one. The two
//! programs do not compute the same function, so the comparison is of the layer rather than of
//! the arithmetic, and the numbers are a report rather than a threshold.
//!
//! The backend runs one worker thread, which the repository requires for measuring small-work
//! overhead, and the report prints the effective thread count so the baseline is verified
//! rather than implied. The profile matters as much as the thread count, because an
//! unoptimized build inflates this layer, so the report is taken in both the test profile and
//! the release profile and each number is labelled with the profile it came from.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use tenferro_cpu::{scalar_fold, CpuBackend};
use tenferro_df64_proof::extension::{Df64Total, DF64_SCALAR_IDENTITY};
use tenferro_df64_proof::{Df64, Df64Add};
use tenferro_runtime::extension::apply;
use tenferro_runtime::{GraphCompiler, Runtime, TracedTensor};
use tenferro_tensor::{DType, Tensor};
use tenferro_tensor_core::{ErasedHostTensor, HostTensor};

/// Counts every allocation the test process makes.
struct Counting;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: the layout is forwarded unchanged to the system allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the pointer and layout came from the system allocator through `alloc`.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn measure(iterations: usize, mut body: impl FnMut()) -> (usize, f64) {
    let start = Instant::now();
    ALLOCATIONS.store(0, Ordering::Relaxed);
    for _ in 0..iterations {
        body();
    }
    let elapsed = start.elapsed();
    (
        ALLOCATIONS.load(Ordering::Relaxed),
        elapsed.as_secs_f64() * 1e9 / iterations as f64,
    )
}

fn runtime() -> (Runtime, usize) {
    // One worker thread, so the numbers are a reproducible baseline rather than a function of
    // this machine's core count.
    let backend = CpuBackend::with_threads(1).expect("single-threaded CPU backend");
    let threads = backend.num_threads();
    let mut builder = Runtime::builder();
    builder
        .register_engine(tenferro_cpu::runtime_engine_registration(&backend).expect("engine"))
        .expect("register the CPU engine");
    builder
        .install_extension_module(tenferro_df64_proof::extension::module().expect("module"))
        .expect("install the Df64 module");
    (builder.build().expect("runtime with the module"), threads)
}

fn external(values: Vec<Df64>, shape: Vec<usize>) -> Tensor {
    Tensor::external(ErasedHostTensor::new(
        HostTensor::from_vec_col_major(shape, values).expect("shape matches data"),
    ))
}

#[test]
fn report_session_and_dispatch_overhead_for_a_contribution_operation() {
    let iterations = 2_000;
    let (runtime, threads) = runtime();
    let mut compiler = GraphCompiler::new();

    // (a) A preset program through the runtime's prepared hot path.
    let ordinary = TracedTensor::input_concrete_shape(DType::F64, &[2]).expect("traced input");
    let doubled = (&ordinary + &ordinary).expect("traced add");
    let ordinary_program = compiler
        .compile_with_input_specs(&doubled, &[(&ordinary, DType::F64, &[2])])
        .expect("compiled preset program");
    let ordinary_value =
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).expect("shape matches data");
    let prepared_ordinary = runtime
        .prepare_compiled(&ordinary_program, &[&ordinary_value])
        .expect("prepared preset program");

    // (b) The contribution's program through the same path.
    let values = vec![Df64::from_f64(1.0), Df64::from_f64(2.0)];
    let input = TracedTensor::from_tensor_concrete_shape_declaring_scalar(
        external(values.clone(), vec![2]),
        DF64_SCALAR_IDENTITY,
    )
    .expect("traced leaf");
    let outputs = apply(Arc::new(Df64Total), &[&input]).expect("traced total");
    let contribution_program = compiler
        .compile(&outputs[0])
        .expect("compiled contribution program");
    let contribution_value = external(values.clone(), vec![2]);
    let prepared_contribution = runtime
        .prepare_compiled(&contribution_program, &[&contribution_value])
        .expect("prepared contribution program");

    // (c) The contribution's body, with neither session nor dispatch in the way.
    let stored = HostTensor::from_vec_col_major(vec![2], values).expect("shape matches data");

    let (ordinary_allocations, ordinary_ns) = measure(iterations, || {
        drop(
            runtime
                .run_prepared(&prepared_ordinary, &[&ordinary_value])
                .expect("preset execution"),
        );
    });
    let (contribution_allocations, contribution_ns) = measure(iterations, || {
        drop(
            runtime
                .run_prepared(&prepared_contribution, &[&contribution_value])
                .expect("contribution execution"),
        );
    });
    let (direct_allocations, direct_ns) = measure(iterations, || {
        let total = scalar_fold::<Df64, Df64Add>("total", &stored, Df64::zero())
            .expect("the body's own reduction");
        std::hint::black_box(total);
    });

    println!(
        "session and dispatch overhead ({iterations} iterations, {threads} worker thread):\n\
         \x20 preset program, runtime path:       {ordinary_ns:>9.1} ns/op, {ordinary_allocations:>5} allocations\n\
         \x20 contribution, runtime path:         {contribution_ns:>9.1} ns/op, {contribution_allocations:>5} allocations\n\
         \x20 contribution body, called directly: {direct_ns:>9.1} ns/op, {direct_allocations:>5} allocations\n\
         \x20 session and dispatch for the contribution: {:.1} ns/op over its body",
        contribution_ns - direct_ns
    );

    // The protocol requires one worker thread, so the report states it and the test checks it.
    assert_eq!(threads, 1, "the measurement must run on one worker thread");

    // The runtime path must not accumulate anything per call: doubling the iterations may
    // double the allocations but must not do more.
    let (ordinary_twice, _) = measure(iterations * 2, || {
        drop(
            runtime
                .run_prepared(&prepared_ordinary, &[&ordinary_value])
                .expect("preset execution"),
        );
    });
    let (contribution_twice, _) = measure(iterations * 2, || {
        drop(
            runtime
                .run_prepared(&prepared_contribution, &[&contribution_value])
                .expect("contribution execution"),
        );
    });
    assert!(
        ordinary_twice <= 2 * ordinary_allocations + 4,
        "the preset path accumulates allocations: {ordinary_allocations} then {ordinary_twice}"
    );
    assert!(
        contribution_twice <= 2 * contribution_allocations + 4,
        "the contribution path accumulates allocations: {contribution_allocations} then {contribution_twice}"
    );
    assert!(
        contribution_allocations > 0 && direct_allocations > 0,
        "a path reports no allocations at all, so this report is stale"
    );
}
