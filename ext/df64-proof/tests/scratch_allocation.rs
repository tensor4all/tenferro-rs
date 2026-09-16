//! Measures how much scratch the contribution's bodies allocate per execution.
//!
//! #1789 asks for a demonstrated gap before any storage boundary is extended, so this
//! reports what the numerical bodies allocate today: how many allocations and how many
//! bytes one factorization and one adjoint cost, and how those numbers grow with repeated
//! execution. A body that reused its scratch would show a much smaller second execution;
//! these numbers are the evidence for what a reusable workspace would have to recover.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tenferro_cpu::CpuBackend;
use tenferro_df64_proof::extension::{module, Df64Qr, Df64QrVjp, DF64_SCALAR_IDENTITY};
use tenferro_df64_proof::Df64;
use tenferro_runtime::extension::apply;
use tenferro_runtime::{GraphCompiler, Runtime, TracedTensor};
use tenferro_tensor::Tensor;
use tenferro_tensor_core::{ErasedHostTensor, HostTensor};

/// Counts every allocation the test process makes.
struct Counting;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(layout.size(), Ordering::Relaxed);
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

fn measure(mut body: impl FnMut()) -> (usize, usize) {
    ALLOCATIONS.store(0, Ordering::Relaxed);
    BYTES.store(0, Ordering::Relaxed);
    body();
    (
        ALLOCATIONS.load(Ordering::Relaxed),
        BYTES.load(Ordering::Relaxed),
    )
}

fn external(values: Vec<Df64>, shape: Vec<usize>) -> Tensor {
    Tensor::external(ErasedHostTensor::new(
        HostTensor::from_vec_col_major(shape, values).expect("shape matches data"),
    ))
}

fn runtime() -> Runtime {
    let backend = CpuBackend::new();
    let mut builder = Runtime::builder();
    builder
        .register_engine(tenferro_cpu::runtime_engine_registration(&backend).expect("engine"))
        .expect("register the CPU engine");
    builder
        .install_extension_module(module().expect("module"))
        .expect("install the Df64 module");
    builder.build().expect("runtime with the module")
}

fn leaf(values: Vec<Df64>, shape: Vec<usize>) -> TracedTensor {
    TracedTensor::from_tensor_concrete_shape_declaring_scalar(
        external(values, shape),
        DF64_SCALAR_IDENTITY,
    )
    .expect("traced leaf")
}

#[test]
fn report_scratch_allocation_per_execution() {
    // A 64 by 64 factorization, so the numerical bodies work on realistic buffers.
    let order = 64;
    let values: Vec<Df64> = (0..order * order)
        .map(|index| Df64::from_f64(((index % order) + 1) as f64))
        .collect();

    let input = leaf(values.clone(), vec![order, order]);
    let outputs = apply(Arc::new(Df64Qr), &[&input]).expect("traced QR");

    // The adjoint is measured on its own, so its factors are leaves of their own.
    let q_leaf = leaf(vec![Df64::from_f64(0.5); order * order], vec![order, order]);
    let r_leaf = leaf(vec![Df64::from_f64(2.0); order * order], vec![order, order]);
    let cotangent_leaf = leaf(vec![Df64::from_f64(1.0); order * order], vec![order, order]);
    let adjoint = apply(
        Arc::new(Df64QrVjp::of(false, true)),
        &[&q_leaf, &r_leaf, &cotangent_leaf],
    )
    .expect("traced adjoint");

    let runtime_value = external(values, vec![order, order]);
    let q_value = external(vec![Df64::from_f64(0.5); order * order], vec![order, order]);
    let r_value = external(vec![Df64::from_f64(2.0); order * order], vec![order, order]);
    let cotangent_value = external(vec![Df64::from_f64(1.0); order * order], vec![order, order]);

    let runtime = runtime();
    let mut compiler = GraphCompiler::new();
    let qr_program = compiler
        .compile_many(&[&outputs[0], &outputs[1]])
        .expect("compiled QR");
    let adjoint_program = compiler.compile(&adjoint[0]).expect("compiled the adjoint");

    // The first execution pays for the runtime's own setup as well as the body's scratch.
    let (first_qr_allocations, first_qr_bytes) = measure(|| {
        drop(
            runtime
                .run_compiled(&qr_program, &[&runtime_value])
                .expect("QR"),
        )
    });
    let (next_qr_allocations, next_qr_bytes) = measure(|| {
        drop(
            runtime
                .run_compiled(&qr_program, &[&runtime_value])
                .expect("QR"),
        )
    });
    let (adjoint_allocations, adjoint_bytes) = measure(|| {
        drop(
            runtime
                .run_compiled(&adjoint_program, &[&q_value, &r_value, &cotangent_value])
                .expect("adjoint"),
        );
    });
    // The second execution can reuse the scratch the first one left in the runtime's
    // accounted extension cache.
    let (steady_adjoint_allocations, steady_adjoint_bytes) = measure(|| {
        drop(
            runtime
                .run_compiled(&adjoint_program, &[&q_value, &r_value, &cotangent_value])
                .expect("adjoint"),
        );
    });

    let elements = order * order;
    println!(
        "df64 scratch per execution ({order}x{order}, {elements} elements):\n\
         \x20 first QR:   {first_qr_allocations:>6} allocations, {first_qr_bytes:>9} bytes\n\
         \x20 steady QR:  {next_qr_allocations:>6} allocations, {next_qr_bytes:>9} bytes\n\
         \x20 adjoint:    {adjoint_allocations:>6} allocations, {adjoint_bytes:>9} bytes\n\
         \x20 steady adj: {steady_adjoint_allocations:>6} allocations, {steady_adjoint_bytes:>9} bytes\n\
         \x20 payload:    {:>6} bytes per matrix",
        elements * std::mem::size_of::<Df64>()
    );

    // A body that reused its workspace would show a much smaller steady-state number than
    // the factorization's own output size, because the scratch for the factors alone is
    // several matrices. Today the steady state is at least the size of the factors it
    // returns, which is what a reusable workspace would have to recover.
    let factor_bytes = 2 * elements * std::mem::size_of::<Df64>();
    assert!(
        next_qr_bytes >= factor_bytes,
        "the steady state allocates less than the factors themselves: {next_qr_bytes} < {factor_bytes}"
    );
    assert!(
        next_qr_allocations > 0 && adjoint_allocations > 0,
        "the bodies allocate no scratch at all, so this report is stale"
    );
    // The accounted scratch is what makes the second adjoint cheaper than the first, so the
    // report fails if the reuse path stops being taken.
    assert!(
        steady_adjoint_bytes < adjoint_bytes,
        "the adjoint's scratch is not reused: {adjoint_bytes} then {steady_adjoint_bytes}"
    );

    // The runtime reports the scratch as an accounted extension cache entry, which is what
    // makes this a declared reuse path rather than a private cache.
    let stats = runtime.cache_stats().expect("runtime cache statistics");
    println!(
        "extension cache: {} entries, {} retained bytes, {} hits, {} misses",
        stats.extensions.entries,
        stats.extensions.retained_bytes,
        stats.extensions.hits,
        stats.extensions.misses
    );
    assert!(
        stats.extensions.entries > 0 && stats.extensions.retained_bytes > 0,
        "the reuse path is not accounted for: {stats:?}"
    );
}
