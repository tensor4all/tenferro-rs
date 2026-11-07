//! Public same-shape wrappers must not allocate input-sized temporaries.
//!
//! The measurement runs on a **fresh** engine, and this is what makes it a gate rather than a
//! comparison of two arms that share their implementation (#2476): the engine's buffer pool
//! recycles payload-sized buffers, so on a warmed engine a steady per-call copy shows up as
//! `0` allocations on both the reference route and the wrapper, and `actual <= baseline` passes
//! no matter how many copies the shared path makes. A pool that starts empty must allocate every
//! payload-sized buffer the call needs, so the count is the call's own concurrent payload
//! demand and the budget is absolute: the call's own result, and nothing else.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use tenferro_cpu::CpuEngine;
use tenferro_runtime::TensorSessionOpsExt;
use tenferro_tensor::TensorRead;
use tenferro_tensor::{CompareDir, DotGeneralConfig, Engine, Tensor};

const LEN: usize = 4096;
const INPUT_BYTES: usize = LEN * size_of::<f64>();
// Only the measuring thread is counted: another thread (a pool worker finishing its
// startup, the test harness) allocating during a measurement is not the op's cost, and
// counted globally it made these budgets flake (#2306). The engines measured here run the
// op on the calling thread.
thread_local! {
    static ENABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
static LARGE_ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static TOTAL_ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static TOTAL_BYTES: AtomicUsize = AtomicUsize::new(0);
struct Allocator;

// SAFETY: all allocations and deallocations are forwarded unchanged to System.
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ENABLED.try_with(std::cell::Cell::get).unwrap_or(false) {
            TOTAL_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            TOTAL_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
            if layout.size() >= INPUT_BYTES {
                LARGE_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            }
        }
        // SAFETY: the original allocation layout is passed to System.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: ptr and layout originate from System.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if ENABLED.try_with(std::cell::Cell::get).unwrap_or(false) {
            TOTAL_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            TOTAL_BYTES.fetch_add(size, Ordering::Relaxed);
            if size >= INPUT_BYTES {
                LARGE_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            }
        }
        // SAFETY: the original allocation and requested size are forwarded.
        unsafe { System.realloc(ptr, layout, size) }
    }
}
#[global_allocator]
static ALLOCATOR: Allocator = Allocator;

/// The allocation counters are process-wide, so the tests of this binary must
/// not measure while another one allocates.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serialize() -> std::sync::MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn allocations(f: impl FnOnce() -> Tensor) -> (Tensor, usize) {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            ENABLED.with(|enabled| enabled.set(false));
        }
    }
    LARGE_ALLOCATIONS.store(0, Ordering::Relaxed);
    TOTAL_ALLOCATIONS.store(0, Ordering::Relaxed);
    TOTAL_BYTES.store(0, Ordering::Relaxed);
    ENABLED.with(|enabled| enabled.set(true));
    let reset = Reset;
    let result = f();
    drop(reset);
    eprintln!(
        "payload-sized={}, total_count={}, requested_bytes={}",
        LARGE_ALLOCATIONS.load(Ordering::Relaxed),
        TOTAL_ALLOCATIONS.load(Ordering::Relaxed),
        TOTAL_BYTES.load(Ordering::Relaxed)
    );
    (result, LARGE_ALLOCATIONS.load(Ordering::Relaxed))
}

/// Counts payload-sized allocations for one call, dropping the result before returning.
///
/// Dropping inside the measurement is the point: a live output keeps its buffer out of the pool,
/// so measuring one arm while the other's result is still held compares two different pool states.
fn count_large_allocations(call: impl FnOnce() -> Tensor) -> usize {
    let (result, large) = allocations(call);
    drop(result);
    large
}

/// The calibrated unit of this gate: one payload-sized buffer, measured through the plain host
/// copy rather than through any path a wrapper could share.
///
/// `Tensor::duplicate` allocates exactly one operand-sized host buffer. That number is the whole
/// budget a wrapper gets for a payload-sized result, so the gate never compares an
/// implementation against itself.
fn payload_buffer_unit(operand: &Tensor) -> usize {
    count_large_allocations(|| operand.duplicate().expect("duplicate"))
}

/// A wrapper may allocate its own payload-sized result and nothing else.
///
/// `result_is_payload_sized` is a property of the operation's declared output dtype and the
/// fixture length, not of the implementation: `add` returns `f64` (`8 * LEN` bytes, payload
/// sized) while `compare` returns `bool` (`LEN` bytes, below the payload threshold).
fn assert_result_buffer_only(name: &str, actual: usize, result_is_payload_sized: bool) {
    let budget = usize::from(result_is_payload_sized);
    assert!(
        actual <= budget,
        "{name} allocated {actual} payload-sized buffers; a call may allocate its own result \
         ({budget}) and no operand copy"
    );
}

fn operands() -> (Tensor, Tensor) {
    (
        Tensor::from_vec_col_major([LEN], vec![2.0_f64; LEN]).unwrap(),
        Tensor::from_vec_col_major([LEN], vec![3.0_f64; LEN]).unwrap(),
    )
}

#[test]
fn same_shape_wrappers_do_not_allocate_operand_copies() {
    let _serial = serialize();
    let (a, b) = operands();
    let mut backend = CpuEngine::serial();
    assert_eq!(backend.num_threads(), 1);
    {
        let session: &mut dyn Engine = &mut backend;
        // The wrapper must forward to the backend route: value equality is what the allocation
        // gate below assumes about the one call it measures.
        macro_rules! check {
            ($name:ident, $read:ident, $payload_result:expr) => {{
                let raw = session
                    .$read(TensorRead::from_tensor(&a), TensorRead::from_tensor(&b))
                    .unwrap();
                let public = a.$name(&b, session).unwrap();
                assert_eq!(
                    public.as_slice::<f64>().unwrap(),
                    raw.as_slice::<f64>().unwrap()
                );
                drop(raw);
                drop(public);

                // A fresh pool per measurement: an empty pool cannot hand the call a recycled
                // buffer, so every payload-sized buffer it holds is counted.
                let mut fresh = CpuEngine::serial();
                let actual = count_large_allocations(|| {
                    let session: &mut dyn Engine = &mut fresh;
                    a.$name(&b, session).unwrap()
                });
                assert_result_buffer_only(stringify!($name), actual, $payload_result);
            }};
        }
        check!(add, add_read, true);
        check!(sub, sub_read, true);
        check!(mul, mul_read, true);
        check!(div, div_read, true);
        check!(rem, rem_read, true);
        check!(pow, pow_read, true);
        check!(maximum, maximum_read, true);
        check!(minimum, minimum_read, true);

        let condition = Tensor::from_vec_col_major([LEN], vec![true; LEN]).unwrap();

        // `compare` and `select` need a differently shaped call, so they are spelled out.
        let raw = session
            .compare_read(
                TensorRead::from_tensor(&a),
                TensorRead::from_tensor(&b),
                &CompareDir::Lt,
            )
            .unwrap();
        let public = a.compare(&b, CompareDir::Lt, session).unwrap();
        assert_eq!(
            public.as_slice::<bool>().unwrap(),
            raw.as_slice::<bool>().unwrap()
        );
        drop(raw);
        drop(public);
        let mut fresh = CpuEngine::serial();
        let actual = count_large_allocations(|| {
            let session: &mut dyn Engine = &mut fresh;
            a.compare(&b, CompareDir::Lt, session).unwrap()
        });
        // A `Bool` result of `LEN` bytes is below the payload threshold: no payload buffer at
        // all is allowed.
        assert_result_buffer_only("compare", actual, false);

        let raw = session
            .clamp_read(
                TensorRead::from_tensor(&a),
                TensorRead::from_tensor(&a),
                TensorRead::from_tensor(&b),
            )
            .unwrap();
        let public = a.clamp(&a, &b, session).unwrap();
        assert_eq!(
            public.as_slice::<f64>().unwrap(),
            raw.as_slice::<f64>().unwrap()
        );
        drop(raw);
        drop(public);
        let mut fresh = CpuEngine::serial();
        let actual = count_large_allocations(|| {
            let session: &mut dyn Engine = &mut fresh;
            a.clamp(&a, &b, session).unwrap()
        });
        assert_result_buffer_only("clamp", actual, true);

        let raw = session
            .select_read(
                TensorRead::from_tensor(&condition),
                TensorRead::from_tensor(&a),
                TensorRead::from_tensor(&b),
            )
            .unwrap();
        let public = condition.where_select(&a, &b, session).unwrap();
        assert_eq!(
            public.as_slice::<f64>().unwrap(),
            raw.as_slice::<f64>().unwrap()
        );
        drop(raw);
        drop(public);
        let mut fresh = CpuEngine::serial();
        let actual = count_large_allocations(|| {
            let session: &mut dyn Engine = &mut fresh;
            condition.where_select(&a, &b, session).unwrap()
        });
        assert_result_buffer_only("select", actual, true);
    }
}

#[test]
fn the_calibrated_unit_is_one_payload_buffer() {
    let _serial = serialize();
    // The unit the gate budgets is itself measured, so a harness that stopped counting (a
    // broken allocator hook, a changed fixture size) fails here instead of silently passing
    // every wrapper.
    let (a, _) = operands();
    assert_eq!(payload_buffer_unit(&a), 1);
}

/// The negative control's measurement half: one injected copy is visible, and it is exactly one
/// buffer, not "some noise above zero".
///
/// `add` on a fresh pool needs one payload buffer for its result. Copying the operand inside the
/// measured window - what a copy added to the shared path would do - raises that to two, which is
/// the number the gate rejects.
#[test]
fn the_gate_sees_exactly_one_injected_payload_copy() {
    let _serial = serialize();
    let (a, b) = operands();

    let mut fresh = CpuEngine::serial();
    let without_copy = count_large_allocations(|| {
        let session: &mut dyn Engine = &mut fresh;
        a.add(&b, session).unwrap()
    });

    // A fresh engine for the injected arm too: the two arms must not share a warmed pool, or the
    // second measurement would inherit the buffers the first one left behind.
    let mut fresh = CpuEngine::serial();
    let with_copy = count_large_allocations(|| {
        let copied = a.duplicate().expect("the injected operand copy");
        let session: &mut dyn Engine = &mut fresh;
        session
            .add_read(
                TensorRead::from_tensor(&copied),
                TensorRead::from_tensor(&b),
            )
            .unwrap()
    });

    assert_eq!(
        without_copy, 1,
        "the reference call's own result is the one payload buffer it allocates"
    );
    assert_eq!(
        with_copy,
        without_copy + 1,
        "an operand copy has to show up as exactly one extra payload-sized buffer"
    );
}

/// The negative control's gate half: with one injected copy the gate fails.
///
/// `copying_add` stands in for the shared path acquiring a copy of its operand: the duplicate
/// happens inside the measured window on a fresh pool, and the same `add_read` entry runs
/// afterwards. It must trip the very assertion the positive test relies on.
#[test]
#[should_panic(expected = "allocated 2 payload-sized buffers")]
fn the_gate_fails_when_one_operand_copy_is_injected() {
    let _serial = serialize();
    let (a, b) = operands();
    let mut fresh = CpuEngine::serial();
    let actual = count_large_allocations(|| {
        let copied = a.duplicate().expect("the injected operand copy");
        let session: &mut dyn Engine = &mut fresh;
        session
            .add_read(
                TensorRead::from_tensor(&copied),
                TensorRead::from_tensor(&b),
            )
            .unwrap()
    });
    assert_result_buffer_only("add with one injected operand copy", actual, true);
}

/// #2141 item 3: the canonical-fallback contraction must not copy its operands.
///
/// A permuted right operand makes the provider take its canonical path. The operand
/// copies that path once made were removed with the plan work
/// (`perf-1938-cpu-followups`: 25 → 15 allocations per call); what remains is the
/// provider's own view and plan metadata (strided-rs `Arc` dims, an HPTT plan rebuild),
/// which is not payload-sized. This pins the payload half: no input-sized temporary
/// with either operand layout.
///
/// Both routes are measured on a fresh pool, so each count is that route's own payload demand
/// rather than the other route's leftovers (#2476). This assertion stays **relative** on purpose:
/// the provider's packing scratch is host-specific (`faer` packs nothing on x86_64 with AVX2 and
/// one buffer on aarch64), so an absolute budget would encode one host's provider. The price is
/// that a copy added to a path *both* routes share still cancels here; the absolute gate above
/// (`same_shape_wrappers_do_not_allocate_operand_copies`) is the one that catches that, and this
/// test pins only the permuted route against the plain one.
///
/// Which scratch stays visible differs by host (measured on an aarch64 macOS host, recorded
/// from Linux CI for x86_64, #2442): faer's `matmul_imp` dispatches to its own
/// `private_gemm_x86` kernel on x86_64 when AVX2 or AVX-512 is present, and that kernel packs
/// nothing. On aarch64 it falls through to the `gemm` crate (`gemm-common`), whose `gemm()`
/// packs the right operand into a `dyn_stack::MemBuffer` it allocates per call - 1629440 B for
/// this fixture, plus a 6-byte flag `Vec` - so both routes count one payload-sized allocation.
#[test]
fn canonical_fallback_contraction_does_not_copy_operands() {
    let _serial = serialize();
    const N: usize = 95;
    let a =
        Tensor::from_vec_col_major([N, N], (0..N * N).map(|i| i as f64 * 0.5).collect()).unwrap();
    let b =
        Tensor::from_vec_col_major([N, N], (0..N * N).map(|i| (i % 7) as f64).collect()).unwrap();
    let config = DotGeneralConfig {
        lhs_contracting_dims: [1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: [].as_slice().into(),
        rhs_batch_dims: [].as_slice().into(),
    };
    let typed_b = b.as_typed::<f64>().expect("an f64 tensor");
    let view = typed_b.as_view();
    let transposed = view
        .transpose_view([1, 0])
        .unwrap()
        .into_tensor_read()
        .unwrap();

    // Each route gets its own fresh engine: one pool shared between them would hide the second
    // route's payload demand behind the buffers the first route already created.
    let mut plain_engine = CpuEngine::serial();
    let plain = count_large_allocations(|| {
        let session: &mut dyn Engine = &mut plain_engine;
        session
            .dot_general_read(
                TensorRead::from_tensor(&a),
                TensorRead::from_tensor(&b),
                &config,
            )
            .unwrap()
    });
    let mut fallback_engine = CpuEngine::serial();
    let fallback = count_large_allocations(|| {
        let session: &mut dyn Engine = &mut fallback_engine;
        session
            .dot_general_read(TensorRead::from_tensor(&a), transposed.clone(), &config)
            .unwrap()
    });

    // The plain route's own result calibrates the budget: the permuted route may pay its
    // provider's packing buffer, but no operand copy on top of it.
    assert!(
        fallback <= plain,
        "the canonical fallback copied an operand: {fallback} payload-sized allocations against \
         {plain} for the plain route"
    );
}
