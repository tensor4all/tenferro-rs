//! Public same-shape wrappers must not allocate input-sized temporaries.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use tenferro_cpu::CpuBackend;
use tenferro_runtime::TensorSessionOpsExt;
use tenferro_tensor::TensorRead;
use tenferro_tensor::{BackendSession, BackendSessionHost, CompareDir, DotGeneralConfig, Tensor};

const LEN: usize = 4096;
const INPUT_BYTES: usize = LEN * size_of::<f64>();
static ENABLED: AtomicBool = AtomicBool::new(false);
static LARGE_ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static TOTAL_ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static TOTAL_BYTES: AtomicUsize = AtomicUsize::new(0);
struct Allocator;

// SAFETY: all allocations and deallocations are forwarded unchanged to System.
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ENABLED.load(Ordering::Relaxed) {
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
        if ENABLED.load(Ordering::Relaxed) {
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

fn allocations(f: impl FnOnce() -> Tensor) -> (Tensor, usize) {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            ENABLED.store(false, Ordering::SeqCst);
        }
    }
    LARGE_ALLOCATIONS.store(0, Ordering::Relaxed);
    TOTAL_ALLOCATIONS.store(0, Ordering::Relaxed);
    TOTAL_BYTES.store(0, Ordering::Relaxed);
    ENABLED.store(true, Ordering::SeqCst);
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

#[test]
fn same_shape_wrappers_do_not_allocate_operand_copies() {
    let a = Tensor::from_vec_col_major([LEN], vec![2.0_f64; LEN]).unwrap();
    let b = Tensor::from_vec_col_major([LEN], vec![3.0_f64; LEN]).unwrap();
    let mut backend = CpuBackend::with_threads(1).unwrap();
    assert_eq!(backend.num_threads(), 1);
    backend
        .with_backend_session(|session: &mut dyn BackendSession| {
            macro_rules! check {
                ($name:ident, $read:ident) => {{
                    drop(
                        session
                            .$read(TensorRead::from_tensor(&a), TensorRead::from_tensor(&b))
                            .unwrap(),
                    );
                    eprintln!(
                        "{}: raw then public (effective backend threads=1)",
                        stringify!($name)
                    );
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
                    let baseline = count_large_allocations(|| {
                        session
                            .$read(TensorRead::from_tensor(&a), TensorRead::from_tensor(&b))
                            .unwrap()
                    });
                    let actual = count_large_allocations(|| a.$name(&b, session).unwrap());
                    assert!(
                        actual <= baseline,
                        "{}: public={actual}, backend={baseline}",
                        stringify!($name)
                    );
                }};
            }
            check!(add, add_read);
            check!(sub, sub_read);
            check!(mul, mul_read);
            check!(div, div_read);
            check!(rem, rem_read);
            check!(pow, pow_read);
            check!(maximum, maximum_read);
            check!(minimum, minimum_read);
            eprintln!("compare: raw then public");
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
            let baseline = count_large_allocations(|| {
                session
                    .compare_read(
                        TensorRead::from_tensor(&a),
                        TensorRead::from_tensor(&b),
                        &CompareDir::Lt,
                    )
                    .unwrap()
            });
            let actual =
                count_large_allocations(|| a.compare(&b, CompareDir::Lt, session).unwrap());
            assert!(
                actual <= baseline,
                "compare: public={actual}, backend={baseline}"
            );
            eprintln!("clamp: raw then public");
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
            let baseline = count_large_allocations(|| {
                session
                    .clamp_read(
                        TensorRead::from_tensor(&a),
                        TensorRead::from_tensor(&a),
                        TensorRead::from_tensor(&b),
                    )
                    .unwrap()
            });
            let actual = count_large_allocations(|| a.clamp(&a, &b, session).unwrap());
            assert!(
                actual <= baseline,
                "clamp: public={actual}, backend={baseline}"
            );
            let condition = Tensor::from_vec_col_major([LEN], vec![true; LEN]).unwrap();
            eprintln!("select: raw then public");
            let raw = session
                .select_read(
                    tenferro_tensor::TensorRead::from_tensor(&condition),
                    tenferro_tensor::TensorRead::from_tensor(&a),
                    tenferro_tensor::TensorRead::from_tensor(&b),
                )
                .unwrap();
            let public = condition.where_select(&a, &b, session).unwrap();
            assert_eq!(
                public.as_slice::<f64>().unwrap(),
                raw.as_slice::<f64>().unwrap()
            );
            drop(raw);
            drop(public);
            let baseline = count_large_allocations(|| {
                session
                    .select_read(
                        tenferro_tensor::TensorRead::from_tensor(&condition),
                        tenferro_tensor::TensorRead::from_tensor(&a),
                        tenferro_tensor::TensorRead::from_tensor(&b),
                    )
                    .unwrap()
            });
            let actual =
                count_large_allocations(|| condition.where_select(&a, &b, session).unwrap());
            assert!(
                actual <= baseline,
                "select: public={actual}, backend={baseline}"
            );
        })
        .unwrap();
}

/// #2141 item 3: the canonical-fallback contraction must not copy its operands.
///
/// A permuted right operand makes the provider take its canonical path. The operand
/// copies that path once made were removed with the plan work
/// (`perf-1938-cpu-followups`: 25 → 15 allocations per call); what remains is the
/// provider's own view and plan metadata (strided-rs `Arc` dims, an HPTT plan rebuild),
/// which is not payload-sized. This pins the payload half: no input-sized temporary
/// with either operand layout.
#[test]
fn canonical_fallback_contraction_does_not_copy_operands() {
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
    let mut backend = CpuBackend::with_threads(1).unwrap();
    backend
        .with_backend_session(|session: &mut dyn BackendSession| {
            let plain = TensorRead::from_tensor(&a);
            let typed_b = b.as_typed::<f64>().expect("an f64 tensor");
            let view = typed_b.as_view();
            let transposed = view
                .transpose_view([1, 0])
                .unwrap()
                .into_tensor_read()
                .unwrap();
            // Warm the pool on both routes before counting.
            drop(
                session
                    .dot_general_read(plain.clone(), TensorRead::from_tensor(&b), &config)
                    .unwrap(),
            );
            drop(
                session
                    .dot_general_read(plain.clone(), transposed.clone(), &config)
                    .unwrap(),
            );

            let baseline = count_large_allocations(|| {
                session
                    .dot_general_read(plain.clone(), TensorRead::from_tensor(&b), &config)
                    .unwrap()
            });
            let fallback = count_large_allocations(|| {
                session
                    .dot_general_read(plain.clone(), transposed.clone(), &config)
                    .unwrap()
            });
            assert_eq!(baseline, 0, "the canonical route copied an operand");
            assert_eq!(fallback, 0, "the canonical fallback copied an operand");
            Ok::<_, tenferro_tensor::Error>(())
        })
        .unwrap()
        .unwrap();
}
