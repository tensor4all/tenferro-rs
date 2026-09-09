//! Public same-shape wrappers must not allocate input-sized temporaries.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use tenferro_cpu::CpuBackend;
use tenferro_runtime::TensorSessionOpsExt;
use tenferro_tensor::{BackendSession, BackendSessionHost, CompareDir, Tensor};

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

#[test]
fn same_shape_wrappers_do_not_allocate_operand_copies() {
    let a = Tensor::from_vec_col_major([LEN], vec![2.0_f64; LEN]).unwrap();
    let b = Tensor::from_vec_col_major([LEN], vec![3.0_f64; LEN]).unwrap();
    let mut backend = CpuBackend::with_threads(1).unwrap();
    assert_eq!(backend.num_threads(), 1);
    backend.with_backend_session(|session: &mut dyn BackendSession| {
        macro_rules! check {
            ($name:ident) => {{
                drop(session.$name(&a, &b).unwrap());
                eprintln!(
                    "{}: raw then public (effective backend threads=1)",
                    stringify!($name)
                );
                let (raw, baseline) = allocations(|| session.$name(&a, &b).unwrap());
                let (public, actual) = allocations(|| a.$name(&b, session).unwrap());
                assert_eq!(
                    public.as_slice::<f64>().unwrap(),
                    raw.as_slice::<f64>().unwrap()
                );
                assert!(
                    actual <= baseline,
                    "{}: public={actual}, backend={baseline}",
                    stringify!($name)
                );
            }};
        }
        check!(add);
        check!(sub);
        check!(mul);
        check!(div);
        check!(rem);
        check!(pow);
        check!(maximum);
        check!(minimum);
        eprintln!("compare: raw then public");
        let (raw, baseline) = allocations(|| session.compare(&a, &b, &CompareDir::Lt).unwrap());
        let (public, actual) = allocations(|| a.compare(&b, CompareDir::Lt, session).unwrap());
        assert_eq!(
            public.as_slice::<bool>().unwrap(),
            raw.as_slice::<bool>().unwrap()
        );
        assert!(
            actual <= baseline,
            "compare: public={actual}, backend={baseline}"
        );
        eprintln!("clamp: raw then public");
        let (raw, baseline) = allocations(|| session.clamp(&a, &a, &b).unwrap());
        let (public, actual) = allocations(|| a.clamp(&a, &b, session).unwrap());
        assert_eq!(
            public.as_slice::<f64>().unwrap(),
            raw.as_slice::<f64>().unwrap()
        );
        assert!(
            actual <= baseline,
            "clamp: public={actual}, backend={baseline}"
        );
        let condition = Tensor::from_vec_col_major([LEN], vec![true; LEN]).unwrap();
        eprintln!("select: raw then public");
        let (raw, baseline) = allocations(|| session.select(&condition, &a, &b).unwrap());
        let (public, actual) = allocations(|| condition.where_select(&a, &b, session).unwrap());
        assert_eq!(
            public.as_slice::<f64>().unwrap(),
            raw.as_slice::<f64>().unwrap()
        );
        assert!(
            actual <= baseline,
            "select: public={actual}, backend={baseline}"
        );
    });
}
