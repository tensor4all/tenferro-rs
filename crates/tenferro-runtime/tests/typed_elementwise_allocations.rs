//! Per-op metadata-allocation budget for the typed elementwise surface (#2139).
//!
//! The umbrella's acceptance asks for zero metadata allocations on the common
//! small-rank elementwise path. This test counts *every* heap allocation made
//! inside one typed op, including the pooled output checkout, so a regression
//! that reintroduces a per-call shape `Vec` fails here even though no
//! payload-sized temporary is allocated (that is `binary_allocations.rs`).
//!
//! Same-shape and same-rank ops are covered at zero. A stride-0 broadcast still
//! pays the upstream `strided-basic` fused-plan builder (its `compress_dims`
//! scratch), which is a pinned-provider change, not a tenferro one; the arm
//! below records that residual rather than hiding it.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use smallvec::smallvec;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
use tenferro_tensor::backend::dot_general_output_shape;
use tenferro_tensor::{BackendSessionHost, DotGeneralConfig};

/// The `strided-basic` fused-plan scratch a stride-0 operand costs today.
const BROADCAST_FUSED_PLAN_ALLOCATIONS: usize = 16;

static ENABLED: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);

struct Allocator;

// SAFETY: every allocation is forwarded unchanged to the system allocator.
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ENABLED.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        // SAFETY: the caller's layout is forwarded verbatim.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` and `layout` originate from `System`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if ENABLED.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(size, Ordering::Relaxed);
        }
        // SAFETY: the caller's pointer and requested size are forwarded verbatim.
        unsafe { System.realloc(ptr, layout, size) }
    }
}

#[global_allocator]
static ALLOCATOR: Allocator = Allocator;

fn measure<T>(call: impl FnOnce() -> T) -> (T, usize, usize) {
    ALLOCATIONS.store(0, Ordering::SeqCst);
    BYTES.store(0, Ordering::SeqCst);
    ENABLED.store(true, Ordering::SeqCst);
    let out = call();
    ENABLED.store(false, Ordering::SeqCst);
    (
        out,
        ALLOCATIONS.load(Ordering::SeqCst),
        BYTES.load(Ordering::SeqCst),
    )
}

#[test]
fn typed_elementwise_allocates_no_heap_metadata_at_small_rank() {
    const LEN: usize = 16;
    let a = TypedTensor::<f64>::from_vec_col_major(vec![LEN], vec![2.0; LEN]).unwrap();
    let b = TypedTensor::<f64>::from_vec_col_major(vec![LEN], vec![3.0; LEN]).unwrap();
    let scalar = TypedTensor::<f64>::from_vec_col_major(vec![], vec![4.0]).unwrap();
    let matrix = TypedTensor::<f64>::from_vec_col_major(vec![4, 4], vec![1.0; 16]).unwrap();
    let row = TypedTensor::<f64>::from_vec_col_major(vec![4], vec![1.0; 4]).unwrap();
    let cube = TypedTensor::<f64>::from_vec_col_major(vec![2, 4, 4], vec![1.0; 32]).unwrap();
    let mut backend = CpuBackend::with_threads(1).unwrap();
    assert_eq!(backend.num_threads(), 1);
    backend
        .with_backend_session(|session| {
            macro_rules! budget {
                ($label:literal, $expected:expr, $call:expr) => {{
                    // Warm the output pool so the pooled checkout is a hit, then
                    // measure one public call: only the op's own metadata counts.
                    for _ in 0..4 {
                        drop($call(session).unwrap());
                    }
                    let (out, allocations, bytes) = measure(|| $call(session).unwrap());
                    drop(out);
                    eprintln!("typed {}: {allocations} allocations, {bytes} bytes", $label);
                    assert_eq!(
                        allocations, $expected,
                        "typed {} made {allocations} heap allocations ({bytes} bytes), \
                         expected {}",
                        $label, $expected
                    );
                    $expected
                }};
            }
            budget!("add n=16", 0, |s| a.add(&b, s));
            // #2139 item 3: the caller-output route allocates nothing at all for the
            // output, and writes it in place. The allocating form above reaches zero
            // only because the warmed pool serves its checkout.
            let mut out =
                TypedTensor::<f64>::from_vec_col_major(vec![LEN], vec![0.0; LEN]).unwrap();
            let ((), into_allocations, into_bytes) =
                measure(|| a.add_into(&b, &mut out, session).unwrap());
            eprintln!("typed add_into n=16: {into_allocations} allocations, {into_bytes} bytes");
            assert_eq!(
                into_allocations, 0,
                "typed add_into made {into_allocations} heap allocations ({into_bytes} bytes)"
            );
            assert_eq!(out.host_data().unwrap(), &[5.0; LEN]);
            // A wrong-shaped output is refused before anything is written.
            let mut wrong =
                TypedTensor::<f64>::from_vec_col_major(vec![LEN + 1], vec![9.0; LEN + 1]).unwrap();
            assert!(a.add_into(&b, &mut wrong, session).is_err());
            assert_eq!(wrong.host_data().unwrap(), &[9.0; LEN + 1]);
            budget!("mul n=16", 0, |s| a.mul(&b, s));
            budget!("sub n=16", 0, |s| a.sub(&b, s));
            budget!("mul rank-3", 0, |s| cube.mul(&cube, s));
            budget!("exp rank-2", 0, |s| matrix.exp(s));
            // Residual: the broadcast arm still pays the upstream fused plan.
            budget!(
                "mul row-into-matrix broadcast",
                BROADCAST_FUSED_PLAN_ALLOCATIONS,
                |s| matrix.mul(&row, s)
            );
            budget!("add rank-0 scalar", BROADCAST_FUSED_PLAN_ALLOCATIONS, |s| a
                .add(&scalar, s));

            // #2139 item 1: the eager `dot_general` path's own shape and axis
            // validation. Only tenferro's allocations are counted here, so the
            // budget does not move with the compiled provider or the pinned
            // `tcontract` plan builder: neither the four rank-sized `Vec<bool>`
            // duplicate-axis scratches nor the output shape itself reach the heap.
            let config = DotGeneralConfig {
                lhs_contracting_dims: smallvec![1],
                rhs_contracting_dims: smallvec![0],
                lhs_batch_dims: smallvec![],
                rhs_batch_dims: smallvec![],
            };
            let (shape, allocations, bytes) =
                measure(|| dot_general_output_shape(&[2, 2], &[2, 2], &config, "matmul").unwrap());
            eprintln!("dot_general_output_shape: {allocations} allocations, {bytes} bytes");
            assert_eq!(shape.as_slice(), &[2, 2]);
            assert_eq!(
                allocations, 0,
                "shape/axis validation must not allocate on the common small rank"
            );
        })
        .unwrap();
}
