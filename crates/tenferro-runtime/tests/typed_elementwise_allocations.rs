//! Per-op metadata-allocation budget for the typed elementwise surface (#2139).
//!
//! The umbrella's acceptance asks for zero metadata allocations on the common
//! small-rank elementwise path. This test counts *every* heap allocation made
//! inside one typed op, including the pooled output checkout, so a regression
//! that reintroduces a per-call shape `Vec` fails here even though no
//! payload-sized temporary is allocated (that is `binary_allocations.rs`).
//!
//! Same-shape and same-rank ops are covered at zero. A stride-0 broadcast is
//! also zero: `strided-fused 0.6.0` builds its plan with inline vectors, so the
//! `strided-basic` fused-plan scratch this arm used to record is gone. The
//! budgets count *Rust* heap allocations only; provider C allocations
//! (LAPACK/BLAS scratch) are outside the counting allocator and out of scope
//! here.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use tenferro_cpu::{scalar_binary_into, AddOp, CpuBackend};
use tenferro_runtime::{TensorSessionOpsExt, TypedTensor, TypedTensorSessionOpsExt};
use tenferro_tensor::backend::dot_general_output_shape;
use tenferro_tensor::{
    Engine, BackendSessionHost, ContractionScalar, DotGeneralAccumulation,
    DotGeneralConfig, DynRank, Host, Tensor, TensorRead, TensorView, TensorWrite,
};
use tinyvec::tiny_vec;

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
                    for _ in 0..16 {
                        let _ = $call(session).unwrap();
                    }
                    let (_out, allocations, bytes) = measure(|| $call(session).unwrap());
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
            // strided-fused 0.6 stores the fused plan inline, so a broadcast
            // add is now metadata-allocation free like the same-shape arms.
            budget!("mul row-into-matrix broadcast", 0, |s| matrix.mul(&row, s));
            budget!("add rank-0 scalar", 0, |s| a.add(&scalar, s));
            // A rank-4 transpose materializes into a pooled output; the
            // permutation validation and the output metadata are allocation
            // free, so the whole call is zero at inline rank (#2178 item 7).
            let rank4 =
                TypedTensor::<f64>::from_vec_col_major(vec![2, 2, 2, 2], vec![1.0; 16]).unwrap();
            budget!("transpose rank-4", 0, |s| rank4.transpose(&[3, 2, 1, 0], s));
            // A permuted borrowed view operand materializes through the same
            // view path and is metadata-allocation free at inline rank.
            let permuted_view = matrix.as_view().transpose_view([1, 0]).unwrap();
            let permuted_read = TensorRead::from_view(TensorView::F64(permuted_view));
            let matrix_read = TensorRead::from_view(TensorView::F64(matrix.as_view()));
            budget!("add permuted view", 0, |s: &mut dyn Engine| s
                .add_read(permuted_read.clone(), matrix_read.clone()));
            // An analytic op over a stride-0 broadcast view stays zero: the
            // view carries its layout inline and the kernel reads through it.
            let broadcast_read = TensorRead::from_view(TensorView::F64(row.as_view()))
                .broadcast_in_dim_view(&[4, 4], &[1])
                .unwrap();
            budget!("exp broadcast view", 0, |s: &mut dyn Engine| s
                .exp_read(broadcast_read.clone()));
            // Concatenate and a beta != 1 contraction on the erased session
            // surface, at inline rank.
            let concat_a = Tensor::from_typed::<f64>(
                TypedTensor::from_vec_col_major(vec![4], vec![1.0; 4]).unwrap(),
            );
            let concat_b = Tensor::from_typed::<f64>(
                TypedTensor::from_vec_col_major(vec![4], vec![2.0; 4]).unwrap(),
            );
            // Concatenate keeps its typed input list and output shape inline and
            // builds a bounded `ErasedConcatenatePlan` through the plan cache.
            budget!("concatenate", 0, |s: &mut dyn Engine| {
                Tensor::concatenate(&[&concat_a, &concat_b], 0, s)
            });
            let dot_lhs = Tensor::from_typed::<f64>(
                TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0; 4]).unwrap(),
            );
            let dot_rhs = Tensor::from_typed::<f64>(
                TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0; 4]).unwrap(),
            );
            let mut dot_out = Tensor::from_typed::<f64>(
                TypedTensor::from_vec_col_major(vec![2, 2], vec![0.0; 4]).unwrap(),
            );
            let dot_config = DotGeneralConfig {
                lhs_contracting_dims: tiny_vec!([usize; 4] => 1),
                rhs_contracting_dims: tiny_vec!([usize; 4] => 0),
                lhs_batch_dims: tiny_vec!([usize; 4]),
                rhs_batch_dims: tiny_vec!([usize; 4]),
            };
            let beta_half = DotGeneralAccumulation::scaled(
                ContractionScalar::F64(1.0),
                ContractionScalar::F64(0.5),
            )
            .unwrap();
            budget!("dot accum beta!=1", 0, |s: &mut dyn Engine| {
                s.dot_general_read_into_accum(
                    TensorRead::from_tensor(&dot_lhs),
                    TensorRead::from_tensor(&dot_rhs),
                    &dot_config,
                    beta_half,
                    TensorWrite::from_tensor(&mut dot_out),
                )
            });

            // #2139 item 1: the eager `dot_general` path's own shape and axis
            // validation. Only tenferro's allocations are counted here, so the
            // budget does not move with the compiled provider or the pinned
            // `tcontract` plan builder: neither the four rank-sized `Vec<bool>`
            // duplicate-axis scratches nor the output shape itself reach the heap.
            let config = DotGeneralConfig {
                lhs_contracting_dims: tiny_vec!([usize; 4] => 1),
                rhs_contracting_dims: tiny_vec!([usize; 4] => 0),
                lhs_batch_dims: tiny_vec!([usize; 4]),
                rhs_batch_dims: tiny_vec!([usize; 4]),
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

    // #2178 item 3: `scalar_binary_into` proves destination and lhs shapes equal
    // before building its views, so the destination metadata borrows the
    // validated lhs shape. Counting every allocation (not only payload-sized
    // ones) catches the former `destination.shape().to_vec()`, which allocated
    // once for any rank in 1..=8. Setup lives outside the measured window and
    // the numeric result is checked so the arm cannot pass by skipping the copy.
    for rank in 1..=8 {
        let shape = vec![2usize; rank];
        let len = shape.iter().product::<usize>();
        let lhs = tenferro_tensor::TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(
            shape.clone(),
            vec![2.0; len],
        )
        .unwrap();
        let rhs = tenferro_tensor::TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(
            shape.clone(),
            vec![3.0; len],
        )
        .unwrap();
        let mut out = tenferro_tensor::TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(
            shape,
            vec![0.0; len],
        )
        .unwrap();
        let (result, allocations, bytes) =
            measure(|| scalar_binary_into::<f64, AddOp>("add", &mut out, &lhs, &rhs));
        result.unwrap();
        eprintln!("scalar_binary_into rank {rank}: {allocations} allocations, {bytes} bytes");
        assert_eq!(
            allocations, 0,
            "scalar_binary_into rank {rank} made {allocations} heap allocations ({bytes} bytes)"
        );
        assert!(
            out.as_slice().iter().all(|&value| value == 5.0),
            "scalar_binary_into rank {rank} numeric result"
        );
    }

    // The same budget under a two-worker backend. The length is above
    // strided-basic's serial floor (lengths <= 32768 stay serial), so these
    // arms exercise worker execution, not just a two-worker configuration.
    // This lives in the same test as the one-worker arms because the counting
    // allocator is process-wide: a second test running in parallel would
    // pollute it. Provider scratch is pooled; C allocations are out of scope.
    const WORKER_LEN: usize = 65_536;
    let two_thread_a =
        TypedTensor::<f64>::from_vec_col_major(vec![WORKER_LEN], vec![2.0; WORKER_LEN]).unwrap();
    let two_thread_b =
        TypedTensor::<f64>::from_vec_col_major(vec![WORKER_LEN], vec![3.0; WORKER_LEN]).unwrap();
    let mut two_thread_backend = CpuBackend::with_threads(2).unwrap();
    assert_eq!(two_thread_backend.num_threads(), 2);
    two_thread_backend
        .with_backend_session(|session| {
            let mut budget =
                |call: &dyn Fn(&mut dyn tenferro_tensor::Engine) -> TypedTensor<f64>| {
                    for _ in 0..16 {
                        drop(call(session));
                    }
                    let (_, allocations, bytes) = measure(|| call(session));
                    (allocations, bytes)
                };
            let (add_allocations, add_bytes) =
                budget(&|s| two_thread_a.add(&two_thread_b, s).unwrap());
            eprintln!("2T add: {add_allocations} allocations, {add_bytes} bytes");
            assert_eq!(add_allocations, 0, "2T add allocated {add_bytes} bytes");
            // The one-axis reduction validates axes and builds its output shape
            // without heap scratch at inline rank.
            let (reduce_allocations, reduce_bytes) =
                budget(&|s| two_thread_a.reduce_sum(Some(&[0]), s).unwrap());
            eprintln!("2T reduce_sum: {reduce_allocations} allocations, {reduce_bytes} bytes");
            assert_eq!(
                reduce_allocations, 0,
                "2T reduce_sum allocated {reduce_bytes} bytes"
            );
            let (exp_allocations, exp_bytes) = budget(&|s| two_thread_a.exp(s).unwrap());
            eprintln!("2T exp: {exp_allocations} allocations, {exp_bytes} bytes");
            assert_eq!(exp_allocations, 0, "2T exp allocated {exp_bytes} bytes");

            let concat_a = Tensor::from_typed::<f64>(
                TypedTensor::from_vec_col_major(vec![4], vec![1.0; 4]).unwrap(),
            );
            let concat_b = Tensor::from_typed::<f64>(
                TypedTensor::from_vec_col_major(vec![4], vec![2.0; 4]).unwrap(),
            );
            for _ in 0..16 {
                drop(Tensor::concatenate(&[&concat_a, &concat_b], 0, session).unwrap());
            }
            let (_, concat_allocations, concat_bytes) =
                measure(|| Tensor::concatenate(&[&concat_a, &concat_b], 0, session).unwrap());
            eprintln!("2T concatenate: {concat_allocations} allocations, {concat_bytes} bytes");
            assert_eq!(
                concat_allocations, 0,
                "2T concatenate allocated {concat_bytes} bytes"
            );

            let dot_lhs = Tensor::from_typed::<f64>(
                TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0; 4]).unwrap(),
            );
            let dot_rhs = Tensor::from_typed::<f64>(
                TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0; 4]).unwrap(),
            );
            let mut dot_out = Tensor::from_typed::<f64>(
                TypedTensor::from_vec_col_major(vec![2, 2], vec![0.0; 4]).unwrap(),
            );
            let dot_config = DotGeneralConfig {
                lhs_contracting_dims: tiny_vec!([usize; 4] => 1),
                rhs_contracting_dims: tiny_vec!([usize; 4] => 0),
                lhs_batch_dims: tiny_vec!([usize; 4]),
                rhs_batch_dims: tiny_vec!([usize; 4]),
            };
            let beta_half = DotGeneralAccumulation::scaled(
                ContractionScalar::F64(1.0),
                ContractionScalar::F64(0.5),
            )
            .unwrap();
            for _ in 0..16 {
                session
                    .dot_general_read_into_accum(
                        TensorRead::from_tensor(&dot_lhs),
                        TensorRead::from_tensor(&dot_rhs),
                        &dot_config,
                        beta_half,
                        TensorWrite::from_tensor(&mut dot_out),
                    )
                    .unwrap();
            }
            let (_, accum_allocations, accum_bytes) = measure(|| {
                session
                    .dot_general_read_into_accum(
                        TensorRead::from_tensor(&dot_lhs),
                        TensorRead::from_tensor(&dot_rhs),
                        &dot_config,
                        beta_half,
                        TensorWrite::from_tensor(&mut dot_out),
                    )
                    .unwrap()
            });
            eprintln!("2T dot accum beta!=1: {accum_allocations} allocations, {accum_bytes} bytes");
            assert_eq!(
                accum_allocations, 0,
                "2T dot accum allocated {accum_bytes} bytes"
            );
        })
        .unwrap();
}
