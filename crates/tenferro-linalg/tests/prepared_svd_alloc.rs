use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

use num_complex::Complex64;
use tenferro_cpu::{CpuBackend, CpuBackendKind};
use tenferro_linalg::{
    PreparedFactorizationBackendExt, PreparedSvdBackendExt, SvdOptions, SvdOutputWrites,
};
use tenferro_tensor::{DType, Tensor, TensorRead, TensorView, TensorWrite, TypedTensorView};

struct CountingAllocator;

static ACTIVE: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static TEST_LOCK: Mutex<()> = Mutex::new(());

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ACTIVE.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: forwarding the allocator contract unchanged to `System`.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` and `layout` came from the forwarded `System` allocation.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if ACTIVE.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: forwarding the allocator contract unchanged to `System`.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn measured_allocations(run: impl FnOnce()) -> usize {
    ALLOCATIONS.store(0, Ordering::SeqCst);
    ACTIVE.store(true, Ordering::SeqCst);
    run();
    ACTIVE.store(false, Ordering::SeqCst);
    ALLOCATIONS.load(Ordering::SeqCst)
}

#[test]
fn prepared_svd_complete_warm_calls_allocate_zero() {
    let _serial = TEST_LOCK.lock().unwrap();
    let mut backend = CpuBackend::with_threads_and_kind(1, CpuBackendKind::Faer).unwrap();

    let input =
        Tensor::from_vec_col_major(vec![3, 2], vec![3.0_f64, 1.0, -2.0, 4.0, -1.0, 2.0]).unwrap();
    let plan = backend
        .prepare_svd([3, 2], DType::F64, SvdOptions::default())
        .unwrap();
    let mut workspace = plan.allocate_workspace(&mut backend).unwrap();
    let mut u = Tensor::from_vec_col_major(vec![3, 2], vec![0.0_f64; 6]).unwrap();
    let mut s = Tensor::from_vec_col_major(vec![2], vec![0.0_f64; 2]).unwrap();
    let mut vt = Tensor::from_vec_col_major(vec![2, 2], vec![0.0_f64; 4]).unwrap();
    plan.execute_into(
        &mut backend,
        &mut workspace,
        TensorRead::from_tensor(&input),
        SvdOutputWrites::new(
            TensorWrite::from_tensor(&mut u),
            TensorWrite::from_tensor(&mut s),
            TensorWrite::from_tensor(&mut vt),
        ),
    )
    .unwrap();
    let contiguous = measured_allocations(|| {
        plan.execute_into(
            &mut backend,
            &mut workspace,
            TensorRead::from_tensor(&input),
            SvdOutputWrites::new(
                TensorWrite::from_tensor(&mut u),
                TensorWrite::from_tensor(&mut s),
                TensorWrite::from_tensor(&mut vt),
            ),
        )
        .unwrap();
    });
    assert_eq!(contiguous, 0, "contiguous F64 warm allocations");

    let mut positive_storage = vec![0.0_f64; 13];
    let mut negative_storage = vec![0.0_f64; 13];
    for col in 0..2 {
        for row in 0..3 {
            let value = 1.0 + row as f64 + 2.0 * col as f64;
            positive_storage[1 + 2 * row + 7 * col] = value;
            negative_storage[5 - 2 * row + 7 * col] = value;
        }
    }
    let positive_prime = TypedTensorView::from_slice([3, 2], [2, 7], 1, &positive_storage).unwrap();
    plan.execute_into(
        &mut backend,
        &mut workspace,
        TensorRead::from_view(TensorView::F64(positive_prime)),
        SvdOutputWrites::new(
            TensorWrite::from_tensor(&mut u),
            TensorWrite::from_tensor(&mut s),
            TensorWrite::from_tensor(&mut vt),
        ),
    )
    .unwrap();
    let positive = TypedTensorView::from_slice([3, 2], [2, 7], 1, &positive_storage).unwrap();
    let positive_allocations = measured_allocations(|| {
        plan.execute_into(
            &mut backend,
            &mut workspace,
            TensorRead::from_view(TensorView::F64(positive)),
            SvdOutputWrites::new(
                TensorWrite::from_tensor(&mut u),
                TensorWrite::from_tensor(&mut s),
                TensorWrite::from_tensor(&mut vt),
            ),
        )
        .unwrap();
    });
    assert_eq!(
        positive_allocations, 0,
        "positive-stride F64 warm allocations"
    );

    let negative_prime =
        TypedTensorView::from_slice([3, 2], [-2, 7], 5, &negative_storage).unwrap();
    plan.execute_into(
        &mut backend,
        &mut workspace,
        TensorRead::from_view(TensorView::F64(negative_prime)),
        SvdOutputWrites::new(
            TensorWrite::from_tensor(&mut u),
            TensorWrite::from_tensor(&mut s),
            TensorWrite::from_tensor(&mut vt),
        ),
    )
    .unwrap();
    let negative = TypedTensorView::from_slice([3, 2], [-2, 7], 5, &negative_storage).unwrap();
    let negative_allocations = measured_allocations(|| {
        plan.execute_into(
            &mut backend,
            &mut workspace,
            TensorRead::from_view(TensorView::F64(negative)),
            SvdOutputWrites::new(
                TensorWrite::from_tensor(&mut u),
                TensorWrite::from_tensor(&mut s),
                TensorWrite::from_tensor(&mut vt),
            ),
        )
        .unwrap();
    });
    assert_eq!(
        negative_allocations, 0,
        "negative-stride F64 warm allocations"
    );

    let complex_input = Tensor::from_vec_col_major(
        vec![2, 2],
        vec![
            Complex64::new(2.0, 1.0),
            Complex64::new(-1.0, 0.5),
            Complex64::new(3.0, -2.0),
            Complex64::new(0.25, 1.5),
        ],
    )
    .unwrap();
    let complex_plan = backend
        .prepare_svd([2, 2], DType::C64, SvdOptions::default())
        .unwrap();
    let mut complex_workspace = complex_plan.allocate_workspace(&mut backend).unwrap();
    let mut complex_u =
        Tensor::from_vec_col_major(vec![2, 2], vec![Complex64::default(); 4]).unwrap();
    let mut complex_s = Tensor::from_vec_col_major(vec![2], vec![0.0_f64; 2]).unwrap();
    let mut complex_vt =
        Tensor::from_vec_col_major(vec![2, 2], vec![Complex64::default(); 4]).unwrap();
    complex_plan
        .execute_into(
            &mut backend,
            &mut complex_workspace,
            TensorRead::from_tensor(&complex_input),
            SvdOutputWrites::new(
                TensorWrite::from_tensor(&mut complex_u),
                TensorWrite::from_tensor(&mut complex_s),
                TensorWrite::from_tensor(&mut complex_vt),
            ),
        )
        .unwrap();
    let complex_allocations = measured_allocations(|| {
        complex_plan
            .execute_into(
                &mut backend,
                &mut complex_workspace,
                TensorRead::from_tensor(&complex_input),
                SvdOutputWrites::new(
                    TensorWrite::from_tensor(&mut complex_u),
                    TensorWrite::from_tensor(&mut complex_s),
                    TensorWrite::from_tensor(&mut complex_vt),
                ),
            )
            .unwrap();
    });
    assert_eq!(complex_allocations, 0, "contiguous C64 warm allocations");
}

#[test]
fn prepared_svd_session_repeated_leaf_calls_allocate_zero() {
    let _serial = TEST_LOCK.lock().unwrap();
    for workers in [1, 2, 4] {
        assert_session_leaf_allocations(workers);
    }
}

fn assert_session_leaf_allocations(workers: usize) {
    let mut backend =
        CpuBackend::with_threads_and_kind(workers, CpuBackendKind::Faer).unwrap();

    let f64_input =
        Tensor::from_vec_col_major(vec![3, 2], vec![3.0_f64, 1.0, -2.0, 4.0, -1.0, 2.0])
            .unwrap();
    let mut f64_strided_storage = vec![0.0_f64; 13];
    for col in 0..2 {
        for row in 0..3 {
            f64_strided_storage[1 + 2 * row + 7 * col] = 1.0 + row as f64 + 2.0 * col as f64;
        }
    }
    let f64_plan = backend
        .prepare_svd([3, 2], DType::F64, SvdOptions::default())
        .unwrap();
    let mut f64_workspace = f64_plan.allocate_workspace(&mut backend).unwrap();
    let mut f64_u = Tensor::from_vec_col_major(vec![3, 2], vec![0.0_f64; 6]).unwrap();
    let mut f64_s = Tensor::from_vec_col_major(vec![2], vec![0.0_f64; 2]).unwrap();
    let mut f64_vt = Tensor::from_vec_col_major(vec![2, 2], vec![0.0_f64; 4]).unwrap();

    let c64_input = Tensor::from_vec_col_major(
        vec![2, 2],
        vec![
            Complex64::new(2.0, 1.0),
            Complex64::new(-1.0, 0.5),
            Complex64::new(3.0, -2.0),
            Complex64::new(0.25, 1.5),
        ],
    )
    .unwrap();
    let mut c64_strided_storage = vec![Complex64::default(); 9];
    for col in 0..2 {
        for row in 0..2 {
            c64_strided_storage[1 + 2 * row + 5 * col] =
                Complex64::new(1.0 + row as f64, 0.5 + col as f64);
        }
    }
    let c64_plan = backend
        .prepare_svd([2, 2], DType::C64, SvdOptions::default())
        .unwrap();
    let mut c64_workspace = c64_plan.allocate_workspace(&mut backend).unwrap();
    let mut c64_u =
        Tensor::from_vec_col_major(vec![2, 2], vec![Complex64::default(); 4]).unwrap();
    let mut c64_s = Tensor::from_vec_col_major(vec![2], vec![0.0_f64; 2]).unwrap();
    let mut c64_vt =
        Tensor::from_vec_col_major(vec![2, 2], vec![Complex64::default(); 4]).unwrap();

    backend.with_prepared_factorization_session(|session| {
        let allocations = measured_allocations(|| {
            for _ in 0..128 {
                f64_plan
                    .execute_into_session(
                        session,
                        &mut f64_workspace,
                        TensorRead::from_tensor(&f64_input),
                        SvdOutputWrites::new(
                            TensorWrite::from_tensor(&mut f64_u),
                            TensorWrite::from_tensor(&mut f64_s),
                            TensorWrite::from_tensor(&mut f64_vt),
                        ),
                    )
                    .unwrap();
                let f64_strided =
                    TypedTensorView::from_slice([3, 2], [2, 7], 1, &f64_strided_storage).unwrap();
                f64_plan
                    .execute_into_session(
                        session,
                        &mut f64_workspace,
                        TensorRead::from_view(TensorView::F64(f64_strided)),
                        SvdOutputWrites::new(
                            TensorWrite::from_tensor(&mut f64_u),
                            TensorWrite::from_tensor(&mut f64_s),
                            TensorWrite::from_tensor(&mut f64_vt),
                        ),
                    )
                    .unwrap();
                c64_plan
                    .execute_into_session(
                        session,
                        &mut c64_workspace,
                        TensorRead::from_tensor(&c64_input),
                        SvdOutputWrites::new(
                            TensorWrite::from_tensor(&mut c64_u),
                            TensorWrite::from_tensor(&mut c64_s),
                            TensorWrite::from_tensor(&mut c64_vt),
                        ),
                    )
                    .unwrap();
                let c64_strided =
                    TypedTensorView::from_slice([2, 2], [2, 5], 1, &c64_strided_storage).unwrap();
                c64_plan
                    .execute_into_session(
                        session,
                        &mut c64_workspace,
                        TensorRead::from_view(TensorView::C64(c64_strided)),
                        SvdOutputWrites::new(
                            TensorWrite::from_tensor(&mut c64_u),
                            TensorWrite::from_tensor(&mut c64_s),
                            TensorWrite::from_tensor(&mut c64_vt),
                        ),
                    )
                    .unwrap();
            }
        });
        assert_eq!(
            allocations, 0,
            "prepared factorization session leaf calls with {workers} workers"
        );
    });
}
