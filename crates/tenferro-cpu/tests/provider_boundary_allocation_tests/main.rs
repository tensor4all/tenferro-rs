use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use tenferro_cpu::provider::{
    __dispatch_gemm_for_allocation_probe, CpuGemmProvider, CpuGemmRequest, CpuGroupedGemmRequest,
    CpuProviderContext, CpuProviderOutcome, CpuProviderUnsupported,
};
use tenferro_cpu::CpuContext;
use tenferro_tensor::{DType, DotGeneralAccumulation, Tensor, TensorRead, TensorWrite};

struct CountingAllocator;

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        // SAFETY: this allocator forwards the unchanged layout to System.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: ptr and layout came from the corresponding System allocation.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(new_size, Ordering::Relaxed);
        }
        // SAFETY: ptr and layout came from System, and new_size is forwarded unchanged.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

#[derive(Debug)]
struct UnsupportedGemm;

impl CpuGemmProvider for UnsupportedGemm {
    fn gemm(
        &self,
        _context: &CpuProviderContext<'_>,
        _request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        Ok(CpuProviderOutcome::Unsupported(
            CpuProviderUnsupported::RuntimeUnavailable,
        ))
    }

    fn strided_batched_gemm(
        &self,
        _context: &CpuProviderContext<'_>,
        _request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        Ok(CpuProviderOutcome::Unsupported(
            CpuProviderUnsupported::RuntimeUnavailable,
        ))
    }

    fn grouped_gemm(
        &self,
        _context: &CpuProviderContext<'_>,
        _request: CpuGroupedGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        Ok(CpuProviderOutcome::Unsupported(
            CpuProviderUnsupported::RuntimeUnavailable,
        ))
    }
}

#[test]
fn warmed_borrowed_request_dispatch_does_not_allocate() {
    let provider = UnsupportedGemm;
    let context = CpuContext::with_threads(1).unwrap();
    let lhs = Tensor::from_vec_col_major(vec![1, 1], vec![2.0_f64]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![1, 1], vec![3.0_f64]).unwrap();
    let mut output = Tensor::from_vec_col_major(vec![1, 1], vec![0.0_f64]).unwrap();
    let lhs = TensorRead::from_tensor(&lhs);
    let rhs = TensorRead::from_tensor(&rhs);
    let mut output = TensorWrite::from_tensor(&mut output);
    let accumulation = DotGeneralAccumulation::overwrite(DType::F64).unwrap();

    for _ in 0..32 {
        let _ = black_box(
            __dispatch_gemm_for_allocation_probe(
                &provider,
                &context,
                &lhs,
                &rhs,
                &mut output,
                accumulation,
            )
            .unwrap(),
        );
    }

    ALLOCATIONS.store(0, Ordering::Relaxed);
    BYTES.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::SeqCst);
    for _ in 0..10_000 {
        let _ = black_box(
            __dispatch_gemm_for_allocation_probe(
                &provider,
                &context,
                &lhs,
                &rhs,
                &mut output,
                accumulation,
            )
            .unwrap(),
        );
    }
    COUNTING.store(false, Ordering::SeqCst);

    assert_eq!(ALLOCATIONS.load(Ordering::Relaxed), 0);
    assert_eq!(BYTES.load(Ordering::Relaxed), 0);
}
