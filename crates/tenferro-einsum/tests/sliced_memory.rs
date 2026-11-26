//! Heap high-water mark of a sliced einsum (#2676) against its plan.
//!
//! A test binary of its own: the counting global allocator sees every
//! allocation of the process, so it runs one serial measurement at a time.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use tenferro_cpu::CpuEngine;
use tenferro_einsum::{EinsumOptions, SliceOptions, SlicedEinsumPlan};
use tenferro_tensor::Tensor;

struct Counting;

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static SERIAL: Mutex<()> = Mutex::new(());

// SAFETY: forwards every call to `System` unchanged and only counts sizes.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller upholds `GlobalAlloc::alloc`'s contract.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            let now = CURRENT.fetch_add(layout.size(), Ordering::SeqCst) + layout.size();
            PEAK.fetch_max(now, Ordering::SeqCst);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `alloc` with this `layout`.
        unsafe { System.dealloc(ptr, layout) };
        CURRENT.fetch_sub(layout.size(), Ordering::SeqCst);
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Bytes allocated above the level at entry, at the highest point of `f`.
fn high_water<R>(f: impl FnOnce() -> R) -> (R, usize) {
    let base = CURRENT.load(Ordering::SeqCst);
    PEAK.store(base, Ordering::SeqCst);
    let result = f();
    (result, PEAK.load(Ordering::SeqCst) - base)
}

const SPEC: &str = "abc,cde,efa,bdf->";
const EXTENT: usize = 16;

fn operands() -> Vec<Tensor> {
    (0..4u64)
        .map(|k| {
            let n = EXTENT * EXTENT * EXTENT;
            let data = (0..n as u64)
                .map(|i| (((i * 2_654_435_761 + k * 97) % 1_000) as f64) / 1_000.0 - 0.5)
                .collect();
            Tensor::from_vec_col_major(vec![EXTENT; 3], data).unwrap()
        })
        .collect()
}

/// Run one GEMM on this thread first: faer's x86 GEMM backend reserves a
/// thread-local page pool (512 MiB) on a thread's first GEMM, once per
/// thread and independent of the contraction, which would mask the run.
fn warm_this_thread() {
    let a = Tensor::from_vec_col_major(vec![64, 64], vec![1.0_f64; 64 * 64]).unwrap();
    let mut engine = CpuEngine::serial();
    tenferro_einsum::EinsumOps::einsum(&[&a, &a, &a][..], "ij,jk,kl->il", &mut engine).unwrap();
}

/// One cold execution on a fresh serial engine: bytes at the high-water mark.
fn measure(plan: &SlicedEinsumPlan, inputs: &[&Tensor]) -> (Tensor, usize) {
    let mut engine = CpuEngine::serial();
    high_water(|| plan.execute(inputs, &mut engine).unwrap())
}

#[test]
fn the_heap_high_water_of_a_sliced_run_follows_the_per_slice_plan() {
    let _serial = SERIAL.lock().unwrap();
    warm_this_thread();
    let tensors = operands();
    let inputs: Vec<&Tensor> = tensors.iter().collect();
    let options = EinsumOptions::default();
    let dense =
        SlicedEinsumPlan::prepare(&inputs, SPEC, &options, &SliceOptions::default()).unwrap();
    let sliced = SlicedEinsumPlan::prepare(
        &inputs,
        SPEC,
        &options,
        &SliceOptions::default().target_log2_elements(10.0),
    )
    .unwrap();

    let (dense_value, dense_bytes) = measure(&dense, &inputs);
    let (sliced_value, sliced_bytes) = measure(&sliced, &inputs);
    // A warm loop allocates no intermediates: the arena is retained.
    let warm_bytes = {
        let mut engine = CpuEngine::serial();
        sliced.execute(&inputs, &mut engine).unwrap();
        high_water(|| sliced.execute(&inputs, &mut engine).unwrap()).1
    };
    let (d, r) = (dense.report(), sliced.report());
    let element = std::mem::size_of::<f64>();
    let planned_bytes = r.slice_peak_intermediate_elements as usize * element;
    let arena = sliced.cpu_arena_elements().unwrap();
    let dense_arena = dense.cpu_arena_elements().unwrap();
    let (a, b) = (
        dense_value.as_slice::<f64>().unwrap()[0],
        sliced_value.as_slice::<f64>().unwrap()[0],
    );
    println!(
        "dense : largest {:>8} peak {:>8} arena {:>8} elements, heap high-water {:>9} B, flops {:e}",
        d.slice_largest_intermediate_elements,
        d.slice_peak_intermediate_elements,
        dense_arena,
        dense_bytes,
        d.total_flops
    );
    println!(
        "sliced: largest {:>8} peak {:>8} arena {:>8} elements, heap high-water {:>9} B, flops {:e} ({} slices of {:?} over labels {:?}, {:.3}x the unsliced tree)",
        r.slice_largest_intermediate_elements,
        r.slice_peak_intermediate_elements,
        arena,
        sliced_bytes,
        r.total_flops,
        r.slice_count,
        r.slice_extents,
        r.sliced_labels
            .iter()
            .map(|&l| char::from_u32(l).unwrap())
            .collect::<String>(),
        r.total_flops / r.unsliced_flops
    );
    println!("warm  : sliced re-run heap high-water {warm_bytes} B");
    println!(
        "value : dense {a:.17e} sliced {b:.17e} |diff| {:e} rel {:e} bit-identical {}",
        (a - b).abs(),
        ((a - b) / a).abs(),
        a.to_bits() == b.to_bits()
    );

    assert!(r.slice_largest_intermediate_elements <= 1024.0);
    assert!(((a - b) / a).abs() <= 1e-12);
    // The executor's arena holds one slice's intermediates as planned.
    assert!(arena as f64 <= r.slice_peak_intermediate_elements);
    // Everything one cold sliced run allocates (arena, per-slice result,
    // views, numerical plan, engine workspaces) stays within the planned
    // per-slice peak plus a fixed 128 KiB of bookkeeping that does not grow
    // with the network (35-80 KiB measured on x86-64 Linux), far below the
    // dense run.
    assert!(
        sliced_bytes <= planned_bytes + 128 * 1024,
        "{sliced_bytes} B against a planned {planned_bytes} B"
    );
    assert!(
        sliced_bytes * 8 < dense_bytes,
        "{sliced_bytes} B vs dense {dense_bytes} B"
    );
    assert!(
        warm_bytes < planned_bytes + 32 * 1024,
        "warm re-run {warm_bytes} B"
    );
}
