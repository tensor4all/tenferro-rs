//! Live-allocation growth of one prepared program with an inline constant.
//!
//! Reproducer for tensor4all/tenferro-rs#2759. Each arm prepares one program
//! once, warms it up, then runs it 1000 times and prints the live bytes the
//! process has allocated but not freed, measured every 100 runs.
//!
//! `cargo run --release -p tenferro-runtime --example scale_leak_repro`
//!
//! A steady program shows the same number at every sample. Growth means the
//! run keeps allocating: `scale_real` and `scale_complex` build an inline
//! scalar `Constant`, whose per-run upload used to be retained by the CPU
//! buffer pool even though no checkout ever asked for a one-element buffer.

use num_complex::Complex64;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicIsize, Ordering::Relaxed};
use tenferro_cpu::CpuEngine;
use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TraceContext, TracedTensor};
use tenferro_tensor::{DType, Tensor, TensorRead};

struct Counting;

static LIVE: AtomicIsize = AtomicIsize::new(0);

// SAFETY: every method forwards to `System`, so the allocator keeps its
// contract; the counter only observes the layout sizes it is given.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        LIVE.fetch_add(layout.size() as isize, Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size() as isize, Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        LIVE.fetch_add(new_size as isize - layout.size() as isize, Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn arm(name: &str, build: impl Fn(&TraceContext, &TracedTensor) -> TracedTensor) {
    let trace = TraceContext::new();
    let x = trace
        .input_concrete_shape(DType::C64, &[2, 2, 2048])
        .unwrap();
    let y = build(&trace, &x);
    let program = GraphCompiler::new().compile_many(&[&y]).unwrap();
    let mut engine = CpuEngine::serial();
    let mut builder = Runtime::builder();
    builder
        .register_engine(tenferro_cpu::runtime_engine_registration(&engine).unwrap())
        .unwrap();
    let runtime = builder.build().unwrap();
    let data = Tensor::from_vec_col_major(
        vec![2, 2, 2048],
        vec![Complex64::new(1.0, 0.5); 2 * 2 * 2048],
    )
    .unwrap();
    let reads = [TensorRead::from_tensor(&data)];
    let target = engine.target();
    let prepared = runtime
        .prepare(&program, &reads, std::slice::from_ref(&target))
        .unwrap();
    for _ in 0..5 {
        drop(runtime.run(&mut [&mut engine], &prepared, &reads).unwrap());
    }
    let base = LIVE.load(Relaxed);
    print!("{name:24} live bytes after every 100 runs:");
    for i in 1..=1000 {
        drop(runtime.run(&mut [&mut engine], &prepared, &reads).unwrap());
        if i % 100 == 0 {
            print!(" {}", LIVE.load(Relaxed) - base);
        }
    }
    println!();
}

fn main() {
    arm("control x * c[1,1,1]", |t, x| {
        let c = Tensor::from_vec_col_major(vec![1, 1, 1], vec![Complex64::new(0.5, 0.0)]).unwrap();
        (x * &t.from_tensor_concrete_shape(c).unwrap()).unwrap()
    });
    arm("scale_real(0.5)", |_, x| x.scale_real(0.5).unwrap());
    arm("scale_complex(-i)", |_, x| {
        x.scale_complex(Complex64::new(0.0, -1.0)).unwrap()
    });
}
