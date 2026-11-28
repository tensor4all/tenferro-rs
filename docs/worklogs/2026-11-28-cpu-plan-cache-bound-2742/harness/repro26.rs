//! Issue #2742's 26-qubit reproducer, generalised to N distinct gate positions (first N of the
//! issue's 8, then pairs enumerated in a fixed order), two rounds. Env: T2742_LIMIT_BYTES.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::Instant;
use num_complex::Complex32;
use tenferro_cpu::{CpuEngine, Pool};
use tenferro_einsum::{EinsumOps, EinsumSubscripts};
use tenferro_tensor::Tensor;
struct Counting;
static CUR: AtomicUsize = AtomicUsize::new(0);
static HIGH: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = System.alloc(l);
        if !p.is_null() { let c = CUR.fetch_add(l.size(), Relaxed) + l.size(); HIGH.fetch_max(c, Relaxed); }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) { System.dealloc(p, l); CUR.fetch_sub(l.size(), Relaxed); }
}
#[global_allocator]
static GLOBAL: Counting = Counting;
fn gib(b: usize) -> f64 { b as f64 / (1u64 << 30) as f64 }
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n: usize = std::env::args().nth(1).map_or(8, |s| s.parse().unwrap());
    let (qubits, threads) = (26usize, 16usize);
    let state = Tensor::from_vec_col_major(vec![2; qubits],
        (0..1usize << qubits).map(|i| Complex32::new((i % 7) as f32, (i % 3) as f32)).collect::<Vec<_>>())?;
    let gate = Tensor::from_vec_col_major(vec![2; 4], (0..16).map(|i| Complex32::new(i as f32, 0.0)).collect::<Vec<_>>())?;
    let mut engine = CpuEngine::new(Pool::builder().threads(threads).build()?);
    if let Ok(v) = std::env::var("T2742_LIMIT_BYTES") { engine.plan_cache().set_contraction_limit_bytes(v.parse().unwrap()); }
    let labels: Vec<u32> = (0..qubits as u32).collect();
    let mut pairs: Vec<(u32, u32)> = vec![(0, 13), (3, 17), (6, 21), (9, 25), (3, 12), (7, 15), (11, 18), (15, 21)];
    'o: for a in 0..qubits as u32 { for b in a + 1..qubits as u32 { if pairs.len() >= n { break 'o; } if !pairs.contains(&(a, b)) { pairs.push((a, b)); } } }
    pairs.truncate(n);
    let base = CUR.load(Relaxed);
    HIGH.store(base, Relaxed);
    let mut rt = [0f64; 2];
    for round in 0..2 {
        let t_round = Instant::now();
        for &(p, q) in &pairs {
            let out: Vec<u32> = labels.iter().map(|&l| if l == p { 1000 } else if l == q { 1001 } else { l }).collect();
            let subs = EinsumSubscripts::new(&[&labels[..], &[1000, 1001, p, q][..]], &out);
            drop([&state, &gate].einsum(&subs, &mut engine)?);
        }
        rt[round] = t_round.elapsed().as_secs_f64();
        let st = engine.plan_cache().contraction_stats();
        println!("T2742 positions {n} round {} seconds {:.3} highwater_over_base_gib {:.3} held_over_base_gib {:.3} entries {} retained_gib {:.4} evictions {} hits {} misses {}",
            round + 1, rt[round], gib(HIGH.load(Relaxed) - base), gib(CUR.load(Relaxed) - base), st.entries, gib(st.retained_bytes), st.evictions, st.hits, st.misses);
    }
    engine.reset_buffer_pool();
    let after_reset = CUR.load(Relaxed) - base;
    engine.plan_cache().clear();
    println!("T2742 after_reset_gib {:.3} after_clear_gib {:.3}", gib(after_reset), gib(CUR.load(Relaxed) - base));
    Ok(())
}
