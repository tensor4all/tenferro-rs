use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::time::{Duration, Instant};

use num_complex::Complex64;
use tenferro_api_grammar_proof::p2_einsum_dest::EinsumSpec;
use tenferro_api_grammar_proof::p6_trace_einsum::{TraceEinsum, TraceEinsumOptions};
use tenferro_cpu::CpuEngine;
use tenferro_einsum::{EinsumOptimize, EinsumSubscripts, TraceContextEinsumExt};
use tenferro_runtime::program::ProgramInputSpec;
use tenferro_runtime::{
    CompiledGraph, DType, ExecutionEngine, GraphCompiler, Runtime, TraceContext, TraceValue,
    TracedTensor,
};
use tenferro_tensor::{Tensor, TensorRead};

type C = Complex64;

// ---- thread-local allocation counter (does not pollute other test threads) -------------

struct Counting;
thread_local! {
    static ALLOCS: Cell<u64> = const { Cell::new(0) };
    static BYTES: Cell<u64> = const { Cell::new(0) };
}
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
        let _ = BYTES.try_with(|c| c.set(c.get() + l.size() as u64));
        System.alloc(l)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l)
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
        let _ = BYTES.try_with(|c| c.set(c.get() + n as u64));
        System.realloc(p, l, n)
    }
}
#[global_allocator]
static A: Counting = Counting;

fn counters() -> (u64, u64) {
    (ALLOCS.with(|c| c.get()), BYTES.with(|c| c.get()))
}

// ---- helpers -----------------------------------------------------------------------------

fn leaf_data(i: usize) -> Vec<C> {
    // near-identity 2x2 complex matrices, column-major, distinct per leaf, non-commuting
    let t = i as f64;
    vec![
        C::new(1.0 + 0.01 * (t * 0.7).sin(), 0.02 * (t * 1.3).cos()),
        C::new(0.03 * (t * 0.9).cos(), -0.02 * (t * 0.4).sin()),
        C::new(-0.025 * (t * 1.1).sin(), 0.03 * (t * 0.5).cos()),
        C::new(1.0 - 0.01 * (t * 0.3).cos(), 0.01 * (t * 1.7).sin()),
    ]
}
fn leaf_tensor(i: usize) -> Tensor {
    Tensor::from_vec_col_major(vec![2, 2], leaf_data(i)).unwrap()
}
fn matmul2(a: &[C], b: &[C]) -> Vec<C> {
    let mut o = vec![C::new(0.0, 0.0); 4];
    for k in 0..2 {
        for i in 0..2 {
            o[i + 2 * k] = (0..2).map(|j| a[i + 2 * j] * b[j + 2 * k]).sum();
        }
    }
    o
}
fn ref_tree(lo: usize, hi: usize) -> Vec<C> {
    if hi - lo == 1 {
        leaf_data(lo)
    } else {
        let mid = lo + (hi - lo) / 2;
        matmul2(&ref_tree(lo, mid), &ref_tree(mid, hi))
    }
}
fn assert_close(a: &[C], b: &[C]) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).norm() <= 1e-10 * (1.0 + y.norm()), "{x} vs {y}");
    }
}

fn try_run(program: &CompiledGraph, inputs: &[Tensor]) -> Result<Vec<Tensor>, String> {
    let e = |x: &dyn std::fmt::Display| x.to_string();
    let mut engine = CpuEngine::serial();
    let mut builder = Runtime::builder();
    builder
        .register_engine(tenferro_cpu::runtime_engine_registration(&engine).map_err(|x| e(&x))?)
        .map_err(|x| e(&x))?;
    builder
        .install_extension_module(
            tenferro_einsum::extension_module::<CpuEngine>(
                tenferro_cpu::runtime_engine_id().map_err(|x| e(&x))?,
            )
            .map_err(|x| e(&x))?,
        )
        .map_err(|x| e(&x))?;
    let runtime = builder.build().map_err(|x| e(&x))?;
    let reads: Vec<TensorRead<'_>> = inputs.iter().map(TensorRead::from_tensor).collect();
    let prepared = runtime
        .prepare(program, &reads, &[engine.target()])
        .map_err(|x| e(&x))?;
    let outs = runtime
        .run(&mut [&mut engine], &prepared, &reads)
        .map_err(|x| e(&x))?;
    outs.into_iter()
        .map(|o| o.into_tensor(&mut engine).map_err(|x| e(&x)))
        .collect()
}
fn run(program: &CompiledGraph, inputs: &[Tensor]) -> Vec<Tensor> {
    try_run(program, inputs).unwrap()
}
fn vals(t: &Tensor) -> Vec<C> {
    t.as_slice::<C>().unwrap().to_vec()
}

fn input_spec() -> ProgramInputSpec {
    ProgramInputSpec::new(DType::C64, vec![2usize.into(), 2usize.into()])
}

fn mm_subs() -> EinsumSubscripts {
    EinsumSubscripts::new(&[&[0, 1], &[1, 2]], &[0, 2])
}

/// Path A: today's downstream pattern (yao-rs): TraceValue handles + TraceContextEinsumExt.
fn build_a(n: usize) -> (TraceContext, Vec<TraceValue>, TraceValue) {
    let mut trace = TraceContext::new();
    let leaves: Vec<TraceValue> = (0..n).map(|_| trace.input(input_spec()).unwrap()).collect();
    fn rec(
        t: &mut TraceContext,
        s: &EinsumSubscripts,
        l: &[TraceValue],
        lo: usize,
        hi: usize,
    ) -> TraceValue {
        if hi - lo == 1 {
            return l[lo];
        }
        let mid = lo + (hi - lo) / 2;
        let a = rec(t, s, l, lo, mid);
        let b = rec(t, s, l, mid, hi);
        t.einsum_subscripts_with(&[a, b], s, EinsumOptimize::False)
            .unwrap()
    }
    let s = mm_subs();
    let out = rec(&mut trace, &s, &leaves, 0, n);
    (trace, leaves, out)
}

/// Path B: owner-receiver over TracedTensor. `text` selects "ij,jk->ik" text vs parsed subscripts.
fn build_b(n: usize, text: bool) -> (TraceContext, Vec<TracedTensor>, TracedTensor) {
    let trace = TraceContext::new();
    let leaves: Vec<TracedTensor> = (0..n)
        .map(|_| trace.input_concrete_shape(DType::C64, &[2, 2]).unwrap())
        .collect();
    let opts = TraceEinsumOptions::default().with_optimize(EinsumOptimize::False);
    let subs = mm_subs();
    fn rec(
        t: &TraceContext,
        l: &[TracedTensor],
        lo: usize,
        hi: usize,
        text: bool,
        subs: &EinsumSubscripts,
        opts: &TraceEinsumOptions,
    ) -> TracedTensor {
        if hi - lo == 1 {
            return l[lo].clone();
        }
        let mid = lo + (hi - lo) / 2;
        let a = rec(t, l, lo, mid, text, subs, opts);
        let b = rec(t, l, mid, hi, text, subs, opts);
        if text {
            t.trace_einsum_with(&[&a, &b], "ab,bc->ac", opts).unwrap()
        } else {
            t.trace_einsum_with(&[a, b], subs, opts).unwrap()
        }
    }
    let out = rec(&trace, &leaves, 0, n, text, &subs, &opts);
    (trace, leaves, out)
}

// ---- correctness ---------------------------------------------------------------------------

#[test]
fn both_paths_compile_to_graphs_with_equal_numbers() {
    for n in [8usize, 64, 512] {
        let inputs: Vec<Tensor> = (0..n).map(leaf_tensor).collect();
        let expected = ref_tree(0, n);

        let (trace_a, _la, out_a) = build_a(n);
        let graph = trace_a.finish(&[out_a]).unwrap();
        let prog_a = GraphCompiler::new().compile_traced_graph(&graph).unwrap();
        assert_eq!(prog_a.input_count(), n);
        let ra = vals(&run(&prog_a, &inputs)[0]);

        let (_tb, _lb, out_b) = build_b(n, true);
        let prog_b_many = GraphCompiler::new().compile_many(&[&out_b]).unwrap();
        assert_eq!(prog_b_many.input_count(), n);
        let rb = vals(&run(&prog_b_many, &inputs)[0]);

        let (tb2, _lb2, out_b2) = build_b(n, false);
        let graph_b = tb2.finish(&[out_b2.trace_value()]).unwrap();
        let prog_b_graph = GraphCompiler::new().compile_traced_graph(&graph_b).unwrap();
        let rb2 = vals(&run(&prog_b_graph, &inputs)[0]);

        assert_close(&ra, &expected);
        assert_close(&rb, &expected);
        assert_close(&rb2, &expected);
        assert_eq!(ra.len(), 4);
    }
}

#[test]
fn input_order_is_creation_order_not_use_order() {
    // inputs created x0, x1 ; out = x1 * x0 (non-commutative)
    let inputs = [leaf_tensor(3), leaf_tensor(11)];
    let expected = matmul2(&leaf_data(11), &leaf_data(3));
    let wrong = matmul2(&leaf_data(3), &leaf_data(11));
    assert!(expected
        .iter()
        .zip(&wrong)
        .any(|(a, b)| (a - b).norm() > 1e-6));

    // TraceValue path
    let mut ta = TraceContext::new();
    let x0 = ta.input(input_spec()).unwrap();
    let x1 = ta.input(input_spec()).unwrap();
    let out = ta
        .einsum_subscripts_with(&[x1, x0], &mm_subs(), EinsumOptimize::False)
        .unwrap();
    let g = ta.finish(&[out]).unwrap();
    let prog = GraphCompiler::new().compile_traced_graph(&g).unwrap();
    assert_close(&vals(&run(&prog, &inputs)[0]), &expected);

    // TracedTensor path, compile_many
    let tb = TraceContext::new();
    let y0 = tb.input_concrete_shape(DType::C64, &[2, 2]).unwrap();
    let y1 = tb.input_concrete_shape(DType::C64, &[2, 2]).unwrap();
    let out = tb
        .trace_einsum_with(&[&y1, &y0], "ab,bc->ac", &TraceEinsumOptions::default())
        .unwrap();
    let prog = GraphCompiler::new().compile_many(&[&out]).unwrap();
    assert_close(&vals(&run(&prog, &inputs)[0]), &expected);
    // compile_many exposes the input keys in semantic (creation) order
    assert_eq!(
        prog.input_keys(),
        &[y0.input_key().unwrap(), y1.input_key().unwrap()]
    );
}

#[test]
fn unused_input_is_pruned_by_finish_and_by_compile_many() {
    // inputs created: x0, unused, x2 ; out = x0 * x2
    let all = [leaf_tensor(1), leaf_tensor(2), leaf_tensor(3)];
    let used = [leaf_tensor(1), leaf_tensor(3)];
    let expected = matmul2(&leaf_data(1), &leaf_data(3));

    // TraceValue + finish + compile_traced_graph
    let mut ta = TraceContext::new();
    let x0 = ta.input(input_spec()).unwrap();
    let unused_a = ta.input(input_spec()).unwrap();
    let x2 = ta.input(input_spec()).unwrap();
    let out = ta
        .einsum_subscripts_with(&[x0, x2], &mm_subs(), EinsumOptimize::False)
        .unwrap();
    let g = ta.finish(&[out]).unwrap();
    let prog_a = GraphCompiler::new().compile_traced_graph(&g).unwrap();
    eprintln!(
        "finish+compile_traced_graph: input_count={} input_keys.len()={}",
        prog_a.input_count(),
        prog_a.input_keys().len()
    );
    assert_eq!(prog_a.input_count(), 2, "unused input must be pruned");
    // the key lookup the TracedTensor path relies on is empty here
    assert!(prog_a.input_keys().is_empty());
    assert!(ta.input_key(x0).is_some() && ta.input_key(unused_a).is_some());
    assert_close(&vals(&run(&prog_a, &used)[0]), &expected);
    // passing all three inputs is rejected
    let rejected = try_run(&prog_a, &all);
    eprintln!(
        "3 inputs into pruned 2-input program: {:?}",
        rejected.as_ref().map(|_| ())
    );
    assert!(rejected.is_err(), "3 inputs to a 2-input program must fail");

    // TracedTensor + compile_many
    let tb = TraceContext::new();
    let y0 = tb.input_concrete_shape(DType::C64, &[2, 2]).unwrap();
    let unused_b = tb.input_concrete_shape(DType::C64, &[2, 2]).unwrap();
    let y2 = tb.input_concrete_shape(DType::C64, &[2, 2]).unwrap();
    let out = tb
        .trace_einsum_with(&[&y0, &y2], "ab,bc->ac", &TraceEinsumOptions::default())
        .unwrap();
    let prog_b = GraphCompiler::new().compile_many(&[&out]).unwrap();
    assert_eq!(prog_b.input_count(), 2);
    assert_eq!(
        prog_b.input_keys(),
        &[y0.input_key().unwrap(), y2.input_key().unwrap()]
    );
    assert_eq!(prog_b.input_key_index(&unused_b.input_key().unwrap()), None);
    assert_eq!(prog_b.input_key_index(&y2.input_key().unwrap()), Some(1));
    assert_close(&vals(&run(&prog_b, &used)[0]), &expected);

    // finish(outputs from TracedTensor) + compile_traced_graph prunes the same way
    let g = tb.finish(&[out.trace_value()]).unwrap();
    let prog_b2 = GraphCompiler::new().compile_traced_graph(&g).unwrap();
    assert_eq!(prog_b2.input_count(), 2);
    assert!(prog_b2.input_keys().is_empty());
}

#[test]
fn output_order_is_preserved() {
    let inputs = [leaf_tensor(5), leaf_tensor(6), leaf_tensor(7)];
    let p01 = matmul2(&leaf_data(5), &leaf_data(6));
    let p12 = matmul2(&leaf_data(6), &leaf_data(7));

    let tb = TraceContext::new();
    let xs: Vec<TracedTensor> = (0..3)
        .map(|_| tb.input_concrete_shape(DType::C64, &[2, 2]).unwrap())
        .collect();
    let o = TraceEinsumOptions::default();
    let a = tb
        .trace_einsum_with(&[&xs[0], &xs[1]], "ab,bc->ac", &o)
        .unwrap();
    let b = tb
        .trace_einsum_with(&[&xs[1], &xs[2]], "ab,bc->ac", &o)
        .unwrap();

    let prog = GraphCompiler::new().compile_many(&[&b, &a]).unwrap();
    let outs = run(&prog, &inputs);
    assert_eq!(outs.len(), 2);
    assert_close(&vals(&outs[0]), &p12);
    assert_close(&vals(&outs[1]), &p01);

    let g = tb.finish(&[a.trace_value(), b.trace_value()]).unwrap();
    let prog = GraphCompiler::new().compile_traced_graph(&g).unwrap();
    let outs = run(&prog, &inputs);
    assert_close(&vals(&outs[0]), &p01);
    assert_close(&vals(&outs[1]), &p12);
}

#[test]
fn foreign_context_is_rejected_on_traced_tensor_path() {
    let t1 = TraceContext::new();
    let t2 = TraceContext::new();
    let a = t1.input_concrete_shape(DType::C64, &[2, 2]).unwrap();
    let b = t2.input_concrete_shape(DType::C64, &[2, 2]).unwrap();
    let opts = TraceEinsumOptions::default();
    let err = t1
        .trace_einsum_with(&[&a, &b], "ab,bc->ac", &opts)
        .unwrap_err();
    eprintln!("foreign tensor error: {err}");
    // the same inputs from the owner's own context are fine
    let a2 = t1.input_concrete_shape(DType::C64, &[2, 2]).unwrap();
    assert!(t1.trace_einsum_with(&[&a, &a2], "ab,bc->ac", &opts).is_ok());
    // the owner must be the receiver: t2 as owner of t1's tensors is also rejected
    assert!(t2
        .trace_einsum_with(&[&a, &a2], "ab,bc->ac", &opts)
        .is_err());
    // typed-kind check
    assert!(matches!(err, tenferro_runtime::Error::Validation { .. }));
}

#[test]
fn foreign_trace_value_on_existing_api_is_already_rejected() {
    // TraceValue handles are context-tagged inside the program builder, so the existing
    // TraceValue API already rejects a foreign handle (same-numbered or out-of-range).
    let mut ta = TraceContext::new();
    let _a0 = ta.input(input_spec()).unwrap();
    let _a1 = ta.input(input_spec()).unwrap();
    let tb = TraceContext::new();
    let b0 = tb.input(input_spec()).unwrap();
    let b1 = tb.input(input_spec()).unwrap();
    let err = ta
        .einsum_subscripts_with(&[b0, b1], &mm_subs(), EinsumOptimize::False)
        .unwrap_err();
    assert!(format!("{err:?}").contains("ForeignValue"), "{err:?}");
    let many: Vec<TraceValue> = (0..5).map(|_| tb.input(input_spec()).unwrap()).collect();
    let err = ta
        .einsum_subscripts_with(&[many[4], many[3]], &mm_subs(), EinsumOptimize::False)
        .unwrap_err();
    assert!(format!("{err:?}").contains("ForeignValue"), "{err:?}");
    // finish() likewise rejects a foreign output
    assert!(ta.finish(&[b0]).is_err());
}

#[test]
fn spec_forms_and_bad_specs_on_traced_path() {
    let tb = TraceContext::new();
    let x = tb.input_concrete_shape(DType::C64, &[2, 2]).unwrap();
    let y = tb.input_concrete_shape(DType::C64, &[2, 2]).unwrap();
    let o = TraceEinsumOptions::default();
    let subs = mm_subs();
    let notation = tenferro_einsum::parse_einsum_notation("...ij,...jk->...ik").unwrap();
    for spec in [
        EinsumSpec::from("ij,jk->ik"),
        EinsumSpec::from(&subs),
        EinsumSpec::from(&notation),
        EinsumSpec::indices(&[&[0, 1], &[1, 2]], &[0, 2]),
    ] {
        let out = tb.trace_einsum_with(&[&x, &y], spec, &o).unwrap();
        assert_eq!(out.concrete_shape().unwrap(), vec![2, 2]);
    }
    // failures are errors, not panics
    assert!(tb.trace_einsum_with(&[&x, &y], "ij,jk-ik", &o).is_err());
    assert!(tb.trace_einsum_with(&[&x], "ij,jk->ik", &o).is_err());
    assert!(tb.trace_einsum_with(&[&x, &y], "ij,jk->iz", &o).is_err());
    let none: [&TracedTensor; 0] = [];
    assert!(tb.trace_einsum_with(&none, "->", &o).is_err());
    // rank mismatch with a 3-index term on a rank-2 tensor
    assert!(tb.trace_einsum_with(&[&x, &y], "ijl,jk->ik", &o).is_err());
    // Tree strategy cannot pass through borrowed options
    let tree = tenferro_einsum::ContractionTree::optimize(
        &tenferro_einsum::Subscripts::from(subs.clone()),
        &[&[2, 2], &[2, 2]],
    )
    .unwrap();
    let o_tree = TraceEinsumOptions::default().with_optimize(EinsumOptimize::Tree(tree));
    assert!(tb
        .trace_einsum_with(&[&x, &y], "ij,jk->ik", &o_tree)
        .is_err());
}

// ---- timing / allocation measurements ------------------------------------------------------

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

struct Row {
    label: &'static str,
    n: usize,
    build: Duration,
    finish: Duration,
    compile: Duration,
    build_allocs: u64,
    build_bytes: u64,
}

fn measure(
    label: &'static str,
    n: usize,
    reps: usize,
    f: impl Fn() -> (Duration, Duration, Duration, u64, u64),
) -> Row {
    let mut b = vec![];
    let mut fi = vec![];
    let mut c = vec![];
    let mut allocs = 0;
    let mut bytes = 0;
    for _ in 0..reps {
        let (bb, ff, cc, al, by) = f();
        b.push(bb);
        fi.push(ff);
        c.push(cc);
        allocs = al;
        bytes = by;
    }
    Row {
        label,
        n,
        build: median(b),
        finish: median(fi),
        compile: median(c),
        build_allocs: allocs,
        build_bytes: bytes,
    }
}

#[test]
fn construction_and_compile_timing_table() {
    let mut rows = vec![];
    for n in [8usize, 64, 512] {
        let reps = if n >= 512 { 5 } else { 11 };
        rows.push(measure("A TraceValue (subscripts)", n, reps, || {
            let before = counters();
            let t = Instant::now();
            let (trace, _l, out) = build_a(n);
            let build = t.elapsed();
            let after = counters();
            let t = Instant::now();
            let g = trace.finish(&[out]).unwrap();
            let finish = t.elapsed();
            let t = Instant::now();
            let _p = GraphCompiler::new().compile_traced_graph(&g).unwrap();
            let compile = t.elapsed();
            (
                build,
                finish,
                compile,
                after.0 - before.0,
                after.1 - before.1,
            )
        }));
        for (label, text) in [
            ("B TracedTensor (text spec)", true),
            ("B TracedTensor (subscripts spec)", false),
        ] {
            rows.push(measure(label, n, reps, move || {
                let before = counters();
                let t = Instant::now();
                let (trace, _l, out) = build_b(n, text);
                let build = t.elapsed();
                let after = counters();
                let t = Instant::now();
                let g = trace.finish(&[out.trace_value()]).unwrap();
                let finish = t.elapsed();
                let t = Instant::now();
                let _p = GraphCompiler::new().compile_traced_graph(&g).unwrap();
                let compile = t.elapsed();
                (
                    build,
                    finish,
                    compile,
                    after.0 - before.0,
                    after.1 - before.1,
                )
            }));
        }
        rows.push(measure("B compile_many (no finish)", n, reps, || {
            let before = counters();
            let t = Instant::now();
            let (_trace, _l, out) = build_b(n, false);
            let build = t.elapsed();
            let after = counters();
            let t = Instant::now();
            let _p = GraphCompiler::new().compile_many(&[&out]).unwrap();
            let compile = t.elapsed();
            (
                build,
                Duration::ZERO,
                compile,
                after.0 - before.0,
                after.1 - before.1,
            )
        }));
    }
    eprintln!("debug_assertions={}", cfg!(debug_assertions));
    eprintln!("| path | N | build (us) | finish (us) | compile (us) | build allocs | build KiB |");
    eprintln!("|---|---|---|---|---|---|---|");
    for r in &rows {
        eprintln!(
            "| {} | {} | {:.0} | {:.0} | {:.0} | {} | {:.0} |",
            r.label,
            r.n,
            r.build.as_secs_f64() * 1e6,
            r.finish.as_secs_f64() * 1e6,
            r.compile.as_secs_f64() * 1e6,
            r.build_allocs,
            r.build_bytes as f64 / 1024.0
        );
    }
}
