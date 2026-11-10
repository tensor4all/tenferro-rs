//! Prototype 4 tests: `MaterializationPolicy` on borrowed SVD and einsum plans.
//!
//! "Did a copy happen" is observed with a counting global allocator: a
//! thread-local count of allocations at least as large as one input operand.
//! Each measurement uses a fresh serial engine, so pooled buffers from earlier
//! calls cannot hide an allocation. Comparisons are differential (strided
//! layout against the compact baseline) so output and workspace allocations
//! cancel.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use num_complex::Complex64;
use tenferro_api_grammar_proof::p4_plan_policy::{PlanPolicyExt, ReadSvdPolicyExt};
use tenferro_api_grammar_proof::policy::MaterializationPolicy::{self, Allow, Forbid};
use tenferro_cpu::linalg_interop::BufferPool;
use tenferro_cpu::CpuEngine;
use tenferro_einsum::ConcreteEinsumPlan;
use tenferro_linalg::{svd_with_exec, LinalgScalar};
use tenferro_tensor::{
    Error, Tensor, TensorRead, TensorScalar, TensorWrite, TypedTensor, TypedTensorView,
};

// ---------------------------------------------------------------- allocator

struct Counting;

thread_local! {
    static MIN_BYTES: Cell<usize> = const { Cell::new(usize::MAX) };
    static BIG_ALLOCS: Cell<usize> = const { Cell::new(0) };
}

fn note(size: usize) {
    let _ = MIN_BYTES.try_with(|min| {
        if size >= min.get() {
            let _ = BIG_ALLOCS.try_with(|n| n.set(n.get() + 1));
        }
    });
}

// SAFETY: forwards to `System`; the bookkeeping touches only const-initialized
// thread-locals without destructors.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note(layout.size());
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note(new_size);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// Run `f`, returning its result and how many allocations of at least
/// `min_bytes` it made on this thread.
fn count_big<R>(min_bytes: usize, f: impl FnOnce() -> R) -> (R, usize) {
    MIN_BYTES.with(|m| m.set(min_bytes));
    BIG_ALLOCS.with(|n| n.set(0));
    let result = f();
    MIN_BYTES.with(|m| m.set(usize::MAX));
    (result, BIG_ALLOCS.with(|n| n.get()))
}

// ------------------------------------------------------------------ scalars

trait Elem: TensorScalar + LinalgScalar + Copy + std::fmt::Debug + PartialEq {
    fn make(i: usize) -> Self;
    fn dist(self, other: Self) -> f64;
}
impl Elem for f64 {
    fn make(i: usize) -> Self {
        ((i * 7919 + 13) % 101) as f64 / 17.0 - 2.5
    }
    fn dist(self, other: Self) -> f64 {
        (self - other).abs()
    }
}
impl Elem for Complex64 {
    fn make(i: usize) -> Self {
        Complex64::new(f64::make(i), f64::make(i * 3 + 1))
    }
    fn dist(self, other: Self) -> f64 {
        (self - other).norm()
    }
}

fn buffer<T: Elem>(len: usize, salt: usize) -> Vec<T> {
    (0..len).map(|i| T::make(i + salt)).collect()
}

fn view<'a, T: Elem>(
    shape: &[usize],
    strides: &[isize],
    offset: isize,
    data: &'a [T],
) -> TensorRead<'a> {
    let typed = TypedTensorView::<T>::from_slice(shape, strides, offset, data).unwrap();
    TensorRead::from_view(T::tensor_view(typed))
}

/// Gather a strided view into a compact column-major tensor (the reference).
fn gather<T: Elem>(shape: &[usize], strides: &[isize], offset: isize, data: &[T]) -> Tensor {
    let typed = TypedTensorView::<T>::from_slice(shape, strides, offset, data).unwrap();
    let total: usize = shape.iter().product();
    let mut out = Vec::with_capacity(total);
    let mut idx = vec![0usize; shape.len()];
    for _ in 0..total {
        out.push(*typed.get(&idx).unwrap());
        for (axis, i) in idx.iter_mut().enumerate() {
            *i += 1;
            if *i < shape[axis] {
                break;
            }
            *i = 0;
        }
    }
    Tensor::from_typed(TypedTensor::<T>::from_vec_col_major(shape.to_vec(), out).unwrap())
}

fn close(a: &Tensor, b: &Tensor) {
    assert_eq!(a.shape(), b.shape());
    let (xa, xb) = (a.as_slice::<f64>().unwrap(), b.as_slice::<f64>().unwrap());
    for (x, y) in xa.iter().zip(xb) {
        assert!((x - y).abs() < 1e-8 * (1.0 + y.abs()), "{x} vs {y}");
    }
}

fn close_t<T: Elem>(a: &Tensor, b: &Tensor) {
    assert_eq!(a.shape(), b.shape());
    let (xa, xb) = (a.as_slice::<T>().unwrap(), b.as_slice::<T>().unwrap());
    for (x, y) in xa.iter().zip(xb) {
        assert!(
            x.dist(*y) < 1e-8 * (1.0 + y.dist(T::make(0))),
            "{x:?} vs {y:?}"
        );
    }
}

fn is_forbidden(err: &Error, op: &str, input: usize) -> bool {
    matches!(err, Error::InputMaterializationForbidden { op: o, input: i } if *o == op && *i == input)
}

// ---------------------------------------------------------------------- SVD

const N: usize = 48;

/// Singular values of `read` under `policy`, plus the big-allocation count and
/// the pool state, all on a fresh engine.
struct SvdRun {
    result: tenferro_tensor::Result<(Tensor, Tensor, Tensor)>,
    big: usize,
    pool_buffers: usize,
    pool_bytes: usize,
}

fn svd_run(read: TensorRead<'_>, bytes: usize, policy: MaterializationPolicy) -> SvdRun {
    let mut engine = CpuEngine::serial();
    let (result, big) = count_big(bytes, || read.svd_with(&mut engine, policy));
    let stats = engine.buffer_pool_stats();
    SvdRun {
        result,
        big,
        pool_buffers: stats.buffers,
        pool_bytes: stats.capacity_bytes,
    }
}

/// One uncounted call so first-use initialization (thread-local state, lazy
/// tables) is not charged to the first measurement on this thread.
fn warm_svd<T: Elem>() {
    let t = gather::<T>(&[N, N], &[1, N as isize], 0, &buffer::<T>(N * N, 1));
    let mut engine = CpuEngine::serial();
    let _ = TensorRead::from_tensor(&t).svd_with(&mut engine, Allow);
}

fn svd_cases<T: Elem>() {
    warm_svd::<T>();
    let bytes = N * N * std::mem::size_of::<T>();
    let reference = |shape: &[usize], strides: &[isize], off: isize, data: &[T]| {
        let compact = gather::<T>(shape, strides, off, data);
        svd_run(TensorRead::from_tensor(&compact), bytes, Allow)
    };

    // 1. compact input: OK under both, and no gather in either.
    let data = buffer::<T>(N * N, 0);
    let compact =
        Tensor::from_typed(TypedTensor::<T>::from_vec_col_major(vec![N, N], data.clone()).unwrap());
    let allow = svd_run(TensorRead::from_tensor(&compact), bytes, Allow);
    let forbid = svd_run(TensorRead::from_tensor(&compact), bytes, Forbid);
    close(
        &forbid.result.as_ref().unwrap().1,
        &allow.result.as_ref().unwrap().1,
    );
    assert_eq!(allow.big, forbid.big);

    // 2. transposed rank-2 view: read in place on native under both policies.
    let t_strides = [N as isize, 1];
    let t_allow = svd_run(view(&[N, N], &t_strides, 0, &data), bytes, Allow);
    let t_forbid = svd_run(view(&[N, N], &t_strides, 0, &data), bytes, Forbid);
    let t_ref = reference(&[N, N], &t_strides, 0, &data);
    close(
        &t_allow.result.as_ref().unwrap().1,
        &t_ref.result.as_ref().unwrap().1,
    );
    close(
        &t_forbid.result.as_ref().unwrap().1,
        &t_ref.result.as_ref().unwrap().1,
    );
    assert_eq!(
        t_allow.big, t_ref.big,
        "transposed view must not be gathered"
    );
    assert_eq!(t_forbid.big, t_ref.big);

    // 3. negative-stride rank-2 view: Allow gathers, Forbid refuses with no side effect.
    let neg_strides = [-1isize, N as isize];
    let neg_off = (N - 1) as isize;
    let n_ref = reference(&[N, N], &neg_strides, neg_off, &data);
    let n_allow = svd_run(view(&[N, N], &neg_strides, neg_off, &data), bytes, Allow);
    let n_forbid = svd_run(view(&[N, N], &neg_strides, neg_off, &data), bytes, Forbid);
    close(
        &n_allow.result.as_ref().unwrap().1,
        &n_ref.result.as_ref().unwrap().1,
    );
    assert!(
        n_allow.big > n_ref.big,
        "Allow must gather a negative-stride view ({} vs {})",
        n_allow.big,
        n_ref.big
    );
    let err = n_forbid.result.as_ref().unwrap_err();
    assert!(is_forbidden(err, "svd", 0), "{err:?}");
    assert!(
        err.to_string().contains("Allow"),
        "remediation in display: {err}"
    );
    assert_eq!(
        n_forbid.big, 0,
        "refusal must not allocate operand-sized buffers"
    );
    assert_eq!(
        (n_forbid.pool_buffers, n_forbid.pool_bytes),
        (0, 0),
        "pool untouched"
    );
    assert!(
        n_allow.pool_bytes > 0,
        "Allow's gather leaves retained pool capacity"
    );

    // 4. rank-3 non-compact batch (batch axis strided by 2 matrices): Allow gathers,
    //    Forbid refuses, although the data is readable in place (see reroute test).
    let b = 2usize;
    let big_data = buffer::<T>(N * N * 3, 5);
    let b_strides = [1isize, N as isize, (2 * N * N) as isize];
    let shape3 = [N, N, b];
    let b_bytes = N * N * std::mem::size_of::<T>();
    let b_ref = {
        let compact = gather::<T>(&shape3, &b_strides, 0, &big_data);
        svd_run(TensorRead::from_tensor(&compact), b_bytes, Allow)
    };
    let b_allow = svd_run(view(&shape3, &b_strides, 0, &big_data), b_bytes, Allow);
    let b_forbid = svd_run(view(&shape3, &b_strides, 0, &big_data), b_bytes, Forbid);
    close(
        &b_allow.result.as_ref().unwrap().1,
        &b_ref.result.as_ref().unwrap().1,
    );
    assert!(b_allow.big > b_ref.big, "Allow gathers the rank-3 batch");
    let err = b_forbid.result.as_ref().unwrap_err();
    assert!(is_forbidden(err, "svd", 0), "{err:?}");
    assert_eq!(b_forbid.big, 0);
    assert_eq!((b_forbid.pool_buffers, b_forbid.pool_bytes), (0, 0));
}

#[test]
fn svd_policy_f64() {
    svd_cases::<f64>();
}

#[test]
fn svd_policy_complex64() {
    svd_cases::<Complex64>();
}

/// The reroute finding: a rank-3 strided (non-negative) batch can be read in
/// place by `svd_with_exec`, so the `Forbid` refusal above is not forced by the
/// math. A negative-stride view has no copy-free route.
fn reroute_cases<T: Elem + LinalgScalar>() {
    warm_svd::<T>();
    let b = 2usize;
    let data = buffer::<T>(N * N * 3, 5);
    let strides = [1isize, N as isize, (2 * N * N) as isize];
    let typed = TypedTensorView::<T>::from_slice([N, N, b], strides, 0, &data).unwrap();
    let mut engine = CpuEngine::serial();
    let bytes = N * N * std::mem::size_of::<T>();
    let (direct, direct_big) = count_big(bytes, || {
        engine.with_exec(|exec| {
            let mut buffers = BufferPool::new();
            svd_with_exec(exec, &mut buffers, &typed, false).map(|(_, s, _)| s)
        })
    });
    let direct = direct.unwrap().unwrap();
    // Same data through the policy-aware read path under Allow (gathers).
    let allow = svd_run(view(&[N, N, b], &strides, 0, &data), bytes, Allow);
    close(
        &Tensor::from_typed(direct),
        &allow.result.as_ref().unwrap().1,
    );
    assert!(
        direct_big < allow.big,
        "direct strided read ({direct_big}) allocates fewer operand-sized buffers than the gather route ({})",
        allow.big
    );

    // Negative strides: the direct route refuses too (no copy-free route exists).
    let neg =
        TypedTensorView::<T>::from_slice([N, N], [-1isize, N as isize], (N - 1) as isize, &data)
            .unwrap();
    let mut engine = CpuEngine::serial();
    let r = engine.with_exec(|exec| {
        let mut buffers = BufferPool::new();
        svd_with_exec(exec, &mut buffers, &neg, false).map(|_| ())
    });
    assert!(
        r.unwrap().is_err(),
        "svd_with_exec cannot read negative strides"
    );
}

#[test]
fn svd_rank3_reroute_possible_f64() {
    reroute_cases::<f64>();
}

#[test]
fn svd_rank3_reroute_possible_complex64() {
    reroute_cases::<Complex64>();
}

// ------------------------------------------------------------------- einsum

/// Operand edge for the plan-reuse tests (small: many executions).
const M: usize = 32;
/// Operand edge for the allocation tests: pack buffers of the packed driver are
/// bounded by cache blocking, so at this size they stay well below one operand.
const BIG: usize = 256;

struct Operands<T> {
    n: usize,
    bufs: Vec<Vec<T>>,
}

fn compact_strides(n: usize) -> [isize; 2] {
    [1, n as isize]
}
fn transposed_strides(n: usize) -> [isize; 2] {
    [n as isize, 1]
}
fn negative_strides(n: usize) -> [isize; 2] {
    [-1, n as isize]
}
fn negative_offset(n: usize) -> isize {
    (n - 1) as isize
}

impl<T: Elem> Operands<T> {
    fn new(count: usize) -> Self {
        Self::sized(count, M)
    }
    fn sized(count: usize, n: usize) -> Self {
        Self {
            n,
            bufs: (0..count).map(|k| buffer::<T>(n * n, 100 * k)).collect(),
        }
    }
    /// Compact column-major owned tensors.
    fn compact(&self) -> Vec<Tensor> {
        self.bufs
            .iter()
            .map(|b| gather::<T>(&[self.n, self.n], &compact_strides(self.n), 0, b))
            .collect()
    }
    fn reads<'a>(&'a self, strides: &[[isize; 2]], offsets: &[isize]) -> Vec<TensorRead<'a>> {
        self.bufs
            .iter()
            .enumerate()
            .map(|(k, b)| view(&[self.n, self.n], &strides[k], offsets[k], b))
            .collect()
    }
    /// The compact tensors holding exactly what `reads(strides, offsets)` sees.
    fn gathered(&self, strides: &[[isize; 2]], offsets: &[isize]) -> Vec<Tensor> {
        self.bufs
            .iter()
            .enumerate()
            .map(|(k, b)| gather::<T>(&[self.n, self.n], &strides[k], offsets[k], b))
            .collect()
    }
}

fn reads_of(tensors: &[Tensor]) -> Vec<TensorRead<'_>> {
    tensors.iter().map(TensorRead::from_tensor).collect()
}

const COMPACT: [isize; 2] = [1, M as isize];
const TRANSPOSED: [isize; 2] = [M as isize, 1];
const NEGATIVE: [isize; 2] = [-1, M as isize];
const NEG_OFF: isize = (M - 1) as isize;

/// Uncounted warm-up (see [`warm_svd`]).
fn warm_einsum<T: Elem>() {
    for (expr, count) in [("ij,jk,kl->il", 3usize), ("ij,jk->ik", 2), ("ij->ji", 1)] {
        let ops = Operands::<T>::new(count);
        let compact = ops.compact();
        let plan = ConcreteEinsumPlan::prepare(compact.iter().collect::<Vec<_>>(), expr).unwrap();
        let mut engine = CpuEngine::serial();
        let _ = plan.execute_with(&reads_of(&compact), &mut engine, Allow);
    }
}

fn plan_reuse_cases<T: Elem>() {
    warm_einsum::<T>();
    let ops = Operands::<T>::new(3);
    let compact = ops.compact();
    let plan =
        ConcreteEinsumPlan::prepare(compact.iter().collect::<Vec<_>>(), "ij,jk,kl->il").unwrap();
    assert_eq!(
        plan.cpu_cache_stats().unwrap().entries,
        0,
        "prepare records no layout"
    );
    let mut engine = CpuEngine::serial();

    // Allow, compact inputs: first execution plans for the compact layouts.
    let base = plan
        .execute_with(&reads_of(&compact), &mut engine, Allow)
        .unwrap();
    let s = plan.cpu_cache_stats().unwrap();
    assert_eq!((s.entries, s.hits, s.misses, s.evictions), (1, 0, 1, 0));
    // Same layouts again: cache hit, same result.
    let again = plan
        .execute_with(&reads_of(&compact), &mut engine, Forbid)
        .unwrap();
    close_t::<T>(&again, &base);
    let s = plan.cpu_cache_stats().unwrap();
    assert_eq!((s.hits, s.misses), (1, 1));

    // Forbid with changed layouts on the same plan: transposed, then negative strides.
    let layouts: [([[isize; 2]; 3], [isize; 3]); 3] = [
        ([TRANSPOSED; 3], [0; 3]),
        ([NEGATIVE; 3], [NEG_OFF; 3]),
        ([TRANSPOSED, NEGATIVE, COMPACT], [0, NEG_OFF, 0]),
    ];
    for (strides, offsets) in layouts {
        let expected_inputs = ops.gathered(&strides, &offsets);
        let expected = plan
            .execute_with(&reads_of(&expected_inputs), &mut engine, Allow)
            .unwrap();
        // The cache now holds the compact layout, so the strided call re-plans.
        let before = plan.cpu_cache_stats().unwrap();
        let got = plan
            .execute_with(&ops.reads(&strides, &offsets), &mut engine, Forbid)
            .unwrap();
        close_t::<T>(&got, &expected);
        let after = plan.cpu_cache_stats().unwrap();
        assert_eq!(
            after.misses,
            before.misses + 1,
            "layout change re-plans once"
        );
        assert_eq!(after.evictions, before.evictions + 1);
        assert_eq!(
            after.entries, 1,
            "single-entry cache: the new layout evicts the old"
        );
        // Repeating the strided layout hits the entry the previous call installed.
        let got2 = plan
            .execute_with(&ops.reads(&strides, &offsets), &mut engine, Forbid)
            .unwrap();
        close_t::<T>(&got2, &expected);
        let again = plan.cpu_cache_stats().unwrap();
        assert_eq!((again.hits, again.misses), (after.hits + 1, after.misses));
    }

    // execute_into_with: written under Forbid with strided inputs.
    let (strides, offsets) = ([NEGATIVE; 3], [NEG_OFF; 3]);
    let expected_inputs = ops.gathered(&strides, &offsets);
    let expected = plan
        .execute_with(&reads_of(&expected_inputs), &mut engine, Allow)
        .unwrap();
    let mut out = Tensor::from_typed(
        TypedTensor::<T>::from_vec_col_major(vec![M, M], buffer::<T>(M * M, 9000)).unwrap(),
    );
    plan.execute_into_with(
        &ops.reads(&strides, &offsets),
        &mut engine,
        Forbid,
        TensorWrite::from_tensor(&mut out),
    )
    .unwrap();
    close_t::<T>(&out, &expected);
}

#[test]
fn plan_reuse_forbid_with_changed_layouts_f64() {
    plan_reuse_cases::<f64>();
}

#[test]
fn plan_reuse_forbid_with_changed_layouts_complex64() {
    plan_reuse_cases::<Complex64>();
}

/// Validation runs before the policy and before any write: `out` is untouched
/// when any operand is rejected, wherever the bad operand sits.
fn validation_before_side_effects<T: Elem>() {
    let ops = Operands::<T>::new(3);
    let compact = ops.compact();
    let plan =
        ConcreteEinsumPlan::prepare(compact.iter().collect::<Vec<_>>(), "ij,jk,kl->il").unwrap();
    let mut engine = CpuEngine::serial();
    let prefill = buffer::<T>(M * M, 777);
    let make_out = || {
        Tensor::from_typed(
            TypedTensor::<T>::from_vec_col_major(vec![M, M], prefill.clone()).unwrap(),
        )
    };
    // Operand 0 has awkward strides (would matter under a copying provider);
    // operand 2 has the wrong shape.
    let wrong = buffer::<T>(M * (M + 1), 3);
    let mut reads = ops.reads(&[NEGATIVE, COMPACT, COMPACT], &[NEG_OFF, 0, 0]);
    reads[2] = view(&[M, M + 1], &[1, M as isize], 0, &wrong);
    let mut out = make_out();
    for policy in [Allow, Forbid] {
        let err = plan
            .execute_into_with(
                &reads,
                &mut engine,
                policy,
                TensorWrite::from_tensor(&mut out),
            )
            .unwrap_err();
        assert!(!matches!(
            err,
            tenferro_einsum::Error::Tensor(Error::InputMaterializationForbidden { .. })
        ));
        assert_eq!(
            out.as_slice::<T>().unwrap(),
            prefill.as_slice(),
            "out untouched ({policy:?})"
        );
    }
    assert!(plan.execute_with(&reads, &mut engine, Forbid).is_err());
    // No plan was built for the rejected call.
    assert_eq!(plan.cpu_cache_stats().unwrap().misses, 0);
}

#[test]
fn validation_precedes_policy_and_writes_f64() {
    validation_before_side_effects::<f64>();
}

#[test]
fn validation_precedes_policy_and_writes_complex64() {
    validation_before_side_effects::<Complex64>();
}

/// Native CPU never copies an input because of its layout: operand-sized
/// allocations with strided inputs equal the compact baseline, on the N-ary
/// path (3 operands) and the binary dot path (2 operands). The eager
/// single-operand path copies unconditionally (compact included), so its copy
/// is not input materialization under the ruling's definition.
fn native_layouts_are_free<T: Elem>() {
    warm_einsum::<T>();
    let n = BIG;
    let (compact, transposed, negative, neg_off) = (
        compact_strides(n),
        transposed_strides(n),
        negative_strides(n),
        negative_offset(n),
    );
    let bytes = n * n * std::mem::size_of::<T>();
    let run = |expr: &str, count: usize, strides: [isize; 2], offset: isize| -> usize {
        let ops = Operands::<T>::sized(count, n);
        let compacts = ops.compact();
        let plan = ConcreteEinsumPlan::prepare(compacts.iter().collect::<Vec<_>>(), expr).unwrap();
        let reads = ops.reads(&vec![strides; count], &vec![offset; count]);
        let mut engine = CpuEngine::serial();
        let (result, big) = count_big(bytes, || plan.execute_with(&reads, &mut engine, Forbid));
        result.unwrap();
        big
    };
    for (expr, count) in [("ij,jk,kl->il", 3usize), ("ij,jk->ik", 2)] {
        let base = run(expr, count, compact, 0);
        // Never more operand-sized allocations than the compact baseline (the
        // compact route can allocate more: it is a different strategy).
        assert!(
            run(expr, count, transposed, 0) <= base,
            "{expr}: transposed"
        );
        assert!(
            run(expr, count, negative, neg_off) <= base,
            "{expr}: negative strides"
        );
    }
    // Eager single-operand fallback: the copy happens for compact input too.
    let base = run("ij->ji", 1, compact, 0);
    assert!(
        base >= 1,
        "the eager route copies even a compact owned input"
    );
    assert!(run("ij->ji", 1, transposed, 0) >= 1);
    assert!(run("ij->ji", 1, negative, neg_off) >= 1);
}

#[test]
fn native_einsum_layouts_are_free_f64() {
    native_layouts_are_free::<f64>();
}

#[test]
fn native_einsum_layouts_are_free_complex64() {
    native_layouts_are_free::<Complex64>();
}
