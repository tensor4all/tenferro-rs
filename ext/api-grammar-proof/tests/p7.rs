//! Prototype 7: the borrowed engine and lane splitting as ordinary grammar (#2552).
//!
//! The four conditions for deleting the six `*_with_exec` functions: lifetimes, buffer reuse
//! (the allocation half is in `p7_allocation.rs`, a separate binary with a counting allocator),
//! nested/concurrent execution, and no retention past the call.

use std::error::Error as _;
use std::sync::mpsc;
use std::time::Duration;

use num_complex::{Complex32, Complex64};
use tcontract_exec::Exec;
use tenferro_api_grammar_proof::p7_borrowed_engine::{
    BorrowedCpuEngine, BorrowedLinalg, CpuEngine, SvdOptions, SvdShape,
};
use tenferro_cpu::linalg_interop::BufferPool;
use tenferro_linalg::{
    eig_values_with_exec, eigh_values_with_exec, eigh_with_exec, qr_with_exec,
    svd_values_with_exec, svd_with_exec, LinalgScalar, TensorLinalgExt,
};
use tenferro_tensor::{
    Engine, ErrorKind, StridedSliceSpec, Tensor, TensorRead, TensorScalar, TypedTensor,
    TypedTensorView,
};

// ---------------------------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------------------------

trait Sample: LinalgScalar + Copy + std::fmt::Debug {
    fn sample(re: f64, im: f64) -> Self;
}
impl Sample for f64 {
    fn sample(re: f64, _im: f64) -> Self {
        re
    }
}
impl Sample for Complex64 {
    fn sample(re: f64, im: f64) -> Self {
        Complex64::new(re, im)
    }
}
impl Sample for f32 {
    fn sample(re: f64, _im: f64) -> Self {
        re as f32
    }
}
impl Sample for Complex32 {
    fn sample(re: f64, im: f64) -> Self {
        Complex32::new(re as f32, im as f32)
    }
}

fn base<T: Sample>(shape: &[usize]) -> TypedTensor<T> {
    let len: usize = shape.iter().product();
    let data = (0..len)
        .map(|index| {
            let x = index as f64;
            T::sample(1.0 + (x * 0.37).sin() + 0.1 * x, 0.25 * (x * 0.71).cos())
        })
        .collect();
    TypedTensor::from_vec_col_major(shape.to_vec(), data).unwrap()
}

fn hermitian<T: Sample>(n: usize, batch: usize) -> TypedTensor<T> {
    let mut data = Vec::with_capacity(n * n * batch);
    for b in 0..batch {
        for col in 0..n {
            for row in 0..n {
                let (lo, hi) = (row.min(col) as f64, row.max(col) as f64);
                let re = if row == col {
                    3.0 + row as f64 + b as f64
                } else {
                    0.5 / (1.0 + hi - lo)
                };
                let im = match row.cmp(&col) {
                    std::cmp::Ordering::Less => 0.125 * (1.0 + lo),
                    std::cmp::Ordering::Greater => -0.125 * (1.0 + lo),
                    std::cmp::Ordering::Equal => 0.0,
                };
                data.push(T::sample(re, im));
            }
        }
    }
    TypedTensor::from_vec_col_major(vec![n, n, batch], data).unwrap()
}

/// Bit identity of two typed tensors: shape, placement and `Debug` text of the data.
fn bits<S: TensorScalar + std::fmt::Debug>(t: &TypedTensor<S>) -> (Vec<usize>, String, String) {
    (
        t.shape().to_vec(),
        format!("{:?}", t.placement()),
        format!("{:?}", t.host_data().unwrap()),
    )
}

/// Run `body` on a borrowed engine built over a fresh serial engine's bundle and a fresh pool.
fn on_borrowed<R>(
    body: impl FnOnce(&mut dyn Engine) -> tenferro_tensor::Result<R>,
) -> tenferro_tensor::Result<R> {
    let mut host = CpuEngine::serial();
    let mut pool = BufferPool::new();
    host.with_exec(|exec| body(&mut BorrowedCpuEngine::new(exec, &mut pool)))?
}

/// The existing borrowed free function, on a fresh bundle and pool.
fn on_free<R>(
    body: impl FnOnce(&mut Exec<'_>, &mut BufferPool) -> tenferro_tensor::Result<R>,
) -> tenferro_tensor::Result<R> {
    let mut host = CpuEngine::serial();
    let mut pool = BufferPool::new();
    host.with_exec(|exec| body(exec, &mut pool))?
}

// ---------------------------------------------------------------------------------------------
// Result parity with the `*_with_exec` entries (and with the ordinary CpuEngine)
// ---------------------------------------------------------------------------------------------

fn general_parity<T: Sample>(label: &str, view: &TypedTensorView<'_, T>)
where
    T::RealScalar: std::fmt::Debug,
{
    let mut cpu = CpuEngine::serial();
    for (shape, full) in [(SvdShape::Thin, false), (SvdShape::Full, true)] {
        let opts = SvdOptions::default().shape(shape);
        let g = on_borrowed(|e| view.svd_with(&opts, e)).unwrap();
        let (u, s, vh) = on_free(|e, b| svd_with_exec(e, b, view, full)).unwrap();
        let c = view.svd_with(&opts, &mut cpu).unwrap();
        for (tag, (gu, gs, gv)) in [
            ("borrowed", (&g.u, &g.s, &g.vh)),
            ("cpu", (&c.u, &c.s, &c.vh)),
        ] {
            assert_eq!(bits(gu), bits(&u), "{label} svd full={full} U ({tag})");
            assert_eq!(bits(gs), bits(&s), "{label} svd full={full} S ({tag})");
            assert_eq!(bits(gv), bits(&vh), "{label} svd full={full} Vh ({tag})");
        }
    }
    let g = on_borrowed(|e| view.svdvals(e)).unwrap();
    let s = on_free(|e, b| svd_values_with_exec(e, b, view)).unwrap();
    assert_eq!(bits(&g), bits(&s), "{label} svdvals");
    assert_eq!(
        bits(&view.svdvals(&mut cpu).unwrap()),
        bits(&s),
        "{label} svdvals cpu"
    );

    let g = on_borrowed(|e| view.qr(e)).unwrap();
    let (q, r) = on_free(|e, b| qr_with_exec(e, b, view)).unwrap();
    assert_eq!(bits(&g.q), bits(&q), "{label} qr Q");
    assert_eq!(bits(&g.r), bits(&r), "{label} qr R");
    let c = view.qr(&mut cpu).unwrap();
    assert_eq!(bits(&c.q), bits(&q), "{label} qr Q cpu");
}

fn square_parity<T: Sample>(label: &str, view: &TypedTensorView<'_, T>)
where
    T::RealScalar: std::fmt::Debug,
    T::Complex: std::fmt::Debug,
{
    let mut cpu = CpuEngine::serial();
    let g = on_borrowed(|e| view.eigh(e)).unwrap();
    let (values, vectors) = on_free(|e, b| eigh_with_exec(e, b, view)).unwrap();
    assert_eq!(bits(&g.values), bits(&values), "{label} eigh values");
    assert_eq!(bits(&g.vectors), bits(&vectors), "{label} eigh vectors");
    let c = view.eigh(&mut cpu).unwrap();
    assert_eq!(bits(&c.vectors), bits(&vectors), "{label} eigh vectors cpu");

    let g = on_borrowed(|e| view.eigvalsh(e)).unwrap();
    let v = on_free(|e, b| eigh_values_with_exec(e, b, view)).unwrap();
    assert_eq!(bits(&g), bits(&v), "{label} eigvalsh");

    let g = on_borrowed(|e| view.eigvals(e)).unwrap();
    let v = on_free(|e, b| eig_values_with_exec(e, b, view)).unwrap();
    assert_eq!(bits(&g), bits(&v), "{label} eigvals");
}

fn parity_for<T: Sample>(dtype: &str)
where
    T::RealScalar: std::fmt::Debug,
    T::Complex: std::fmt::Debug,
{
    for shape in [[3, 2], [2, 3]] {
        let a = base::<T>(&shape);
        general_parity(&format!("{dtype} {shape:?}"), &a.as_view());
    }
    let tall = base::<T>(&[6, 3]);
    let strided = tall
        .as_view()
        .slice_view(&[StridedSliceSpec::new(0, None, 2), StridedSliceSpec::all()])
        .unwrap()
        .transpose_view([1, 0])
        .unwrap();
    general_parity(&format!("{dtype} strided"), &strided);
    for shape in [[0, 3], [3, 0], [0, 0]] {
        let a = base::<T>(&shape);
        general_parity(&format!("{dtype} empty {shape:?}"), &a.as_view());
    }
    let stacked = base::<T>(&[2, 4, 3]);
    let batched = stacked.as_view().transpose_view([1, 2, 0]).unwrap();
    general_parity(&format!("{dtype} batched"), &batched);
    let herm = hermitian::<T>(3, 2);
    square_parity(&format!("{dtype} batched square"), &herm.as_view());
    let herm_t = herm.as_view().transpose_view([1, 0, 2]).unwrap();
    square_parity(&format!("{dtype} strided square"), &herm_t);
    let empty = base::<T>(&[0, 0]);
    square_parity(&format!("{dtype} empty square"), &empty.as_view());
}

#[test]
fn grammar_results_are_bit_identical_to_the_with_exec_entries_f64() {
    parity_for::<f64>("f64");
}

#[test]
fn grammar_results_are_bit_identical_to_the_with_exec_entries_complex64() {
    parity_for::<Complex64>("c64");
}

#[test]
fn grammar_results_are_bit_identical_to_the_with_exec_entries_single_precision() {
    parity_for::<f32>("f32");
    parity_for::<Complex32>("c32");
}

// ---------------------------------------------------------------------------------------------
// Error parity
// ---------------------------------------------------------------------------------------------

fn same_error(label: &str, grammar: tenferro_tensor::Error, free: tenferro_tensor::Error) {
    assert_eq!(grammar.kind(), free.kind(), "{label}: kind");
    assert_eq!(
        format!("{grammar:?}"),
        format!("{free:?}"),
        "{label}: variant"
    );
    let payload = |e: &tenferro_tensor::Error| {
        e.source()
            .and_then(|s| s.downcast_ref::<tenferro_linalg::Error>())
            .map(|p| format!("{p:?}"))
    };
    assert_eq!(payload(&grammar), payload(&free), "{label}: source");
}

#[test]
fn errors_have_the_same_kind_variant_and_source_as_the_with_exec_entries() {
    // Rank 1 and non-square.
    let vector = base::<f64>(&[3]);
    let v = vector.as_view();
    same_error(
        "rank-1 svd",
        on_borrowed(|e| v.svd(e)).unwrap_err(),
        on_free(|e, b| svd_with_exec(e, b, &v, false)).unwrap_err(),
    );
    same_error(
        "rank-1 qr",
        on_borrowed(|e| v.qr(e)).unwrap_err(),
        on_free(|e, b| qr_with_exec(e, b, &v)).unwrap_err(),
    );
    let rect = base::<f64>(&[3, 2]);
    let r = rect.as_view();
    same_error(
        "non-square eigh",
        on_borrowed(|e| r.eigh(e)).unwrap_err(),
        on_free(|e, b| eigh_with_exec(e, b, &r)).unwrap_err(),
    );
    same_error(
        "non-square eigvals",
        on_borrowed(|e| r.eigvals(e)).unwrap_err(),
        on_free(|e, b| eig_values_with_exec(e, b, &r)).unwrap_err(),
    );

    // Non-finite input: the provider's own mapped failure, wherever it reports one.
    let mut data = base::<f64>(&[3, 3]).host_data().unwrap().to_vec();
    data[4] = f64::NAN;
    let nan = TypedTensor::<f64>::from_vec_col_major(vec![3, 3], data).unwrap();
    let n = nan.as_view();
    let mut failures = 0;
    let cases = [
        (
            "nan svd",
            on_borrowed(|e| n.svd(e)).map(|_| ()),
            on_free(|e, b| svd_with_exec(e, b, &n, false)).map(|_| ()),
        ),
        (
            "nan qr",
            on_borrowed(|e| n.qr(e)).map(|_| ()),
            on_free(|e, b| qr_with_exec(e, b, &n)).map(|_| ()),
        ),
        (
            "nan eigh",
            on_borrowed(|e| n.eigh(e)).map(|_| ()),
            on_free(|e, b| eigh_with_exec(e, b, &n)).map(|_| ()),
        ),
        (
            "nan eigvals",
            on_borrowed(|e| n.eigvals(e)).map(|_| ()),
            on_free(|e, b| eig_values_with_exec(e, b, &n)).map(|_| ()),
        ),
    ];
    for (label, g, f) in cases {
        match (g, f) {
            (Ok(()), Ok(())) => {}
            (Err(g), Err(f)) => {
                failures += 1;
                same_error(label, g, f);
            }
            (g, f) => panic!("{label}: grammar {g:?} but free {f:?}"),
        }
    }
    assert!(
        failures > 0,
        "the NaN fixture must reach a provider failure"
    );
}

#[test]
fn a_negative_stride_is_unsupported_with_the_same_message_for_every_entry() {
    let a = base::<f64>(&[3, 3]);
    let reversed = a
        .as_view()
        .slice_view(&[StridedSliceSpec::reverse(), StridedSliceSpec::all()])
        .unwrap();
    assert!(reversed.strides()[0] < 0);
    let empty = reversed
        .slice_view(&[
            StridedSliceSpec::new(0, Some(0), 1),
            StridedSliceSpec::new(0, Some(0), 1),
        ])
        .unwrap();

    let mut cpu = CpuEngine::serial();
    for (tag, view) in [("non-empty", &reversed), ("empty", &empty)] {
        let g: Vec<(&str, tenferro_tensor::Error)> = vec![
            ("svd", on_borrowed(|e| view.svd(e)).unwrap_err()),
            (
                "svd_full",
                on_borrowed(|e| view.svd_with(&SvdOptions::default().shape(SvdShape::Full), e))
                    .unwrap_err(),
            ),
            ("svdvals", on_borrowed(|e| view.svdvals(e)).unwrap_err()),
            ("qr", on_borrowed(|e| view.qr(e)).unwrap_err()),
            ("eigh", on_borrowed(|e| view.eigh(e)).unwrap_err()),
            ("eigvalsh", on_borrowed(|e| view.eigvalsh(e)).unwrap_err()),
            ("eigvals", on_borrowed(|e| view.eigvals(e)).unwrap_err()),
        ];
        let f: Vec<tenferro_tensor::Error> = vec![
            on_free(|e, b| svd_with_exec(e, b, view, false)).unwrap_err(),
            on_free(|e, b| svd_with_exec(e, b, view, true)).unwrap_err(),
            on_free(|e, b| svd_values_with_exec(e, b, view)).unwrap_err(),
            on_free(|e, b| qr_with_exec(e, b, view)).unwrap_err(),
            on_free(|e, b| eigh_with_exec(e, b, view)).unwrap_err(),
            on_free(|e, b| eigh_values_with_exec(e, b, view)).unwrap_err(),
            on_free(|e, b| eig_values_with_exec(e, b, view)).unwrap_err(),
        ];
        for ((op, g), f) in g.into_iter().zip(f) {
            assert_eq!(g.kind(), ErrorKind::Unsupported, "{tag} {op}: {g}");
            same_error(&format!("{tag} {op}"), g, f);
        }
    }
    // The same grammar call on an ordinary CpuEngine keeps the borrowed rule too: the typed view
    // receiver does not gather, whichever engine carries it.
    assert_eq!(
        reversed.svd(&mut cpu).unwrap_err().kind(),
        ErrorKind::Unsupported
    );
    // ...while the erased engine entry of the same data gathers (finding F1 in p7.md).
    let read = TensorRead::from_view(tenferro_tensor::TensorView::F64(reversed.clone()));
    assert!(tenferro_linalg::TensorReadLinalgExt::svd_read(read, &mut cpu).is_ok());
}

// ---------------------------------------------------------------------------------------------
// Condition 1: lifetimes
// ---------------------------------------------------------------------------------------------

#[test]
fn the_borrowed_engine_is_a_dyn_engine_over_borrowed_exec_and_pool_without_static() {
    // `Exec` and `BufferPool` are locals of this frame: nothing here is `'static`.
    let a = base::<f64>(&[4, 3]);
    let mut exec = Exec::serial();
    let mut pool = BufferPool::new();
    {
        let mut borrowed = BorrowedCpuEngine::new(&mut exec, &mut pool);
        let engine: &mut dyn Engine = &mut borrowed;
        let first = a.as_view().qr(engine).unwrap();
        let second = a.as_view().qr(engine).unwrap();
        assert_eq!(bits(&first.q), bits(&second.q));
    }
    // The borrow ended with the block: the pool and the bundle are usable again.
    assert!(pool.stats().buffers > 0);
    assert_eq!(exec.budget(), 1);
    drop(
        a.as_view()
            .qr(&mut BorrowedCpuEngine::new(&mut exec, &mut pool))
            .unwrap(),
    );
}

#[test]
fn the_borrowed_engine_refuses_every_other_operation_and_the_erased_entries() {
    let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let r = on_borrowed(|e| e.add_read(TensorRead::from_tensor(&a), TensorRead::from_tensor(&a)));
    assert_eq!(r.unwrap_err().kind(), ErrorKind::Unsupported);
    // The erased engine entry looks for a CPU engine and does not find one.
    let r = on_borrowed(|e| a.svd(e).map(|_| ()));
    assert_eq!(r.unwrap_err().kind(), ErrorKind::Unsupported);
}

// ---------------------------------------------------------------------------------------------
// Condition 2 (pool half): the caller's pool, not the engine's
// ---------------------------------------------------------------------------------------------

fn run_all(engine: &mut dyn Engine) {
    let a = base::<Complex64>(&[4, 3, 2]);
    let h = hermitian::<Complex64>(3, 2);
    let (a, h) = (a.as_view(), h.as_view());
    drop(a.svd(engine).unwrap());
    drop(
        a.svd_with(&SvdOptions::default().shape(SvdShape::Full), engine)
            .unwrap(),
    );
    drop(a.svdvals(engine).unwrap());
    drop(a.qr(engine).unwrap());
    drop(h.eigh(engine).unwrap());
    drop(h.eigvalsh(engine).unwrap());
    drop(h.eigvals(engine).unwrap());
}

#[test]
fn every_buffer_comes_from_the_callers_pool_and_the_engines_pool_is_untouched() {
    let mut host = CpuEngine::serial();
    let engine_before = host.buffer_pool_stats();
    let mut lane_pool = BufferPool::new();
    host.with_exec(|exec| run_all(&mut BorrowedCpuEngine::new(exec, &mut lane_pool)))
        .unwrap();
    let warm = lane_pool.stats();
    assert!(warm.buffers > 0, "outputs recycle into the caller's pool");
    assert_eq!(
        host.buffer_pool_stats(),
        engine_before,
        "engine pool untouched"
    );
    host.with_exec(|exec| run_all(&mut BorrowedCpuEngine::new(exec, &mut lane_pool)))
        .unwrap();
    assert_eq!(
        lane_pool.stats(),
        warm,
        "steady state: the pool does not grow"
    );
    assert_eq!(host.buffer_pool_stats(), engine_before);
    println!("p7 pool: caller pool {warm:?}, engine pool {engine_before:?} before and after");
}

// ---------------------------------------------------------------------------------------------
// Condition 3: nested and concurrent execution
// ---------------------------------------------------------------------------------------------

fn team_engine(threads: usize) -> CpuEngine {
    let pool = tenferro_cpu::Pool::builder()
        .threads(threads)
        .acquire_timeout(Duration::from_secs(10))
        .build()
        .expect("pool");
    CpuEngine::new(pool)
}

/// Run `body` on a helper thread and fail instead of hanging if it deadlocks.
fn within_a_minute<R: Send + 'static>(body: impl FnOnce() -> R + Send + 'static) -> R {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(body());
    });
    rx.recv_timeout(Duration::from_secs(60))
        .expect("no result within 60 s: deadlock or lost lane")
}

#[test]
fn nested_in_with_exec_and_with_contraction_exec() {
    let a = base::<f64>(&[5, 4]);
    let want = on_free(|e, b| qr_with_exec(e, b, &a.as_view())).unwrap();

    // `with_exec`, serial and team engines.
    for mut host in [CpuEngine::serial(), team_engine(2)] {
        let mut pool = BufferPool::new();
        let got = host
            .with_exec(|exec| a.as_view().qr(&mut BorrowedCpuEngine::new(exec, &mut pool)))
            .unwrap()
            .unwrap();
        assert_eq!(bits(&got.q), bits(&want.0));
        assert!(!host.is_broken());
        // The outer engine still answers ordinary calls: it was never re-entered or left locked.
        let owned = Tensor::from_typed::<f64>(
            TypedTensor::from_vec_col_major(vec![5, 4], a.host_data().unwrap().to_vec()).unwrap(),
        );
        assert!(owned.qr(&mut host).is_ok());
    }

    // `with_contraction_exec` lends the engine's own pool: the borrowed engine fills *that* pool.
    let mut host = CpuEngine::serial();
    let got = host
        .with_contraction_exec(|exec, engine_pool, _| {
            a.as_view()
                .qr(&mut BorrowedCpuEngine::new(exec, engine_pool))
        })
        .unwrap();
    assert_eq!(bits(&got.q), bits(&want.0));
    drop(got);
    assert!(host.buffer_pool_stats().buffers > 0);
    assert!(!host.is_broken());
}

/// Two factorizations on two sub-teams of one engine, built by hand from the public pieces:
/// `Exec::token().fork_sized` plus `Exec::from_token` (docs/guides/cpu-execution.md "Lanes").
#[test]
fn concurrent_lanes_from_fork_sized_and_from_token() {
    let a = base::<f64>(&[24, 24]);
    let b = hermitian::<Complex64>(20, 1);
    let want_qr = on_free(|e, p| qr_with_exec(e, p, &a.as_view())).unwrap();
    let want_eigh = on_free(|e, p| eigh_with_exec(e, p, &b.as_view())).unwrap();

    within_a_minute(move || {
        let mut host = team_engine(4);
        assert!(host.has_team(), "the engine must own a team for this test");
        let pools = [
            std::sync::Mutex::new(BufferPool::new()),
            std::sync::Mutex::new(BufferPool::new()),
        ];
        let results = std::sync::Mutex::new((None, None));
        host.with_exec(|exec| {
            let workspace = exec.workspace();
            let policy = *exec.batch_policy();
            exec.token()
                .fork_sized(&[3, 1], |lane, child| {
                    let mut bundle = Exec::from_token(child, workspace, policy);
                    let mut pool = pools[lane].lock().unwrap();
                    let mut engine = BorrowedCpuEngine::new(&mut bundle, &mut pool);
                    let budget = engine_budget(&mut engine);
                    assert_eq!(budget, [3, 1][lane]);
                    if lane == 0 {
                        let r = a.as_view().qr(&mut engine).unwrap();
                        results.lock().unwrap().0 = Some(r);
                    } else {
                        let r = b.as_view().eigh(&mut engine).unwrap();
                        results.lock().unwrap().1 = Some(r);
                    }
                })
                .unwrap();
        })
        .unwrap();
        let (qr, eigh) = results.into_inner().unwrap();
        let (qr, eigh) = (qr.unwrap(), eigh.unwrap());
        assert_eq!(bits(&qr.q), bits(&want_qr.0));
        assert_eq!(bits(&qr.r), bits(&want_qr.1));
        assert_eq!(bits(&eigh.values), bits(&want_eigh.0));
        assert_eq!(bits(&eigh.vectors), bits(&want_eigh.1));
        assert!(!host.is_broken());
        // Both lane pools were used, the engine's was not.
        for pool in &pools {
            assert!(pool.lock().unwrap().stats().buffers > 0);
        }
        assert_eq!(host.buffer_pool_stats().buffers, 0);
        // The outer engine is intact afterwards.
        let owned = Tensor::from_typed::<f64>(
            TypedTensor::from_vec_col_major(vec![24, 24], a.host_data().unwrap().to_vec()).unwrap(),
        );
        owned.qr(&mut host).unwrap();
    });
}

fn engine_budget(engine: &mut dyn Engine) -> usize {
    tenferro_cpu::with_borrowed_cpu_engine(engine, |exec, _| exec.budget()).unwrap()
}

/// The same split through the `CpuEngine::with_lanes` hook: one call, lanes are `&mut dyn Engine`.
#[test]
fn with_lanes_runs_ordinary_grammar_calls_concurrently() {
    let a = base::<Complex64>(&[24, 18]);
    let h = hermitian::<Complex64>(16, 3);
    let want_svd = on_free(|e, p| svd_with_exec(e, p, &a.as_view(), false)).unwrap();
    let want_vals = on_free(|e, p| eigh_values_with_exec(e, p, &h.as_view())).unwrap();

    within_a_minute(move || {
        let mut host = team_engine(4);
        assert!(host.has_team());
        let mut pools = [BufferPool::new(), BufferPool::new()];
        let out = host
            .with_lanes(&[2, 2], &mut pools, |lane, engine| {
                assert_eq!(engine_budget(engine), 2);
                if lane == 0 {
                    let r = a.as_view().svd(engine).unwrap();
                    (Some(r), None)
                } else {
                    (None, Some(h.as_view().eigvalsh(engine).unwrap()))
                }
            })
            .unwrap();
        let svd = out[0].0.as_ref().unwrap();
        assert_eq!(bits(&svd.u), bits(&want_svd.0));
        assert_eq!(bits(&svd.s), bits(&want_svd.1));
        assert_eq!(bits(&svd.vh), bits(&want_svd.2));
        assert_eq!(bits(out[1].1.as_ref().unwrap()), bits(&want_vals));
        assert!(pools.iter().all(|p| p.stats().buffers > 0));
        assert_eq!(host.buffer_pool_stats().buffers, 0);
        assert!(!host.is_broken());
        // Invalid splits are typed errors and leave the engine usable.
        let mut pools3 = [BufferPool::new(), BufferPool::new()];
        for shares in [&[3usize, 2][..], &[0, 1][..], &[1][..]] {
            let err = host.with_lanes(shares, &mut pools3, |_, _| ()).unwrap_err();
            assert!(
                matches!(err.kind(), ErrorKind::Validation(_)),
                "{shares:?}: {err}"
            );
        }
        assert!(!host.is_broken());
        // Serial engines run lanes inline, in order.
        let mut serial = CpuEngine::serial();
        let mut one = [BufferPool::new()];
        let got = serial
            .with_lanes(&[1], &mut one, |_, e| a.as_view().svdvals(e).unwrap())
            .unwrap();
        assert_eq!(got.len(), 1);
    });
}

#[test]
fn a_panic_in_a_lane_breaks_the_engine_like_a_panic_in_with_exec() {
    let mut host = team_engine(2);
    let mut pools = [BufferPool::new(), BufferPool::new()];
    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = host.with_lanes(&[1, 1], &mut pools, |lane, _| {
            if lane == 1 {
                panic!("a failing lane");
            }
        });
    }));
    assert!(unwind.is_err());
    assert!(host.is_broken());
    assert!(host.with_lanes(&[1, 1], &mut pools, |_, _| ()).is_err());
}

// ---------------------------------------------------------------------------------------------
// Condition 4: retention past the call
// ---------------------------------------------------------------------------------------------

#[test]
fn outputs_may_outlive_the_borrowed_engine_and_the_pool() {
    let a = base::<f64>(&[6, 5]);
    let want = on_free(|e, b| qr_with_exec(e, b, &a.as_view())).unwrap();

    // (1) Engine gone, pool alive: the output still returns its buffer to the pool on drop.
    let mut host = CpuEngine::serial();
    let mut pool = BufferPool::new();
    let qr = host
        .with_exec(|exec| a.as_view().qr(&mut BorrowedCpuEngine::new(exec, &mut pool)))
        .unwrap()
        .unwrap();
    // The provider workspace has already returned; Q and R are still checked out.
    let while_held = pool.stats().buffers;
    let q_bits = bits(&qr.q);
    drop(qr);
    let returned = pool.stats();
    assert!(
        returned.buffers == while_held + 2,
        "dropping the outputs returns Q and R: {while_held} held, {returned:?}"
    );

    // (2) Engine and pool gone: the output is plain owned storage. Its recycler is a Weak handle,
    // so nothing is dangling and nothing is kept alive; it frees like any Vec.
    let qr = {
        let mut host = CpuEngine::serial();
        let mut pool = BufferPool::new();
        host.with_exec(|exec| a.as_view().qr(&mut BorrowedCpuEngine::new(exec, &mut pool)))
            .unwrap()
            .unwrap()
    };
    assert_eq!(bits(&qr.q), q_bits);
    assert_eq!(bits(&qr.q), bits(&want.0));
    assert_eq!(bits(&qr.r), bits(&want.1));
    drop(qr); // would crash or leak into a dead pool if the recycler were a strong or raw handle

    // (3) The same holds for the existing `*_with_exec` entry: same behaviour, nothing new here.
    let q = {
        let mut host = CpuEngine::serial();
        let mut pool = BufferPool::new();
        host.with_exec(|exec| qr_with_exec(exec, &mut pool, &a.as_view()))
            .unwrap()
            .unwrap()
    };
    assert_eq!(bits(&q.0), bits(&want.0));

    // (4) The pool does not keep a strong reference to what it handed out either: dropping the
    // pool while an output is alive, then the output, leaves nothing retained anywhere.
    let mut host = CpuEngine::serial();
    let mut pool = BufferPool::new();
    let held = host
        .with_exec(|exec| {
            a.as_view()
                .svdvals(&mut BorrowedCpuEngine::new(exec, &mut pool))
        })
        .unwrap()
        .unwrap();
    drop(pool);
    assert_eq!(held.shape(), &[5]);
    drop(held);
}

/// Output tensors keep working when moved to another thread after the lanes are gone.
#[test]
fn lane_outputs_are_sendable_after_the_lanes_end() {
    let a = base::<f64>(&[8, 8]);
    let mut host = team_engine(2);
    let mut pools = [BufferPool::new(), BufferPool::new()];
    let outs = host
        .with_lanes(&[1, 1], &mut pools, |_, e| a.as_view().qr(e).unwrap())
        .unwrap();
    drop(pools);
    drop(host);
    let handle = std::thread::spawn(move || {
        outs.iter()
            .map(|o| o.q.shape().to_vec())
            .collect::<Vec<_>>()
    });
    assert_eq!(handle.join().unwrap(), vec![vec![8, 8], vec![8, 8]]);
}

// ---------------------------------------------------------------------------------------------
// Design (C): no borrowed engine, one ordinary serial CpuEngine per lane
// ---------------------------------------------------------------------------------------------

/// Per-lane `CpuEngine::serial()` values give correct concurrent results through the ordinary
/// grammar, but they cannot take the lane's share of the outer budget (a serial engine owns no
/// token), and their pool is the lane engine's own, not a caller-chosen one. So (C) meets the
/// "concurrent factorizations inside one entered engine call" part only as unshared serial work:
/// a lane that was given 3 of 4 slots computes on 1.
#[test]
fn design_c_per_lane_serial_engines_run_concurrently_but_ignore_the_share() {
    let a = base::<f64>(&[24, 24]);
    let want = on_free(|e, p| qr_with_exec(e, p, &a.as_view())).unwrap();
    within_a_minute(move || {
        let mut host = team_engine(4);
        let lanes = [
            std::sync::Mutex::new(CpuEngine::serial()),
            std::sync::Mutex::new(CpuEngine::serial()),
        ];
        let seen = std::sync::Mutex::new(Vec::new());
        host.with_exec(|exec| {
            exec.token()
                .fork_sized(&[3, 1], |lane, child| {
                    let mut engine = lanes[lane].lock().unwrap();
                    let r = a.as_view().qr(&mut *engine).unwrap();
                    seen.lock().unwrap().push((
                        lane,
                        child.budget().get(),
                        engine.num_threads(),
                        bits(&r.q),
                    ));
                })
                .unwrap();
        })
        .unwrap();
        let mut seen = seen.into_inner().unwrap();
        seen.sort_by_key(|s| s.0);
        for (lane, share, used, q) in &seen {
            assert_eq!(q, &bits(&want.0), "lane {lane}");
            assert_eq!(*used, 1, "a serial lane engine computes on one thread");
            println!("p7 (C): lane {lane} had a share of {share}, its engine used {used}");
        }
        assert_eq!(seen[0].1, 3);
        // Each lane engine kept its own pool, not a caller-chosen one.
        assert!(lanes
            .iter()
            .all(|l| l.lock().unwrap().buffer_pool_stats().buffers > 0));
    });
}
