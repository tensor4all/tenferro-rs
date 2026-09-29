//! A/B: every routed operation gives the default backend's result with the
//! tprims provider installed, over dtypes, layouts and thread counts.
use std::sync::Arc;

use num_complex::{Complex32, Complex64};
use std::sync::atomic::{AtomicUsize, Ordering};

use tenferro_cpu::provider::{
    CpuExecutionContext, CpuGemmProvider, CpuGemmRequest, CpuGroupedGemmRequest, CpuProviderOutcome,
};
use tenferro_cpu::{
    CpuBackend, CpuBackendKind, CpuProviderBundle, CpuProviderExecutionCapabilities,
};
use tenferro_cpu_tprims::TprimsProvider;
use tenferro_tensor::backend::{GroupedGemmConfig, GroupedGemmJob};
use tenferro_tensor::{
    BackendSessionHost, ContractionScalar, DotGeneralAccumulation, DotGeneralConfig, Tensor,
    TensorRead, TensorScalar, TensorView, TensorViewMut, TensorWrite, TypedTensorView,
    TypedTensorViewMut,
};

#[derive(Clone, Copy, Debug)]
enum Route {
    /// tprims `dot_general` (general-contraction slot, required so a silent
    /// fallback fails the test).
    Contract,
    /// tprims GEMM slot only: `dot_general` lowers to tprims `gemm` /
    /// `strided_batched_gemm`.
    Gemm,
}

/// Counts the tprims GEMM-slot calls that executed, so a silent fallback to
/// the default backend fails the tests.
#[derive(Debug, Default)]
struct Counting {
    executed: AtomicUsize,
    unsupported: AtomicUsize,
}
static GEMM_CALLS: Counting = Counting {
    executed: AtomicUsize::new(0),
    unsupported: AtomicUsize::new(0),
};

#[derive(Debug)]
struct CountingTprims;

fn count(
    r: tenferro_tensor::Result<CpuProviderOutcome>,
) -> tenferro_tensor::Result<CpuProviderOutcome> {
    match &r {
        Ok(CpuProviderOutcome::Executed) => GEMM_CALLS.executed.fetch_add(1, Ordering::Relaxed),
        _ => GEMM_CALLS.unsupported.fetch_add(1, Ordering::Relaxed),
    };
    r
}

impl CpuGemmProvider for CountingTprims {
    fn execution_capabilities(&self) -> CpuProviderExecutionCapabilities {
        CpuGemmProvider::execution_capabilities(&TprimsProvider::new())
    }
    fn gemm(
        &self,
        c: &CpuExecutionContext<'_>,
        r: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        count(TprimsProvider::new().gemm(c, r))
    }
    fn strided_batched_gemm(
        &self,
        c: &CpuExecutionContext<'_>,
        r: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        count(TprimsProvider::new().strided_batched_gemm(c, r))
    }
    fn grouped_gemm(
        &self,
        c: &CpuExecutionContext<'_>,
        r: CpuGroupedGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        count(TprimsProvider::new().grouped_gemm(c, r))
    }
}

fn backend(threads: usize, route: Option<Route>) -> CpuBackend {
    let base = CpuBackend::with_threads(threads).unwrap();
    let Some(route) = route else { return base };
    let builder = CpuProviderBundle::builder(CpuBackendKind::default_compiled());
    let builder = match route {
        Route::Contract => {
            builder.require_general_contraction_provider(Arc::new(TprimsProvider::new()))
        }
        Route::Gemm => builder.gemm_provider(Arc::new(CountingTprims)),
    };
    base.with_provider_bundle(builder.build().unwrap()).unwrap()
}

trait T: TensorScalar + Copy + std::fmt::Debug {
    fn make(re: f64, im: f64) -> Self;
    fn mag(self) -> f64;
    fn sub(self, o: Self) -> Self;
    fn scalar(self) -> ContractionScalar;
    fn view<'a>(v: TypedTensorView<'a, Self>) -> TensorView<'a>;
    fn view_mut<'a>(v: TypedTensorViewMut<'a, Self>) -> TensorViewMut<'a>;
    const TOL: f64;
}
macro_rules! real {
    ($t:ty, $v:ident, $tol:expr) => {
        impl T for $t {
            fn make(re: f64, _: f64) -> Self {
                re as $t
            }
            fn mag(self) -> f64 {
                (self as f64).abs()
            }
            fn sub(self, o: Self) -> Self {
                self - o
            }
            fn scalar(self) -> ContractionScalar {
                ContractionScalar::$v(self)
            }
            fn view<'a>(v: TypedTensorView<'a, Self>) -> TensorView<'a> {
                TensorView::$v(v)
            }
            fn view_mut<'a>(v: TypedTensorViewMut<'a, Self>) -> TensorViewMut<'a> {
                TensorViewMut::$v(v)
            }
            const TOL: f64 = $tol;
        }
    };
}
macro_rules! cplx {
    ($t:ty, $r:ty, $v:ident, $tol:expr) => {
        impl T for $t {
            fn make(re: f64, im: f64) -> Self {
                <$t>::new(re as $r, im as $r)
            }
            fn mag(self) -> f64 {
                self.norm() as f64
            }
            fn sub(self, o: Self) -> Self {
                self - o
            }
            fn scalar(self) -> ContractionScalar {
                ContractionScalar::$v(self)
            }
            fn view<'a>(v: TypedTensorView<'a, Self>) -> TensorView<'a> {
                TensorView::$v(v)
            }
            fn view_mut<'a>(v: TypedTensorViewMut<'a, Self>) -> TensorViewMut<'a> {
                TensorViewMut::$v(v)
            }
            const TOL: f64 = $tol;
        }
    };
}
real!(f32, F32, 1e-4);
real!(f64, F64, 1e-12);
cplx!(Complex32, f32, C32, 1e-4);
cplx!(Complex64, f64, C64, 1e-12);

fn data<X: T>(len: usize, seed: u64) -> Vec<X> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            X::make(
                (s % 1000) as f64 / 500.0 - 1.0,
                ((s >> 20) % 1000) as f64 / 500.0 - 1.0,
            )
        })
        .collect()
}

/// An operand layout: shape, element strides, offset and backing length.
#[derive(Clone)]
struct Lay {
    shape: Vec<usize>,
    strides: Vec<isize>,
    offset: isize,
    len: usize,
}

fn col_major(shape: &[usize]) -> Lay {
    let mut strides = Vec::new();
    let mut acc = 1isize;
    for &d in shape {
        strides.push(acc);
        acc *= d.max(1) as isize;
    }
    Lay {
        shape: shape.to_vec(),
        strides,
        offset: 0,
        len: shape.iter().product(),
    }
}

/// `shape` stored with axes in reverse order (row-major), 3 elements of
/// padding in front.
fn transposed(shape: &[usize]) -> Lay {
    let rev: Vec<usize> = shape.iter().rev().copied().collect();
    let base = col_major(&rev);
    Lay {
        shape: shape.to_vec(),
        strides: base.strides.iter().rev().copied().collect(),
        offset: 3,
        len: base.len + 3,
    }
}

/// Column-major with axis 0 reversed (negative stride).
fn reversed(shape: &[usize]) -> Lay {
    let mut l = col_major(shape);
    if shape[0] > 1 {
        l.offset = (shape[0] - 1) as isize;
        l.strides[0] = -1;
    }
    l
}

struct Case {
    name: &'static str,
    a: Lay,
    b: Lay,
    cfg: DotGeneralConfig,
    out: Vec<usize>,
}

fn cfg(lc: &[usize], rc: &[usize], lb: &[usize], rb: &[usize]) -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: lc.into(),
        rhs_contracting_dims: rc.into(),
        lhs_batch_dims: lb.into(),
        rhs_batch_dims: rb.into(),
    }
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "matmul",
            a: col_major(&[6, 4]),
            b: col_major(&[4, 5]),
            cfg: cfg(&[1], &[0], &[], &[]),
            out: vec![6, 5],
        },
        Case {
            name: "matmul_transposed_a",
            a: transposed(&[6, 4]),
            b: col_major(&[4, 5]),
            cfg: cfg(&[1], &[0], &[], &[]),
            out: vec![6, 5],
        },
        Case {
            name: "matmul_reversed_b",
            a: col_major(&[6, 4]),
            b: reversed(&[4, 5]),
            cfg: cfg(&[1], &[0], &[], &[]),
            out: vec![6, 5],
        },
        Case {
            name: "batched",
            a: col_major(&[5, 3, 7]),
            b: col_major(&[3, 4, 7]),
            cfg: cfg(&[1], &[0], &[2], &[2]),
            out: vec![5, 4, 7],
        },
        Case {
            name: "batched_transposed",
            a: transposed(&[5, 3, 7]),
            b: transposed(&[3, 4, 7]),
            cfg: cfg(&[1], &[0], &[2], &[2]),
            out: vec![5, 4, 7],
        },
        Case {
            name: "two_contracted",
            a: col_major(&[4, 5, 6]),
            b: col_major(&[6, 5, 3]),
            cfg: cfg(&[1, 2], &[1, 0], &[], &[]),
            out: vec![4, 3],
        },
        Case {
            name: "outer",
            a: col_major(&[3, 2]),
            b: col_major(&[4]),
            cfg: cfg(&[], &[], &[], &[]),
            out: vec![3, 2, 4],
        },
        Case {
            name: "large",
            a: col_major(&[96, 80]),
            b: col_major(&[80, 72]),
            cfg: cfg(&[1], &[0], &[], &[]),
            out: vec![96, 72],
        },
        Case {
            name: "empty_free",
            a: col_major(&[0, 4]),
            b: col_major(&[4, 5]),
            cfg: cfg(&[1], &[0], &[], &[]),
            out: vec![0, 5],
        },
        Case {
            name: "empty_contracted",
            a: col_major(&[3, 0]),
            b: col_major(&[0, 2]),
            cfg: cfg(&[1], &[0], &[], &[]),
            out: vec![3, 2],
        },
    ]
}

/// `alpha * op(A) op(B) + beta * C0` into a column-major output.
fn run<X: T>(
    threads: usize,
    route: Option<Route>,
    c: &Case,
    conj: (bool, bool),
    alpha: X,
    beta: X,
) -> Vec<X> {
    let ad: Vec<X> = data(c.a.len, 1);
    let bd: Vec<X> = data(c.b.len, 2);
    let o = col_major(&c.out);
    let mut out: Vec<X> = data(o.len, 3);
    let av = TypedTensorView::from_slice(&c.a.shape, &c.a.strides, c.a.offset, &ad).unwrap();
    let bv = TypedTensorView::from_slice(&c.b.shape, &c.b.strides, c.b.offset, &bd).unwrap();
    let ov = TypedTensorViewMut::from_slice(&o.shape, &o.strides, 0, &mut out).unwrap();
    let acc = DotGeneralAccumulation {
        lhs_conj: conj.0,
        rhs_conj: conj.1,
        alpha: alpha.scalar(),
        beta: beta.scalar(),
    };
    let mut be = backend(threads, route);
    be.with_backend_session(|s| {
        s.dot_general_read_into_accum(
            TensorRead::from_view(X::view(av)),
            TensorRead::from_view(X::view(bv)),
            &c.cfg,
            acc,
            TensorWrite::from_view(X::view_mut(ov)),
        )
    })
    .unwrap()
    .unwrap_or_else(|e| panic!("{} {route:?}: {e}", c.name));
    out
}

fn close<X: T>(what: &str, got: &[X], want: &[X]) {
    let scale = want.iter().map(|x| x.mag()).fold(1.0, f64::max);
    let err = got
        .iter()
        .zip(want)
        .map(|(g, w)| g.sub(*w).mag())
        .fold(0.0, f64::max)
        / scale;
    assert!(err < X::TOL, "{what}: rel err {err:e}");
}

fn sweep<X: T>(threads: usize) {
    let complex = X::make(0.0, 1.0).mag() > 0.0;
    for c in cases() {
        for route in [Route::Contract, Route::Gemm] {
            for (conj, alpha, beta) in [
                ((false, false), X::make(1.0, 0.0), X::make(0.0, 0.0)),
                ((complex, false), X::make(0.5, 0.25), X::make(-1.5, 0.5)),
            ] {
                let want = run::<X>(threads, None, &c, conj, alpha, beta);
                let got = run::<X>(threads, Some(route), &c, conj, alpha, beta);
                close(
                    &format!("{} {route:?} {threads}T conj={conj:?}", c.name),
                    &got,
                    &want,
                );
            }
        }
    }
}

#[test]
fn dot_general_and_gemm_match_the_default_backend_1t() {
    sweep::<f64>(1);
    sweep::<Complex64>(1);
    sweep::<f32>(1);
    sweep::<Complex32>(1);
    assert!(
        GEMM_CALLS.executed.load(Ordering::Relaxed) > 0,
        "the GEMM route never reached tprims"
    );
    assert_eq!(
        GEMM_CALLS.unsupported.load(Ordering::Relaxed),
        0,
        "tprims declined a GEMM call"
    );
}

#[test]
fn dot_general_and_gemm_match_the_default_backend_4t() {
    sweep::<f64>(4);
    sweep::<Complex64>(4);
}

#[test]
fn allocating_dot_general_matches_too() {
    let a = Tensor::from_vec_col_major(vec![3, 4], (0..12).map(|x| x as f64).collect()).unwrap();
    let b =
        Tensor::from_vec_col_major(vec![4, 2], (0..8).map(|x| x as f64 - 3.0).collect()).unwrap();
    let c = cfg(&[1], &[0], &[], &[]);
    let go = |route| {
        let mut be = backend(2, route);
        be.with_backend_session(|s| {
            s.dot_general_read(TensorRead::from_tensor(&a), TensorRead::from_tensor(&b), &c)
        })
        .unwrap()
        .unwrap()
    };
    let want = go(None);
    for route in [Route::Contract, Route::Gemm] {
        let got = go(Some(route));
        assert_eq!(
            got.as_slice::<f64>().unwrap(),
            want.as_slice::<f64>().unwrap(),
            "{route:?}"
        );
    }
}

fn grouped<X: T>(threads: usize, tprims: bool) -> Vec<X> {
    let shapes = [
        (3, 4, 2),
        (1, 1, 1),
        (7, 0, 3),
        (0, 2, 5),
        (16, 9, 11),
        (5, 5, 5),
    ];
    let (mut oa, mut ob, mut oc) = (0, 0, 0);
    let mut jobs = Vec::new();
    for &(m, k, n) in &shapes {
        jobs.push(GroupedGemmJob::new(oc, oa, ob, m, k, n));
        oa += m * k;
        ob += k * n;
        oc += m * n;
    }
    let (ad, bd): (Vec<X>, Vec<X>) = (data(oa, 4), data(ob, 5));
    let mut out: Vec<X> = data(oc, 6);
    let av = TypedTensorView::from_slice([oa], [1], 0, &ad).unwrap();
    let bv = TypedTensorView::from_slice([ob], [1], 0, &bd).unwrap();
    let ov = TypedTensorViewMut::from_slice([oc], [1], 0, &mut out).unwrap();
    let acc = DotGeneralAccumulation {
        lhs_conj: false,
        rhs_conj: false,
        alpha: X::make(0.5, 0.0).scalar(),
        beta: X::make(2.0, 0.0).scalar(),
    };
    let mut be = backend(threads, tprims.then_some(Route::Gemm));
    be.with_backend_session(|s| {
        s.grouped_gemm_cached(
            None,
            TensorRead::from_view(X::view(av)),
            TensorRead::from_view(X::view(bv)),
            &GroupedGemmConfig::new(&jobs, acc),
            TensorWrite::from_view(X::view_mut(ov)),
        )
    })
    .unwrap()
    .unwrap();
    out
}

#[test]
fn grouped_gemm_matches_the_default_backend() {
    for threads in [1, 4] {
        close(
            "grouped f64",
            &grouped::<f64>(threads, true),
            &grouped::<f64>(threads, false),
        );
        close(
            "grouped c64",
            &grouped::<Complex64>(threads, true),
            &grouped::<Complex64>(threads, false),
        );
    }
}
