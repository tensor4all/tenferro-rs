//! A/B: the tprims linalg kernels give the built-in kernels' results
//! (exactly where the result is unique, through reconstruction where it is
//! gauge dependent), and decline what they do not handle.
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use num_complex::{Complex32, Complex64};
use tenferro_cpu::{
    with_cpu_exec_session, CpuBackend, CpuBackendKind, CpuExecSession, CpuExecutionContext,
    CpuProviderBundle,
};
use tenferro_cpu_tprims::TprimsProvider;
use tenferro_linalg::backend::LinalgBackend;
use tenferro_linalg::cpu_kernels::{
    install_linalg_kernels, CpuLinalgKernels, CpuLinalgOutcome, TriangularSolveOptions,
};
use tenferro_tensor::{
    BackendSessionHost, Result, Tensor, TensorRead, TensorScalar, TensorView, TypedTensorView,
};

static EXECUTED: AtomicUsize = AtomicUsize::new(0);
static DECLINED: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug)]
struct Counting;

fn count<R>(r: Result<CpuLinalgOutcome<R>>) -> Result<CpuLinalgOutcome<R>> {
    match &r {
        Ok(CpuLinalgOutcome::Executed(_)) => EXECUTED.fetch_add(1, Ordering::Relaxed),
        _ => DECLINED.fetch_add(1, Ordering::Relaxed),
    };
    r
}

macro_rules! delegate {
    ($($name:ident($($arg:ident: $ty:ty),*) -> $out:ty;)*) => {
        impl CpuLinalgKernels for Counting {
            $(fn $name(&self, c: &CpuExecutionContext<'_>, $($arg: $ty),*) -> Result<CpuLinalgOutcome<$out>> {
                count(TprimsProvider::new().$name(c, $($arg),*))
            })*
        }
    };
}
delegate! {
    cholesky(input: TensorView<'_>) -> Tensor;
    triangular_solve(a: TensorView<'_>, b: TensorView<'_>, o: TriangularSolveOptions) -> Tensor;
    solve(a: TensorView<'_>, b: TensorView<'_>) -> Tensor;
    svd(input: TensorView<'_>) -> Vec<Tensor>;
    svd_values(input: TensorView<'_>) -> Tensor;
    qr(input: TensorView<'_>) -> Vec<Tensor>;
    eigh(input: TensorView<'_>) -> Vec<Tensor>;
    eigh_values(input: TensorView<'_>) -> Tensor;
}

fn backend(threads: usize, tprims: bool) -> CpuBackend {
    let base = CpuBackend::with_threads(threads).unwrap();
    if !tprims {
        return base;
    }
    let builder = CpuProviderBundle::builder(CpuBackendKind::default_compiled());
    base.with_provider_bundle(
        install_linalg_kernels(builder, Arc::new(Counting))
            .build()
            .unwrap(),
    )
    .unwrap()
}

fn run<R: Send>(b: &mut CpuBackend, f: impl FnOnce(&mut CpuExecSession<'_>) -> R + Send) -> R {
    b.with_backend_session(|s| with_cpu_exec_session(s, f).expect("a CPU backend session"))
        .unwrap()
}

trait X: TensorScalar + Copy + std::fmt::Debug + Send + Sync {
    fn c(self) -> Complex64;
    fn make(re: f64, im: f64) -> Self;
    const TOL: f64;
}
impl X for f32 {
    fn c(self) -> Complex64 {
        Complex64::new(self as f64, 0.0)
    }
    fn make(r: f64, _: f64) -> Self {
        r as f32
    }
    const TOL: f64 = 2e-4;
}
impl X for f64 {
    fn c(self) -> Complex64 {
        Complex64::new(self, 0.0)
    }
    fn make(r: f64, _: f64) -> Self {
        r
    }
    const TOL: f64 = 1e-10;
}
impl X for Complex32 {
    fn c(self) -> Complex64 {
        Complex64::new(self.re as f64, self.im as f64)
    }
    fn make(r: f64, i: f64) -> Self {
        Complex32::new(r as f32, i as f32)
    }
    const TOL: f64 = 2e-4;
}
impl X for Complex64 {
    fn c(self) -> Complex64 {
        self
    }
    fn make(r: f64, i: f64) -> Self {
        Complex64::new(r, i)
    }
    const TOL: f64 = 1e-10;
}

fn data<T: X>(len: usize, seed: u64) -> Vec<T> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            T::make(
                (s % 1000) as f64 / 500.0 - 1.0,
                ((s >> 20) % 1000) as f64 / 500.0 - 1.0,
            )
        })
        .collect()
}

/// Column-major `m x n` as Complex64.
fn cm(t: &Tensor) -> Vec<Complex64> {
    match t.dtype() {
        tenferro_tensor::DType::F32 => t
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .map(|&x| x.c())
            .collect(),
        tenferro_tensor::DType::F64 => t
            .as_slice::<f64>()
            .unwrap()
            .iter()
            .map(|&x| x.c())
            .collect(),
        tenferro_tensor::DType::C32 => t
            .as_slice::<Complex32>()
            .unwrap()
            .iter()
            .map(|&x| x.c())
            .collect(),
        _ => t.as_slice::<Complex64>().unwrap().to_vec(),
    }
}

fn matmul(a: &[Complex64], b: &[Complex64], m: usize, k: usize, n: usize) -> Vec<Complex64> {
    let mut c = vec![Complex64::new(0.0, 0.0); m * n];
    for j in 0..n {
        for l in 0..k {
            for i in 0..m {
                c[i + j * m] += a[i + l * m] * b[l + j * k];
            }
        }
    }
    c
}

fn close(what: &str, got: &[Complex64], want: &[Complex64], tol: f64) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let scale = want.iter().map(|z| z.norm()).fold(1.0, f64::max);
    let err = got
        .iter()
        .zip(want)
        .map(|(g, w)| (g - w).norm())
        .fold(0.0, f64::max)
        / scale;
    assert!(err < tol, "{what}: rel err {err:e}");
}

fn same_meta(what: &str, got: &[Tensor], want: &[Tensor]) {
    assert_eq!(got.len(), want.len(), "{what}: outputs");
    for (g, w) in got.iter().zip(want) {
        assert_eq!(
            (g.dtype(), g.shape()),
            (w.dtype(), w.shape()),
            "{what}: dtype/shape"
        );
    }
}

/// Hermitian positive definite `n x n`: `M Mᴴ + n I`.
fn hpd<T: X>(n: usize) -> Tensor {
    let m: Vec<Complex64> = data::<T>(n * n, 7).iter().map(|x| x.c()).collect();
    let mut a = vec![Complex64::new(0.0, 0.0); n * n];
    for j in 0..n {
        for i in 0..n {
            let mut s = Complex64::new(0.0, 0.0);
            for l in 0..n {
                s += m[i + l * n] * m[j + l * n].conj();
            }
            a[i + j * n] = s + if i == j {
                Complex64::new(n as f64, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            };
        }
    }
    let v: Vec<T> = a.iter().map(|z| T::make(z.re, z.im)).collect();
    Tensor::from_vec_col_major(vec![n, n], v).unwrap()
}

fn rect<T: X>(m: usize, n: usize, seed: u64) -> Tensor {
    Tensor::from_vec_col_major(vec![m, n], data::<T>(m * n, seed)).unwrap()
}

fn sweep<T: X>(threads: usize) {
    let (mut d, mut t) = (backend(threads, false), backend(threads, true));
    let a = hpd::<T>(6);
    // Cholesky, solve: unique results.
    let (w, g) = (
        run(&mut d, |c| c.cholesky(&a)).unwrap(),
        run(&mut t, |c| c.cholesky(&a)).unwrap(),
    );
    assert_eq!(
        (g.dtype(), g.shape()),
        (w.dtype(), w.shape()),
        "cholesky meta"
    );
    close("cholesky", &cm(&g), &cm(&w), T::TOL);
    for b in [
        rect::<T>(6, 3, 11),
        Tensor::from_vec_col_major(vec![6], data::<T>(6, 12)).unwrap(),
    ] {
        let (w, g) = (
            run(&mut d, |c| c.solve(&a, &b)).unwrap(),
            run(&mut t, |c| c.solve(&a, &b)).unwrap(),
        );
        assert_eq!((g.dtype(), g.shape()), (w.dtype(), w.shape()), "solve meta");
        close("solve", &cm(&g), &cm(&w), T::TOL * 10.0);
    }
    // Triangular solve, every flag combination, on the Cholesky factor.
    let l = run(&mut d, |c| c.cholesky(&a)).unwrap();
    for bits in 0..16u32 {
        let (left, lower, trans, unit) =
            (bits & 1 != 0, bits & 2 != 0, bits & 4 != 0, bits & 8 != 0);
        let b = if left {
            rect::<T>(6, 2, 13)
        } else {
            rect::<T>(2, 6, 13)
        };
        let w = run(&mut d, |c| {
            c.triangular_solve(&l, &b, left, lower, trans, unit)
        })
        .unwrap();
        let g = run(&mut t, |c| {
            c.triangular_solve(&l, &b, left, lower, trans, unit)
        })
        .unwrap();
        close(
            &format!("trsm {bits:04b}"),
            &cm(&g),
            &cm(&w),
            T::TOL * 100.0,
        );
    }
    // Eigh: values unique, vectors through A V = V diag(w).
    let (w, g) = (
        run(&mut d, |c| c.eigh(&a)).unwrap(),
        run(&mut t, |c| c.eigh(&a)).unwrap(),
    );
    same_meta("eigh", &g, &w);
    close("eigh values", &cm(&g[0]), &cm(&w[0]), T::TOL * 10.0);
    let (av, v, vals) = (cm(&a), cm(&g[1]), cm(&g[0]));
    let lhs = matmul(&av, &v, 6, 6, 6);
    let rhs: Vec<Complex64> = (0..36).map(|i| v[i] * vals[i / 6]).collect();
    close("eigh A V = V w", &lhs, &rhs, T::TOL * 100.0);
    let (w, g) = (
        run(&mut d, |c| c.eigh_values(&a)).unwrap(),
        run(&mut t, |c| c.eigh_values(&a)).unwrap(),
    );
    close("eigh_values", &cm(&g), &cm(&w), T::TOL * 10.0);
    // SVD and QR on tall and wide matrices: values unique, factors through reconstruction.
    for (m, n) in [(7, 4), (4, 7)] {
        let x = rect::<T>(m, n, 21);
        let k = m.min(n);
        let (w, g) = (
            run(&mut d, |c| c.svd(&x)).unwrap(),
            run(&mut t, |c| c.svd(&x)).unwrap(),
        );
        same_meta("svd", &g, &w);
        close("svd S", &cm(&g[1]), &cm(&w[1]), T::TOL * 10.0);
        let (u, s, vh) = (cm(&g[0]), cm(&g[1]), cm(&g[2]));
        let us: Vec<Complex64> = (0..m * k).map(|i| u[i] * s[i / m]).collect();
        close(
            "svd U S Vh = A",
            &matmul(&us, &vh, m, k, n),
            &cm(&x),
            T::TOL * 100.0,
        );
        let (w, g) = (
            run(&mut d, |c| c.svd_values(&x)).unwrap(),
            run(&mut t, |c| c.svd_values(&x)).unwrap(),
        );
        close("svd_values", &cm(&g), &cm(&w), T::TOL * 10.0);
        let (w, g) = (
            run(&mut d, |c| c.qr(&x)).unwrap(),
            run(&mut t, |c| c.qr(&x)).unwrap(),
        );
        same_meta("qr", &g, &w);
        close(
            "qr Q R = A",
            &matmul(&cm(&g[0]), &cm(&g[1]), m, k, n),
            &cm(&x),
            T::TOL * 100.0,
        );
        let r = cm(&g[1]);
        for j in 0..n {
            for i in (j + 1)..k {
                assert!(r[i + j * k].norm() == 0.0, "R upper triangular");
            }
        }
    }
}

#[test]
fn linalg_kernels_match_the_default_backend_1t() {
    EXECUTED.store(0, Ordering::Relaxed);
    sweep::<f64>(1);
    sweep::<Complex64>(1);
    sweep::<f32>(1);
    sweep::<Complex32>(1);
    assert!(
        EXECUTED.load(Ordering::Relaxed) >= 4 * 30,
        "tprims ran {} calls",
        EXECUTED.load(Ordering::Relaxed)
    );
}

#[test]
fn linalg_kernels_match_the_default_backend_4t() {
    sweep::<f64>(4);
    sweep::<Complex64>(4);
}

#[test]
fn strided_reads_and_declines_fall_back_to_the_default_result() {
    let (mut d, mut t) = (backend(2, false), backend(2, true));
    // A transposed view of a Hermitian matrix through the read path.
    let a = hpd::<f64>(5);
    let data = a.as_slice::<f64>().unwrap().to_vec();
    let view = || {
        TensorRead::from_view(TensorView::F64(
            TypedTensorView::from_slice([5, 5], [5, 1], 0, &data).unwrap(),
        ))
    };
    let (w, g) = (
        run(&mut d, |c| c.cholesky_read(view())).unwrap(),
        run(&mut t, |c| c.cholesky_read(view())).unwrap(),
    );
    close("cholesky_read transposed", &cm(&g), &cm(&w), 1e-10);
    // Not positive definite: tprims declines and the built-in error comes back.
    let bad = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 2.0, 1.0]).unwrap();
    let (w, g) = (
        run(&mut d, |c| c.cholesky(&bad)),
        run(&mut t, |c| c.cholesky(&bad)),
    );
    assert_eq!(
        format!("{:?}", w.map(|x| x.shape().to_vec())),
        format!("{:?}", g.map(|x| x.shape().to_vec()))
    );
    // Batched input: declined, built-in result.
    let batch = Tensor::from_vec_col_major(
        vec![2, 2, 3],
        (0..12)
            .map(|i| if i % 4 == 0 || i % 4 == 3 { 3.0 } else { 1.0 })
            .collect::<Vec<f64>>(),
    )
    .unwrap();
    let before = DECLINED.load(Ordering::Relaxed);
    let (w, g) = (
        run(&mut d, |c| c.cholesky(&batch)).unwrap(),
        run(&mut t, |c| c.cholesky(&batch)).unwrap(),
    );
    assert_eq!(g.as_slice::<f64>().unwrap(), w.as_slice::<f64>().unwrap());
    assert!(
        DECLINED.load(Ordering::Relaxed) > before,
        "batched input declined"
    );
}
