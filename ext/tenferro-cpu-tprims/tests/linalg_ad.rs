//! Linalg AD through the tprims kernels: gradients of Cholesky-, eigh-, SVD-,
//! QR- and solve-based losses equal the default backend's.
use std::sync::Arc;

use std::sync::atomic::{AtomicUsize, Ordering};
use tenferro_ad::{AdContext, EagerRuntime, EagerTensor, Tensor};
use tenferro_cpu::{CpuBackend, CpuBackendKind, CpuProviderBundle};
use tenferro_cpu_tprims::TprimsProvider;

use tenferro_cpu::CpuExecutionContext;
use tenferro_linalg::cpu_kernels::{install_linalg_kernels, CpuLinalgKernels, CpuLinalgOutcome};
use tenferro_linalg::EagerSessionLinalgExt;
use tenferro_tensor::TensorView;

static CHOLESKY: AtomicUsize = AtomicUsize::new(0);

/// tprims kernels, counting Cholesky calls that executed in tprims.
#[derive(Debug)]
struct Counted;

impl CpuLinalgKernels for Counted {
    fn cholesky(
        &self,
        c: &CpuExecutionContext<'_>,
        i: TensorView<'_>,
    ) -> tenferro_tensor::Result<CpuLinalgOutcome<Tensor>> {
        let r = TprimsProvider::new().cholesky(c, i);
        if matches!(r, Ok(CpuLinalgOutcome::Executed(_))) {
            CHOLESKY.fetch_add(1, Ordering::Relaxed);
        }
        r
    }
    fn eigh(
        &self,
        c: &CpuExecutionContext<'_>,
        i: TensorView<'_>,
    ) -> tenferro_tensor::Result<CpuLinalgOutcome<Vec<Tensor>>> {
        TprimsProvider::new().eigh(c, i)
    }
    fn svd(
        &self,
        c: &CpuExecutionContext<'_>,
        i: TensorView<'_>,
    ) -> tenferro_tensor::Result<CpuLinalgOutcome<Vec<Tensor>>> {
        TprimsProvider::new().svd(c, i)
    }
    fn qr(
        &self,
        c: &CpuExecutionContext<'_>,
        i: TensorView<'_>,
    ) -> tenferro_tensor::Result<CpuLinalgOutcome<Vec<Tensor>>> {
        TprimsProvider::new().qr(c, i)
    }
    fn solve(
        &self,
        c: &CpuExecutionContext<'_>,
        a: TensorView<'_>,
        b: TensorView<'_>,
    ) -> tenferro_tensor::Result<CpuLinalgOutcome<Tensor>> {
        TprimsProvider::new().solve(c, a, b)
    }
    fn triangular_solve(
        &self,
        c: &CpuExecutionContext<'_>,
        a: TensorView<'_>,
        b: TensorView<'_>,
        o: tenferro_linalg::cpu_kernels::TriangularSolveOptions,
    ) -> tenferro_tensor::Result<CpuLinalgOutcome<Tensor>> {
        TprimsProvider::new().triangular_solve(c, a, b, o)
    }
}

fn runtime(tprims: bool) -> Arc<EagerRuntime> {
    let mut backend = CpuBackend::with_threads(2).unwrap();
    if tprims {
        let builder = CpuProviderBundle::builder(CpuBackendKind::default_compiled());
        let bundle = install_linalg_kernels(builder, Arc::new(Counted))
            .build()
            .unwrap();
        backend = backend.with_provider_bundle(bundle).unwrap();
    }
    let ad = AdContext::builder()
        .with_semantic_extension_rules(tenferro_linalg::semantic_ad_rules().unwrap())
        .unwrap()
        .build()
        .unwrap();
    EagerRuntime::with_cpu_backend_and_ad_context(backend, &ad).unwrap()
}

/// `sum(w .* x .* x)`: sign- and phase-invariant for gauge-dependent factors.
fn weighted(x: &EagerTensor, seed: f64) -> EagerTensor {
    let n: usize = x.shape().iter().product();
    let w: Vec<f64> = (0..n).map(|i| ((i as f64 + seed) * 0.37).sin()).collect();
    let w = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(x.shape().to_vec(), w).unwrap(),
        x.runtime().clone(),
    )
    .unwrap();
    let axes: Vec<usize> = (0..x.shape().len()).collect();
    x.runtime()
        .with_eager_session(|s| {
            let sq = s.mul(x, x)?;
            let ws = s.mul(&sq, &w)?;
            s.reduce_sum(&ws, Some(&axes))
        })
        .unwrap()
        .unwrap()
}

fn spd() -> Vec<f64> {
    vec![
        4.0, 1.0, 0.5, 0.2, 1.0, 3.0, 0.3, 0.1, 0.5, 0.3, 2.5, 0.4, 0.2, 0.1, 0.4, 2.0,
    ]
}

fn grad(
    tprims: bool,
    loss: fn(&EagerTensor) -> EagerTensor,
    data: Vec<f64>,
    shape: [usize; 2],
) -> Vec<f64> {
    let a = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(shape.to_vec(), data).unwrap(),
        runtime(tprims),
    )
    .unwrap();
    loss(&a).backward().unwrap();
    a.grad()
        .unwrap()
        .unwrap()
        .to_tensor()
        .unwrap()
        .as_slice::<f64>()
        .unwrap()
        .to_vec()
}

fn check(what: &str, loss: fn(&EagerTensor) -> EagerTensor, data: Vec<f64>, shape: [usize; 2]) {
    let (want, got) = (
        grad(false, loss, data.clone(), shape),
        grad(true, loss, data, shape),
    );
    let err = want
        .iter()
        .zip(&got)
        .map(|(w, g)| (w - g).abs())
        .fold(0.0, f64::max);
    assert!(
        err < 1e-9,
        "{what}: max grad diff {err:e}\n want {want:?}\n got {got:?}"
    );
}

#[test]
fn cholesky_gradient() {
    let before = CHOLESKY.load(Ordering::Relaxed);
    check(
        "cholesky",
        |a| {
            weighted(
                &a.runtime()
                    .with_eager_session(|s| s.cholesky(a))
                    .unwrap()
                    .unwrap(),
                1.0,
            )
        },
        spd(),
        [4, 4],
    );
    assert!(
        CHOLESKY.load(Ordering::Relaxed) > before,
        "the eager runtime reached the tprims kernels"
    );
}

#[test]
fn eigh_gradient() {
    check(
        "eigh",
        |a| {
            let (w, v) = a
                .runtime()
                .with_eager_session(|s| s.eigh(a))
                .unwrap()
                .unwrap();
            let (lw, lv) = (weighted(&w, 2.0), weighted(&v, 3.0));
            a.runtime()
                .with_eager_session(|s| s.add(&lw, &lv))
                .unwrap()
                .unwrap()
        },
        spd(),
        [4, 4],
    );
}

#[test]
fn svd_gradient() {
    check(
        "svd",
        |a| {
            let (u, s, vt) = a
                .runtime()
                .with_eager_session(|x| x.svd(a))
                .unwrap()
                .unwrap();
            let (lu, ls, lv) = (weighted(&u, 4.0), weighted(&s, 5.0), weighted(&vt, 6.0));
            a.runtime()
                .with_eager_session(|x| {
                    let t = x.add(&lu, &ls)?;
                    x.add(&t, &lv)
                })
                .unwrap()
                .unwrap()
        },
        vec![
            1.2, -0.3, 0.7, 0.4, 1.5, -0.8, 0.2, 0.9, -1.1, 0.6, 0.3, 1.4,
        ],
        [4, 3],
    );
}

#[test]
fn qr_and_solve_gradients() {
    check(
        "qr",
        |a| {
            let (q, r) = a
                .runtime()
                .with_eager_session(|s| s.qr(a))
                .unwrap()
                .unwrap();
            let (lq, lr) = (weighted(&q, 7.0), weighted(&r, 8.0));
            a.runtime()
                .with_eager_session(|s| s.add(&lq, &lr))
                .unwrap()
                .unwrap()
        },
        vec![
            1.2, -0.3, 0.7, 0.4, 1.5, -0.8, 0.2, 0.9, -1.1, 0.6, 0.3, 1.4,
        ],
        [4, 3],
    );
    check(
        "solve",
        |a| {
            let b = EagerTensor::from_tensor_in(
                Tensor::from_vec_col_major(
                    vec![4, 2],
                    vec![1.0, 2.0, -1.0, 0.5, 0.3, -0.7, 1.1, 0.2],
                )
                .unwrap(),
                a.runtime().clone(),
            )
            .unwrap();
            weighted(
                &a.runtime()
                    .with_eager_session(|s| s.solve(a, &b))
                    .unwrap()
                    .unwrap(),
                9.0,
            )
        },
        spd(),
        [4, 4],
    );
}
