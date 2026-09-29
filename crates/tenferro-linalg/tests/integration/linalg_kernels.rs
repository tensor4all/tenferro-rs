//! Injected CPU linalg kernels (`tenferro_linalg::cpu_kernels`): executed
//! kernels replace the built-in ones, declined ones fall through, errors
//! propagate.
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tenferro_cpu::provider::CpuProviderUnsupported;
use tenferro_cpu::{
    with_cpu_exec_session, CpuBackend, CpuBackendKind, CpuExecutionContext, CpuProviderBundle,
};
use tenferro_linalg::backend::LinalgBackend;
use tenferro_linalg::cpu_kernels::{
    install_linalg_kernels, CpuLinalgKernels, CpuLinalgOutcome, TriangularSolveOptions,
};
use tenferro_tensor::{BackendSessionHost, Error, Tensor, TensorRead, TensorView};

#[derive(Debug, Default)]
struct Mock {
    cholesky: AtomicUsize,
    declined: AtomicUsize,
    solve_flags: std::sync::Mutex<Option<TriangularSolveOptions>>,
}

impl CpuLinalgKernels for Mock {
    fn cholesky(
        &self,
        _: &CpuExecutionContext<'_>,
        input: TensorView<'_>,
    ) -> tenferro_tensor::Result<CpuLinalgOutcome<Tensor>> {
        self.cholesky.fetch_add(1, Ordering::Relaxed);
        let n: usize = input.shape().iter().product();
        Ok(CpuLinalgOutcome::Executed(Tensor::from_vec_col_major(
            input.shape().to_vec(),
            vec![42.0_f64; n],
        )?))
    }
    fn qr(
        &self,
        _: &CpuExecutionContext<'_>,
        _: TensorView<'_>,
    ) -> tenferro_tensor::Result<CpuLinalgOutcome<Vec<Tensor>>> {
        self.declined.fetch_add(1, Ordering::Relaxed);
        Ok(CpuLinalgOutcome::Unsupported(
            CpuProviderUnsupported::RuntimeUnavailable,
        ))
    }
    fn triangular_solve(
        &self,
        _: &CpuExecutionContext<'_>,
        _: TensorView<'_>,
        _: TensorView<'_>,
        options: TriangularSolveOptions,
    ) -> tenferro_tensor::Result<CpuLinalgOutcome<Tensor>> {
        *self.solve_flags.lock().unwrap() = Some(options);
        Ok(CpuLinalgOutcome::Unsupported(
            CpuProviderUnsupported::RuntimeUnavailable,
        ))
    }
    fn eig(
        &self,
        _: &CpuExecutionContext<'_>,
        _: TensorView<'_>,
    ) -> tenferro_tensor::Result<CpuLinalgOutcome<Vec<Tensor>>> {
        Err(Error::backend_failure("eig", "mock kernel failure"))
    }
}

fn backend_with(mock: Arc<Mock>) -> CpuBackend {
    let bundle = install_linalg_kernels(
        CpuProviderBundle::builder(CpuBackendKind::default_compiled()),
        mock,
    )
    .build()
    .unwrap();
    CpuBackend::with_threads(2)
        .unwrap()
        .with_provider_bundle(bundle)
        .unwrap()
}

fn spd() -> Tensor {
    Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 1.0, 1.0, 3.0]).unwrap()
}

fn with_session<R: Send>(
    backend: &mut CpuBackend,
    f: impl FnOnce(&mut tenferro_cpu::CpuExecSession<'_>) -> R + Send,
) -> R {
    backend
        .with_backend_session(|s| with_cpu_exec_session(s, f).expect("a CPU backend session"))
        .unwrap()
}

#[test]
fn an_executed_kernel_replaces_the_builtin_one() {
    let mock = Arc::new(Mock::default());
    let mut backend = backend_with(Arc::clone(&mock));
    let a = spd();
    let owned = with_session(&mut backend, |cpu| cpu.cholesky(&a)).unwrap();
    let read = with_session(&mut backend, |cpu| {
        cpu.cholesky_read(TensorRead::from_tensor(&a))
    })
    .unwrap();
    for out in [owned, read] {
        assert_eq!(out.as_slice::<f64>().unwrap(), &[42.0; 4]);
    }
    assert_eq!(mock.cholesky.load(Ordering::Relaxed), 2);
}

#[test]
fn a_declined_kernel_falls_through_to_the_builtin_result() {
    let mock = Arc::new(Mock::default());
    let mut backend = backend_with(Arc::clone(&mock));
    let mut plain = CpuBackend::with_threads(2).unwrap();
    let a = spd();
    let got = with_session(&mut backend, |cpu| cpu.qr(&a)).unwrap();
    let want = with_session(&mut plain, |cpu| cpu.qr(&a)).unwrap();
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(&want) {
        assert_eq!(g.as_slice::<f64>().unwrap(), w.as_slice::<f64>().unwrap());
    }
    assert_eq!(mock.declined.load(Ordering::Relaxed), 1);
    // Methods the mock does not override decline by default.
    let got = with_session(&mut backend, |cpu| cpu.eigh(&a)).unwrap();
    let want = with_session(&mut plain, |cpu| cpu.eigh(&a)).unwrap();
    assert_eq!(
        got[0].as_slice::<f64>().unwrap(),
        want[0].as_slice::<f64>().unwrap()
    );
}

#[test]
fn triangular_solve_flags_reach_the_kernel() {
    let mock = Arc::new(Mock::default());
    let mut backend = backend_with(Arc::clone(&mock));
    let (a, b) = (
        spd(),
        Tensor::from_vec_col_major(vec![2, 1], vec![1.0_f64, 2.0]).unwrap(),
    );
    with_session(&mut backend, |cpu| {
        cpu.triangular_solve(&a, &b, true, false, true, false)
    })
    .unwrap();
    assert_eq!(
        *mock.solve_flags.lock().unwrap(),
        Some(TriangularSolveOptions {
            left_side: true,
            lower: false,
            transpose_a: true,
            unit_diagonal: false
        })
    );
}

#[test]
fn kernel_errors_propagate() {
    let mut backend = backend_with(Arc::new(Mock::default()));
    let err = with_session(&mut backend, |cpu| cpu.eig(&spd())).unwrap_err();
    assert!(err.to_string().contains("mock kernel failure"), "{err}");
}

#[test]
fn a_backend_without_kernels_is_unchanged() {
    let mut plain = CpuBackend::with_threads(1).unwrap();
    let l = with_session(&mut plain, |cpu| cpu.cholesky(&spd())).unwrap();
    assert!((l.as_slice::<f64>().unwrap()[0] - 2.0).abs() < 1e-12);
}

/// Declines every call and counts it.
#[derive(Debug, Default)]
struct CountingDecline(AtomicUsize);

fn decline<T>(c: &AtomicUsize) -> tenferro_tensor::Result<CpuLinalgOutcome<T>> {
    c.fetch_add(1, Ordering::Relaxed);
    Ok(CpuLinalgOutcome::Unsupported(
        CpuProviderUnsupported::RuntimeUnavailable,
    ))
}

impl CpuLinalgKernels for CountingDecline {
    fn cholesky(
        &self,
        _: &CpuExecutionContext<'_>,
        _: TensorView<'_>,
    ) -> tenferro_tensor::Result<CpuLinalgOutcome<Tensor>> {
        decline(&self.0)
    }
    fn triangular_solve(
        &self,
        _: &CpuExecutionContext<'_>,
        _: TensorView<'_>,
        _: TensorView<'_>,
        _: TriangularSolveOptions,
    ) -> tenferro_tensor::Result<CpuLinalgOutcome<Tensor>> {
        decline(&self.0)
    }
    fn lu(
        &self,
        _: &CpuExecutionContext<'_>,
        _: TensorView<'_>,
    ) -> tenferro_tensor::Result<CpuLinalgOutcome<Vec<Tensor>>> {
        decline(&self.0)
    }
    fn full_piv_lu(
        &self,
        _: &CpuExecutionContext<'_>,
        _: TensorView<'_>,
    ) -> tenferro_tensor::Result<CpuLinalgOutcome<Vec<Tensor>>> {
        decline(&self.0)
    }
    fn solve(
        &self,
        _: &CpuExecutionContext<'_>,
        _: TensorView<'_>,
        _: TensorView<'_>,
    ) -> tenferro_tensor::Result<CpuLinalgOutcome<Tensor>> {
        decline(&self.0)
    }
    fn svd(
        &self,
        _: &CpuExecutionContext<'_>,
        _: TensorView<'_>,
    ) -> tenferro_tensor::Result<CpuLinalgOutcome<Vec<Tensor>>> {
        decline(&self.0)
    }
    fn svd_full(
        &self,
        _: &CpuExecutionContext<'_>,
        _: TensorView<'_>,
    ) -> tenferro_tensor::Result<CpuLinalgOutcome<Vec<Tensor>>> {
        decline(&self.0)
    }
    fn svd_values(
        &self,
        _: &CpuExecutionContext<'_>,
        _: TensorView<'_>,
    ) -> tenferro_tensor::Result<CpuLinalgOutcome<Tensor>> {
        decline(&self.0)
    }
    fn qr(
        &self,
        _: &CpuExecutionContext<'_>,
        _: TensorView<'_>,
    ) -> tenferro_tensor::Result<CpuLinalgOutcome<Vec<Tensor>>> {
        decline(&self.0)
    }
    fn rank_revealing_qr(
        &self,
        _: &CpuExecutionContext<'_>,
        _: TensorView<'_>,
        _: tenferro_linalg::RankRevealingQrOptions,
    ) -> tenferro_tensor::Result<CpuLinalgOutcome<Vec<Tensor>>> {
        decline(&self.0)
    }
    fn eigh(
        &self,
        _: &CpuExecutionContext<'_>,
        _: TensorView<'_>,
    ) -> tenferro_tensor::Result<CpuLinalgOutcome<Vec<Tensor>>> {
        decline(&self.0)
    }
    fn eigh_values(
        &self,
        _: &CpuExecutionContext<'_>,
        _: TensorView<'_>,
    ) -> tenferro_tensor::Result<CpuLinalgOutcome<Tensor>> {
        decline(&self.0)
    }
    fn eig(
        &self,
        _: &CpuExecutionContext<'_>,
        _: TensorView<'_>,
    ) -> tenferro_tensor::Result<CpuLinalgOutcome<Vec<Tensor>>> {
        decline(&self.0)
    }
    fn eig_values(
        &self,
        _: &CpuExecutionContext<'_>,
        _: TensorView<'_>,
    ) -> tenferro_tensor::Result<CpuLinalgOutcome<Tensor>> {
        decline(&self.0)
    }
}

/// Declines through the trait's default methods.
#[derive(Debug)]
struct Defaults;
impl CpuLinalgKernels for Defaults {}

fn r(t: &Tensor) -> TensorRead<'_> {
    TensorRead::from_tensor(t)
}

/// Every hooked method, owned and read, on small matrices; returns the output
/// count so results are compared across backends.
fn call_every_hook(cpu: &mut tenferro_cpu::CpuExecSession<'_>) -> Vec<usize> {
    let a = spd();
    let b = Tensor::from_vec_col_major(vec![2, 1], vec![1.0_f64, 2.0]).unwrap();
    let o = tenferro_linalg::RankRevealingQrOptions::default();
    vec![
        cpu.cholesky(&a).map(|_| 1).unwrap(),
        cpu.cholesky_read(r(&a)).map(|_| 1).unwrap(),
        cpu.triangular_solve(&a, &b, true, true, false, false)
            .map(|_| 1)
            .unwrap(),
        cpu.triangular_solve_read(r(&a), r(&b), true, true, false, false)
            .map(|_| 1)
            .unwrap(),
        cpu.lu(&a).unwrap().len(),
        cpu.lu_read(r(&a)).unwrap().len(),
        cpu.full_piv_lu(&a).unwrap().len(),
        cpu.full_piv_lu_read(r(&a)).unwrap().len(),
        cpu.solve(&a, &b).map(|_| 1).unwrap(),
        cpu.solve_read(r(&a), r(&b)).map(|_| 1).unwrap(),
        cpu.svd(&a).unwrap().len(),
        cpu.svd_read(r(&a)).unwrap().len(),
        cpu.svd_full(&a).unwrap().len(),
        cpu.svd_full_read(r(&a)).unwrap().len(),
        cpu.svd_values(&a).map(|_| 1).unwrap(),
        cpu.svd_values_read(r(&a)).map(|_| 1).unwrap(),
        cpu.qr(&a).unwrap().len(),
        cpu.qr_read(r(&a)).unwrap().len(),
        cpu.rank_revealing_qr(&a, o).unwrap().len(),
        cpu.rank_revealing_qr_read(r(&a), o).unwrap().len(),
        cpu.eigh(&a).unwrap().len(),
        cpu.eigh_read(r(&a)).unwrap().len(),
        cpu.eigh_values(&a).map(|_| 1).unwrap(),
        cpu.eigh_values_read(r(&a)).map(|_| 1).unwrap(),
        cpu.eig(&a).unwrap().len(),
        cpu.eig_read(r(&a)).unwrap().len(),
        cpu.eig_values(&a).map(|_| 1).unwrap(),
        cpu.eig_values_read(r(&a)).map(|_| 1).unwrap(),
    ]
}

#[test]
fn every_hooked_method_consults_the_kernels_and_falls_back() {
    let counting = Arc::new(CountingDecline::default());
    let bundle = install_linalg_kernels(
        CpuProviderBundle::builder(CpuBackendKind::default_compiled()),
        counting.clone(),
    )
    .build()
    .unwrap();
    let mut with_kernels = CpuBackend::with_threads(1)
        .unwrap()
        .with_provider_bundle(bundle)
        .unwrap();
    let mut plain = CpuBackend::with_threads(1).unwrap();
    let got = with_session(&mut with_kernels, call_every_hook);
    let want = with_session(&mut plain, call_every_hook);
    assert_eq!(got, want);
    assert_eq!(
        counting.0.load(Ordering::Relaxed),
        28,
        "each of the 28 hooks asks once"
    );

    let bundle = install_linalg_kernels(
        CpuProviderBundle::builder(CpuBackendKind::default_compiled()),
        Arc::new(Defaults),
    )
    .build()
    .unwrap();
    let mut defaults = CpuBackend::with_threads(1)
        .unwrap()
        .with_provider_bundle(bundle)
        .unwrap();
    assert_eq!(with_session(&mut defaults, call_every_hook), want);
}
