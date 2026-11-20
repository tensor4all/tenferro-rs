#![cfg(feature = "fftw")]

use std::num::NonZeroUsize;
use std::path::Path;

use num_complex::{Complex32, Complex64};
use tenferro_cpu::CpuEngine;
use tenferro_fft::{FftExecutor, FftNorm, FftOptions, Fftw, FftwError, FftwPlanning};
use tenferro_tensor::Tensor;

#[test]
fn missing_library_has_a_typed_error_with_its_name() {
    let directory = Path::new("/tenferro-test-missing-fftw");
    let error = Fftw::load_from(directory, FftwPlanning::Estimate, NonZeroUsize::MIN).unwrap_err();
    let FftwError::Load { library, .. } = error else {
        panic!("expected a library load error, got {error}");
    };
    assert_eq!(library.parent(), Some(directory));
    assert!(library
        .file_name()
        .unwrap()
        .to_string_lossy()
        .contains("fftw3"));
}

fn load(planning: FftwPlanning, threads: NonZeroUsize) -> Option<Fftw> {
    let directory = std::env::var_os("TENFERRO_TEST_FFTW_DIR");
    let result = match directory.as_deref() {
        Some(directory) => Fftw::load_from(Path::new(directory), planning, threads),
        None => Fftw::load(planning, threads),
    };
    match result {
        Ok(fftw) => Some(fftw),
        Err(error)
            if directory.is_none()
                && std::env::var_os("TENFERRO_TEST_REQUIRE_FFTW").is_none()
                && matches!(error, FftwError::Load { .. }) =>
        {
            eprintln!("FFTW integration requires a runtime: {error}");
            None
        }
        Err(error) => panic!("{error}"),
    }
}

macro_rules! parity {
    ($name:ident, $real:ty, $complex:ty, $tol:expr) => {
        #[test]
        fn $name() {
            let Some(provider) = load(FftwPlanning::Estimate, NonZeroUsize::MIN) else {
                return;
            };
            let mut fftw = CpuEngine::serial();
            provider.install(&mut fftw);
            let mut rustfft = CpuEngine::serial();
            let mut executor = FftExecutor::default();
            for n in [1, 4, 5] {
                let real: Vec<$real> = (0..2 * n)
                    .map(|i| ((i * 3 % 7) as $real - 2.0) / 5.0)
                    .collect();
                let x = Tensor::from_vec_col_major([2, n], real.clone()).unwrap();
                let z = Tensor::from_vec_col_major(
                    [2, n],
                    real.iter()
                        .enumerate()
                        .map(|(i, v)| <$complex>::new(*v, i as $real / 8.0))
                        .collect::<Vec<_>>(),
                )
                .unwrap();
                for norm in [FftNorm::Backward, FftNorm::Forward, FftNorm::Ortho] {
                    let options = FftOptions::default().axis(1).n(Some(n)).norm(norm);
                    for (op, input) in [
                        (
                            FftExecutor::fft_with
                                as fn(
                                    &mut FftExecutor,
                                    &Tensor,
                                    &FftOptions,
                                    &mut dyn tenferro_tensor::Engine,
                                ) -> _,
                            &z,
                        ),
                        (FftExecutor::ifft_with, &z),
                        (FftExecutor::fft_with, &x),
                        (FftExecutor::rfft_with, &x),
                    ] {
                        let a = op(&mut executor, input, &options, &mut fftw).unwrap();
                        let b = op(&mut executor, input, &options, &mut rustfft).unwrap();
                        assert_eq!(a.shape(), b.shape());
                        for (a, b) in a
                            .as_slice::<$complex>()
                            .unwrap()
                            .iter()
                            .zip(b.as_slice::<$complex>().unwrap())
                        {
                            assert!((*a - *b).norm() < $tol, "{a} != {b}");
                        }
                    }
                    let spectrum = executor.rfft_with(&x, &options, &mut fftw).unwrap();
                    let a = executor.irfft_with(&spectrum, &options, &mut fftw).unwrap();
                    let b = executor
                        .irfft_with(&spectrum, &options, &mut rustfft)
                        .unwrap();
                    for ((a, b), expected) in a
                        .as_slice::<$real>()
                        .unwrap()
                        .iter()
                        .zip(b.as_slice::<$real>().unwrap())
                        .zip(&real)
                    {
                        assert!((a - b).abs() < $tol && (a - expected).abs() < $tol);
                    }
                }
            }
        }
    };
}
parity!(double_precision, f64, Complex64, 1e-12);
parity!(single_precision, f32, Complex32, 2e-6);

#[test]
fn measure_and_threaded_plans_are_reused_and_do_not_modify_input() {
    let Some(provider) = load(FftwPlanning::Measure, NonZeroUsize::new(2).unwrap()) else {
        return;
    };
    let mut engine = CpuEngine::serial();
    provider.install(&mut engine);
    let mut executor = FftExecutor::default();
    let x = Tensor::from_vec_col_major([4], vec![1.0_f64, 0.0, 0.0, 0.0]).unwrap();
    for hits in 0..2 {
        let y = executor
            .rfft_with(&x, &FftOptions::default(), &mut engine)
            .unwrap();
        assert_eq!(
            y.as_slice::<Complex64>().unwrap(),
            &[Complex64::new(1.0, 0.0); 3]
        );
        assert_eq!(x.as_slice::<f64>().unwrap(), &[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(executor.cache_stats().entries, 1);
        assert_eq!(executor.cache_stats().hits, hits);
    }
}

fn fftw_plan_count(engine: &mut CpuEngine) -> usize {
    use tenferro_runtime::{ExecutionEngine, ExtensionCacheSelector};

    let cache = engine.take_extension_caches();
    let entries = cache
        .stats(ExtensionCacheSelector::Cache {
            family_id: tenferro_fft::FFT_EXTENSION_FAMILY_ID,
            cache_name: "fftw-plans",
        })
        .entries;
    engine.restore_extension_caches(cache);
    entries
}

#[test]
fn traced_fft_selects_the_installed_provider() {
    use tenferro_fft::TracedFftOps;
    use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TraceContext};

    let Some(provider) = load(FftwPlanning::Estimate, NonZeroUsize::MIN) else {
        return;
    };
    let trace = TraceContext::new();
    let x = trace
        .from_vec_col_major([2], vec![Complex64::new(1.0, 0.0); 2])
        .unwrap();
    let program = GraphCompiler::new().compile(&x.fft().unwrap()).unwrap();
    let mut engine = CpuEngine::serial();
    provider.install(&mut engine);
    let mut builder = Runtime::builder();
    builder
        .register_engine(tenferro_cpu::runtime_engine_registration(&engine).unwrap())
        .unwrap();
    builder
        .install_extension_module(
            tenferro_fft::extension_module::<CpuEngine>(tenferro_cpu::runtime_engine_id().unwrap())
                .unwrap(),
        )
        .unwrap();
    let runtime = builder.build().unwrap();
    let prepared = runtime.prepare(&program, &[], &[engine.target()]).unwrap();
    let y = runtime
        .run(&mut [&mut engine], &prepared, &[])
        .unwrap()
        .pop()
        .unwrap()
        .into_tensor(&mut engine)
        .unwrap();
    assert_eq!(
        y.as_slice::<Complex64>().unwrap(),
        &[Complex64::new(2.0, 0.0), Complex64::new(0.0, 0.0)]
    );
    assert_eq!(fftw_plan_count(&mut engine), 1);
}

#[cfg(feature = "autodiff")]
#[test]
fn eager_in_place_fft_selects_the_installed_provider() {
    use tenferro_ad::EagerRuntime;
    use tenferro_fft::EagerFftOps;

    let Some(provider) = load(FftwPlanning::Estimate, NonZeroUsize::MIN) else {
        return;
    };
    let mut engine = CpuEngine::serial();
    provider.install(&mut engine);
    let runtime = EagerRuntime::new().unwrap();
    let x = runtime
        .constant_from(Tensor::from_vec_col_major([2], vec![Complex64::new(1.0, 0.0); 2]).unwrap())
        .unwrap();
    let y = runtime
        .with_eager_session(&mut engine, |session| {
            Ok::<_, tenferro_ad::Error>(session.fft_in_place(x, &FftOptions::default()))
        })
        .unwrap()
        .unwrap();
    assert_eq!(
        y.value().unwrap().as_slice::<Complex64>().unwrap(),
        &[Complex64::new(2.0, 0.0), Complex64::new(0.0, 0.0)]
    );
    assert_eq!(fftw_plan_count(&mut engine), 1);
}

#[test]
fn threaded_provider_runs_many_lanes_in_parallel_jobs() {
    let Some(provider) = load(FftwPlanning::Estimate, NonZeroUsize::new(2).unwrap()) else {
        return;
    };
    let parallel = || CpuEngine::new(tenferro_cpu::Pool::builder().threads(4).build().unwrap());
    let mut fftw = parallel();
    assert!(
        fftw.num_threads() >= 2,
        "lane-parallel dispatch needs at least two engine threads"
    );
    provider.install(&mut fftw);
    let mut rustfft = parallel();
    // With FFTW installed, `fftw_plans` holds only FFTW plans.
    let mut fftw_plans = FftExecutor::default();
    let mut rustfft_plans = FftExecutor::default();
    // 4096 lanes of length 16 fill every engine thread, so they run as lane jobs
    // with a single-threaded plan; one short lane runs FFTW's threaded plan.
    for shape in [vec![16, 4096], vec![16]] {
        let len = shape.iter().product::<usize>();
        let x = Tensor::from_vec_col_major(
            shape,
            (0..len)
                .map(|i| Complex64::new((i % 7) as f64, (i % 3) as f64))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let options = FftOptions::default().axis(0);
        let a = fftw_plans.fft_with(&x, &options, &mut fftw).unwrap();
        let b = rustfft_plans.fft_with(&x, &options, &mut rustfft).unwrap();
        for (a, b) in a
            .as_slice::<Complex64>()
            .unwrap()
            .iter()
            .zip(b.as_slice::<Complex64>().unwrap())
        {
            assert!((*a - *b).norm() < 1e-12, "{a} != {b}");
        }
    }
    assert_eq!(
        fftw_plans.cache_stats().entries,
        2,
        "expected single-threaded and threaded length-16 plans"
    );
}
