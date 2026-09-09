use super::*;

#[test]
fn execution_cache_debug_identifies_both_owners_and_exposes_the_store() {
    let mut caller = FftPlanCache::default();
    let mut caller_cache = FftExecutionCache::caller_owned(&mut caller);
    assert!(format!("{caller_cache:?}").contains("CallerOwned"));
    assert_eq!(
        caller_cache
            .store_mut()
            .stats(tenferro_runtime::ExtensionCacheSelector::All)
            .entries,
        0
    );

    let mut runtime = ExtensionCacheStore::default();
    let mut runtime_cache = FftExecutionCache::runtime_owned(&mut runtime);
    assert!(format!("{runtime_cache:?}").contains("RuntimeOwned"));
    assert_eq!(
        runtime_cache
            .store_mut()
            .stats(tenferro_runtime::ExtensionCacheSelector::All)
            .entries,
        0
    );
}

// Exercise the trait defaults with the existing backend's real structural
// operations; only the final FFT execution delegates to its CPU session.
impl FftBackend for tenferro_cpu::CpuBackend {
    fn execute_fft(
        &mut self,
        input: &Tensor,
        spec: &FftPlanSpec,
        cache: FftExecutionCache<'_>,
    ) -> tenferro_tensor::Result<Tensor> {
        use tenferro_tensor::BackendSessionHost;
        self.with_backend_session(|session| {
            tenferro_cpu::with_cpu_exec_session(session, |cpu| cpu.execute_fft(input, spec, cache))
                .unwrap()
        })
    }
}

#[test]
fn default_read_execution_preserves_owned_input_and_canonicalizes_a_view() {
    use crate::{FftNorm, FftOperation};
    use num_complex::Complex64;
    use tenferro_tensor::{DType, TensorView, TypedTensorView};
    let data = [1., 2., 3., 4.].map(|x| Complex64::new(x, 0.));
    let input = Tensor::from_vec_col_major([2, 2], data.to_vec()).unwrap();
    let view = TensorRead::from_view(TensorView::C64(
        TypedTensorView::from_col_major(&[2, 2], &data)
            .unwrap()
            .transpose_view([1, 0])
            .unwrap(),
    ));
    let spec = crate::concrete_fft_spec(
        "fft",
        FftOperation::C2cForward,
        DType::C64,
        &[2, 2],
        None,
        0,
        FftNorm::Backward,
    )
    .unwrap();
    let mut backend = tenferro_cpu::CpuBackend::with_threads(1).unwrap();
    let mut cache = FftPlanCache::default();
    for (read, expected) in [
        (TensorRead::from_tensor(&input), [3., -1., 7., -1.]),
        (view, [4., -2., 6., -2.]),
    ] {
        backend.validate_fft_read_input("fft", &read).unwrap();
        let output = backend
            .execute_fft_read(read, &spec, FftExecutionCache::caller_owned(&mut cache))
            .unwrap();
        assert_eq!(
            output.as_slice::<Complex64>().unwrap(),
            expected.map(|x| Complex64::new(x, 0.))
        );
    }
    assert_eq!(input.as_slice::<Complex64>().unwrap(), data);
}
