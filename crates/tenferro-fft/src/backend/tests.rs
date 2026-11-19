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
    let mut runtime_cache = FftExecutionCache::engine_owned(&mut runtime);
    assert!(format!("{runtime_cache:?}").contains("EngineOwned"));
    assert!(runtime_cache.is_engine_owned());
    assert_eq!(
        runtime_cache
            .store_mut()
            .stats(tenferro_runtime::ExtensionCacheSelector::All)
            .entries,
        0
    );
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
    let mut backend = tenferro_cpu::CpuEngine::serial();
    let mut cache = FftPlanCache::default();
    for (read, expected) in [
        (TensorRead::from_tensor(&input), [3., -1., 7., -1.]),
        (view, [4., -2., 6., -2.]),
    ] {
        let output = {
            let session: &mut dyn tenferro_tensor::Engine = &mut backend;
            tenferro_cpu::with_cpu_engine(session, |cpu| {
                cpu.validate_fft_read_input("fft", &read).unwrap();
                cpu.execute_fft_read(read, &spec, FftExecutionCache::caller_owned(&mut cache))
            })
            .expect("the CPU session exposes the FFT capability")
        }
        .unwrap();
        assert_eq!(
            output.as_slice::<Complex64>().unwrap(),
            expected.map(|x| Complex64::new(x, 0.))
        );
    }
    assert_eq!(input.as_slice::<Complex64>().unwrap(), data);
}

/// What the default read dispatch handed to [`FftBackend::execute_fft`].
#[derive(Debug)]
struct RecordedFft {
    data_ptr: *const num_complex::Complex64,
    shape: Vec<usize>,
    values: Vec<num_complex::Complex64>,
    spec: FftPlanSpec,
}

/// A session that keeps every `FftBackend` default and records its `execute_fft` input.
///
/// The CPU session overrides both read hooks, so the trait defaults are only
/// observable through a session like this one. Only the methods a session must
/// define are stubbed; `to_contiguous_read` keeps the host-duplicating default.
#[derive(Debug, Default)]
struct DefaultFftSession {
    recorded: Vec<RecordedFft>,
}

macro_rules! unreachable_methods {
    ($($name:ident($($arg:ident : $argty:ty),*) -> $ret:ty;)+) => {
        $(
            fn $name(&mut self, $($arg: $argty),*) -> $ret {
                $(let _ = &$arg;)*
                panic!(concat!(stringify!($name), " is not used by the default FFT dispatch"))
            }
        )+
    };
}

mod default_session_impls {
    use super::DefaultFftSession;
    use tenferro_tensor::{
        CompareDir, DType, DotGeneralConfig, ElementwiseReadOp, Engine, GatherConfig, PadConfig,
        ScatterConfig, SessionCachedDot, SliceConfig, Tensor, TensorAnalytic, TensorBuffer,
        TensorDeviceTransfer, TensorDot, TensorElementwise, TensorFusion, TensorIndexing,
        TensorRead, TensorReduction, TensorStructural, TensorWrite,
    };

    type Result<T> = tenferro_tensor::Result<T>;

    impl TensorElementwise for DefaultFftSession {
        unreachable_methods! {
            elementwise_read_into(op: ElementwiseReadOp, inputs: &[TensorRead<'_>], out: TensorWrite<'_>) -> Result<()>;
            add_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> Result<Tensor>;
            sub_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> Result<Tensor>;
            mul_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> Result<Tensor>;
            div_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> Result<Tensor>;
            maximum_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> Result<Tensor>;
            minimum_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> Result<Tensor>;
            neg_read(input: TensorRead<'_>) -> Result<Tensor>;
            conj_read(input: TensorRead<'_>) -> Result<Tensor>;
            abs_read(input: TensorRead<'_>) -> Result<Tensor>;
            sign_read(input: TensorRead<'_>) -> Result<Tensor>;
            compare_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>, dir: &CompareDir) -> Result<Tensor>;
            select_read(pred: TensorRead<'_>, on_true: TensorRead<'_>, on_false: TensorRead<'_>) -> Result<Tensor>;
            clamp_read(input: TensorRead<'_>, lower: TensorRead<'_>, upper: TensorRead<'_>) -> Result<Tensor>;
        }
    }

    impl TensorAnalytic for DefaultFftSession {
        unreachable_methods! {
            exp_read(input: TensorRead<'_>) -> Result<Tensor>;
            log_read(input: TensorRead<'_>) -> Result<Tensor>;
            sin_read(input: TensorRead<'_>) -> Result<Tensor>;
            cos_read(input: TensorRead<'_>) -> Result<Tensor>;
            tanh_read(input: TensorRead<'_>) -> Result<Tensor>;
            sqrt_read(input: TensorRead<'_>) -> Result<Tensor>;
            rsqrt_read(input: TensorRead<'_>) -> Result<Tensor>;
            expm1_read(input: TensorRead<'_>) -> Result<Tensor>;
            log1p_read(input: TensorRead<'_>) -> Result<Tensor>;
            erf_read(input: TensorRead<'_>) -> Result<Tensor>;
            pow_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> Result<Tensor>;
        }
    }

    impl TensorStructural for DefaultFftSession {
        unreachable_methods! {
            cast(input: &Tensor, to: DType) -> Result<Tensor>;
            extract_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> Result<Tensor>;
            embed_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> Result<Tensor>;
            tril(input: &Tensor, k: i64) -> Result<Tensor>;
            triu(input: &Tensor, k: i64) -> Result<Tensor>;
            transpose_read(input: TensorRead<'_>, perm: &[usize]) -> Result<Tensor>;
            reshape_read(input: TensorRead<'_>, shape: &[usize]) -> Result<Tensor>;
            broadcast_in_dim_read(input: TensorRead<'_>, shape: &[usize], dims: &[usize]) -> Result<Tensor>;
        }
    }

    impl TensorReduction for DefaultFftSession {
        unreachable_methods! {
            reduce_sum_read(input: TensorRead<'_>, axes: &[usize]) -> Result<Tensor>;
            reduce_prod_read(input: TensorRead<'_>, axes: &[usize]) -> Result<Tensor>;
            reduce_max_read(input: TensorRead<'_>, axes: &[usize]) -> Result<Tensor>;
            reduce_min_read(input: TensorRead<'_>, axes: &[usize]) -> Result<Tensor>;
        }
    }

    impl TensorIndexing for DefaultFftSession {
        unreachable_methods! {
            gather(operand: &Tensor, start_indices: &Tensor, config: &GatherConfig) -> Result<Tensor>;
            scatter(operand: &Tensor, scatter_indices: &Tensor, updates: &Tensor, config: &ScatterConfig) -> Result<Tensor>;
            slice(input: &Tensor, config: &SliceConfig) -> Result<Tensor>;
            dynamic_slice(input: &Tensor, starts: &Tensor, slice_sizes: &[usize]) -> Result<Tensor>;
            dynamic_update_slice(operand: &Tensor, update: &Tensor, starts: &Tensor) -> Result<Tensor>;
            pad(input: &Tensor, config: &PadConfig) -> Result<Tensor>;
            concatenate(inputs: &[&Tensor], axis: usize) -> Result<Tensor>;
            reverse(input: &Tensor, axes: &[usize]) -> Result<Tensor>;
        }
    }

    impl TensorDot for DefaultFftSession {
        unreachable_methods! {
            dot_general_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>, config: &DotGeneralConfig) -> Result<Tensor>;
        }
    }
    impl TensorFusion for DefaultFftSession {}
    impl TensorBuffer for DefaultFftSession {}

    impl TensorDeviceTransfer for DefaultFftSession {
        unreachable_methods! {
            download_to_host(tensor: TensorRead<'_>) -> Result<Tensor>;
            upload_host_tensor(tensor: TensorRead<'_>) -> Result<Tensor>;
        }
    }

    impl SessionCachedDot for DefaultFftSession {}
    impl Engine for DefaultFftSession {}
}

impl FftBackend for DefaultFftSession {
    fn execute_fft(
        &mut self,
        input: &Tensor,
        spec: &FftPlanSpec,
        _cache: FftExecutionCache<'_>,
    ) -> tenferro_tensor::Result<Tensor> {
        let values = input.as_slice::<num_complex::Complex64>()?;
        self.recorded.push(RecordedFft {
            data_ptr: values.as_ptr(),
            shape: input.shape().to_vec(),
            values: values.to_vec(),
            spec: spec.clone(),
        });
        input.duplicate()
    }
}

#[test]
fn trait_default_read_hooks_accept_input_pass_tensors_through_and_canonicalize_views() {
    use crate::{FftNorm, FftOperation};
    use num_complex::Complex64;
    use tenferro_tensor::{DType, TensorView, TypedTensorView};

    let data = [1., 2., 3., 4.].map(|x| Complex64::new(x, 0.));
    let input = Tensor::from_vec_col_major([2, 2], data.to_vec()).unwrap();
    let borrowed = || {
        TensorRead::from_view(TensorView::C64(
            TypedTensorView::from_col_major(&[2, 2], &data).unwrap(),
        ))
    };
    let transposed = || {
        TensorRead::from_view(TensorView::C64(
            TypedTensorView::from_col_major(&[2, 2], &data)
                .unwrap()
                .transpose_view([1, 0])
                .unwrap(),
        ))
    };
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
    let mut session = DefaultFftSession::default();
    let mut cache = FftPlanCache::default();

    // The default validation hook accepts owned and borrowed reads unchanged.
    for read in [TensorRead::from_tensor(&input), borrowed(), transposed()] {
        session.validate_fft_read_input("fft", &read).unwrap();
    }

    // An owned tensor reaches execute_fft as the caller's own storage: no copy.
    let owned_output = session
        .execute_fft_read(
            TensorRead::from_tensor(&input),
            &spec,
            FftExecutionCache::caller_owned(&mut cache),
        )
        .unwrap();
    // A borrowed view is canonicalized through to_contiguous_read into a fresh
    // owned tensor before execute_fft sees it.
    let view_output = session
        .execute_fft_read(
            borrowed(),
            &spec,
            FftExecutionCache::caller_owned(&mut cache),
        )
        .unwrap();

    let [owned, view] = session.recorded.as_slice() else {
        panic!(
            "execute_fft must run once per read, got {:?}",
            session.recorded
        )
    };
    let input_ptr = input.as_slice::<Complex64>().unwrap().as_ptr();
    assert_eq!(owned.data_ptr, input_ptr);
    assert_eq!(owned.shape, [2, 2]);
    assert_eq!(owned.values, data);
    assert_eq!(owned.spec, spec);
    assert_eq!(owned_output.as_slice::<Complex64>().unwrap(), data);

    assert_ne!(view.data_ptr, data.as_ptr());
    assert_ne!(view.data_ptr, input_ptr);
    assert_eq!(view.shape, [2, 2]);
    assert_eq!(view.values, data);
    assert_eq!(view.spec, spec);
    assert_eq!(view_output.as_slice::<Complex64>().unwrap(), data);

    // The default structural hook now gathers a strided host view over its
    // layout, so the transposed read reaches execute_fft as compact storage.
    let transposed_output = session
        .execute_fft_read(
            transposed(),
            &spec,
            FftExecutionCache::caller_owned(&mut cache),
        )
        .unwrap();
    assert_eq!(session.recorded.len(), 3);
    let compact = &session.recorded[2];
    assert_eq!(compact.shape, [2, 2]);
    assert_eq!(compact.values, [data[0], data[2], data[1], data[3]]);
    assert_eq!(transposed_output.as_slice::<Complex64>().unwrap().len(), 4);
}

#[test]
fn cpu_into_writes_compact_offset_views_for_complex_and_real_transforms() {
    use crate::{FftNorm, FftOperation};
    use num_complex::{Complex32, Complex64};
    use tenferro_tensor::{TensorWrite, TypedTensorViewMut};
    let mut cpu = tenferro_cpu::CpuEngine::serial();
    let mut cache = FftPlanCache::default();
    macro_rules! check {
        ($input:expr, $operation:expr, $scalar:ty, $zero:expr) => {{
            let input = $input;
            let spec = crate::concrete_fft_spec(
                "fft",
                $operation,
                input.dtype(),
                input.shape(),
                None,
                1,
                FftNorm::Ortho,
            )
            .unwrap();
            let expected = cpu
                .execute_fft(&input, &spec, FftExecutionCache::caller_owned(&mut cache))
                .unwrap();
            let len = expected.as_slice::<$scalar>().unwrap().len();
            let mut data = vec![$zero; len + 2];
            let view =
                TypedTensorViewMut::from_slice(expected.shape(), [1, 2], 1, &mut data).unwrap();
            let write = TensorWrite::from_view(
                <$scalar as tenferro_tensor::TensorScalar>::tensor_view_mut(view),
            );
            cpu.execute_fft_read_into(
                TensorRead::from_tensor(&input),
                &spec,
                FftExecutionCache::caller_owned(&mut cache),
                write,
            )
            .unwrap();
            assert_eq!(&data[1..len + 1], expected.as_slice::<$scalar>().unwrap());
            assert_eq!(data[0], $zero);
            assert_eq!(data[len + 1], $zero);
        }};
    }
    check!(
        Tensor::from_vec_col_major(
            [2, 4],
            (0..8).map(|x| Complex64::new(x as f64, 0.25)).collect()
        )
        .unwrap(),
        FftOperation::C2cForward,
        Complex64,
        Complex64::new(-9., 0.)
    );
    check!(
        Tensor::from_vec_col_major(
            [2, 4],
            (0..8).map(|x| Complex32::new(x as f32, 0.25)).collect()
        )
        .unwrap(),
        FftOperation::C2cInverse,
        Complex32,
        Complex32::new(-9., 0.)
    );
    check!(
        Tensor::from_vec_col_major([2, 4], (0..8).map(|x| x as f64).collect()).unwrap(),
        FftOperation::R2cOnesided,
        Complex64,
        Complex64::new(-9., 0.)
    );
    check!(
        Tensor::from_vec_col_major([2, 4], (0..8).map(|x| x as f32).collect()).unwrap(),
        FftOperation::R2cFull,
        Complex32,
        Complex32::new(-9., 0.)
    );
    check!(
        Tensor::from_vec_col_major(
            [2, 3],
            (0..6).map(|x| Complex64::new(x as f64, 0.25)).collect()
        )
        .unwrap(),
        FftOperation::C2r,
        f64,
        -9.
    );
    check!(
        Tensor::from_vec_col_major(
            [2, 3],
            (0..6).map(|x| Complex32::new(x as f32, 0.25)).collect()
        )
        .unwrap(),
        FftOperation::C2r,
        f32,
        -9.
    );
}

#[test]
fn cpu_into_rejects_noncompact_destination_before_writes() {
    use crate::{FftNorm, FftOperation};
    use num_complex::Complex64;
    use tenferro_tensor::{TensorViewMut, TensorWrite, TypedTensorViewMut};
    let input = Tensor::from_vec_col_major([2, 2], vec![Complex64::new(1., 0.); 4]).unwrap();
    let spec = crate::concrete_fft_spec(
        "fft",
        FftOperation::C2cForward,
        input.dtype(),
        input.shape(),
        None,
        1,
        FftNorm::Backward,
    )
    .unwrap();
    let sentinel = Complex64::new(17., -4.);
    let mut data = [sentinel; 8];
    let view = TypedTensorViewMut::from_slice([2, 2], [2, 4], 0, &mut data).unwrap();
    let mut cpu = tenferro_cpu::CpuEngine::serial();
    let mut cache = FftPlanCache::default();
    let error = cpu
        .execute_fft_read_into(
            TensorRead::from_tensor(&input),
            &spec,
            FftExecutionCache::caller_owned(&mut cache),
            TensorWrite::from_view(TensorViewMut::C64(view)),
        )
        .unwrap_err();
    assert!(matches!(error, tenferro_tensor::Error::Unsupported { .. }));
    assert_eq!(data, [sentinel; 8]);
    assert_eq!(cache.stats().entries, 0);
}

#[test]
fn default_into_is_unsupported_without_allocating_execution() {
    use crate::{FftNorm, FftOperation};
    use num_complex::Complex64;
    use tenferro_tensor::TensorWrite;
    let input = Tensor::from_vec_col_major([2], vec![Complex64::new(1., 0.); 2]).unwrap();
    let spec = crate::concrete_fft_spec(
        "fft",
        FftOperation::C2cForward,
        input.dtype(),
        input.shape(),
        None,
        0,
        FftNorm::Backward,
    )
    .unwrap();
    let mut output = Tensor::from_vec_col_major([2], vec![Complex64::new(7., 0.); 2]).unwrap();
    let mut backend = DefaultFftSession::default();
    let mut cache = FftPlanCache::default();
    let error = backend
        .execute_fft_read_into(
            TensorRead::from_tensor(&input),
            &spec,
            FftExecutionCache::caller_owned(&mut cache),
            TensorWrite::from_tensor(&mut output),
        )
        .unwrap_err();
    assert!(matches!(error, tenferro_tensor::Error::Unsupported { .. }));
    assert!(backend.recorded.is_empty());
    assert_eq!(
        output.as_slice::<Complex64>().unwrap(),
        &[Complex64::new(7., 0.); 2]
    );
}
