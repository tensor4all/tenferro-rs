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
    use tenferro_tensor::BackendSessionHost;
    let mut backend = tenferro_cpu::CpuBackend::with_threads(1).unwrap();
    let mut cache = FftPlanCache::default();
    for (read, expected) in [
        (TensorRead::from_tensor(&input), [3., -1., 7., -1.]),
        (view, [4., -2., 6., -2.]),
    ] {
        let output = backend
            .with_backend_session(|session| {
                tenferro_cpu::with_cpu_exec_session(session, |cpu| {
                    cpu.validate_fft_read_input("fft", &read).unwrap();
                    cpu.execute_fft_read(read, &spec, FftExecutionCache::caller_owned(&mut cache))
                })
                .expect("the CPU session exposes the FFT capability")
            })
            .unwrap()
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
        BackendCachedDot, BackendRuntimeCache, BackendSession, BackendSessionHost, CompareDir,
        DType, DotGeneralConfig, ElementwiseReadOp, GatherConfig, PadConfig, ScatterConfig,
        SliceConfig, Tensor, TensorAnalytic, TensorBackend, TensorBuffer, TensorDeviceTransfer,
        TensorDot, TensorElementwise, TensorFusion, TensorIndexing, TensorRead, TensorReduction,
        TensorStructural, TensorWrite,
    };

    type Result<T> = tenferro_tensor::Result<T>;

    impl BackendRuntimeCache for DefaultFftSession {
        type RuntimeCache = ();
    }

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

    impl BackendCachedDot for DefaultFftSession {}
    impl BackendSession for DefaultFftSession {}

    impl BackendSessionHost for DefaultFftSession {
        fn with_backend_session<R>(
            &mut self,
            f: impl FnOnce(&mut dyn BackendSession) -> R,
        ) -> std::result::Result<R, tenferro_tensor::SessionEntryError> {
            tenferro_tensor::with_session_entry_guard("default FFT test session", || f(self))
        }
    }

    impl TensorBackend for DefaultFftSession {}
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
