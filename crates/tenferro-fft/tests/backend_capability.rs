use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};

use num_complex::{Complex32, Complex64};
use tenferro_cpu::{with_cpu_exec_session, CpuBackend, CpuExecSession};
use tenferro_fft::{
    FftBackend, FftExecutionCache, FftExecutor, FftNorm, FftOperation, FftPlanSpec, TensorFftExt,
    FFT_EXTENSION_FAMILY_ID,
};
use tenferro_runtime::{ExtensionCacheKey, Runtime};
use tenferro_tensor::{
    BackendCachedDot, BackendRuntimeCache, BackendSessionHost, Buffer, BufferHandle, CompareDir,
    DType, DeviceId, DeviceKind, DotGeneralConfig, ErrorKind, GatherConfig, GpuBackendKind,
    MemoryKind, PadConfig, Placement, ScatterConfig, SliceConfig, Tensor, TensorAnalytic,
    TensorBackend, TensorBuffer, TensorDeviceTransfer, TensorDot, TensorElementwise, TensorFusion,
    TensorIndexing, TensorReduction, TensorStructural, TypedTensor,
};

macro_rules! unreachable_backend_methods {
    ($($name:ident($($arg:ident : $argty:ty),*) -> $ret:ty;)+) => {
        $(
            fn $name(&mut self, $($arg: $argty),*) -> $ret {
                $(let _ = &$arg;)*
                panic!(concat!(stringify!($name), " should not be called by this test"))
            }
        )+
    };
}

macro_rules! impl_minimal_tensor_backend {
    ($ty:ty) => {
        impl BackendRuntimeCache for $ty {
            type RuntimeCache = ();
        }

        impl TensorElementwise for $ty {
            unreachable_backend_methods! {
                add(lhs: &Tensor, rhs: &Tensor) -> tenferro_tensor::Result<Tensor>;
                sub(lhs: &Tensor, rhs: &Tensor) -> tenferro_tensor::Result<Tensor>;
                mul(lhs: &Tensor, rhs: &Tensor) -> tenferro_tensor::Result<Tensor>;
                neg(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
                conj(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
                div(lhs: &Tensor, rhs: &Tensor) -> tenferro_tensor::Result<Tensor>;
                abs(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
                sign(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
                maximum(lhs: &Tensor, rhs: &Tensor) -> tenferro_tensor::Result<Tensor>;
                minimum(lhs: &Tensor, rhs: &Tensor) -> tenferro_tensor::Result<Tensor>;
                compare(lhs: &Tensor, rhs: &Tensor, dir: &CompareDir) -> tenferro_tensor::Result<Tensor>;
                select(pred: &Tensor, on_true: &Tensor, on_false: &Tensor) -> tenferro_tensor::Result<Tensor>;
                clamp(input: &Tensor, lower: &Tensor, upper: &Tensor) -> tenferro_tensor::Result<Tensor>;
            }
        }

        impl TensorAnalytic for $ty {
            unreachable_backend_methods! {
                exp(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
                log(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
                sin(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
                cos(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
                tanh(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
                sqrt(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
                rsqrt(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
                pow(lhs: &Tensor, rhs: &Tensor) -> tenferro_tensor::Result<Tensor>;
                expm1(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
                log1p(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
            }
        }

        impl TensorStructural for $ty {
            unreachable_backend_methods! {
                transpose(input: &Tensor, perm: &[usize]) -> tenferro_tensor::Result<Tensor>;
                reshape(input: &Tensor, shape: &[usize]) -> tenferro_tensor::Result<Tensor>;
                broadcast_in_dim(input: &Tensor, shape: &[usize], dims: &[usize]) -> tenferro_tensor::Result<Tensor>;
                cast(input: &Tensor, to: DType) -> tenferro_tensor::Result<Tensor>;
                extract_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> tenferro_tensor::Result<Tensor>;
                embed_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> tenferro_tensor::Result<Tensor>;
                tril(input: &Tensor, k: i64) -> tenferro_tensor::Result<Tensor>;
                triu(input: &Tensor, k: i64) -> tenferro_tensor::Result<Tensor>;
            }
        }

        impl TensorReduction for $ty {
            unreachable_backend_methods! {
                reduce_sum(input: &Tensor, axes: &[usize]) -> tenferro_tensor::Result<Tensor>;
                reduce_prod(input: &Tensor, axes: &[usize]) -> tenferro_tensor::Result<Tensor>;
                reduce_max(input: &Tensor, axes: &[usize]) -> tenferro_tensor::Result<Tensor>;
                reduce_min(input: &Tensor, axes: &[usize]) -> tenferro_tensor::Result<Tensor>;
            }
        }

        impl TensorIndexing for $ty {
            unreachable_backend_methods! {
                gather(operand: &Tensor, start_indices: &Tensor, config: &GatherConfig) -> tenferro_tensor::Result<Tensor>;
                scatter(operand: &Tensor, scatter_indices: &Tensor, updates: &Tensor, config: &ScatterConfig) -> tenferro_tensor::Result<Tensor>;
                slice(input: &Tensor, config: &SliceConfig) -> tenferro_tensor::Result<Tensor>;
                dynamic_slice(input: &Tensor, starts: &Tensor, slice_sizes: &[usize]) -> tenferro_tensor::Result<Tensor>;
                dynamic_update_slice(operand: &Tensor, update: &Tensor, starts: &Tensor) -> tenferro_tensor::Result<Tensor>;
                pad(input: &Tensor, config: &PadConfig) -> tenferro_tensor::Result<Tensor>;
                concatenate(inputs: &[&Tensor], axis: usize) -> tenferro_tensor::Result<Tensor>;
                reverse(input: &Tensor, axes: &[usize]) -> tenferro_tensor::Result<Tensor>;
            }
        }

        impl TensorDot for $ty {
            unreachable_backend_methods! {
                dot_general(lhs: &Tensor, rhs: &Tensor, config: &DotGeneralConfig) -> tenferro_tensor::Result<Tensor>;
            }
        }

        impl TensorFusion for $ty {}
        impl TensorBuffer for $ty {}
        impl TensorDeviceTransfer for $ty {
            fn download_to_host(&mut self, tensor: &Tensor) -> tenferro_tensor::Result<Tensor> {
                self.record_transfer();
                Ok(tensor.clone())
            }

            fn upload_host_tensor(&mut self, tensor: &Tensor) -> tenferro_tensor::Result<Tensor> {
                self.record_transfer();
                Ok(tensor.clone())
            }
        }
        impl BackendCachedDot for $ty {}
        impl BackendSessionHost for $ty {}
        impl TensorBackend for $ty {}
    };
}

#[derive(Debug, Default)]
struct TensorOnlyBackend;

impl TensorOnlyBackend {
    fn record_transfer(&self) {}
}

impl_minimal_tensor_backend!(TensorOnlyBackend);

#[derive(Debug, Default)]
struct MockNonCpuSession {
    plan_builds: usize,
    plan_reuses: usize,
}

impl MockNonCpuSession {
    fn record_transfer(&self) {}
}

impl_minimal_tensor_backend!(MockNonCpuSession);

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct MockNonCpuPlanKey {
    operation: FftOperation,
    normalized_axis: usize,
    requested_len: Option<usize>,
    input_dtype: DType,
    input_shape: Vec<usize>,
}

#[derive(Debug)]
struct MockNonCpuPlan {
    key: MockNonCpuPlanKey,
}

impl FftBackend for MockNonCpuSession {
    fn execute_fft(
        &mut self,
        input: &Tensor,
        spec: &FftPlanSpec,
        mut cache: FftExecutionCache<'_>,
    ) -> tenferro_tensor::Result<Tensor> {
        let plan_key = MockNonCpuPlanKey {
            operation: spec.operation(),
            normalized_axis: spec.normalized_axis(),
            requested_len: spec.requested_len(),
            input_dtype: spec.input_dtype(),
            input_shape: spec.input_shape().to_vec(),
        };
        let mut hasher = DefaultHasher::new();
        plan_key.hash(&mut hasher);
        let cache_key = ExtensionCacheKey::new(
            FFT_EXTENSION_FAMILY_ID,
            "mock-non-cpu-plans",
            hasher.finish(),
        );
        let store = cache.store_mut();
        if store
            .get::<MockNonCpuPlan>(&cache_key)
            .is_some_and(|plan| plan.key == plan_key)
        {
            self.plan_reuses += 1;
        } else {
            self.plan_builds += 1;
            store.put(cache_key, MockNonCpuPlan { key: plan_key }, 96);
        }
        Ok(input.clone())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RecordedSpec {
    operation: FftOperation,
    normalized_axis: usize,
    requested_len: Option<usize>,
    norm: FftNorm,
    input_dtype: DType,
    input_shape: Vec<usize>,
    requires_compact_column_major: bool,
}

#[derive(Debug)]
struct RecordingFftSession {
    cpu_owner: CpuBackend,
    specs: Arc<Mutex<Vec<RecordedSpec>>>,
    transfers: Arc<AtomicUsize>,
}

impl RecordingFftSession {
    fn new() -> (Self, Arc<Mutex<Vec<RecordedSpec>>>, Arc<AtomicUsize>) {
        let specs = Arc::new(Mutex::new(Vec::new()));
        let transfers = Arc::new(AtomicUsize::new(0));
        (
            Self {
                cpu_owner: CpuBackend::new(),
                specs: Arc::clone(&specs),
                transfers: Arc::clone(&transfers),
            },
            specs,
            transfers,
        )
    }

    fn record_transfer(&self) {
        self.transfers.fetch_add(1, Ordering::Relaxed);
    }
}

impl_minimal_tensor_backend!(RecordingFftSession);

impl FftBackend for RecordingFftSession {
    fn execute_fft(
        &mut self,
        input: &Tensor,
        spec: &FftPlanSpec,
        cache: FftExecutionCache<'_>,
    ) -> tenferro_tensor::Result<Tensor> {
        self.specs.lock().unwrap().push(RecordedSpec {
            operation: spec.operation(),
            normalized_axis: spec.normalized_axis(),
            requested_len: spec.requested_len(),
            norm: spec.norm(),
            input_dtype: spec.input_dtype(),
            input_shape: spec.input_shape().to_vec(),
            requires_compact_column_major: spec.requires_compact_column_major(),
        });
        self.cpu_owner
            .with_backend_session(|session| {
                with_cpu_exec_session(session, |session| session.execute_fft(input, spec, cache))
            })
            .expect("CpuBackend must expose a CPU execution session")
    }
}

fn assert_fft_capability<B: FftBackend>() {}
fn assert_tensor_backend<B: TensorBackend>() {}

fn with_cpu_session<R>(
    backend: &mut CpuBackend,
    f: impl for<'a> FnOnce(&'a mut CpuExecSession<'a>) -> R + Send,
) -> R
where
    R: Send,
{
    backend.with_backend_session(|session| {
        with_cpu_exec_session(session, f).expect("CpuBackend must expose a CPU execution session")
    })
}

#[test]
fn cpu_session_is_fft_capable_and_runtime_registration_accepts_it() {
    assert_fft_capability::<CpuExecSession<'static>>();
    assert_tensor_backend::<TensorOnlyBackend>();

    let owner = CpuBackend::new();
    let mut builder = Runtime::builder();
    builder
        .register_engine(tenferro_cpu::runtime_engine_registration(&owner).unwrap())
        .unwrap();
    builder
        .install_extension_module(
            tenferro_fft::extension_module::<CpuBackend>(
                tenferro_cpu::runtime_engine_id().unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
    let runtime = builder.build().unwrap();

    assert_eq!(runtime.snapshot().unwrap().extension_module_count(), 1);
}

#[test]
fn cpu_fft_is_invoked_through_the_borrowed_provider_session() {
    let input = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let mut owner = CpuBackend::new();
    let output = owner
        .with_backend_session(|session| {
            with_cpu_exec_session(session, |session: &mut CpuExecSession<'_>| {
                input.fft(None, -1, FftNorm::Backward, session)
            })
        })
        .expect("CpuBackend must expose a CPU execution session")
        .unwrap();

    assert_eq!(
        output.as_slice::<Complex64>().unwrap()[0],
        Complex64::new(3.0, 0.0)
    );
}

#[test]
fn caller_owned_cache_is_backend_neutral_and_reports_reuse_clear_and_stats() {
    let input = Tensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
    )
    .unwrap();
    let mut session = MockNonCpuSession::default();
    let mut executor = FftExecutor::default();

    executor
        .fft(&input, None, -1, FftNorm::Backward, &mut session)
        .unwrap();
    executor
        .fft(&input, None, -1, FftNorm::Backward, &mut session)
        .unwrap();

    assert_eq!(session.plan_builds, 1);
    assert_eq!(session.plan_reuses, 1);
    assert_eq!(executor.cache_stats().entries, 1);
    assert_eq!(executor.cache_stats().retained_bytes, 96);

    executor.clear_cache();
    assert_eq!(executor.cache_stats().entries, 0);
    assert_eq!(executor.cache_stats().retained_bytes, 0);

    executor
        .fft(&input, None, -1, FftNorm::Backward, &mut session)
        .unwrap();
    assert_eq!(session.plan_builds, 2);
}

#[test]
fn direct_concrete_api_uses_call_local_one_shot_plan_cache() {
    let input = Tensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
    )
    .unwrap();
    let mut session = MockNonCpuSession::default();

    input
        .fft(None, -1, FftNorm::Backward, &mut session)
        .unwrap();
    input
        .fft(None, -1, FftNorm::Backward, &mut session)
        .unwrap();

    assert_eq!(session.plan_builds, 2);
    assert_eq!(session.plan_reuses, 0);
}

#[test]
fn concrete_cpu_execution_preserves_all_four_scalar_dtypes() {
    let mut backend = CpuBackend::new();

    with_cpu_session(&mut backend, |backend| {
        let f32_input = Tensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]).unwrap();
        let f32_output = f32_input.fft(None, -1, FftNorm::Backward, backend).unwrap();
        assert_eq!(
            f32_output.as_slice::<Complex32>().unwrap()[0],
            Complex32::new(3.0, 0.0)
        );

        let f64_input = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
        let f64_output = f64_input.fft(None, -1, FftNorm::Backward, backend).unwrap();
        assert_eq!(
            f64_output.as_slice::<Complex64>().unwrap()[0],
            Complex64::new(3.0, 0.0)
        );

        let c32_input = Tensor::from_vec_col_major(
            vec![2],
            vec![Complex32::new(1.0, 0.0), Complex32::new(2.0, 0.0)],
        )
        .unwrap();
        let c32_output = c32_input.fft(None, -1, FftNorm::Backward, backend).unwrap();
        assert_eq!(
            c32_output.as_slice::<Complex32>().unwrap()[0],
            Complex32::new(3.0, 0.0)
        );

        let c64_input = Tensor::from_vec_col_major(
            vec![2],
            vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
        )
        .unwrap();
        let c64_output = c64_input.fft(None, -1, FftNorm::Backward, backend).unwrap();
        assert_eq!(
            c64_output.as_slice::<Complex64>().unwrap()[0],
            Complex64::new(3.0, 0.0)
        );
    });
}

#[test]
fn concrete_execution_delegates_the_validated_plan_spec() {
    let (mut session, specs, _) = RecordingFftSession::new();
    let input = Tensor::from_vec_col_major(
        vec![2, 3],
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
            Complex64::new(5.0, 0.0),
            Complex64::new(6.0, 0.0),
        ],
    )
    .unwrap();

    let output = input
        .fft(Some(4), -1, FftNorm::Ortho, &mut session)
        .unwrap();
    assert_eq!(output.shape(), &[2, 4]);

    assert_eq!(
        *specs.lock().unwrap(),
        vec![RecordedSpec {
            operation: FftOperation::C2cForward,
            normalized_axis: 1,
            requested_len: Some(4),
            norm: FftNorm::Ortho,
            input_dtype: DType::C64,
            input_shape: vec![2, 3],
            requires_compact_column_major: true,
        }]
    );
}

fn cuda_c64_tensor(shape: Vec<usize>) -> Tensor {
    let len = shape.iter().product();
    Tensor::C64(
        TypedTensor::from_buffer_col_major(
            shape,
            Buffer::Backend(Arc::new(BufferHandle::<Complex64>::new_with_len(7, len))),
            Placement {
                memory_kind: MemoryKind::Device,
                device: Some(DeviceId {
                    kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
                    ordinal: 0,
                }),
                cpu_affinity: None,
            },
        )
        .unwrap(),
    )
}

#[test]
fn foreign_placement_is_unsupported_without_transfer() {
    let (mut session, _, transfers) = RecordingFftSession::new();
    let input = cuda_c64_tensor(vec![2]);

    let error = input
        .fft(None, -1, FftNorm::Backward, &mut session)
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert_eq!(transfers.load(Ordering::Relaxed), 0);
}
