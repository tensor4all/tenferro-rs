use num_complex::{Complex32, Complex64};
use tenferro_cpu::{CpuBackend, CpuExecSession};
use tenferro_fft::{FftBackend, FftExecutor, FftNorm, TensorFftExt};
#[cfg(feature = "cuda")]
use tenferro_gpu::cuda::CudaExecSession;
use tenferro_runtime::Runtime;
use tenferro_tensor::{
    BackendCachedDot, BackendRuntimeCache, BackendSession, BackendSessionHost,
    BackendStorageHandle, CompareDir, DType, DeviceId, DeviceKind, DotGeneralConfig, ErrorKind,
    GatherConfig, GpuBackendKind, MemoryKind, PadConfig, Placement, ScatterConfig, SliceConfig,
    StorageBuffer, Tensor, TensorAnalytic, TensorBackend, TensorBuffer, TensorDeviceTransfer,
    TensorDot, TensorElementwise, TensorFusion, TensorIndexing, TensorRead, TensorReduction,
    TensorStructural, TypedTensor,
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
    ($ty:ty, $marker:ty) => {
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
            fn download_to_host(
                &mut self,
                tensor: TensorRead<'_>,
            ) -> tenferro_tensor::Result<Tensor> {
                self.record_transfer();
                tensor.tensor_view().duplicate()
            }

            fn upload_host_tensor(
                &mut self,
                tensor: TensorRead<'_>,
            ) -> tenferro_tensor::Result<Tensor> {
                self.record_transfer();
                tensor.tensor_view().duplicate()
            }
        }
        impl BackendCachedDot for $ty {}
        impl BackendSession for $ty {
            fn session_type_id(&self) -> std::any::TypeId {
                std::any::TypeId::of::<$marker>()
            }

            unsafe fn session_data_mut(&mut self) -> *mut () {
                self as *mut Self as *mut ()
            }
        }
        impl BackendSessionHost for $ty {}
        impl TensorBackend for $ty {}
    };
}

#[derive(Debug, Default)]
struct TensorOnlyBackend;

impl TensorOnlyBackend {
    fn record_transfer(&self) {}
}

#[doc(hidden)]
struct TensorOnlyBackendSessionMarker;
impl_minimal_tensor_backend!(TensorOnlyBackend, TensorOnlyBackendSessionMarker);

fn assert_fft_capability<B: FftBackend>() {}
fn assert_tensor_backend<B: TensorBackend>() {}

#[cfg(feature = "cuda")]
#[test]
fn cuda_session_is_fft_capable() {
    assert_fft_capability::<CudaExecSession<'static>>();
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
        .with_backend_session(|session| input.fft(None, -1, FftNorm::Backward, session))
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
    let mut owner = CpuBackend::new();
    let mut executor = FftExecutor::default();

    owner.with_backend_session(|session| {
        executor
            .fft(&input, None, -1, FftNorm::Backward, session)
            .unwrap();
        executor
            .fft(&input, None, -1, FftNorm::Backward, session)
            .unwrap();
    });

    let stats = executor.cache_stats();
    assert_eq!(stats.entries, 1);
    assert!(stats.hits >= 1, "warm call should hit the retained plan");
    assert!(stats.retained_bytes > 0);

    executor.clear_cache();
    assert_eq!(executor.cache_stats().entries, 0);
    assert_eq!(executor.cache_stats().retained_bytes, 0);

    owner.with_backend_session(|session| {
        executor
            .fft(&input, None, -1, FftNorm::Backward, session)
            .unwrap();
    });
    assert_eq!(executor.cache_stats().entries, 1);
}

#[test]
fn direct_concrete_api_returns_typed_capability_error_without_fft_capability() {
    // The concrete FFT traits dispatch internally to the built-in FFT exec
    // sessions (CPU/CUDA/WebGPU); a backend session that exposes no FFT
    // capability must return a typed capability error (issue #1680 Phase 3).
    let input = Tensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
    )
    .unwrap();
    let mut session = TensorOnlyBackend;

    let error = input
        .fft(None, -1, FftNorm::Backward, &mut session)
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error
        .to_string()
        .contains("does not expose an FFT execution capability"));
}

#[test]
fn concrete_cpu_execution_preserves_all_four_scalar_dtypes() {
    let mut backend = CpuBackend::new();

    backend.with_backend_session(|session| {
        let f32_input = Tensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]).unwrap();
        let f32_output = f32_input.fft(None, -1, FftNorm::Backward, session).unwrap();
        assert_eq!(
            f32_output.as_slice::<Complex32>().unwrap()[0],
            Complex32::new(3.0, 0.0)
        );

        let f64_input = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
        let f64_output = f64_input.fft(None, -1, FftNorm::Backward, session).unwrap();
        assert_eq!(
            f64_output.as_slice::<Complex64>().unwrap()[0],
            Complex64::new(3.0, 0.0)
        );

        let c32_input = Tensor::from_vec_col_major(
            vec![2],
            vec![Complex32::new(1.0, 0.0), Complex32::new(2.0, 0.0)],
        )
        .unwrap();
        let c32_output = c32_input.fft(None, -1, FftNorm::Backward, session).unwrap();
        assert_eq!(
            c32_output.as_slice::<Complex32>().unwrap()[0],
            Complex32::new(3.0, 0.0)
        );

        let c64_input = Tensor::from_vec_col_major(
            vec![2],
            vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
        )
        .unwrap();
        let c64_output = c64_input.fft(None, -1, FftNorm::Backward, session).unwrap();
        assert_eq!(
            c64_output.as_slice::<Complex64>().unwrap()[0],
            Complex64::new(3.0, 0.0)
        );
    });
}

fn cuda_c64_tensor(shape: Vec<usize>) -> Tensor {
    let len = shape.iter().product();
    Tensor::C64(
        TypedTensor::from_buffer_col_major(
            shape,
            StorageBuffer::Backend(Box::new(BackendStorageHandle::<Complex64>::new_with_len(
                7, len,
            ))),
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
    let input = cuda_c64_tensor(vec![2]);
    let mut owner = CpuBackend::new();

    let error = owner
        .with_backend_session(|session| input.fft(None, -1, FftNorm::Backward, session))
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Unsupported);
}
