use num_complex::{Complex32, Complex64};
use tenferro_cpu::{CpuBackend, CpuExecSession};
use tenferro_fft::{FftBackend, FftExecutor, FftNorm, TensorFftExt};
#[cfg(feature = "cuda")]
use tenferro_gpu::cuda::CudaExecSession;
use tenferro_runtime::Runtime;
use tenferro_tensor::{
    BackendCachedDot, BackendRuntimeCache, BackendSession, BackendSessionHost,
    BackendStorageHandle, CompareDir, DType, DeviceId, DeviceKind, DotGeneralConfig,
    ElementwiseReadOp, ErrorKind, GatherConfig, GpuBackendKind, MemoryKind, PadConfig, Placement,
    ScatterConfig, SliceConfig, StorageBuffer, Tensor, TensorAnalytic, TensorBackend, TensorBuffer,
    TensorDeviceTransfer, TensorDot, TensorElementwise, TensorFusion, TensorIndexing, TensorRead,
    TensorReduction, TensorStructural, TensorWrite, TypedTensor,
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
            fn elementwise_read_into(
                &mut self,
                op: ElementwiseReadOp,
                inputs: &[TensorRead<'_>],
                out: TensorWrite<'_>,
            ) -> tenferro_tensor::Result<()> {
                let _ = (op, inputs, out);
                panic!("elementwise_read_into should not be called by this test")
            }

                // Reproduce the previous read-half default: delegate an owned tensor and
                // reject a borrowed view.
                fn add_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> tenferro_tensor::Result<Tensor> {
                    let _ = tenferro_tensor::backend::read_owned_tensor("add", lhs)?;
                    let _ = tenferro_tensor::backend::read_owned_tensor("add", rhs)?;
                    panic!("add should not be called by this test")
                }

                // Reproduce the previous read-half default: delegate an owned tensor and
                // reject a borrowed view.
                fn sub_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> tenferro_tensor::Result<Tensor> {
                    let _ = tenferro_tensor::backend::read_owned_tensor("sub", lhs)?;
                    let _ = tenferro_tensor::backend::read_owned_tensor("sub", rhs)?;
                    panic!("sub should not be called by this test")
                }

                // Reproduce the previous read-half default: delegate an owned tensor and
                // reject a borrowed view.
                fn mul_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> tenferro_tensor::Result<Tensor> {
                    let _ = tenferro_tensor::backend::read_owned_tensor("mul", lhs)?;
                    let _ = tenferro_tensor::backend::read_owned_tensor("mul", rhs)?;
                    panic!("mul should not be called by this test")
                }

                // Reproduce the previous read-half default: delegate an owned tensor and
                // reject a borrowed view.
                fn neg_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Tensor> {
                    let _ = tenferro_tensor::backend::read_owned_tensor("neg", input)?;
                    panic!("neg should not be called by this test")
                }

                // Reproduce the previous read-half default: delegate an owned tensor and
                // reject a borrowed view.
                fn conj_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Tensor> {
                    let _ = tenferro_tensor::backend::read_owned_tensor("conj", input)?;
                    panic!("conj should not be called by this test")
                }

                // Reproduce the previous read-half default: delegate an owned tensor and
                // reject a borrowed view.
                fn div_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> tenferro_tensor::Result<Tensor> {
                    let _ = tenferro_tensor::backend::read_owned_tensor("div", lhs)?;
                    let _ = tenferro_tensor::backend::read_owned_tensor("div", rhs)?;
                    panic!("div should not be called by this test")
                }

                // Reproduce the previous read-half default: delegate an owned tensor and
                // reject a borrowed view.
                fn abs_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Tensor> {
                    let _ = tenferro_tensor::backend::read_owned_tensor("abs", input)?;
                    panic!("abs should not be called by this test")
                }

                // Reproduce the previous read-half default: delegate an owned tensor and
                // reject a borrowed view.
                fn sign_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Tensor> {
                    let _ = tenferro_tensor::backend::read_owned_tensor("sign", input)?;
                    panic!("sign should not be called by this test")
                }

                // Reproduce the previous read-half default: delegate an owned tensor and
                // reject a borrowed view.
                fn maximum_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> tenferro_tensor::Result<Tensor> {
                    let _ = tenferro_tensor::backend::read_owned_tensor("maximum", lhs)?;
                    let _ = tenferro_tensor::backend::read_owned_tensor("maximum", rhs)?;
                    panic!("maximum should not be called by this test")
                }

                // Reproduce the previous read-half default: delegate an owned tensor and
                // reject a borrowed view.
                fn minimum_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> tenferro_tensor::Result<Tensor> {
                    let _ = tenferro_tensor::backend::read_owned_tensor("minimum", lhs)?;
                    let _ = tenferro_tensor::backend::read_owned_tensor("minimum", rhs)?;
                    panic!("minimum should not be called by this test")
                }

                // Reproduce the previous read-half default: delegate an owned tensor and
                // reject a borrowed view.
                fn compare_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>, dir: &CompareDir) -> tenferro_tensor::Result<Tensor> {
                                        let _ = tenferro_tensor::backend::read_owned_tensor("compare", lhs)?;
                        let _ = tenferro_tensor::backend::read_owned_tensor("compare", rhs)?;
                        let _ = dir;
                        panic!("compare should not be called in this test")
                }

                // Reproduce the previous read-half default: delegate an owned tensor and
                // reject a borrowed view.
                fn select_read(&mut self, pred: TensorRead<'_>, on_true: TensorRead<'_>, on_false: TensorRead<'_>) -> tenferro_tensor::Result<Tensor> {
                    {
                        let _ = tenferro_tensor::backend::read_owned_tensor("select", pred)?;
                        let _ = tenferro_tensor::backend::read_owned_tensor("select", on_true)?;
                        let _ = tenferro_tensor::backend::read_owned_tensor("select", on_false)?;
                        panic!("select should not be called by this test")
                    }
                }

                // Reproduce the previous read-half default: delegate an owned tensor and
                // reject a borrowed view.
                fn clamp_read(&mut self, input: TensorRead<'_>, lower: TensorRead<'_>, upper: TensorRead<'_>) -> tenferro_tensor::Result<Tensor> {
                                        let _ = tenferro_tensor::backend::read_owned_tensor("clamp", input)?;
                        let _ = tenferro_tensor::backend::read_owned_tensor("clamp", lower)?;
                        let _ = tenferro_tensor::backend::read_owned_tensor("clamp", upper)?;
                        panic!("clamp should not be called in this test")
                }

        }

        impl TensorAnalytic for $ty {
            // Reproduce the previous read-half default: delegate an owned tensor and
            // reject a borrowed view.
            fn exp_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> tenferro_tensor::Result<Tensor> {
                let _ = tenferro_tensor::backend::read_owned_tensor("exp", input)?;
                panic!("exp should not be called by this test")
            }

            // Reproduce the previous read-half default: delegate an owned tensor and
            // reject a borrowed view.
            fn log_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> tenferro_tensor::Result<Tensor> {
                let _ = tenferro_tensor::backend::read_owned_tensor("log", input)?;
                panic!("log should not be called by this test")
            }

            // Reproduce the previous read-half default: delegate an owned tensor and
            // reject a borrowed view.
            fn sin_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> tenferro_tensor::Result<Tensor> {
                let _ = tenferro_tensor::backend::read_owned_tensor("sin", input)?;
                panic!("sin should not be called by this test")
            }

            // Reproduce the previous read-half default: delegate an owned tensor and
            // reject a borrowed view.
            fn cos_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> tenferro_tensor::Result<Tensor> {
                let _ = tenferro_tensor::backend::read_owned_tensor("cos", input)?;
                panic!("cos should not be called by this test")
            }

            // Reproduce the previous read-half default: delegate an owned tensor and
            // reject a borrowed view.
            fn tanh_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> tenferro_tensor::Result<Tensor> {
                let _ = tenferro_tensor::backend::read_owned_tensor("tanh", input)?;
                panic!("tanh should not be called by this test")
            }

            // Reproduce the previous read-half default: delegate an owned tensor and
            // reject a borrowed view.
            fn sqrt_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> tenferro_tensor::Result<Tensor> {
                let _ = tenferro_tensor::backend::read_owned_tensor("sqrt", input)?;
                panic!("sqrt should not be called by this test")
            }

            // Reproduce the previous read-half default: delegate an owned tensor and
            // reject a borrowed view.
            fn rsqrt_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> tenferro_tensor::Result<Tensor> {
                let _ = tenferro_tensor::backend::read_owned_tensor("rsqrt", input)?;
                panic!("rsqrt should not be called by this test")
            }

            // Reproduce the previous read-half default: delegate an owned tensor and
            // reject a borrowed view.
            fn pow_read(&mut self, lhs: tenferro_tensor::TensorRead<'_>, rhs: tenferro_tensor::TensorRead<'_>) -> tenferro_tensor::Result<Tensor> {
                let _ = tenferro_tensor::backend::read_owned_tensor("pow", lhs)?;
                let _ = tenferro_tensor::backend::read_owned_tensor("pow", rhs)?;
                panic!("pow should not be called by this test")
            }

            // Reproduce the previous read-half default: delegate an owned tensor and
            // reject a borrowed view.
            fn expm1_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> tenferro_tensor::Result<Tensor> {
                let _ = tenferro_tensor::backend::read_owned_tensor("expm1", input)?;
                panic!("expm1 should not be called by this test")
            }

            // Reproduce the previous read-half default: delegate an owned tensor and
            // reject a borrowed view.
            fn log1p_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> tenferro_tensor::Result<Tensor> {
                let _ = tenferro_tensor::backend::read_owned_tensor("log1p", input)?;
                panic!("log1p should not be called by this test")
            }

        }

        impl TensorStructural for $ty {
            unreachable_backend_methods! {
                cast(input: &Tensor, to: DType) -> tenferro_tensor::Result<Tensor>;
                extract_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> tenferro_tensor::Result<Tensor>;
                embed_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> tenferro_tensor::Result<Tensor>;
                tril(input: &Tensor, k: i64) -> tenferro_tensor::Result<Tensor>;
                triu(input: &Tensor, k: i64) -> tenferro_tensor::Result<Tensor>;
            }

            // The previous read-half default delegated owned tensors to the one-shot
            // method and rejected borrowed views. Reproduce it explicitly rather than
            // forwarding a view, which would widen the accepted input surface.
            fn transpose_read(&mut self, input: TensorRead<'_>, perm: &[usize]) -> tenferro_tensor::Result<Tensor> {
                                let _ = tenferro_tensor::backend::read_owned_tensor("transpose", input)?;
                    let _ = perm;
                    panic!("transpose should not be called in this test")
            }

            // The previous read-half default delegated owned tensors to the one-shot
            // method and rejected borrowed views. Reproduce it explicitly rather than
            // forwarding a view, which would widen the accepted input surface.
            fn reshape_read(&mut self, input: TensorRead<'_>, shape: &[usize]) -> tenferro_tensor::Result<Tensor> {
                                let _ = tenferro_tensor::backend::read_owned_tensor("reshape", input)?;
                    let _ = shape;
                    panic!("reshape should not be called in this test")
            }

            // The previous read-half default delegated owned tensors to the one-shot
            // method and rejected borrowed views. Reproduce it explicitly rather than
            // forwarding a view, which would widen the accepted input surface.
            fn broadcast_in_dim_read(&mut self, input: TensorRead<'_>, shape: &[usize], dims: &[usize]) -> tenferro_tensor::Result<Tensor> {
                                let _ = tenferro_tensor::backend::read_owned_tensor("broadcast_in_dim", input)?;
                    let _ = (shape, dims);
                    panic!("broadcast_in_dim should not be called in this test")
            }
        }

        impl TensorReduction for $ty {
            unreachable_backend_methods! {
                reduce_prod(input: &Tensor, axes: &[usize]) -> tenferro_tensor::Result<Tensor>;
            }

            // The previous read-half default delegated owned tensors to the
            // one-shot method and rejected borrowed views.
            fn reduce_sum_read(
                &mut self,
                input: tenferro_tensor::TensorRead<'_>,
                axes: &[usize],
            ) -> tenferro_tensor::Result<Tensor> {
                                let _ = tenferro_tensor::backend::read_owned_tensor("reduce_sum", input)?;
                    let _ = axes;
                    panic!("reduce_sum should not be called in this test")
            }

            // The previous read-half default delegated owned tensors to the
            // one-shot method and rejected borrowed views.
            fn reduce_prod_read(
                &mut self,
                input: tenferro_tensor::TensorRead<'_>,
                axes: &[usize],
            ) -> tenferro_tensor::Result<Tensor> {
                self.reduce_prod(
                    tenferro_tensor::backend::read_owned_tensor("reduce_prod", input)?,
                    axes,
                )
            }

            // The previous read-half default delegated owned tensors to the
            // one-shot method and rejected borrowed views.
            fn reduce_max_read(
                &mut self,
                input: tenferro_tensor::TensorRead<'_>,
                axes: &[usize],
            ) -> tenferro_tensor::Result<Tensor> {
                                let _ = tenferro_tensor::backend::read_owned_tensor("reduce_max", input)?;
                    let _ = axes;
                    panic!("reduce_max should not be called in this test")
            }

            // The previous read-half default delegated owned tensors to the
            // one-shot method and rejected borrowed views.
            fn reduce_min_read(
                &mut self,
                input: tenferro_tensor::TensorRead<'_>,
                axes: &[usize],
            ) -> tenferro_tensor::Result<Tensor> {
                                let _ = tenferro_tensor::backend::read_owned_tensor("reduce_min", input)?;
                    let _ = axes;
                    panic!("reduce_min should not be called in this test")
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
        // The previous read-half default delegated an owned pair to the one-shot
        // method and materialized borrowed views through to_contiguous_read before
        // contracting. Reproduce that exactly rather than forwarding a view.
        fn dot_general_read(
            &mut self,
            lhs: tenferro_tensor::TensorRead<'_>,
            rhs: tenferro_tensor::TensorRead<'_>,
            config: &DotGeneralConfig,
        ) -> tenferro_tensor::Result<Tensor> {
            match (lhs.as_tensor(), rhs.as_tensor()) {
                (Some(lhs), Some(rhs)) => self.dot_general(lhs, rhs, config),
                _ => {
                    let lhs = self.to_contiguous_read(lhs)?;
                    let rhs = self.to_contiguous_read(rhs)?;
                    self.dot_general(&lhs, &rhs, config)
                }
            }
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
    Tensor::from_typed::<tenferro_tensor::Complex64>(
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
