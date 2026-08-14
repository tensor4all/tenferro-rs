use crate::{
    backend::{
        validate_dot_general_read_into, validate_grouped_gemm, GroupedGemmConfig, GroupedGemmJob,
    },
    BackendCachedDot, BackendRuntimeCache, BackendSession, BackendSessionHost, CompareDir,
    ContractionScalar, DType, DotGeneralAccumulation, DotGeneralConfig, GatherConfig, PadConfig,
    ScatterConfig, SliceConfig, Tensor, TensorAnalytic, TensorBackend, TensorBuffer,
    TensorDeviceTransfer, TensorDot, TensorElementwise, TensorFusion, TensorIndexing, TensorRead,
    TensorReduction, TensorScalar, TensorStructural, TensorView, TensorViewMut, TensorWrite,
    TypedTensor, TypedTensorView, TypedTensorViewMut,
};
use num_complex::{Complex32, Complex64};

#[doc(hidden)]
struct DefaultReadBackendSessionMarker;

pub(crate) struct DefaultReadBackend {
    calls: Vec<&'static str>,
    dot_result: Option<Tensor>,
    gather_indices: Option<Tensor>,
    gather_config: Option<GatherConfig>,
    reshape_shapes: Vec<Vec<usize>>,
    concatenate_axis: Option<usize>,
    structural_runtime_enabled: bool,
}

impl Default for DefaultReadBackend {
    fn default() -> Self {
        Self {
            calls: Vec::new(),
            dot_result: None,
            gather_indices: None,
            gather_config: None,
            reshape_shapes: Vec::new(),
            concatenate_axis: None,
            structural_runtime_enabled: true,
        }
    }
}

fn marker() -> Tensor {
    Tensor::from_vec_col_major(vec![1], vec![42.0_f64]).unwrap()
}

fn for_each_index_col_major(shape: &[usize], mut visit: impl FnMut(&[usize])) {
    if shape.contains(&0) {
        return;
    }
    let mut index = vec![0; shape.len()];
    loop {
        visit(&index);
        let Some(axis) = (0..shape.len()).find(|&axis| {
            index[axis] += 1;
            if index[axis] < shape[axis] {
                true
            } else {
                index[axis] = 0;
                false
            }
        }) else {
            break;
        };
        let _ = axis;
    }
}

fn materialize_host_view<T: TensorScalar>(view: TypedTensorView<'_, T>) -> crate::Result<Tensor> {
    let shape = view.shape().to_vec();
    let mut data = Vec::with_capacity(shape.iter().product());
    let mut error = None;
    for_each_index_col_major(&shape, |index| match view.get(index) {
        Some(value) => data.push(*value),
        None => {
            error = Some(crate::Error::backend_failure(
                "to_contiguous_read",
                "test backend could not read a host view element",
            ))
        }
    });
    if let Some(error) = error {
        return Err(error);
    }
    Tensor::from_vec_col_major(shape, data)
}

fn copy_host_view<T: TensorScalar>(
    src: &TypedTensor<T>,
    mut dst: TypedTensorViewMut<'_, T>,
) -> crate::Result<()> {
    let shape = src.shape().to_vec();
    let mut error = None;
    for_each_index_col_major(&shape, |index| {
        let value = src.get(index).cloned();
        match (value, dst.get_mut(index)) {
            (Ok(value), Some(slot)) => *slot = value,
            _ => {
                error = Some(crate::Error::backend_failure(
                    "copy_read_into",
                    "test backend could not copy a host view element",
                ));
            }
        }
    });
    error.map_or(Ok(()), Err)
}

impl TensorElementwise for DefaultReadBackend {
    fn add(&mut self, _lhs: &Tensor, _rhs: &Tensor) -> crate::Result<Tensor> {
        self.calls.push("add");
        Ok(marker())
    }

    fn sub(&mut self, _lhs: &Tensor, _rhs: &Tensor) -> crate::Result<Tensor> {
        self.calls.push("sub");
        Ok(marker())
    }

    fn mul(&mut self, _lhs: &Tensor, _rhs: &Tensor) -> crate::Result<Tensor> {
        self.calls.push("mul");
        Ok(marker())
    }

    fn neg(&mut self, _input: &Tensor) -> crate::Result<Tensor> {
        self.calls.push("neg");
        Ok(marker())
    }

    fn conj(&mut self, _input: &Tensor) -> crate::Result<Tensor> {
        self.calls.push("conj");
        Ok(marker())
    }

    fn div(&mut self, _lhs: &Tensor, _rhs: &Tensor) -> crate::Result<Tensor> {
        self.calls.push("div");
        Ok(marker())
    }

    fn abs(&mut self, _input: &Tensor) -> crate::Result<Tensor> {
        self.calls.push("abs");
        Ok(marker())
    }

    fn sign(&mut self, _input: &Tensor) -> crate::Result<Tensor> {
        self.calls.push("sign");
        Ok(marker())
    }

    fn maximum(&mut self, _lhs: &Tensor, _rhs: &Tensor) -> crate::Result<Tensor> {
        self.calls.push("maximum");
        Ok(marker())
    }

    fn minimum(&mut self, _lhs: &Tensor, _rhs: &Tensor) -> crate::Result<Tensor> {
        self.calls.push("minimum");
        Ok(marker())
    }

    fn compare(
        &mut self,
        _lhs: &Tensor,
        _rhs: &Tensor,
        _dir: &CompareDir,
    ) -> crate::Result<Tensor> {
        self.calls.push("compare");
        Ok(marker())
    }

    fn select(
        &mut self,
        _pred: &Tensor,
        _on_true: &Tensor,
        _on_false: &Tensor,
    ) -> crate::Result<Tensor> {
        self.calls.push("select");
        Ok(marker())
    }

    fn clamp(
        &mut self,
        _input: &Tensor,
        _lower: &Tensor,
        _upper: &Tensor,
    ) -> crate::Result<Tensor> {
        self.calls.push("clamp");
        Ok(marker())
    }
}

impl TensorAnalytic for DefaultReadBackend {
    fn exp(&mut self, _input: &Tensor) -> crate::Result<Tensor> {
        self.calls.push("exp");
        Ok(marker())
    }

    fn log(&mut self, _input: &Tensor) -> crate::Result<Tensor> {
        self.calls.push("log");
        Ok(marker())
    }

    fn sin(&mut self, _input: &Tensor) -> crate::Result<Tensor> {
        self.calls.push("sin");
        Ok(marker())
    }

    fn cos(&mut self, _input: &Tensor) -> crate::Result<Tensor> {
        self.calls.push("cos");
        Ok(marker())
    }

    fn tanh(&mut self, _input: &Tensor) -> crate::Result<Tensor> {
        self.calls.push("tanh");
        Ok(marker())
    }

    fn sqrt(&mut self, _input: &Tensor) -> crate::Result<Tensor> {
        self.calls.push("sqrt");
        Ok(marker())
    }

    fn rsqrt(&mut self, _input: &Tensor) -> crate::Result<Tensor> {
        self.calls.push("rsqrt");
        Ok(marker())
    }

    fn pow(&mut self, _lhs: &Tensor, _rhs: &Tensor) -> crate::Result<Tensor> {
        self.calls.push("pow");
        Ok(marker())
    }

    fn expm1(&mut self, _input: &Tensor) -> crate::Result<Tensor> {
        self.calls.push("expm1");
        Ok(marker())
    }

    fn log1p(&mut self, _input: &Tensor) -> crate::Result<Tensor> {
        self.calls.push("log1p");
        Ok(marker())
    }
}

impl TensorStructural for DefaultReadBackend {
    fn to_contiguous_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        if !self.structural_runtime_enabled {
            let TensorRead::Tensor(tensor) = input else {
                return Err(crate::Error::unsupported(
                    "to_contiguous_read",
                    "backend does not accept borrowed tensor views at this execution boundary",
                ));
            };
            if tensor.is_backend_buffer()
                || !matches!(
                    tensor.placement().memory_kind,
                    crate::MemoryKind::PinnedHost | crate::MemoryKind::UnpinnedHost
                )
            {
                return Err(crate::Error::runtime_state(
                    "to_contiguous_read",
                    "default materialization accepts only host-owned tensors; use the storage's owning backend",
                ));
            }
            return tensor.duplicate();
        }
        match input {
            TensorRead::Tensor(tensor) => tensor.duplicate(),
            TensorRead::View(TensorView::F32(view)) => materialize_host_view(view),
            TensorRead::View(TensorView::F64(view)) => materialize_host_view(view),
            TensorRead::View(TensorView::I32(view)) => materialize_host_view(view),
            TensorRead::View(TensorView::I64(view)) => materialize_host_view(view),
            TensorRead::View(TensorView::Bool(view)) => materialize_host_view(view),
            TensorRead::View(TensorView::C32(view)) => materialize_host_view(view),
            TensorRead::View(TensorView::C64(view)) => materialize_host_view(view),
        }
    }

    fn copy_read_into(&mut self, src: TensorRead<'_>, dst: TensorWrite<'_>) -> crate::Result<()> {
        if !self.structural_runtime_enabled {
            return Err(crate::Error::unsupported(
                "copy_read_into",
                "backend-owned runtime copy is unsupported by this backend",
            ));
        }
        if src.dtype() != dst.dtype() {
            return Err(crate::Error::validation(
                "copy_read_into",
                crate::ValidationError::DTypeMismatch {
                    expected: crate::core_dtype(dst.dtype()),
                    actual: crate::core_dtype(src.dtype()),
                },
            ));
        }
        if src.shape() != dst.shape() {
            return Err(crate::Error::validation(
                "copy_read_into",
                crate::ShapeMismatch::IncompatibleShapes {
                    lhs: src.shape().to_vec().into(),
                    rhs: dst.shape().to_vec().into(),
                }
                .into(),
            ));
        }
        let src = self.to_contiguous_read(src)?;
        macro_rules! copy_typed {
            ($src:expr, $dst:expr, $variant:ident) => {
                match $dst {
                    TensorWrite::Tensor(dst) => *dst = Tensor::$variant($src),
                    TensorWrite::View(TensorViewMut::$variant(dst)) => copy_host_view(&$src, dst)?,
                    _ => unreachable!("dtype was validated before copy dispatch"),
                }
            };
        }
        match src {
            Tensor::F32(src) => copy_typed!(src, dst, F32),
            Tensor::F64(src) => copy_typed!(src, dst, F64),
            Tensor::I32(src) => copy_typed!(src, dst, I32),
            Tensor::I64(src) => copy_typed!(src, dst, I64),
            Tensor::Bool(src) => copy_typed!(src, dst, Bool),
            Tensor::C32(src) => copy_typed!(src, dst, C32),
            Tensor::C64(src) => copy_typed!(src, dst, C64),
        }
        Ok(())
    }

    fn transpose(&mut self, _input: &Tensor, _perm: &[usize]) -> crate::Result<Tensor> {
        self.calls.push("transpose");
        Ok(marker())
    }

    fn reshape(&mut self, _input: &Tensor, _shape: &[usize]) -> crate::Result<Tensor> {
        self.calls.push("reshape");
        self.reshape_shapes.push(_shape.to_vec());
        Ok(marker())
    }

    fn broadcast_in_dim(
        &mut self,
        _input: &Tensor,
        _shape: &[usize],
        _dims: &[usize],
    ) -> crate::Result<Tensor> {
        self.calls.push("broadcast_in_dim");
        Ok(marker())
    }

    fn cast(&mut self, _input: &Tensor, _to: DType) -> crate::Result<Tensor> {
        self.calls.push("cast");
        Ok(marker())
    }

    fn extract_diagonal(
        &mut self,
        _input: &Tensor,
        _axis_a: usize,
        _axis_b: usize,
    ) -> crate::Result<Tensor> {
        self.calls.push("extract_diagonal");
        Ok(marker())
    }

    fn embed_diagonal(
        &mut self,
        _input: &Tensor,
        _axis_a: usize,
        _axis_b: usize,
    ) -> crate::Result<Tensor> {
        self.calls.push("embed_diagonal");
        Ok(marker())
    }

    fn tril(&mut self, _input: &Tensor, _k: i64) -> crate::Result<Tensor> {
        self.calls.push("tril");
        Ok(marker())
    }

    fn triu(&mut self, _input: &Tensor, _k: i64) -> crate::Result<Tensor> {
        self.calls.push("triu");
        Ok(marker())
    }
}

impl TensorReduction for DefaultReadBackend {
    fn reduce_sum(&mut self, _input: &Tensor, _axes: &[usize]) -> crate::Result<Tensor> {
        self.calls.push("reduce_sum");
        Ok(marker())
    }

    fn reduce_prod(&mut self, _input: &Tensor, _axes: &[usize]) -> crate::Result<Tensor> {
        self.calls.push("reduce_prod");
        Ok(marker())
    }

    fn reduce_max(&mut self, _input: &Tensor, _axes: &[usize]) -> crate::Result<Tensor> {
        self.calls.push("reduce_max");
        Ok(marker())
    }

    fn reduce_min(&mut self, _input: &Tensor, _axes: &[usize]) -> crate::Result<Tensor> {
        self.calls.push("reduce_min");
        Ok(marker())
    }
}

#[test]
fn reduce_sum_squares_default_requires_an_explicit_backend_override() {
    let input = Tensor::from_vec_col_major(vec![1], vec![3.0_f64]).unwrap();
    let mut backend = DefaultReadBackend::default();

    let error = backend
        .reduce_sum_squares_read(TensorRead::from_tensor(&input), &[0])
        .unwrap_err();

    assert!(matches!(
        error,
        crate::Error::Unsupported {
            op: "reduce_sum_squares",
            ..
        }
    ));
    assert!(backend.calls.is_empty());
}

impl TensorIndexing for DefaultReadBackend {
    fn gather(
        &mut self,
        operand: &Tensor,
        start_indices: &Tensor,
        config: &GatherConfig,
    ) -> crate::Result<Tensor> {
        let _ = operand;
        self.calls.push("gather");
        self.gather_indices = Some(start_indices.duplicate().unwrap());
        self.gather_config = Some(config.clone());
        Ok(marker())
    }

    fn scatter(
        &mut self,
        _operand: &Tensor,
        _scatter_indices: &Tensor,
        _updates: &Tensor,
        _config: &ScatterConfig,
    ) -> crate::Result<Tensor> {
        Ok(marker())
    }

    fn slice(&mut self, _input: &Tensor, _config: &SliceConfig) -> crate::Result<Tensor> {
        Ok(marker())
    }

    fn dynamic_slice(
        &mut self,
        _input: &Tensor,
        _starts: &Tensor,
        _slice_sizes: &[usize],
    ) -> crate::Result<Tensor> {
        Ok(marker())
    }

    fn dynamic_update_slice(
        &mut self,
        _operand: &Tensor,
        _update: &Tensor,
        _starts: &Tensor,
    ) -> crate::Result<Tensor> {
        Ok(marker())
    }

    fn pad(&mut self, _input: &Tensor, _config: &PadConfig) -> crate::Result<Tensor> {
        Ok(marker())
    }

    fn concatenate(&mut self, _inputs: &[&Tensor], axis: usize) -> crate::Result<Tensor> {
        self.calls.push("concatenate");
        self.concatenate_axis = Some(axis);
        Ok(marker())
    }

    fn reverse(&mut self, _input: &Tensor, _axes: &[usize]) -> crate::Result<Tensor> {
        Ok(marker())
    }
}

impl TensorDot for DefaultReadBackend {
    fn dot_general(
        &mut self,
        _lhs: &Tensor,
        _rhs: &Tensor,
        _config: &DotGeneralConfig,
    ) -> crate::Result<Tensor> {
        self.calls.push("dot_general");
        self.dot_result
            .as_ref()
            .map(Tensor::duplicate)
            .transpose()
            .map(|result| result.unwrap_or_else(marker))
    }
}

impl TensorFusion for DefaultReadBackend {}

impl TensorBuffer for DefaultReadBackend {}

impl TensorDeviceTransfer for DefaultReadBackend {
    fn download_to_host(&mut self, _tensor: TensorRead<'_>) -> crate::Result<Tensor> {
        Err(crate::Error::unsupported(
            "DefaultReadBackend::download_to_host",
            "test backend does not transfer tensors",
        ))
    }

    fn upload_host_tensor(&mut self, _tensor: TensorRead<'_>) -> crate::Result<Tensor> {
        Err(crate::Error::unsupported(
            "DefaultReadBackend::upload_host_tensor",
            "test backend does not transfer tensors",
        ))
    }
}

impl BackendRuntimeCache for DefaultReadBackend {
    type RuntimeCache = ();
}

impl BackendCachedDot for DefaultReadBackend {}

impl BackendSession for DefaultReadBackend {
    fn session_type_id(&self) -> std::any::TypeId {
        std::any::TypeId::of::<DefaultReadBackendSessionMarker>()
    }

    unsafe fn session_data_mut(&mut self) -> *mut () {
        self as *mut Self as *mut ()
    }
}

impl BackendSessionHost for DefaultReadBackend {}

impl TensorBackend for DefaultReadBackend {}

#[test]
fn default_read_methods_delegate_owned_tensors_and_reject_views() {
    let a = Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();
    let b = Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    let pred = Tensor::from_vec_col_major(vec![1], vec![true]).unwrap();
    let view_source = TypedTensor::<f64>::from_vec_col_major(vec![1], vec![3.0]).unwrap();
    let mut backend = DefaultReadBackend::default();

    backend
        .add_read(TensorRead::from_tensor(&a), TensorRead::from_tensor(&b))
        .unwrap();
    backend
        .mul_read(TensorRead::from_tensor(&a), TensorRead::from_tensor(&b))
        .unwrap();
    backend.neg_read(TensorRead::from_tensor(&a)).unwrap();
    backend.conj_read(TensorRead::from_tensor(&a)).unwrap();
    backend
        .div_read(TensorRead::from_tensor(&a), TensorRead::from_tensor(&b))
        .unwrap();
    backend.abs_read(TensorRead::from_tensor(&a)).unwrap();
    backend.sign_read(TensorRead::from_tensor(&a)).unwrap();
    backend
        .maximum_read(TensorRead::from_tensor(&a), TensorRead::from_tensor(&b))
        .unwrap();
    backend
        .minimum_read(TensorRead::from_tensor(&a), TensorRead::from_tensor(&b))
        .unwrap();
    backend
        .compare_read(
            TensorRead::from_tensor(&a),
            TensorRead::from_tensor(&b),
            &CompareDir::Lt,
        )
        .unwrap();
    backend
        .select_read(
            TensorRead::from_tensor(&pred),
            TensorRead::from_tensor(&a),
            TensorRead::from_tensor(&b),
        )
        .unwrap();
    backend
        .clamp_read(
            TensorRead::from_tensor(&a),
            TensorRead::from_tensor(&a),
            TensorRead::from_tensor(&b),
        )
        .unwrap();

    backend.exp_read(TensorRead::from_tensor(&a)).unwrap();
    backend.log_read(TensorRead::from_tensor(&a)).unwrap();
    backend.sin_read(TensorRead::from_tensor(&a)).unwrap();
    backend.cos_read(TensorRead::from_tensor(&a)).unwrap();
    backend.tanh_read(TensorRead::from_tensor(&a)).unwrap();
    backend.sqrt_read(TensorRead::from_tensor(&a)).unwrap();
    backend.rsqrt_read(TensorRead::from_tensor(&a)).unwrap();
    backend
        .pow_read(TensorRead::from_tensor(&a), TensorRead::from_tensor(&b))
        .unwrap();
    backend.expm1_read(TensorRead::from_tensor(&a)).unwrap();
    backend.log1p_read(TensorRead::from_tensor(&a)).unwrap();

    backend
        .reshape_read(TensorRead::from_tensor(&a), &[1])
        .unwrap();
    backend
        .broadcast_in_dim_read(TensorRead::from_tensor(&a), &[1], &[0])
        .unwrap();
    backend
        .reduce_sum_read(TensorRead::from_tensor(&a), &[0])
        .unwrap();
    backend
        .reduce_prod_read(TensorRead::from_tensor(&a), &[0])
        .unwrap();
    backend
        .reduce_max_read(TensorRead::from_tensor(&a), &[0])
        .unwrap();
    backend
        .reduce_min_read(TensorRead::from_tensor(&a), &[0])
        .unwrap();

    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![0],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    backend
        .dot_general_read(
            TensorRead::from_tensor(&a),
            TensorRead::from_tensor(&b),
            &config,
        )
        .unwrap();
    backend
        .dot_general_read(
            TensorRead::from_view(TensorView::F64(view_source.as_view())),
            TensorRead::from_tensor(&b),
            &config,
        )
        .unwrap();
    backend
        .dot_general_with_conj(&a, &b, &config, true, false)
        .unwrap();
    let mut cache = ();
    BackendCachedDot::dot_general_read_cached(
        &mut backend,
        &mut cache,
        Some(0),
        TensorRead::from_tensor(&a),
        TensorRead::from_tensor(&b),
        &config,
    )
    .unwrap();
    BackendCachedDot::dot_general_read_cached(
        &mut backend,
        &mut cache,
        Some(1),
        TensorRead::from_view(TensorView::F64(view_source.as_view())),
        TensorRead::from_tensor(&b),
        &config,
    )
    .unwrap();
    BackendCachedDot::dot_general_with_conj_read_cached(
        &mut backend,
        &mut cache,
        Some(2),
        TensorRead::from_tensor(&a),
        TensorRead::from_tensor(&b),
        &config,
        true,
        false,
    )
    .unwrap();

    let err = backend
        .add_read(
            TensorRead::from_view(TensorView::F64(view_source.as_view())),
            TensorRead::from_tensor(&b),
        )
        .unwrap_err();
    assert!(err.to_string().contains("borrowed tensor views"));

    assert!(backend.calls.contains(&"add"));
    assert!(backend.calls.contains(&"dot_general"));
}

#[test]
fn elementwise_into_defaults_overwrite_outputs_and_validate_output() {
    let a = Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();
    let b = Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    let mut backend = DefaultReadBackend::default();

    let mut out = Tensor::from_vec_col_major(vec![1], vec![0.0_f64]).unwrap();
    backend
        .add_into(&a, &b, TensorWrite::from_tensor(&mut out))
        .unwrap();
    assert_eq!(out.as_slice::<f64>().unwrap(), &[3.0]);

    let mut out = Tensor::from_vec_col_major(vec![1], vec![0.0_f64]).unwrap();
    backend
        .sub_into(&a, &b, TensorWrite::from_tensor(&mut out))
        .unwrap();
    assert_eq!(out.as_slice::<f64>().unwrap(), &[-1.0]);

    let mut out = Tensor::from_vec_col_major(vec![1], vec![0.0_f64]).unwrap();
    backend
        .mul_into(&a, &b, TensorWrite::from_tensor(&mut out))
        .unwrap();
    assert_eq!(out.as_slice::<f64>().unwrap(), &[2.0]);

    let mut out = Tensor::from_vec_col_major(vec![1], vec![0.0_f64]).unwrap();
    backend
        .neg_into(&a, TensorWrite::from_tensor(&mut out))
        .unwrap();
    assert_eq!(out.as_slice::<f64>().unwrap(), &[-1.0]);

    let mut out = Tensor::from_vec_col_major(vec![1], vec![0.0_f64]).unwrap();
    backend
        .conj_into(&a, TensorWrite::from_tensor(&mut out))
        .unwrap();
    assert_eq!(out.as_slice::<f64>().unwrap(), &[1.0]);

    let mut out = Tensor::from_vec_col_major(vec![1], vec![0.0_f64]).unwrap();
    backend
        .div_read_into(
            TensorRead::from_tensor(&a),
            TensorRead::from_tensor(&b),
            TensorWrite::from_tensor(&mut out),
        )
        .unwrap();
    assert_eq!(out.as_slice::<f64>().unwrap(), &[0.5]);
    assert!(
        backend.calls.is_empty(),
        "host elementwise into defaults must not allocate through the backend"
    );

    let mut wrong_shape = Tensor::from_vec_col_major(vec![2], vec![0.0_f64, 0.0]).unwrap();
    let err = backend
        .add_into(&a, &b, TensorWrite::from_tensor(&mut wrong_shape))
        .unwrap_err();
    assert!(matches!(
        err,
        crate::Error::Validation {
            source: crate::ValidationError::ShapeMismatch(_),
            ..
        }
    ));

    let mut wrong_dtype = Tensor::from_vec_col_major(vec![1], vec![0_i32]).unwrap();
    let err = backend
        .add_into(&a, &b, TensorWrite::from_tensor(&mut wrong_dtype))
        .unwrap_err();
    assert!(matches!(
        err,
        crate::Error::Validation {
            source: crate::ValidationError::DTypeMismatch { .. },
            ..
        }
    ));
}

#[test]
fn elementwise_into_accepts_independent_backend_destinations() {
    let placement = crate::Placement {
        memory_kind: crate::MemoryKind::Device,
        device: Some(crate::DeviceId {
            kind: crate::DeviceKind::Gpu(crate::GpuBackendKind::Cuda),
            ordinal: 0,
        }),
        cpu_affinity: None,
    };
    let lhs = Tensor::F64(
        TypedTensor::from_buffer_col_major(
            vec![1],
            crate::StorageBuffer::Backend(Box::new(
                crate::BackendStorageHandle::<f64>::new_with_len(17, 1),
            )),
            placement.clone(),
        )
        .unwrap(),
    );
    let rhs = Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    let mut out = Tensor::F64(
        TypedTensor::from_buffer_col_major(
            vec![1],
            crate::StorageBuffer::Backend(Box::new(
                crate::BackendStorageHandle::<f64>::new_with_len(18, 1),
            )),
            placement,
        )
        .unwrap(),
    );

    DefaultReadBackend::default()
        .add_into(&lhs, &rhs, TensorWrite::from_tensor(&mut out))
        .unwrap();
}

#[test]
fn elementwise_into_updates_caller_view_storage_after_handoff() {
    let lhs = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap();
    let mut storage = vec![-1.0_f64, 0.0, 0.0, -1.0];

    {
        let view = TypedTensorViewMut::from_slice(vec![2], vec![1], 1, &mut storage).unwrap();
        DefaultReadBackend::default()
            .add_into(&lhs, &rhs, TensorWrite::from_view(TensorViewMut::F64(view)))
            .unwrap();
    }

    assert_eq!(storage, vec![-1.0, 4.0, 6.0, -1.0]);
    storage[2] = 9.0;
    assert_eq!(storage, vec![-1.0, 4.0, 9.0, -1.0]);
}

#[test]
fn contraction_scalar_helpers_cover_supported_and_rejected_dtypes() {
    assert_eq!(ContractionScalar::F32(1.0).dtype(), DType::F32);
    assert_eq!(ContractionScalar::F64(1.0).dtype(), DType::F64);
    assert_eq!(
        ContractionScalar::C32(Complex32::new(1.0, 0.0)).dtype(),
        DType::C32
    );
    assert_eq!(
        ContractionScalar::C64(Complex64::new(1.0, 0.0)).dtype(),
        DType::C64
    );

    assert_eq!(
        ContractionScalar::one(DType::C32).unwrap(),
        ContractionScalar::C32(Complex32::new(1.0, 0.0))
    );
    assert_eq!(
        ContractionScalar::zero(DType::C64).unwrap(),
        ContractionScalar::C64(Complex64::new(0.0, 0.0))
    );
    assert!(ContractionScalar::one(DType::I64).is_err());
    assert!(ContractionScalar::zero(DType::Bool).is_err());

    let overwrite = DotGeneralAccumulation::overwrite(DType::F32).unwrap();
    assert_eq!(overwrite.alpha, ContractionScalar::F32(1.0));
    assert_eq!(overwrite.beta, ContractionScalar::F32(0.0));
    let add_to = DotGeneralAccumulation::add_to(DType::F64).unwrap();
    assert_eq!(add_to.alpha, ContractionScalar::F64(1.0));
    assert_eq!(add_to.beta, ContractionScalar::F64(1.0));
    let scaled =
        DotGeneralAccumulation::scaled(ContractionScalar::F64(0.5), ContractionScalar::F64(2.0))
            .unwrap();
    assert_eq!(scaled.alpha, ContractionScalar::F64(0.5));
    assert_eq!(scaled.beta, ContractionScalar::F64(2.0));
    assert!(DotGeneralAccumulation::scaled(
        ContractionScalar::F32(1.0),
        ContractionScalar::F64(1.0)
    )
    .is_err());
    assert!(DotGeneralAccumulation::overwrite(DType::I32).is_err());
}

fn vector_contract_config() -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: vec![0],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    }
}

#[test]
fn dot_general_read_into_dtype_error_reports_output_as_actual() {
    let lhs = Tensor::from_vec_col_major(vec![1], vec![1.0_f32]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![1], vec![2.0_f32]).unwrap();
    let mut out = Tensor::from_vec_col_major(vec![], vec![0.0_f64]).unwrap();
    let err = validate_dot_general_read_into(
        &TensorRead::from_tensor(&lhs),
        &TensorRead::from_tensor(&rhs),
        &vector_contract_config(),
        &TensorWrite::from_tensor(&mut out),
        "test_dot_general",
    )
    .unwrap_err();

    match err {
        crate::Error::Validation {
            source: crate::ValidationError::DTypeMismatch { expected, actual },
            ..
        } => {
            assert_eq!(expected, crate::core_dtype(DType::F32));
            assert_eq!(actual, crate::core_dtype(DType::F64));
        }
        other => panic!("expected dtype mismatch, got {other:?}"),
    }
}

#[test]
fn dot_general_accum_default_fallback_updates_tensor_and_view_outputs() {
    let lhs = Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    let config = vector_contract_config();
    let mut backend = DefaultReadBackend {
        dot_result: Some(Tensor::from_vec_col_major(vec![], vec![42.0_f64]).unwrap()),
        ..Default::default()
    };
    let accumulation = DotGeneralAccumulation {
        lhs_conj: true,
        rhs_conj: false,
        alpha: ContractionScalar::F64(2.0),
        beta: ContractionScalar::F64(0.5),
    };
    let mut out = Tensor::from_vec_col_major(vec![], vec![10.0_f64]).unwrap();

    backend
        .dot_general_read_into_accum(
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &config,
            accumulation,
            TensorWrite::from_tensor(&mut out),
        )
        .unwrap();

    assert_eq!(out.as_slice::<f64>().unwrap(), &[89.0]);
    assert!(backend.calls.contains(&"conj"));
    assert!(backend.calls.contains(&"dot_general"));

    let mut cache = ();
    let mut out_view = TypedTensor::<f64>::from_vec_col_major(vec![], vec![1.0]).unwrap();
    BackendCachedDot::dot_general_read_into_accum_cached(
        &mut backend,
        &mut cache,
        Some(7),
        TensorRead::from_tensor(&lhs),
        TensorRead::from_tensor(&rhs),
        &config,
        DotGeneralAccumulation {
            lhs_conj: false,
            rhs_conj: false,
            alpha: ContractionScalar::F64(1.0),
            beta: ContractionScalar::F64(1.0),
        },
        TensorWrite::from_view(TensorViewMut::F64(out_view.as_view_mut())),
    )
    .unwrap();

    assert_eq!(out_view.as_slice().unwrap(), &[43.0]);
}

#[test]
fn dot_general_accum_beta_zero_does_not_read_compact_output() {
    let dot = Tensor::from_vec_col_major([2], vec![2.0_f64, -3.0]).unwrap();
    let mut out = Tensor::from_vec_col_major([2], vec![f64::NAN, f64::NAN]).unwrap();
    let mut write = TensorWrite::from_tensor(&mut out);

    crate::backend::accumulate_dot_result_into(
        &dot,
        DotGeneralAccumulation {
            lhs_conj: false,
            rhs_conj: false,
            alpha: ContractionScalar::F64(2.0),
            beta: ContractionScalar::F64(0.0),
        },
        &mut write,
    )
    .unwrap();

    assert_eq!(out.as_slice::<f64>().unwrap(), &[4.0, -6.0]);
}

#[test]
fn dot_general_accum_keeps_strided_output_fallback() {
    let dot = Tensor::from_vec_col_major([2], vec![2.0_f64, 3.0]).unwrap();
    let mut storage = [10.0_f64, 99.0, 20.0];
    let view = TypedTensorViewMut::from_slice([2], [2], 0, &mut storage).unwrap();
    let mut write = TensorWrite::from_view(TensorViewMut::F64(view));

    crate::backend::accumulate_dot_result_into(
        &dot,
        DotGeneralAccumulation {
            lhs_conj: false,
            rhs_conj: false,
            alpha: ContractionScalar::F64(1.0),
            beta: ContractionScalar::F64(1.0),
        },
        &mut write,
    )
    .unwrap();

    assert_eq!(storage, [12.0, 99.0, 23.0]);
}

#[test]
fn dot_general_accum_default_fallback_covers_supported_scalar_dtypes() {
    let config = vector_contract_config();

    let lhs = Tensor::from_vec_col_major(vec![1], vec![1.0_f32]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![1], vec![2.0_f32]).unwrap();
    let mut out = Tensor::from_vec_col_major(vec![], vec![10.0_f32]).unwrap();
    let mut backend = DefaultReadBackend {
        dot_result: Some(Tensor::from_vec_col_major(vec![], vec![4.0_f32]).unwrap()),
        ..Default::default()
    };
    backend
        .dot_general_read_into_accum(
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &config,
            DotGeneralAccumulation {
                lhs_conj: false,
                rhs_conj: false,
                alpha: ContractionScalar::F32(2.0),
                beta: ContractionScalar::F32(0.5),
            },
            TensorWrite::from_tensor(&mut out),
        )
        .unwrap();
    assert_eq!(out.as_slice::<f32>().unwrap(), &[13.0]);

    let lhs = Tensor::from_vec_col_major(vec![1], vec![Complex32::new(1.0, 0.0)]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![1], vec![Complex32::new(2.0, 0.0)]).unwrap();
    let mut out = Tensor::from_vec_col_major(vec![], vec![Complex32::new(3.0, -1.0)]).unwrap();
    let mut backend = DefaultReadBackend {
        dot_result: Some(
            Tensor::from_vec_col_major(vec![], vec![Complex32::new(1.0, 2.0)]).unwrap(),
        ),
        ..Default::default()
    };
    backend
        .dot_general_read_into_accum(
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &config,
            DotGeneralAccumulation {
                lhs_conj: false,
                rhs_conj: false,
                alpha: ContractionScalar::C32(Complex32::new(2.0, 0.0)),
                beta: ContractionScalar::C32(Complex32::new(0.0, 1.0)),
            },
            TensorWrite::from_tensor(&mut out),
        )
        .unwrap();
    assert_eq!(
        out.as_slice::<Complex32>().unwrap(),
        &[Complex32::new(3.0, 7.0)]
    );

    let lhs = Tensor::from_vec_col_major(vec![1], vec![Complex64::new(1.0, 0.0)]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![1], vec![Complex64::new(2.0, 0.0)]).unwrap();
    let mut out = Tensor::from_vec_col_major(vec![], vec![Complex64::new(5.0, 0.0)]).unwrap();
    let mut backend = DefaultReadBackend {
        dot_result: Some(
            Tensor::from_vec_col_major(vec![], vec![Complex64::new(4.0, -2.0)]).unwrap(),
        ),
        ..Default::default()
    };
    backend
        .dot_general_read_into_accum(
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &config,
            DotGeneralAccumulation {
                lhs_conj: false,
                rhs_conj: false,
                alpha: ContractionScalar::C64(Complex64::new(1.0, 0.0)),
                beta: ContractionScalar::C64(Complex64::new(0.0, 0.0)),
            },
            TensorWrite::from_tensor(&mut out),
        )
        .unwrap();
    assert_eq!(
        out.as_slice::<Complex64>().unwrap(),
        &[Complex64::new(4.0, -2.0)]
    );
}

#[test]
fn dot_general_accum_default_fallback_rejects_scalar_dtype_mismatch() {
    let lhs = Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    let mut out = Tensor::from_vec_col_major(vec![], vec![0.0_f64]).unwrap();
    let config = vector_contract_config();
    let mut backend = DefaultReadBackend::default();

    let err = backend
        .dot_general_read_into_accum(
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &config,
            DotGeneralAccumulation {
                lhs_conj: false,
                rhs_conj: false,
                alpha: ContractionScalar::F32(1.0),
                beta: ContractionScalar::F64(0.0),
            },
            TensorWrite::from_tensor(&mut out),
        )
        .unwrap_err();

    assert!(err.to_string().contains("dtype mismatch"));
}

fn scalar_grouped_config<'a>(jobs: &'a [GroupedGemmJob], dtype: DType) -> GroupedGemmConfig<'a> {
    GroupedGemmConfig::new(jobs, DotGeneralAccumulation::overwrite(dtype).unwrap())
}

#[test]
fn grouped_gemm_job_and_config_metadata_are_preserved() {
    let job = GroupedGemmJob::new(1, 2, 3, 4, 5, 6);

    assert_eq!(job.out_offset(), 1);
    assert_eq!(job.lhs_offset(), 2);
    assert_eq!(job.rhs_offset(), 3);
    assert_eq!(job.rows(), 4);
    assert_eq!(job.contracted(), 5);
    assert_eq!(job.cols(), 6);

    let jobs = [job];
    let accumulation = DotGeneralAccumulation {
        lhs_conj: false,
        rhs_conj: true,
        alpha: ContractionScalar::F64(2.0),
        beta: ContractionScalar::F64(3.0),
    };
    let config = GroupedGemmConfig::new(&jobs, accumulation);

    assert_eq!(config.jobs(), &jobs);
    assert_eq!(config.accumulation(), accumulation);
}

fn run_grouped_f64_default_combo(
    lhs_as_view: bool,
    rhs_as_view: bool,
    out_as_view: bool,
) -> Vec<f64> {
    let lhs_tensor = Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();
    let rhs_tensor = Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    let lhs_storage = [0.0_f64, 1.0];
    let rhs_storage = [0.0_f64, 2.0];
    let lhs_view =
        TypedTensorView::from_slice(vec![1], vec![1], 1, lhs_storage.as_slice()).unwrap();
    let rhs_view =
        TypedTensorView::from_slice(vec![1], vec![1], 1, rhs_storage.as_slice()).unwrap();

    let lhs_read = if lhs_as_view {
        TensorRead::from_view(TensorView::F64(lhs_view))
    } else {
        TensorRead::from_tensor(&lhs_tensor)
    };
    let rhs_read = if rhs_as_view {
        TensorRead::from_view(TensorView::F64(rhs_view))
    } else {
        TensorRead::from_tensor(&rhs_tensor)
    };

    let jobs = [GroupedGemmJob::new(0, 0, 0, 1, 1, 1)];
    let config = scalar_grouped_config(&jobs, DType::F64);
    let mut cache = ();
    let mut backend = DefaultReadBackend {
        dot_result: Some(Tensor::from_vec_col_major(vec![1, 1], vec![4.0_f64]).unwrap()),
        ..Default::default()
    };

    if out_as_view {
        let mut out_storage = vec![100.0_f64, 9.0];
        {
            let out_view =
                TypedTensorViewMut::from_slice(vec![1], vec![1], 1, out_storage.as_mut_slice())
                    .unwrap();
            BackendCachedDot::grouped_gemm_cached(
                &mut backend,
                &mut cache,
                Some(3),
                lhs_read,
                rhs_read,
                &config,
                TensorWrite::from_view(TensorViewMut::F64(out_view)),
            )
            .unwrap();
        }
        out_storage
    } else {
        let mut out = Tensor::from_vec_col_major(vec![1], vec![9.0_f64]).unwrap();
        BackendCachedDot::grouped_gemm_cached(
            &mut backend,
            &mut cache,
            Some(3),
            lhs_read,
            rhs_read,
            &config,
            TensorWrite::from_tensor(&mut out),
        )
        .unwrap();
        out.as_slice::<f64>().unwrap().to_vec()
    }
}

#[test]
fn grouped_gemm_default_fallback_covers_tensor_and_view_dispatch() {
    for lhs_as_view in [false, true] {
        for rhs_as_view in [false, true] {
            for out_as_view in [false, true] {
                let values = run_grouped_f64_default_combo(lhs_as_view, rhs_as_view, out_as_view);
                if out_as_view {
                    assert_eq!(values, vec![100.0, 4.0]);
                } else {
                    assert_eq!(values, vec![4.0]);
                }
            }
        }
    }
}

#[test]
fn grouped_gemm_default_fallback_updates_shared_buffer_offsets() {
    let lhs = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![3], vec![4.0_f64, 5.0, 6.0]).unwrap();
    let mut out = Tensor::from_vec_col_major(vec![4], vec![10.0_f64, 20.0, 30.0, 40.0]).unwrap();
    let jobs = [
        GroupedGemmJob::new(0, 0, 0, 1, 1, 1),
        GroupedGemmJob::new(2, 2, 2, 1, 1, 1),
    ];
    let config = GroupedGemmConfig::new(
        &jobs,
        DotGeneralAccumulation {
            lhs_conj: false,
            rhs_conj: false,
            alpha: ContractionScalar::F64(2.0),
            beta: ContractionScalar::F64(0.5),
        },
    );
    let mut cache = ();
    let mut backend = DefaultReadBackend {
        dot_result: Some(Tensor::from_vec_col_major(vec![1, 1], vec![4.0_f64]).unwrap()),
        ..Default::default()
    };

    BackendCachedDot::grouped_gemm_cached(
        &mut backend,
        &mut cache,
        Some(9),
        TensorRead::from_tensor(&lhs),
        TensorRead::from_tensor(&rhs),
        &config,
        TensorWrite::from_tensor(&mut out),
    )
    .unwrap();

    assert_eq!(out.as_slice::<f64>().unwrap(), &[13.0, 20.0, 23.0, 40.0]);
    assert_eq!(
        backend
            .calls
            .iter()
            .filter(|&&call| call == "dot_general")
            .count(),
        2
    );
}

#[test]
fn grouped_gemm_default_fallback_covers_supported_dtypes() {
    let jobs = [GroupedGemmJob::new(0, 0, 0, 1, 1, 1)];
    let mut cache = ();

    let lhs = Tensor::from_vec_col_major(vec![1], vec![1.0_f32]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![1], vec![2.0_f32]).unwrap();
    let mut out = Tensor::from_vec_col_major(vec![1], vec![5.0_f32]).unwrap();
    let mut backend = DefaultReadBackend {
        dot_result: Some(Tensor::from_vec_col_major(vec![1, 1], vec![2.0_f32]).unwrap()),
        ..Default::default()
    };
    BackendCachedDot::grouped_gemm_cached(
        &mut backend,
        &mut cache,
        None,
        TensorRead::from_tensor(&lhs),
        TensorRead::from_tensor(&rhs),
        &GroupedGemmConfig::new(
            &jobs,
            DotGeneralAccumulation {
                lhs_conj: false,
                rhs_conj: false,
                alpha: ContractionScalar::F32(3.0),
                beta: ContractionScalar::F32(1.0),
            },
        ),
        TensorWrite::from_tensor(&mut out),
    )
    .unwrap();
    assert_eq!(out.as_slice::<f32>().unwrap(), &[11.0]);

    let lhs = Tensor::from_vec_col_major(vec![1], vec![Complex32::new(1.0, 0.0)]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![1], vec![Complex32::new(2.0, 0.0)]).unwrap();
    let mut out = Tensor::from_vec_col_major(vec![1], vec![Complex32::new(3.0, -1.0)]).unwrap();
    let mut backend = DefaultReadBackend {
        dot_result: Some(
            Tensor::from_vec_col_major(vec![1, 1], vec![Complex32::new(1.0, 1.0)]).unwrap(),
        ),
        ..Default::default()
    };
    BackendCachedDot::grouped_gemm_cached(
        &mut backend,
        &mut cache,
        None,
        TensorRead::from_tensor(&lhs),
        TensorRead::from_tensor(&rhs),
        &GroupedGemmConfig::new(
            &jobs,
            DotGeneralAccumulation {
                lhs_conj: false,
                rhs_conj: false,
                alpha: ContractionScalar::C32(Complex32::new(2.0, 0.0)),
                beta: ContractionScalar::C32(Complex32::new(0.0, 1.0)),
            },
        ),
        TensorWrite::from_tensor(&mut out),
    )
    .unwrap();
    assert_eq!(
        out.as_slice::<Complex32>().unwrap(),
        &[Complex32::new(3.0, 5.0)]
    );

    let lhs = Tensor::from_vec_col_major(vec![1], vec![Complex64::new(1.0, 0.0)]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![1], vec![Complex64::new(2.0, 0.0)]).unwrap();
    let mut out = Tensor::from_vec_col_major(vec![1], vec![Complex64::new(0.0, 0.0)]).unwrap();
    let mut backend = DefaultReadBackend {
        dot_result: Some(
            Tensor::from_vec_col_major(vec![1, 1], vec![Complex64::new(4.0, -2.0)]).unwrap(),
        ),
        ..Default::default()
    };
    BackendCachedDot::grouped_gemm_cached(
        &mut backend,
        &mut cache,
        None,
        TensorRead::from_tensor(&lhs),
        TensorRead::from_tensor(&rhs),
        &GroupedGemmConfig::new(
            &jobs,
            DotGeneralAccumulation {
                lhs_conj: false,
                rhs_conj: false,
                alpha: ContractionScalar::C64(Complex64::new(1.0, 0.0)),
                beta: ContractionScalar::C64(Complex64::new(0.0, 0.0)),
            },
        ),
        TensorWrite::from_tensor(&mut out),
    )
    .unwrap();
    assert_eq!(
        out.as_slice::<Complex64>().unwrap(),
        &[Complex64::new(4.0, -2.0)]
    );
}

#[test]
fn grouped_gemm_validation_rejects_invalid_metadata() {
    let lhs = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap();
    let out = Tensor::from_vec_col_major(vec![2], vec![0.0_f64, 0.0]).unwrap();
    let mut out_mut = out.duplicate().unwrap();
    let jobs = [GroupedGemmJob::new(0, 0, 0, 1, 1, 1)];

    let rhs_f32 = Tensor::from_vec_col_major(vec![2], vec![3.0_f32, 4.0]).unwrap();
    let err = validate_grouped_gemm(
        &TensorRead::from_tensor(&lhs),
        &TensorRead::from_tensor(&rhs_f32),
        &TensorWrite::from_tensor(&mut out_mut),
        &scalar_grouped_config(&jobs, DType::F64),
        "test_grouped_gemm",
    )
    .unwrap_err();
    assert!(err.to_string().contains("dtype mismatch"));

    let mut out_f32 = Tensor::from_vec_col_major(vec![2], vec![0.0_f32, 0.0]).unwrap();
    let err = validate_grouped_gemm(
        &TensorRead::from_tensor(&lhs),
        &TensorRead::from_tensor(&rhs),
        &TensorWrite::from_tensor(&mut out_f32),
        &scalar_grouped_config(&jobs, DType::F64),
        "test_grouped_gemm",
    )
    .unwrap_err();
    assert!(err.to_string().contains("dtype mismatch"));

    let err = validate_grouped_gemm(
        &TensorRead::from_tensor(&lhs),
        &TensorRead::from_tensor(&rhs),
        &TensorWrite::from_tensor(&mut out_mut),
        &GroupedGemmConfig::new(
            &jobs,
            DotGeneralAccumulation {
                lhs_conj: false,
                rhs_conj: false,
                alpha: ContractionScalar::F32(1.0),
                beta: ContractionScalar::F64(0.0),
            },
        ),
        "test_grouped_gemm",
    )
    .unwrap_err();
    assert!(err.to_string().contains("dtype mismatch"));

    let lhs_range_jobs = [GroupedGemmJob::new(0, 1, 0, 2, 1, 1)];
    let err = validate_grouped_gemm(
        &TensorRead::from_tensor(&lhs),
        &TensorRead::from_tensor(&rhs),
        &TensorWrite::from_tensor(&mut out_mut),
        &scalar_grouped_config(&lhs_range_jobs, DType::F64),
        "test_grouped_gemm",
    )
    .unwrap_err();
    assert!(matches!(
        err,
        crate::Error::Validation {
            source: crate::ValidationError::InvalidArgument {
                argument: "lhs",
                ..
            },
            ..
        }
    ));

    let rhs_range_jobs = [GroupedGemmJob::new(0, 0, 1, 1, 1, 2)];
    let err = validate_grouped_gemm(
        &TensorRead::from_tensor(&lhs),
        &TensorRead::from_tensor(&rhs),
        &TensorWrite::from_tensor(&mut out_mut),
        &scalar_grouped_config(&rhs_range_jobs, DType::F64),
        "test_grouped_gemm",
    )
    .unwrap_err();
    assert!(matches!(
        err,
        crate::Error::Validation {
            source: crate::ValidationError::InvalidArgument {
                argument: "rhs",
                ..
            },
            ..
        }
    ));

    let out_range_jobs = [GroupedGemmJob::new(1, 0, 0, 1, 1, 2)];
    let err = validate_grouped_gemm(
        &TensorRead::from_tensor(&lhs),
        &TensorRead::from_tensor(&rhs),
        &TensorWrite::from_tensor(&mut out_mut),
        &scalar_grouped_config(&out_range_jobs, DType::F64),
        "test_grouped_gemm",
    )
    .unwrap_err();
    assert!(matches!(
        err,
        crate::Error::Validation {
            source: crate::ValidationError::InvalidArgument {
                argument: "out",
                ..
            },
            ..
        }
    ));

    let overlapping_jobs = [
        GroupedGemmJob::new(0, 0, 0, 1, 1, 1),
        GroupedGemmJob::new(0, 1, 1, 1, 1, 1),
    ];
    let err = validate_grouped_gemm(
        &TensorRead::from_tensor(&lhs),
        &TensorRead::from_tensor(&rhs),
        &TensorWrite::from_tensor(&mut out_mut),
        &scalar_grouped_config(&overlapping_jobs, DType::F64),
        "test_grouped_gemm",
    )
    .unwrap_err();
    assert!(matches!(
        err,
        crate::Error::Validation {
            source: crate::ValidationError::InvalidArgument {
                argument: "jobs",
                ..
            },
            ..
        }
    ));

    let overflow_jobs = [GroupedGemmJob::new(0, 0, 0, usize::MAX, 0, 2)];
    let err = validate_grouped_gemm(
        &TensorRead::from_tensor(&lhs),
        &TensorRead::from_tensor(&rhs),
        &TensorWrite::from_tensor(&mut out_mut),
        &scalar_grouped_config(&overflow_jobs, DType::F64),
        "test_grouped_gemm",
    )
    .unwrap_err();
    assert!(matches!(
        err,
        crate::Error::Validation {
            source: crate::ValidationError::InvalidArgument { .. },
            ..
        }
    ));

    let empty_jobs = [GroupedGemmJob::new(
        usize::MAX,
        usize::MAX,
        usize::MAX,
        0,
        0,
        2,
    )];
    validate_grouped_gemm(
        &TensorRead::from_tensor(&lhs),
        &TensorRead::from_tensor(&rhs),
        &TensorWrite::from_tensor(&mut out_mut),
        &scalar_grouped_config(&empty_jobs, DType::F64),
        "test_grouped_gemm",
    )
    .unwrap();
}

#[test]
fn grouped_gemm_default_fallback_rejects_offsets_that_do_not_fit_isize() {
    let lhs = Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    let mut out = Tensor::from_vec_col_major(vec![1], vec![0.0_f64]).unwrap();
    let jobs = [GroupedGemmJob::new(0, usize::MAX, 0, 0, 0, 0)];
    let config = scalar_grouped_config(&jobs, DType::F64);
    let mut cache = ();
    let mut backend = DefaultReadBackend {
        dot_result: Some(Tensor::from_vec_col_major(vec![0, 0], Vec::<f64>::new()).unwrap()),
        ..Default::default()
    };

    let err = BackendCachedDot::grouped_gemm_cached(
        &mut backend,
        &mut cache,
        None,
        TensorRead::from_tensor(&lhs),
        TensorRead::from_tensor(&rhs),
        &config,
        TensorWrite::from_tensor(&mut out),
    )
    .unwrap_err();

    assert!(err.to_string().contains("offset"));
}

#[test]
fn tensor_index_select_builds_gather_config_and_validates_inputs() {
    let input = Tensor::from_vec_col_major(vec![2, 3], vec![0.0_f64; 6]).unwrap();
    let mut backend = DefaultReadBackend::default();

    backend
        .with_backend_session(|session| input.index_select(-1, &[2, 0], session))
        .unwrap();

    let indices = backend.gather_indices.as_ref().unwrap();
    assert_eq!(indices.shape(), &[2, 1]);
    assert_eq!(indices.as_slice::<i64>().unwrap(), &[2, 0]);

    let config = backend.gather_config.as_ref().unwrap();
    assert_eq!(config.offset_dims, vec![0]);
    assert_eq!(config.collapsed_slice_dims, vec![1]);
    assert_eq!(config.start_index_map, vec![1]);
    assert_eq!(config.index_vector_dim, 1);
    assert_eq!(config.slice_sizes, vec![2, 1]);

    let axis_err = backend
        .with_backend_session(|session| input.index_select(2, &[0], session))
        .unwrap_err();
    assert!(axis_err.to_string().contains("axis 2"));

    let position_err = backend
        .with_backend_session(|session| input.index_select(1, &[3], session))
        .unwrap_err();
    assert!(position_err
        .to_string()
        .contains("position 3 out of bounds"));
}

#[test]
fn tensor_stack_reshapes_then_concatenates_and_validates_inputs() {
    let a = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let b = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap();
    let mut backend = DefaultReadBackend::default();

    backend
        .with_backend_session(|session| Tensor::stack(&[&a, &b], -1, session))
        .unwrap();

    assert_eq!(backend.reshape_shapes, vec![vec![2, 1], vec![2, 1]]);
    assert_eq!(backend.concatenate_axis, Some(1));

    let empty: [&Tensor; 0] = [];
    let empty_err = backend
        .with_backend_session(|session| Tensor::stack(&empty, 0, session))
        .unwrap_err();
    assert!(empty_err.to_string().contains("at least one input"));

    let c = Tensor::from_vec_col_major(vec![3], vec![0.0_f64; 3]).unwrap();
    let shape_err = backend
        .with_backend_session(|session| Tensor::stack(&[&a, &c], 0, session))
        .unwrap_err();
    assert!(matches!(
        shape_err,
        crate::Error::Validation {
            source: crate::ValidationError::ShapeMismatch(_),
            ..
        }
    ));

    let axis_err = backend
        .with_backend_session(|session| Tensor::stack(&[&a], 2, session))
        .unwrap_err();
    assert!(axis_err.to_string().contains("axis 2"));
}

#[test]
fn structural_runtime_materialization_is_object_safe_and_clones_owned_input_by_default() {
    let input = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let mut backend = DefaultReadBackend {
        structural_runtime_enabled: false,
        ..Default::default()
    };
    let session: &mut dyn BackendSession = &mut backend;

    let output = session
        .to_contiguous_read(TensorRead::from_tensor(&input))
        .unwrap();

    assert_eq!(output.as_slice::<f64>().unwrap(), &[1.0, 2.0]);
    assert_eq!(input.as_slice::<f64>().unwrap(), &[1.0, 2.0]);
}

#[test]
fn structural_runtime_materialization_rejects_views_by_default() {
    let data = [1.0_f64, 2.0];
    let view = TensorView::f64(&[2], &data).unwrap();
    let mut backend = DefaultReadBackend {
        structural_runtime_enabled: false,
        ..Default::default()
    };
    let session: &mut dyn BackendSession = &mut backend;

    let err = session
        .to_contiguous_read(TensorRead::from_view(view))
        .unwrap_err();

    assert!(matches!(
        err,
        crate::Error::Unsupported {
            op: "to_contiguous_read",
            ref message,
        } if message.contains("borrowed tensor views")
    ));
}

#[test]
fn structural_runtime_materialization_rejects_foreign_backend_storage_by_default() {
    let input = Tensor::F64(
        TypedTensor::from_buffer_col_major(
            vec![2],
            crate::StorageBuffer::Backend(Box::new(
                crate::BackendStorageHandle::<f64>::new_with_len(41, 2),
            )),
            crate::Placement {
                memory_kind: crate::MemoryKind::Device,
                device: Some(crate::DeviceId {
                    kind: crate::DeviceKind::Gpu(crate::GpuBackendKind::Cuda),
                    ordinal: 0,
                }),
                cpu_affinity: None,
            },
        )
        .unwrap(),
    );
    let mut backend = DefaultReadBackend {
        structural_runtime_enabled: false,
        ..Default::default()
    };
    let session: &mut dyn BackendSession = &mut backend;

    let err = session
        .to_contiguous_read(TensorRead::from_tensor(&input))
        .unwrap_err();

    assert!(matches!(
        err,
        crate::Error::RuntimeState {
            op: "to_contiguous_read",
            ref message,
        } if message.contains("host-owned") && message.contains("owning backend")
    ));
}

#[test]
fn structural_runtime_copy_is_explicitly_unsupported_by_default() {
    let src = Tensor::from_vec_col_major(vec![2], vec![1_i32, 2]).unwrap();
    let mut dst = Tensor::from_vec_col_major(vec![2], vec![0_i32, 0]).unwrap();
    let mut backend = DefaultReadBackend {
        structural_runtime_enabled: false,
        ..Default::default()
    };
    let session: &mut dyn BackendSession = &mut backend;

    let err = session
        .copy_read_into(
            TensorRead::from_tensor(&src),
            TensorWrite::from_tensor(&mut dst),
        )
        .unwrap_err();

    assert!(matches!(
        err,
        crate::Error::Unsupported {
            op: "copy_read_into",
            ref message,
        } if message.contains("unsupported")
    ));
    assert_eq!(dst.as_slice::<i32>().unwrap(), &[0, 0]);
}
