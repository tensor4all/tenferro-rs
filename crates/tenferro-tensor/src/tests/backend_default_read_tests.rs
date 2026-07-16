use crate::{
    backend::{validate_grouped_gemm, GroupedGemmConfig, GroupedGemmJob},
    BackendCachedDot, BackendRuntimeCache, BackendSession, BackendSessionHost, CompareDir,
    ContractionScalar, DType, DotGeneralAccumulation, DotGeneralConfig, GatherConfig, PadConfig,
    ScatterConfig, SliceConfig, Tensor, TensorAnalytic, TensorBackend, TensorBuffer,
    TensorDeviceTransfer, TensorDot, TensorElementwise, TensorFusion, TensorIndexing, TensorRead,
    TensorReduction, TensorStructural, TensorView, TensorViewMut, TensorWrite, TypedTensor,
    TypedTensorView, TypedTensorViewMut,
};
use num_complex::{Complex32, Complex64};

#[derive(Default)]
struct DefaultReadBackend {
    calls: Vec<&'static str>,
    dot_result: Option<Tensor>,
    gather_indices: Option<Tensor>,
    gather_config: Option<GatherConfig>,
    reshape_shapes: Vec<Vec<usize>>,
    concatenate_axis: Option<usize>,
}

fn marker() -> Tensor {
    Tensor::from_vec_col_major(vec![1], vec![42.0_f64]).unwrap()
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

impl TensorIndexing for DefaultReadBackend {
    fn gather(
        &mut self,
        operand: &Tensor,
        start_indices: &Tensor,
        config: &GatherConfig,
    ) -> crate::Result<Tensor> {
        let _ = operand;
        self.calls.push("gather");
        self.gather_indices = Some(start_indices.clone());
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
        Ok(self.dot_result.clone().unwrap_or_else(marker))
    }
}

impl TensorFusion for DefaultReadBackend {}

impl TensorBuffer for DefaultReadBackend {}

impl TensorDeviceTransfer for DefaultReadBackend {}

impl BackendRuntimeCache for DefaultReadBackend {
    type RuntimeCache = ();
}

impl BackendCachedDot for DefaultReadBackend {}

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
    assert_eq!(out.as_slice::<f64>().unwrap(), &[42.0]);

    let mut out = Tensor::from_vec_col_major(vec![1], vec![0.0_f64]).unwrap();
    backend
        .sub_into(&a, &b, TensorWrite::from_tensor(&mut out))
        .unwrap();
    assert_eq!(out.as_slice::<f64>().unwrap(), &[42.0]);

    let mut out = Tensor::from_vec_col_major(vec![1], vec![0.0_f64]).unwrap();
    backend
        .mul_into(&a, &b, TensorWrite::from_tensor(&mut out))
        .unwrap();
    assert_eq!(out.as_slice::<f64>().unwrap(), &[42.0]);

    let mut out = Tensor::from_vec_col_major(vec![1], vec![0.0_f64]).unwrap();
    backend
        .neg_into(&a, TensorWrite::from_tensor(&mut out))
        .unwrap();
    assert_eq!(out.as_slice::<f64>().unwrap(), &[42.0]);

    let mut out = Tensor::from_vec_col_major(vec![1], vec![0.0_f64]).unwrap();
    backend
        .conj_into(&a, TensorWrite::from_tensor(&mut out))
        .unwrap();
    assert_eq!(out.as_slice::<f64>().unwrap(), &[42.0]);

    let mut out = Tensor::from_vec_col_major(vec![1], vec![0.0_f64]).unwrap();
    backend
        .div_read_into(
            TensorRead::from_tensor(&a),
            TensorRead::from_tensor(&b),
            TensorWrite::from_tensor(&mut out),
        )
        .unwrap();
    assert_eq!(out.as_slice::<f64>().unwrap(), &[42.0]);

    let mut wrong_shape = Tensor::from_vec_col_major(vec![2], vec![0.0_f64, 0.0]).unwrap();
    let err = backend
        .add_into(&a, &b, TensorWrite::from_tensor(&mut wrong_shape))
        .unwrap_err();
    assert!(matches!(err, crate::Error::ShapeMismatch { .. }));

    let mut wrong_dtype = Tensor::from_vec_col_major(vec![1], vec![0_i32]).unwrap();
    let err = backend
        .add_into(&a, &b, TensorWrite::from_tensor(&mut wrong_dtype))
        .unwrap_err();
    assert!(matches!(err, crate::Error::DTypeMismatch { .. }));
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
    let mut out_mut = out.clone();
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
    assert!(err.to_string().contains("lhs matrix range"));

    let rhs_range_jobs = [GroupedGemmJob::new(0, 0, 1, 1, 1, 2)];
    let err = validate_grouped_gemm(
        &TensorRead::from_tensor(&lhs),
        &TensorRead::from_tensor(&rhs),
        &TensorWrite::from_tensor(&mut out_mut),
        &scalar_grouped_config(&rhs_range_jobs, DType::F64),
        "test_grouped_gemm",
    )
    .unwrap_err();
    assert!(err.to_string().contains("rhs matrix range"));

    let out_range_jobs = [GroupedGemmJob::new(1, 0, 0, 1, 1, 2)];
    let err = validate_grouped_gemm(
        &TensorRead::from_tensor(&lhs),
        &TensorRead::from_tensor(&rhs),
        &TensorWrite::from_tensor(&mut out_mut),
        &scalar_grouped_config(&out_range_jobs, DType::F64),
        "test_grouped_gemm",
    )
    .unwrap_err();
    assert!(err.to_string().contains("out matrix range"));

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
    assert!(err.to_string().contains("overlaps job"));

    let overflow_jobs = [GroupedGemmJob::new(0, 0, 0, usize::MAX, 0, 2)];
    let err = validate_grouped_gemm(
        &TensorRead::from_tensor(&lhs),
        &TensorRead::from_tensor(&rhs),
        &TensorWrite::from_tensor(&mut out_mut),
        &scalar_grouped_config(&overflow_jobs, DType::F64),
        "test_grouped_gemm",
    )
    .unwrap_err();
    assert!(err.to_string().contains("element count overflows"));

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

    input.index_select(-1, &[2, 0], &mut backend).unwrap();

    let indices = backend.gather_indices.as_ref().unwrap();
    assert_eq!(indices.shape(), &[2, 1]);
    assert_eq!(indices.as_slice::<i64>().unwrap(), &[2, 0]);

    let config = backend.gather_config.as_ref().unwrap();
    assert_eq!(config.offset_dims, vec![0]);
    assert_eq!(config.collapsed_slice_dims, vec![1]);
    assert_eq!(config.start_index_map, vec![1]);
    assert_eq!(config.index_vector_dim, 1);
    assert_eq!(config.slice_sizes, vec![2, 1]);

    let axis_err = input.index_select(2, &[0], &mut backend).unwrap_err();
    assert!(axis_err.to_string().contains("axis 2"));

    let position_err = input.index_select(1, &[3], &mut backend).unwrap_err();
    assert!(position_err
        .to_string()
        .contains("position 3 out of bounds"));
}

#[test]
fn tensor_stack_reshapes_then_concatenates_and_validates_inputs() {
    let a = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let b = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap();
    let mut backend = DefaultReadBackend::default();

    Tensor::stack(&[&a, &b], -1, &mut backend).unwrap();

    assert_eq!(backend.reshape_shapes, vec![vec![2, 1], vec![2, 1]]);
    assert_eq!(backend.concatenate_axis, Some(1));

    let empty: [&Tensor; 0] = [];
    let empty_err = Tensor::stack(&empty, 0, &mut backend).unwrap_err();
    assert!(empty_err.to_string().contains("at least one input"));

    let c = Tensor::from_vec_col_major(vec![3], vec![0.0_f64; 3]).unwrap();
    let shape_err = Tensor::stack(&[&a, &c], 0, &mut backend).unwrap_err();
    assert!(shape_err.to_string().contains("shape mismatch"));

    let axis_err = Tensor::stack(&[&a], 2, &mut backend).unwrap_err();
    assert!(axis_err.to_string().contains("axis 2"));
}

#[test]
fn structural_runtime_materialization_is_object_safe_and_clones_owned_input_by_default() {
    let input = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let mut backend = DefaultReadBackend::default();
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
    let mut backend = DefaultReadBackend::default();
    let session: &mut dyn BackendSession = &mut backend;

    let err = session
        .to_contiguous_read(TensorRead::from_view(view))
        .unwrap_err();

    assert!(matches!(
        err,
        crate::Error::BackendFailure {
            op: "to_contiguous_read",
            ref message,
        } if message.contains("borrowed tensor views")
    ));
}

#[test]
fn structural_runtime_copy_is_explicitly_unsupported_by_default() {
    let src = Tensor::from_vec_col_major(vec![2], vec![1_i32, 2]).unwrap();
    let mut dst = Tensor::from_vec_col_major(vec![2], vec![0_i32, 0]).unwrap();
    let mut backend = DefaultReadBackend::default();
    let session: &mut dyn BackendSession = &mut backend;

    let err = session
        .copy_read_into(
            TensorRead::from_tensor(&src),
            TensorWrite::from_tensor(&mut dst),
        )
        .unwrap_err();

    assert!(matches!(
        err,
        crate::Error::BackendFailure {
            op: "copy_read_into",
            ref message,
        } if message.contains("unsupported")
    ));
    assert_eq!(dst.as_slice::<i32>().unwrap(), &[0, 0]);
}
