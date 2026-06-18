use crate::{
    BackendCachedDot, BackendRuntimeCache, BackendSessionHost, CompareDir, DType, DotGeneralConfig,
    GatherConfig, PadConfig, ScatterConfig, SliceConfig, Tensor, TensorAnalytic, TensorBackend,
    TensorBuffer, TensorDeviceTransfer, TensorDot, TensorElementwise, TensorFusion, TensorIndexing,
    TensorRead, TensorReduction, TensorStructural, TensorView, TypedTensor,
};

#[derive(Default)]
struct DefaultReadBackend {
    calls: Vec<&'static str>,
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

    fn convert(&mut self, _input: &Tensor, _to: DType) -> crate::Result<Tensor> {
        self.calls.push("convert");
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
        Ok(marker())
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
