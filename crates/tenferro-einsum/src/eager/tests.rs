use tenferro_cpu::CpuBackend;
use tenferro_tensor::{
    BackendSession, BackendSessionHost, CompareDir, DotGeneralConfig, ElementwiseReadOp, Error,
    GatherConfig, PadConfig, Result, ScatterConfig, SessionCachedDot, SliceConfig, Tensor,
    TensorAnalytic, TensorBuffer, TensorDeviceTransfer, TensorDot, TensorElementwise, TensorFusion,
    TensorIndexing, TensorRead, TensorReduction, TensorStructural, TensorView, TensorWrite,
};

use super::{
    binary_contract, eager_einsum_exec_read, eager_einsum_read_subscripts, LabeledTensor,
    TensorValue,
};
use crate::{ContractionTree, Subscripts};

#[test]
fn tensor_value_view_paths_materialize_and_read() {
    let tensor = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let view_shape = [2usize];
    let view_data = [3.0_f64, 4.0];
    let view = TensorView::f64(&view_shape, &view_data).unwrap();

    let borrowed = TensorValue::Borrowed(&tensor);
    assert_eq!(borrowed.as_tensor().unwrap().shape(), &[2]);
    assert_eq!(borrowed.tensor_read().shape(), &[2]);

    let owned = TensorValue::Owned(tensor.duplicate().unwrap());
    assert_eq!(owned.as_tensor().unwrap().shape(), &[2]);
    assert_eq!(owned.tensor_read().shape(), &[2]);

    let view_value = TensorValue::View(view);
    assert!(view_value.as_tensor().is_none());
    assert_eq!(view_value.tensor_read().shape(), &[2]);
    let mut backend = CpuBackend::new();
    assert_eq!(
        backend
            .with_backend_session(|exec| view_value.into_tensor(exec))
            .unwrap()
            .as_slice::<f64>()
            .unwrap(),
        &[3.0, 4.0]
    );
}

#[test]
fn generic_outer_product_with_views_uses_broadcast_path() {
    let lhs_shape = [2usize];
    let lhs_data = [1.0_f64, 2.0];
    let rhs_shape = [3usize];
    let rhs_data = [3.0_f64, 4.0, 5.0];
    let lhs_view = TensorView::f64(&lhs_shape, &lhs_data).unwrap();
    let rhs_view = TensorView::f64(&rhs_shape, &rhs_data).unwrap();
    let lhs = LabeledTensor {
        tensor: TensorValue::View(lhs_view),
        labels: vec![0],
    };
    let rhs = LabeledTensor {
        tensor: TensorValue::View(rhs_view),
        labels: vec![1],
    };

    let mut ctx = CpuBackend::new();
    let result = ctx
        .with_backend_session(|exec| binary_contract(exec, lhs, rhs, &[0, 1], true))
        .unwrap();
    let labels = result.labels;
    let tensor = ctx
        .with_backend_session(|exec| result.tensor.into_tensor(exec))
        .unwrap();

    assert_eq!(labels, vec![0, 1]);
    assert_eq!(tensor.shape(), &[2, 3]);
    assert_eq!(
        tensor.as_slice::<f64>().unwrap(),
        &[3.0, 6.0, 4.0, 8.0, 5.0, 10.0]
    );
}

#[test]
fn generic_outer_product_uses_broadcast_views_without_materialized_broadcast_ops() {
    let lhs = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![3], vec![3.0_f64, 4.0, 5.0]).unwrap();
    let lhs = LabeledTensor {
        tensor: TensorValue::Borrowed(&lhs),
        labels: vec![0],
    };
    let rhs = LabeledTensor {
        tensor: TensorValue::Borrowed(&rhs),
        labels: vec![1],
    };
    let mut backend = NoBroadcastMaterializationBackend {
        shape: &[2, 3],
        lhs_strides: &[1, 0],
        rhs_strides: &[0, 1],
    };

    let result = binary_contract(&mut backend, lhs, rhs, &[0, 1], true).unwrap();
    let tensor = result.tensor.into_tensor(&mut backend).unwrap();

    assert_eq!(result.labels, vec![0, 1]);
    assert_eq!(tensor.shape(), &[2, 3]);
    assert_eq!(
        tensor.as_slice::<f64>().unwrap(),
        &[3.0, 6.0, 4.0, 8.0, 5.0, 10.0]
    );
}

#[test]
fn generic_outer_product_uses_target_order_without_final_transpose() {
    let lhs = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![3], vec![3.0_f64, 4.0, 5.0]).unwrap();
    let lhs = LabeledTensor {
        tensor: TensorValue::Borrowed(&lhs),
        labels: vec![0],
    };
    let rhs = LabeledTensor {
        tensor: TensorValue::Borrowed(&rhs),
        labels: vec![1],
    };
    let mut backend = NoBroadcastMaterializationBackend {
        shape: &[3, 2],
        lhs_strides: &[0, 1],
        rhs_strides: &[1, 0],
    };

    let result = binary_contract(&mut backend, lhs, rhs, &[1, 0], true).unwrap();
    let tensor = result.tensor.into_tensor(&mut backend).unwrap();

    assert_eq!(result.labels, vec![1, 0]);
    assert_eq!(tensor.shape(), &[3, 2]);
    assert_eq!(
        tensor.as_slice::<f64>().unwrap(),
        &[3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    );
}

#[test]
fn generic_binary_contract_reduces_then_builds_dot_config() {
    let lhs = Tensor::from_vec_col_major(vec![2, 3, 4], vec![1.0_f64; 24]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![3, 5], vec![2.0_f64; 15]).unwrap();
    let lhs = LabeledTensor {
        tensor: TensorValue::Borrowed(&lhs),
        labels: vec![0, 1, 9],
    };
    let rhs = LabeledTensor {
        tensor: TensorValue::Borrowed(&rhs),
        labels: vec![1, 2],
    };

    let mut ctx = CpuBackend::new();
    let result = ctx
        .with_backend_session(|exec| binary_contract(exec, lhs, rhs, &[0, 2], false))
        .unwrap();
    let labels = result.labels;
    let tensor = ctx
        .with_backend_session(|exec| result.tensor.into_tensor(exec))
        .unwrap();

    assert_eq!(labels, vec![0, 2]);
    assert_eq!(tensor.shape(), &[2, 5]);
    assert_eq!(tensor.as_slice::<f64>().unwrap(), &[24.0; 10]);
}

#[doc(hidden)]
struct NoBroadcastMaterializationBackendSessionMarker;

struct NoBroadcastMaterializationBackend {
    shape: &'static [usize],
    lhs_strides: &'static [isize],
    rhs_strides: &'static [isize],
}

impl BackendSession for NoBroadcastMaterializationBackend {
    fn session_type_id(&self) -> std::any::TypeId {
        std::any::TypeId::of::<NoBroadcastMaterializationBackendSessionMarker>()
    }

    unsafe fn session_data_mut(&mut self) -> *mut () {
        self as *mut Self as *mut ()
    }
}

fn unexpected(op: &'static str) -> Error {
    Error::backend_failure(op, "unexpected backend operation in outer-product test")
}

impl TensorElementwise for NoBroadcastMaterializationBackend {
    fn elementwise_read_into(
        &mut self,
        op: ElementwiseReadOp,
        inputs: &[TensorRead<'_>],
        out: TensorWrite<'_>,
    ) -> Result<()> {
        let _ = (op, inputs, out);
        Err(unexpected("elementwise_read_into"))
    }

    fn add(&mut self, _lhs: &Tensor, _rhs: &Tensor) -> Result<Tensor> {
        Err(unexpected("add"))
    }

    fn sub(&mut self, _lhs: &Tensor, _rhs: &Tensor) -> Result<Tensor> {
        Err(unexpected("sub"))
    }

    fn mul(&mut self, _lhs: &Tensor, _rhs: &Tensor) -> Result<Tensor> {
        Err(unexpected("mul"))
    }

    fn mul_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> Result<Tensor> {
        match (&lhs, &rhs) {
            (
                TensorRead::View(TensorView::F64(lhs_view)),
                TensorRead::View(TensorView::F64(rhs_view)),
            ) => {
                assert_eq!(lhs_view.shape(), self.shape);
                assert_eq!(rhs_view.shape(), self.shape);
                assert_eq!(lhs_view.strides(), self.lhs_strides);
                assert_eq!(rhs_view.strides(), self.rhs_strides);
            }
            _ => panic!("outer product should pass f64 broadcast views to mul_read"),
        }
        CpuBackend::new().mul_read(lhs, rhs)
    }

    fn neg(&mut self, _input: &Tensor) -> Result<Tensor> {
        Err(unexpected("neg"))
    }

    fn conj(&mut self, _input: &Tensor) -> Result<Tensor> {
        Err(unexpected("conj"))
    }

    fn div(&mut self, _lhs: &Tensor, _rhs: &Tensor) -> Result<Tensor> {
        Err(unexpected("div"))
    }

    fn abs(&mut self, _input: &Tensor) -> Result<Tensor> {
        Err(unexpected("abs"))
    }

    fn sign(&mut self, _input: &Tensor) -> Result<Tensor> {
        Err(unexpected("sign"))
    }

    fn maximum(&mut self, _lhs: &Tensor, _rhs: &Tensor) -> Result<Tensor> {
        Err(unexpected("maximum"))
    }

    fn minimum(&mut self, _lhs: &Tensor, _rhs: &Tensor) -> Result<Tensor> {
        Err(unexpected("minimum"))
    }

    fn compare(&mut self, _lhs: &Tensor, _rhs: &Tensor, _dir: &CompareDir) -> Result<Tensor> {
        Err(unexpected("compare"))
    }

    fn clamp(&mut self, _input: &Tensor, _lower: &Tensor, _upper: &Tensor) -> Result<Tensor> {
        Err(unexpected("clamp"))
    }
    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn add_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> Result<Tensor> {
        self.add(
            tenferro_tensor::backend::read_owned_tensor("add", lhs)?,
            tenferro_tensor::backend::read_owned_tensor("add", rhs)?,
        )
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn sub_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> Result<Tensor> {
        self.sub(
            tenferro_tensor::backend::read_owned_tensor("sub", lhs)?,
            tenferro_tensor::backend::read_owned_tensor("sub", rhs)?,
        )
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn neg_read(&mut self, input: TensorRead<'_>) -> Result<Tensor> {
        self.neg(tenferro_tensor::backend::read_owned_tensor("neg", input)?)
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn conj_read(&mut self, input: TensorRead<'_>) -> Result<Tensor> {
        self.conj(tenferro_tensor::backend::read_owned_tensor("conj", input)?)
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn div_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> Result<Tensor> {
        self.div(
            tenferro_tensor::backend::read_owned_tensor("div", lhs)?,
            tenferro_tensor::backend::read_owned_tensor("div", rhs)?,
        )
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn abs_read(&mut self, input: TensorRead<'_>) -> Result<Tensor> {
        self.abs(tenferro_tensor::backend::read_owned_tensor("abs", input)?)
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn sign_read(&mut self, input: TensorRead<'_>) -> Result<Tensor> {
        self.sign(tenferro_tensor::backend::read_owned_tensor("sign", input)?)
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn maximum_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> Result<Tensor> {
        self.maximum(
            tenferro_tensor::backend::read_owned_tensor("maximum", lhs)?,
            tenferro_tensor::backend::read_owned_tensor("maximum", rhs)?,
        )
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn minimum_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> Result<Tensor> {
        self.minimum(
            tenferro_tensor::backend::read_owned_tensor("minimum", lhs)?,
            tenferro_tensor::backend::read_owned_tensor("minimum", rhs)?,
        )
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn compare_read(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        dir: &CompareDir,
    ) -> Result<Tensor> {
        self.compare(
            tenferro_tensor::backend::read_owned_tensor("compare", lhs)?,
            tenferro_tensor::backend::read_owned_tensor("compare", rhs)?,
            dir,
        )
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn select_read(
        &mut self,
        pred: TensorRead<'_>,
        on_true: TensorRead<'_>,
        on_false: TensorRead<'_>,
    ) -> Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("select", pred)?;
        let _ = tenferro_tensor::backend::read_owned_tensor("select", on_true)?;
        let _ = tenferro_tensor::backend::read_owned_tensor("select", on_false)?;
        Err(unexpected("select"))
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn clamp_read(
        &mut self,
        input: TensorRead<'_>,
        lower: TensorRead<'_>,
        upper: TensorRead<'_>,
    ) -> Result<Tensor> {
        self.clamp(
            tenferro_tensor::backend::read_owned_tensor("clamp", input)?,
            tenferro_tensor::backend::read_owned_tensor("clamp", lower)?,
            tenferro_tensor::backend::read_owned_tensor("clamp", upper)?,
        )
    }
}

impl TensorAnalytic for NoBroadcastMaterializationBackend {
    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn exp_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("exp", input)?;
        Err(unexpected("exp"))
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn log_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("log", input)?;
        Err(unexpected("log"))
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn sin_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("sin", input)?;
        Err(unexpected("sin"))
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn cos_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("cos", input)?;
        Err(unexpected("cos"))
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn tanh_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("tanh", input)?;
        Err(unexpected("tanh"))
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn sqrt_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("sqrt", input)?;
        Err(unexpected("sqrt"))
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn rsqrt_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("rsqrt", input)?;
        Err(unexpected("rsqrt"))
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn pow_read(
        &mut self,
        lhs: tenferro_tensor::TensorRead<'_>,
        rhs: tenferro_tensor::TensorRead<'_>,
    ) -> Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("pow", lhs)?;
        let _ = tenferro_tensor::backend::read_owned_tensor("pow", rhs)?;
        Err(unexpected("pow"))
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn expm1_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("expm1", input)?;
        Err(unexpected("expm1"))
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn log1p_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> Result<Tensor> {
        let _ = tenferro_tensor::backend::read_owned_tensor("log1p", input)?;
        Err(unexpected("log1p"))
    }
}

impl TensorStructural for NoBroadcastMaterializationBackend {
    fn transpose(&mut self, _input: &Tensor, _perm: &[usize]) -> Result<Tensor> {
        Err(unexpected("transpose"))
    }

    fn reshape(&mut self, _input: &Tensor, _shape: &[usize]) -> Result<Tensor> {
        Err(unexpected("reshape"))
    }

    fn broadcast_in_dim(
        &mut self,
        _input: &Tensor,
        _shape: &[usize],
        _dims: &[usize],
    ) -> Result<Tensor> {
        Err(Error::backend_failure(
            "broadcast_in_dim",
            "outer product should use broadcast views, not materialized broadcast ops",
        ))
    }

    fn cast(&mut self, _input: &Tensor, _to: tenferro_tensor::DType) -> Result<Tensor> {
        Err(unexpected("cast"))
    }

    fn extract_diagonal(
        &mut self,
        _input: &Tensor,
        _axis_a: usize,
        _axis_b: usize,
    ) -> Result<Tensor> {
        Err(unexpected("extract_diagonal"))
    }

    fn embed_diagonal(
        &mut self,
        _input: &Tensor,
        _axis_a: usize,
        _axis_b: usize,
    ) -> Result<Tensor> {
        Err(unexpected("embed_diagonal"))
    }

    fn tril(&mut self, _input: &Tensor, _k: i64) -> Result<Tensor> {
        Err(unexpected("tril"))
    }

    fn triu(&mut self, _input: &Tensor, _k: i64) -> Result<Tensor> {
        Err(unexpected("triu"))
    }

    // The previous read-half default delegated owned tensors to the one-shot
    // method and rejected borrowed views. Reproduce it explicitly rather than
    // forwarding a view, which would widen the accepted input surface.
    fn transpose_read(&mut self, input: TensorRead<'_>, perm: &[usize]) -> Result<Tensor> {
        self.transpose(
            tenferro_tensor::backend::read_owned_tensor("transpose", input)?,
            perm,
        )
    }

    // The previous read-half default delegated owned tensors to the one-shot
    // method and rejected borrowed views. Reproduce it explicitly rather than
    // forwarding a view, which would widen the accepted input surface.
    fn reshape_read(&mut self, input: TensorRead<'_>, shape: &[usize]) -> Result<Tensor> {
        self.reshape(
            tenferro_tensor::backend::read_owned_tensor("reshape", input)?,
            shape,
        )
    }

    // The previous read-half default delegated owned tensors to the one-shot
    // method and rejected borrowed views. Reproduce it explicitly rather than
    // forwarding a view, which would widen the accepted input surface.
    fn broadcast_in_dim_read(
        &mut self,
        input: TensorRead<'_>,
        shape: &[usize],
        dims: &[usize],
    ) -> Result<Tensor> {
        self.broadcast_in_dim(
            tenferro_tensor::backend::read_owned_tensor("broadcast_in_dim", input)?,
            shape,
            dims,
        )
    }
}

impl TensorReduction for NoBroadcastMaterializationBackend {
    fn reduce_sum(&mut self, _input: &Tensor, _axes: &[usize]) -> Result<Tensor> {
        Err(unexpected("reduce_sum"))
    }

    fn reduce_prod(&mut self, _input: &Tensor, _axes: &[usize]) -> Result<Tensor> {
        Err(unexpected("reduce_prod"))
    }

    fn reduce_max(&mut self, _input: &Tensor, _axes: &[usize]) -> Result<Tensor> {
        Err(unexpected("reduce_max"))
    }

    fn reduce_min(&mut self, _input: &Tensor, _axes: &[usize]) -> Result<Tensor> {
        Err(unexpected("reduce_min"))
    }

    // The previous read-half default delegated owned tensors to the one-shot
    // method and rejected borrowed views. Reproduce it explicitly.
    fn reduce_sum_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> Result<Tensor> {
        self.reduce_sum(
            tenferro_tensor::backend::read_owned_tensor("reduce_sum", input)?,
            axes,
        )
    }

    // The previous read-half default delegated owned tensors to the one-shot
    // method and rejected borrowed views. Reproduce it explicitly.
    fn reduce_prod_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> Result<Tensor> {
        self.reduce_prod(
            tenferro_tensor::backend::read_owned_tensor("reduce_prod", input)?,
            axes,
        )
    }

    // The previous read-half default delegated owned tensors to the one-shot
    // method and rejected borrowed views. Reproduce it explicitly.
    fn reduce_max_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> Result<Tensor> {
        self.reduce_max(
            tenferro_tensor::backend::read_owned_tensor("reduce_max", input)?,
            axes,
        )
    }

    // The previous read-half default delegated owned tensors to the one-shot
    // method and rejected borrowed views. Reproduce it explicitly.
    fn reduce_min_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> Result<Tensor> {
        self.reduce_min(
            tenferro_tensor::backend::read_owned_tensor("reduce_min", input)?,
            axes,
        )
    }
}

impl TensorIndexing for NoBroadcastMaterializationBackend {
    fn gather(
        &mut self,
        _operand: &Tensor,
        _start_indices: &Tensor,
        _config: &GatherConfig,
    ) -> Result<Tensor> {
        Err(unexpected("gather"))
    }

    fn scatter(
        &mut self,
        _operand: &Tensor,
        _scatter_indices: &Tensor,
        _updates: &Tensor,
        _config: &ScatterConfig,
    ) -> Result<Tensor> {
        Err(unexpected("scatter"))
    }

    fn slice(&mut self, _input: &Tensor, _config: &SliceConfig) -> Result<Tensor> {
        Err(unexpected("slice"))
    }

    fn dynamic_slice(
        &mut self,
        _input: &Tensor,
        _starts: &Tensor,
        _slice_sizes: &[usize],
    ) -> Result<Tensor> {
        Err(unexpected("dynamic_slice"))
    }

    fn dynamic_update_slice(
        &mut self,
        _operand: &Tensor,
        _update: &Tensor,
        _starts: &Tensor,
    ) -> Result<Tensor> {
        Err(unexpected("dynamic_update_slice"))
    }

    fn pad(&mut self, _input: &Tensor, _config: &PadConfig) -> Result<Tensor> {
        Err(unexpected("pad"))
    }

    fn concatenate(&mut self, _inputs: &[&Tensor], _axis: usize) -> Result<Tensor> {
        Err(unexpected("concatenate"))
    }

    fn reverse(&mut self, _input: &Tensor, _axes: &[usize]) -> Result<Tensor> {
        Err(unexpected("reverse"))
    }
}

impl TensorDot for NoBroadcastMaterializationBackend {
    fn dot_general(
        &mut self,
        _lhs: &Tensor,
        _rhs: &Tensor,
        _config: &DotGeneralConfig,
    ) -> Result<Tensor> {
        Err(unexpected("dot_general"))
    }
    // The previous read-half default delegated an owned pair to the one-shot
    // method and materialized borrowed views through to_contiguous_read before
    // contracting. Reproduce that exactly rather than forwarding a view.
    fn dot_general_read(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
    ) -> Result<Tensor> {
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

impl TensorFusion for NoBroadcastMaterializationBackend {}
impl TensorBuffer for NoBroadcastMaterializationBackend {}
impl SessionCachedDot for NoBroadcastMaterializationBackend {}
impl TensorDeviceTransfer for NoBroadcastMaterializationBackend {
    fn download_to_host(&mut self, _tensor: TensorRead<'_>) -> Result<Tensor> {
        Err(unexpected("download_to_host"))
    }

    fn upload_host_tensor(&mut self, _tensor: TensorRead<'_>) -> Result<Tensor> {
        Err(unexpected("upload_host_tensor"))
    }
}

#[test]
fn generic_read_exec_reduces_single_view_input() {
    let shape = [2usize, 3];
    let data = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let view = TensorView::f64(&shape, &data).unwrap();
    let inputs = [TensorRead::from_view(view)];
    let subscripts = Subscripts::parse("ij->i").unwrap();
    let tree = ContractionTree::optimize(&subscripts, &[&shape]).unwrap();

    let mut ctx = CpuBackend::new();
    let result = ctx
        .with_backend_session(|exec| eager_einsum_exec_read(exec, &inputs, &tree))
        .unwrap();

    assert_eq!(result.shape(), &[2]);
    assert_eq!(result.as_slice::<f64>().unwrap(), &[9.0, 12.0]);
}

#[test]
fn read_subscripts_routes_non_fast_cases_through_plan() {
    let lhs = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let rhs =
        Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let mut ctx = CpuBackend::new();

    // Single-input call against a two-input contract: clean plan validation error.
    let subscripts = Subscripts::parse("ij,jk->ik").unwrap();
    let one_input = [TensorRead::from_tensor(&lhs)];
    assert!(eager_einsum_read_subscripts(&mut ctx, &one_input, &subscripts).is_err());

    // Rank-mismatched labels: clean plan validation error.
    let flat_rhs = Tensor::from_vec_col_major(vec![6], vec![1.0_f64; 6]).unwrap();
    let rank_mismatch = [
        TensorRead::from_tensor(&lhs),
        TensorRead::from_tensor(&flat_rhs),
    ];
    assert!(eager_einsum_read_subscripts(&mut ctx, &rank_mismatch, &subscripts).is_err());

    // Duplicate labels decline the internal binary dot plan and execute through
    // the plan's generic path: diag(ii) * sum_j(rhs[jk]).
    let duplicate_labels = Subscripts::parse("ii,jk->ik").unwrap();
    let inputs = [TensorRead::from_tensor(&lhs), TensorRead::from_tensor(&rhs)];
    let result = eager_einsum_read_subscripts(&mut ctx, &inputs, &duplicate_labels).unwrap();
    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.as_slice::<f64>().unwrap(), &[6.0, 24.0, 15.0, 60.0]);
}
