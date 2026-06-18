use tenferro_cpu::CpuBackend;
use tenferro_tensor::{
    BackendSessionHost, CompareDir, DotGeneralConfig, Error, GatherConfig, PadConfig, Result,
    ScatterConfig, SessionCachedDot, SliceConfig, Tensor, TensorAnalytic, TensorBuffer, TensorDot,
    TensorElementwise, TensorFusion, TensorIndexing, TensorRead, TensorReduction, TensorStructural,
    TensorView,
};

use super::{
    binary_contract, eager_einsum_exec_read, try_eager_einsum_binary_read_fast, LabeledTensor,
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

    let owned = TensorValue::Owned(tensor.clone());
    assert_eq!(owned.as_tensor().unwrap().shape(), &[2]);
    assert_eq!(owned.tensor_read().shape(), &[2]);

    let view_value = TensorValue::View(view);
    assert!(view_value.as_tensor().is_none());
    assert_eq!(view_value.tensor_read().shape(), &[2]);
    assert_eq!(
        view_value.into_tensor().unwrap().as_slice::<f64>().unwrap(),
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
    let tensor = result.tensor.into_tensor().unwrap();

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
    let tensor = result.tensor.into_tensor().unwrap();

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
    let tensor = result.tensor.into_tensor().unwrap();

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
    let tensor = result.tensor.into_tensor().unwrap();

    assert_eq!(labels, vec![0, 2]);
    assert_eq!(tensor.shape(), &[2, 5]);
    assert_eq!(tensor.as_slice::<f64>().unwrap(), &[24.0; 10]);
}

struct NoBroadcastMaterializationBackend {
    shape: &'static [usize],
    lhs_strides: &'static [isize],
    rhs_strides: &'static [isize],
}

fn unexpected(op: &'static str) -> Error {
    Error::backend_failure(op, "unexpected backend operation in outer-product test")
}

impl TensorElementwise for NoBroadcastMaterializationBackend {
    fn add(&mut self, _lhs: &Tensor, _rhs: &Tensor) -> Result<Tensor> {
        Err(unexpected("add"))
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

    fn select(&mut self, _pred: &Tensor, _on_true: &Tensor, _on_false: &Tensor) -> Result<Tensor> {
        Err(unexpected("select"))
    }

    fn clamp(&mut self, _input: &Tensor, _lower: &Tensor, _upper: &Tensor) -> Result<Tensor> {
        Err(unexpected("clamp"))
    }
}

impl TensorAnalytic for NoBroadcastMaterializationBackend {
    fn exp(&mut self, _input: &Tensor) -> Result<Tensor> {
        Err(unexpected("exp"))
    }

    fn log(&mut self, _input: &Tensor) -> Result<Tensor> {
        Err(unexpected("log"))
    }

    fn sin(&mut self, _input: &Tensor) -> Result<Tensor> {
        Err(unexpected("sin"))
    }

    fn cos(&mut self, _input: &Tensor) -> Result<Tensor> {
        Err(unexpected("cos"))
    }

    fn tanh(&mut self, _input: &Tensor) -> Result<Tensor> {
        Err(unexpected("tanh"))
    }

    fn sqrt(&mut self, _input: &Tensor) -> Result<Tensor> {
        Err(unexpected("sqrt"))
    }

    fn rsqrt(&mut self, _input: &Tensor) -> Result<Tensor> {
        Err(unexpected("rsqrt"))
    }

    fn pow(&mut self, _lhs: &Tensor, _rhs: &Tensor) -> Result<Tensor> {
        Err(unexpected("pow"))
    }

    fn expm1(&mut self, _input: &Tensor) -> Result<Tensor> {
        Err(unexpected("expm1"))
    }

    fn log1p(&mut self, _input: &Tensor) -> Result<Tensor> {
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

    fn convert(&mut self, _input: &Tensor, _to: tenferro_tensor::DType) -> Result<Tensor> {
        Err(unexpected("convert"))
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
}

impl TensorFusion for NoBroadcastMaterializationBackend {}
impl TensorBuffer for NoBroadcastMaterializationBackend {}
impl SessionCachedDot for NoBroadcastMaterializationBackend {}

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
fn binary_read_fast_path_rejects_non_fast_shapes_and_labels() {
    let lhs = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64; 6]).unwrap();
    let mut ctx = CpuBackend::new();

    let subscripts = Subscripts::parse("ij,jk->ik").unwrap();
    let one_input = [TensorRead::from_tensor(&lhs)];
    assert!(try_eager_einsum_binary_read_fast(&mut ctx, &one_input, &subscripts).is_none());

    let flat_rhs = Tensor::from_vec_col_major(vec![6], vec![1.0_f64; 6]).unwrap();
    let rank_mismatch = [
        TensorRead::from_tensor(&lhs),
        TensorRead::from_tensor(&flat_rhs),
    ];
    assert!(try_eager_einsum_binary_read_fast(&mut ctx, &rank_mismatch, &subscripts).is_none());

    let duplicate_labels = Subscripts::parse("ii,jk->ik").unwrap();
    let inputs = [TensorRead::from_tensor(&lhs), TensorRead::from_tensor(&rhs)];
    assert!(try_eager_einsum_binary_read_fast(&mut ctx, &inputs, &duplicate_labels).is_none());
}
