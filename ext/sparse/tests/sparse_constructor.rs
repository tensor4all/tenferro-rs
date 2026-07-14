use tenferro_ext_sparse::SparseCooTracedTensor;
use tenferro_runtime::{Error as RuntimeError, TracedTensor};
use tenferro_tensor::{DType, Error as TensorError, Tensor};

fn one_coordinate() -> Tensor {
    Tensor::from_vec_col_major(vec![2, 1], vec![0_i64, 0]).unwrap()
}

#[test]
fn traced_constructor_accepts_rank_one_unknown_symbolic_extent() {
    let values = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();

    let sparse = SparseCooTracedTensor::from_parts(vec![1, 1], one_coordinate(), values)
        .expect("unknown symbolic extent is validated by the extension constraint");

    assert_eq!(sparse.values().rank, 1);
    assert!(!sparse.values().is_concrete_shape());
}

#[test]
fn traced_constructor_rejects_known_concrete_nnz_mismatch() {
    let values = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();

    let error = SparseCooTracedTensor::from_parts(vec![1, 1], one_coordinate(), values)
        .expect_err("known concrete value extent must match coordinate nnz");

    assert!(matches!(
        error,
        RuntimeError::TensorRuntime(TensorError::InvalidConfig {
            op: "tenferro-ext-sparse",
            message,
        }) if message == "value tensor must have shape [1], got [2]"
    ));
}
