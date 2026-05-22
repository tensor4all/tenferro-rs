mod support;
use support::RunTraced;
use tenferro::error::Error;
use tenferro::GraphExecutor;
use tenferro::{CpuBackend, Tensor, TracedTensor, TypedTensor};

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec_col_major(shape, data))
}

fn get_f64_data(tensor: &Tensor) -> &[f64] {
    match tensor {
        Tensor::F64(inner) => inner.host_data(),
        other => panic!("expected f64 tensor, got {other:?}"),
    }
}

#[test]
fn reshape_sym_uses_symbolic_input_axes() {
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![2, 3, 4],
        (1..=24).map(|value| value as f64).collect(),
    ));
    let rows = x.sym_size(0) * x.sym_size(1);
    let cols = x.sym_size(2);
    let y = x.reshape_sym(&[rows, cols]).unwrap();

    assert_eq!(y.rank, 2);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let result = y.run_with(&mut engine).unwrap();
    assert_eq!(result.shape(), &[6, 4]);
    assert_eq!(
        get_f64_data(&result),
        &(1..=24).map(|value| value as f64).collect::<Vec<_>>()
    );
}

#[test]
fn reshape_sym_supports_mixed_usize_arithmetic() {
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![2, 3, 4],
        (1..=24).map(|value| value as f64).collect(),
    ));
    let y = x
        .reshape_sym(&[
            2usize * x.sym_size(0).max(1usize),
            (x.sym_size(1) * x.sym_size(2).min(4usize)) / 2usize,
        ])
        .unwrap();

    assert_eq!(y.rank, 2);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let result = y.run_with(&mut engine).unwrap();
    assert_eq!(result.shape(), &[4, 6]);
    assert_eq!(
        get_f64_data(&result),
        &(1..=24).map(|value| value as f64).collect::<Vec<_>>()
    );
}

#[test]
fn reshape_sym_rejects_symbolic_dims_from_another_tensor() {
    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![2, 3],
        (1..=6).map(|v| v as f64).collect(),
    ));
    let b = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![3, 2],
        (1..=6).rev().map(|v| v as f64).collect(),
    ));

    let err = match b.reshape_sym(&[a.sym_size(0), b.sym_size(1)]) {
        Ok(_) => panic!("reshape_sym should reject symbolic dimensions from another tensor"),
        Err(err) => err,
    };
    assert!(
        matches!(err, Error::Internal(message) if message.contains("unknown symbolic tensor id"))
    );
}

#[test]
fn sym_dim_sub_and_div_operators() {
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![12],
        (1..=12).map(|v| v as f64).collect(),
    ));
    // reshape [12] -> [dim0 - 2, dim0 / 2] = [10, 6]... that doesn't multiply to 12
    // Instead: reshape [12] -> [dim0 / 4, dim0 / 3] = [3, 4]
    let a = x.sym_size(0) / 4usize;
    let b = x.sym_size(0) / 3usize;
    let y = x.reshape_sym(&[a, b]).unwrap();

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let result = y.run_with(&mut engine).unwrap();
    assert_eq!(result.shape(), &[3, 4]);
}

#[test]
fn sym_dim_sub_operator() {
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![6],
        (1..=6).map(|v| v as f64).collect(),
    ));
    // reshape [6] -> [dim0 - 4, dim0 - 3] = [2, 3]
    let a = x.sym_size(0) - 4usize;
    let b = x.sym_size(0) - 3usize;
    let y = x.reshape_sym(&[a, b]).unwrap();

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let result = y.run_with(&mut engine).unwrap();
    assert_eq!(result.shape(), &[2, 3]);
}

#[test]
fn sym_dim_usize_lhs_operators() {
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![6],
        (1..=6).map(|v| v as f64).collect(),
    ));
    // 12usize / dim0 = 2, dim0 + 0usize = 6 -- but need product = 6
    // Use: 3usize + (dim0 - dim0) = 3, dim0 - 4usize = 2 => [3, 2]
    let a = 3usize + (x.sym_size(0) - x.sym_size(0));
    let b = x.sym_size(0) - 4usize;
    let y = x.reshape_sym(&[a, b]).unwrap();

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let result = y.run_with(&mut engine).unwrap();
    assert_eq!(result.shape(), &[3, 2]);
}

#[test]
fn sym_dim_usize_sub_and_div_lhs() {
    // Test usize - SymDim and usize / SymDim
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![2, 3],
        (1..=6).map(|v| v as f64).collect(),
    ));
    // 5usize - dim0 = 3, 6usize / dim1 = 2 => [3, 2]
    let a = 5usize - x.sym_size(0);
    let b = 6usize / x.sym_size(1);
    let y = x.reshape_sym(&[a, b]).unwrap();

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let result = y.run_with(&mut engine).unwrap();
    assert_eq!(result.shape(), &[3, 2]);
}

#[test]
fn sym_dim_min_max_methods() {
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![2, 3],
        (1..=6).map(|v| v as f64).collect(),
    ));
    // min(dim0, dim1) = 2, max(dim0, dim1) = 3 => [2, 3]
    let a = x.sym_size(0).min(x.sym_size(1));
    let b = x.sym_size(0).max(x.sym_size(1));
    let y = x.reshape_sym(&[a, b]).unwrap();

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let result = y.run_with(&mut engine).unwrap();
    assert_eq!(result.shape(), &[2, 3]);
}

/// Build a symbolic flatten graph (reshape to [dim0 * dim1]) ONCE and
/// execute it with two tensors of different shapes, verifying that the
/// DimExpr-based reshape resolves correctly each time.
#[test]
fn reshape_sym_graph_reuse_with_different_shapes() {
    // Helper: build a flatten graph from a 2-D input tensor.
    fn build_flatten(input: &TracedTensor) -> TracedTensor {
        let total = input.sym_size(0) * input.sym_size(1);
        input.reshape_sym(&[total]).unwrap()
    }

    let mut engine = GraphExecutor::new(CpuBackend::new());

    // First execution: shape [2, 3]
    let data_a: Vec<f64> = (1..=6).map(|v| v as f64).collect();
    let x_a = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 3], data_a.clone()));
    let y_a = build_flatten(&x_a);
    let result_a = y_a.run_with(&mut engine).unwrap();
    assert_eq!(result_a.shape(), &[6]);
    assert_eq!(get_f64_data(&result_a), &data_a);

    // Second execution: shape [4, 5] — same graph pattern, different sizes
    let data_b: Vec<f64> = (1..=20).map(|v| v as f64).collect();
    let x_b = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![4, 5], data_b.clone()));
    let y_b = build_flatten(&x_b);
    let result_b = y_b.run_with(&mut engine).unwrap();
    assert_eq!(result_b.shape(), &[20]);
    assert_eq!(get_f64_data(&result_b), &data_b);
}
