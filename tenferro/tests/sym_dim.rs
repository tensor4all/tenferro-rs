use tenferro::engine::Engine;
use tenferro::error::Error;
use tenferro::{CpuBackend, Tensor, TracedTensor, TypedTensor};

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec(shape, data))
}

fn get_f64_data(tensor: &Tensor) -> &[f64] {
    match tensor {
        Tensor::F64(inner) => inner.host_data(),
        other => panic!("expected f64 tensor, got {other:?}"),
    }
}

#[test]
fn reshape_sym_uses_symbolic_input_axes() {
    let x = TracedTensor::from_tensor(f64_tensor(
        vec![2, 3, 4],
        (1..=24).map(|value| value as f64).collect(),
    ));
    let rows = x.sym_size(0) * x.sym_size(1);
    let cols = x.sym_size(2);
    let mut y = x.reshape_sym(&[rows, cols]).unwrap();

    assert_eq!(y.rank, 2);

    let mut engine = Engine::new(CpuBackend::new());
    let result = y.eval(&mut engine).unwrap();
    assert_eq!(result.shape(), &[6, 4]);
    assert_eq!(
        get_f64_data(result),
        &(1..=24).map(|value| value as f64).collect::<Vec<_>>()
    );
}

#[test]
fn reshape_sym_supports_mixed_usize_arithmetic() {
    let x = TracedTensor::from_tensor(f64_tensor(
        vec![2, 3, 4],
        (1..=24).map(|value| value as f64).collect(),
    ));
    let mut y = x
        .reshape_sym(&[
            2usize * x.sym_size(0).max(1usize),
            (x.sym_size(1) * x.sym_size(2).min(4usize)) / 2usize,
        ])
        .unwrap();

    assert_eq!(y.rank, 2);

    let mut engine = Engine::new(CpuBackend::new());
    let result = y.eval(&mut engine).unwrap();
    assert_eq!(result.shape(), &[4, 6]);
    assert_eq!(
        get_f64_data(result),
        &(1..=24).map(|value| value as f64).collect::<Vec<_>>()
    );
}

#[test]
fn reshape_sym_rejects_symbolic_dims_from_another_tensor() {
    let a = TracedTensor::from_tensor(f64_tensor(vec![2, 3], (1..=6).map(|v| v as f64).collect()));
    let b = TracedTensor::from_tensor(f64_tensor(
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
