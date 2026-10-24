use tenferro_tensor::{Engine, Tensor, TensorRead};

use crate::test_support::cpu_engine;

/// Scalar f64 reference for one column-major `[axis_len, columns]` array.
fn reference(input: &[f64], axis_len: usize) -> Vec<f64> {
    let mut out = input.to_vec();
    for column in out.chunks_mut(axis_len) {
        let max = column.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0;
        for value in column.iter_mut() {
            *value = (*value - max).exp();
            sum += *value;
        }
        for value in column.iter_mut() {
            *value /= sum;
        }
    }
    out
}

#[test]
fn softmax_covers_the_f32_and_f64_paths() {
    let values = vec![1.0f64, 2.0, 3.0, -1.0, 0.0, 1.0, 0.5, -0.5, 4.0];
    let want = reference(&values, 3);
    let mut engine = cpu_engine(1).unwrap();
    let input = Tensor::from_vec_col_major(vec![3, 3], values.clone()).unwrap();
    let output = engine
        .softmax_read(TensorRead::from_tensor(&input), 0)
        .unwrap()
        .unwrap();
    for (got, &want) in output.as_slice::<f64>().unwrap().iter().zip(&want) {
        assert!((got - want).abs() < 1e-12, "f64: {got} vs {want}");
    }

    let values32: Vec<f32> = values.iter().map(|&value| value as f32).collect();
    let input = Tensor::from_vec_col_major(vec![3, 3], values32).unwrap();
    let output = engine
        .softmax_read(TensorRead::from_tensor(&input), 0)
        .unwrap()
        .unwrap();
    for (got, &want) in output.as_slice::<f32>().unwrap().iter().zip(&want) {
        assert!(
            (f64::from(*got) - want).abs() < 1e-6,
            "f32: {got} vs {want}"
        );
    }
}

#[test]
fn a_pooled_engine_matches_the_serial_result_on_a_ragged_tensor() {
    // Enough columns for several tasks and a ragged tail.
    let axis_len = 17;
    let columns = 4_000;
    let values: Vec<f64> = (0..axis_len * columns)
        .map(|i| f64::from((i % axis_len) as u32) * 0.25 - 1.0)
        .collect();
    let input = Tensor::from_vec_col_major(vec![axis_len, columns], values.clone()).unwrap();
    let mut serial = cpu_engine(1).unwrap();
    let mut pooled = cpu_engine(4).unwrap();
    let a = serial
        .softmax_read(TensorRead::from_tensor(&input), 0)
        .unwrap()
        .unwrap();
    let b = pooled
        .softmax_read(TensorRead::from_tensor(&input), 0)
        .unwrap()
        .unwrap();
    assert_eq!(a.as_slice::<f64>().unwrap(), b.as_slice::<f64>().unwrap());
    for (got, &want) in a
        .as_slice::<f64>()
        .unwrap()
        .iter()
        .zip(reference(&values, axis_len).iter())
    {
        assert!((got - want).abs() < 1e-12, "f64: {got} vs {want}");
    }
}

#[test]
fn a_nan_column_stays_nan_and_leaves_the_others_alone() {
    let values = vec![f64::NAN, 1.0, 2.0, 1.0, 2.0, 3.0];
    let input = Tensor::from_vec_col_major(vec![3, 2], values).unwrap();
    let mut engine = cpu_engine(1).unwrap();
    let output = engine
        .softmax_read(TensorRead::from_tensor(&input), 0)
        .unwrap()
        .unwrap();
    let got = output.as_slice::<f64>().unwrap();
    assert!(got[..3].iter().all(|value| value.is_nan()));
    let sum: f64 = got[3..].iter().sum();
    assert!((sum - 1.0).abs() < 1e-12);
}

#[test]
fn declined_shapes_and_dtypes_keep_the_composite_path() {
    let mut engine = cpu_engine(2).unwrap();
    // A non-axis-0 reduction, an empty axis and a non-float dtype decline.
    let matrix = Tensor::from_vec_col_major(vec![2, 3], vec![1.0f64; 6]).unwrap();
    assert!(engine
        .softmax_read(TensorRead::from_tensor(&matrix), 1)
        .unwrap()
        .is_none());
    let empty = Tensor::from_vec_col_major(vec![0, 3], Vec::<f64>::new()).unwrap();
    assert!(engine
        .softmax_read(TensorRead::from_tensor(&empty), 0)
        .unwrap()
        .is_none());
    let ints = Tensor::from_vec_col_major(vec![3], vec![1i32, 2, 3]).unwrap();
    assert!(engine
        .softmax_read(TensorRead::from_tensor(&ints), 0)
        .unwrap()
        .is_none());
}
