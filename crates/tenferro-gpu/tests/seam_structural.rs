#![cfg(feature = "cuda")]

//! U5 first slice: the `TensorStructural` surface on the new engine — canonical
//! materialization, a same-placement copy and a dtype cast, through the trait rather than the
//! typed dispatcher (issue #2208 U5).

use tenferro_gpu::cuda::{download_tensor, gpu_available, upload_tensor, CudaDeviceId, CudaEngine};
use tenferro_tensor::{
    DType, Tensor, TensorIndexing, TensorRead, TensorReduction, TensorStructural, TensorWrite,
};

fn engine() -> Option<CudaEngine> {
    if !gpu_available() {
        return None;
    }
    Some(CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("CUDA engine"))
}

fn values(len: usize) -> Vec<f64> {
    (0..len).map(|index| index as f64 * 0.5 - 3.0).collect()
}

fn host(data: Vec<f64>) -> Tensor {
    let len = data.len();
    Tensor::from_vec_col_major(vec![len], data).expect("host tensor")
}

fn device(engine: &CudaEngine, data: Vec<f64>) -> Tensor {
    upload_tensor(engine.runtime(), &host(data)).expect("upload")
}

#[test]
#[ignore = "requires CUDA"]
fn a_device_tensor_materializes_into_fresh_storage() {
    let Some(mut engine) = engine() else { return };
    let data = values(64);
    let source = device(&engine, data.clone());
    let copy = engine
        .to_contiguous_read(TensorRead::from_tensor(&source))
        .expect("materialize");
    assert_eq!(copy.shape(), source.shape());
    assert!(
        copy.is_backend_buffer() && source.is_backend_buffer(),
        "both stay on the device"
    );
    let back = download_tensor(engine.runtime(), &copy).expect("download");
    assert_eq!(back.as_slice::<f64>().expect("f64 payload"), &data[..]);
}

#[test]
#[ignore = "requires CUDA"]
fn a_host_tensor_is_duplicated_without_device_work() {
    let Some(mut engine) = engine() else { return };
    let data = values(16);
    let source = host(data.clone());
    let copy = engine
        .to_contiguous_read(TensorRead::from_tensor(&source))
        .expect("duplicate");
    // The host path is the trait's own canonicalization: same placement, same values.
    assert_eq!(copy.as_slice::<f64>().expect("f64 payload"), &data[..]);
    assert!(!copy.is_backend_buffer());
}

#[test]
#[ignore = "requires CUDA"]
fn copy_read_into_overwrites_the_destination() {
    let Some(mut engine) = engine() else { return };
    let data = values(32);
    let source = device(&engine, data.clone());
    let mut destination = device(&engine, vec![0.0; data.len()]);
    engine
        .copy_read_into(
            TensorRead::from_tensor(&source),
            TensorWrite::from_tensor(&mut destination),
        )
        .expect("copy");
    let back = download_tensor(engine.runtime(), &destination).expect("download");
    assert_eq!(back.as_slice::<f64>().expect("f64 payload"), &data[..]);
}

#[test]
#[ignore = "requires CUDA"]
fn a_cast_converts_on_the_device() {
    let Some(mut engine) = engine() else { return };
    let data = values(24);
    let source = device(&engine, data.clone());
    let converted = engine.cast(&source, DType::F32).expect("cast");
    assert_eq!(converted.dtype(), DType::F32);
    assert_eq!(converted.shape(), source.shape());
    let back = download_tensor(engine.runtime(), &converted).expect("download");
    let observed = back.as_slice::<f32>().expect("f32 payload");
    let expected: Vec<f32> = data.iter().map(|value| *value as f32).collect();
    assert_eq!(observed, &expected[..]);
}

#[test]
#[ignore = "requires CUDA"]
fn a_transposed_read_matches_the_host_transpose() {
    let Some(mut engine) = engine() else { return };
    // A 3x4 column-major matrix, read as its 4x3 transpose.
    let data: Vec<f64> = (0..12).map(|index| index as f64).collect();
    let source = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![3, 4], data.clone()).expect("matrix"),
    )
    .expect("upload");
    let transposed = engine
        .transpose_read(TensorRead::from_tensor(&source), &[1, 0])
        .expect("transpose");
    assert_eq!(transposed.shape(), &[4, 3]);
    let back = download_tensor(engine.runtime(), &transposed).expect("download");
    let observed = back.as_slice::<f64>().expect("f64 payload");
    let mut expected = Vec::with_capacity(12);
    for row in 0..3 {
        for column in 0..4 {
            expected.push(data[row + column * 3]);
        }
    }
    assert_eq!(observed, &expected[..]);
}

#[test]
#[ignore = "requires CUDA"]
fn a_reshape_read_relabels_the_same_elements() {
    let Some(mut engine) = engine() else { return };
    let data = values(12);
    let source = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![3, 4], data.clone()).expect("matrix"),
    )
    .expect("upload");
    let reshaped = engine
        .reshape_read(TensorRead::from_tensor(&source), &[2, 6])
        .expect("reshape");
    assert_eq!(reshaped.shape(), &[2, 6]);
    let back = download_tensor(engine.runtime(), &reshaped).expect("download");
    assert_eq!(back.as_slice::<f64>().expect("f64 payload"), &data[..]);
    // A reshape that changes the element count is a typed refusal.
    assert!(engine
        .reshape_read(TensorRead::from_tensor(&source), &[5, 5])
        .is_err());
}

#[test]
#[ignore = "requires CUDA"]
fn a_broadcast_in_dim_read_matches_the_host() {
    let Some(mut engine) = engine() else { return };
    // A 3-vector broadcast along the second axis of a 3x2 output.
    let data = vec![1.0f64, 2.0, 3.0];
    let source = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![3], data.clone()).expect("vector"),
    )
    .expect("upload");
    let broadcast = engine
        .broadcast_in_dim_read(TensorRead::from_tensor(&source), &[3, 2], &[0])
        .expect("broadcast");
    assert_eq!(broadcast.shape(), &[3, 2]);
    let back = download_tensor(engine.runtime(), &broadcast).expect("download");
    let observed = back.as_slice::<f64>().expect("f64 payload");
    let expected: Vec<f64> = (0..2).flat_map(|_| data.clone()).collect();
    assert_eq!(observed, &expected[..]);
}

#[test]
#[ignore = "requires CUDA"]
fn a_diagonal_extract_and_embed_round_trip() {
    let Some(mut engine) = engine() else { return };
    // A 3x3 column-major matrix whose diagonal is (1, 5, 9).
    let data: Vec<f64> = (1..=9).map(|value| value as f64).collect();
    let source = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![3, 3], data.clone()).expect("matrix"),
    )
    .expect("upload");
    let diagonal = engine.extract_diagonal(&source, 0, 1).expect("extract");
    assert_eq!(diagonal.shape(), &[3]);
    let back = download_tensor(engine.runtime(), &diagonal).expect("download");
    assert_eq!(
        back.as_slice::<f64>().expect("f64 payload"),
        &[1.0, 5.0, 9.0]
    );

    // Embedding that diagonal back into a 3x3 gives the zero-filled matrix with only the
    // diagonal written.
    let embedded = engine.embed_diagonal(&diagonal, 0, 1).expect("embed");
    assert_eq!(embedded.shape(), &[3, 3]);
    let back = download_tensor(engine.runtime(), &embedded).expect("download");
    let observed = back.as_slice::<f64>().expect("f64 payload");
    let mut expected = [0.0f64; 9];
    for (index, value) in [1.0, 5.0, 9.0].iter().enumerate() {
        expected[index + index * 3] = *value;
    }
    assert_eq!(observed, &expected[..]);
    // Two distinct in-range axes are required.
    assert!(engine.extract_diagonal(&source, 0, 0).is_err());
}

#[test]
#[ignore = "requires CUDA"]
fn the_triangular_masks_match_the_host() {
    let Some(mut engine) = engine() else { return };
    let data: Vec<f64> = (1..=12).map(|value| value as f64).collect();
    let source = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![3, 4], data.clone()).expect("matrix"),
    )
    .expect("upload");
    let lower = engine.tril(&source, 0).expect("tril");
    let back = download_tensor(engine.runtime(), &lower).expect("download");
    let observed = back.as_slice::<f64>().expect("f64 payload");
    let mut expected = [0.0f64; 12];
    for column in 0..4 {
        for row in 0..3 {
            if row >= column {
                expected[row + column * 3] = data[row + column * 3];
            }
        }
    }
    assert_eq!(observed, &expected[..]);

    let upper = engine.triu(&source, 0).expect("triu");
    let back = download_tensor(engine.runtime(), &upper).expect("download");
    let observed = back.as_slice::<f64>().expect("f64 payload");
    let mut expected = [0.0f64; 12];
    for column in 0..4 {
        for row in 0..3 {
            if row <= column {
                expected[row + column * 3] = data[row + column * 3];
            }
        }
    }
    assert_eq!(observed, &expected[..]);
    // A rank-1 tensor has no triangular mask.
    let vector = device(&engine, values(4));
    assert!(engine.tril(&vector, 0).is_err());
}

#[test]
#[ignore = "requires CUDA"]
fn the_reductions_match_the_host() {
    use tenferro_tensor::TensorReduction as _;

    let Some(mut engine) = engine() else { return };
    // A 3x4 column-major matrix with distinct positive entries.
    let data: Vec<f64> = (1..=12).map(|value| value as f64 * 0.5).collect();
    let source = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![3, 4], data.clone()).expect("matrix"),
    )
    .expect("upload");

    // Summing axis 1 leaves one value per row.
    let summed = engine
        .reduce_sum_read(TensorRead::from_tensor(&source), &[1])
        .expect("sum");
    assert_eq!(summed.shape(), &[3]);
    let back = download_tensor(engine.runtime(), &summed).expect("download");
    let observed = back.as_slice::<f64>().expect("f64 payload");
    let expected: Vec<f64> = (0..3)
        .map(|row| (0..4).map(|column| data[row + column * 3]).sum())
        .collect();
    assert_eq!(observed, &expected[..]);

    // Summing both axes reduces to a scalar.
    let total = engine
        .reduce_sum_read(TensorRead::from_tensor(&source), &[0, 1])
        .expect("total");
    assert_eq!(total.shape(), &[] as &[usize]);
    let back = download_tensor(engine.runtime(), &total).expect("download");
    let observed = back.as_slice::<f64>().expect("f64 payload");
    let expected: f64 = data.iter().sum();
    assert!((observed[0] - expected).abs() < 1e-12);

    // Max over axis 0, and sum of squares over axis 1.
    let maximum = engine
        .reduce_max_read(TensorRead::from_tensor(&source), &[0])
        .expect("max");
    let back = download_tensor(engine.runtime(), &maximum).expect("download");
    let observed = back.as_slice::<f64>().expect("f64 payload");
    let expected: Vec<f64> = (0..4)
        .map(|column| {
            data[column * 3..column * 3 + 3]
                .iter()
                .fold(f64::MIN, |a, b| a.max(*b))
        })
        .collect();
    assert_eq!(observed, &expected[..]);

    let squares = engine
        .reduce_sum_squares_read(TensorRead::from_tensor(&source), &[1])
        .expect("squares");
    let back = download_tensor(engine.runtime(), &squares).expect("download");
    let observed = back.as_slice::<f64>().expect("f64 payload");
    let expected: Vec<f64> = (0..3)
        .map(|row| (0..4).map(|column| data[row + column * 3].powi(2)).sum())
        .collect();
    assert_eq!(observed, &expected[..]);

    // No axes is the identity copy, and an out-of-range axis is refused.
    let identity = engine
        .reduce_sum_read(TensorRead::from_tensor(&source), &[])
        .expect("identity");
    assert_eq!(identity.shape(), source.shape());
    assert!(engine
        .reduce_min_read(TensorRead::from_tensor(&source), &[7])
        .is_err());
}

#[test]
#[ignore = "requires CUDA"]
fn the_elementwise_family_matches_the_host() {
    use tenferro_tensor::TensorElementwise as _;

    let Some(mut engine) = engine() else { return };
    let lhs_data: Vec<f64> = (0..12).map(|index| index as f64 * 0.5 - 2.0).collect();
    let rhs_data: Vec<f64> = (0..12).map(|index| (index % 5) as f64 + 0.25).collect();
    let lhs = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![3, 4], lhs_data.clone()).expect("lhs"),
    )
    .expect("upload");
    let rhs = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![3, 4], rhs_data.clone()).expect("rhs"),
    )
    .expect("upload");

    let sum = engine
        .add_read(TensorRead::from_tensor(&lhs), TensorRead::from_tensor(&rhs))
        .expect("add");
    let back = download_tensor(engine.runtime(), &sum).expect("download");
    let observed = back.as_slice::<f64>().expect("f64 payload");
    let expected: Vec<f64> = lhs_data
        .iter()
        .zip(rhs_data.iter())
        .map(|(left, right)| left + right)
        .collect();
    assert_eq!(observed, &expected[..]);

    let product = engine
        .mul_read(TensorRead::from_tensor(&lhs), TensorRead::from_tensor(&rhs))
        .expect("mul");
    let back = download_tensor(engine.runtime(), &product).expect("download");
    let observed = back.as_slice::<f64>().expect("f64 payload");
    let expected: Vec<f64> = lhs_data
        .iter()
        .zip(rhs_data.iter())
        .map(|(left, right)| left * right)
        .collect();
    assert_eq!(observed, &expected[..]);

    let quotient = engine
        .div_read(TensorRead::from_tensor(&lhs), TensorRead::from_tensor(&rhs))
        .expect("div");
    let back = download_tensor(engine.runtime(), &quotient).expect("download");
    let observed = back.as_slice::<f64>().expect("f64 payload");
    let expected: Vec<f64> = lhs_data
        .iter()
        .zip(rhs_data.iter())
        .map(|(left, right)| left / right)
        .collect();
    assert_eq!(observed, &expected[..]);

    let difference = engine
        .sub_read(TensorRead::from_tensor(&lhs), TensorRead::from_tensor(&rhs))
        .expect("sub");
    let back = download_tensor(engine.runtime(), &difference).expect("download");
    let observed = back.as_slice::<f64>().expect("f64 payload");
    let expected: Vec<f64> = lhs_data
        .iter()
        .zip(rhs_data.iter())
        .map(|(left, right)| left - right)
        .collect();
    assert_eq!(observed, &expected[..]);

    // Unary operations: negation, absolute value and the sign.
    for (op, reference) in [
        ("neg", (|value: f64| -value) as fn(f64) -> f64),
        ("abs", f64::abs as fn(f64) -> f64),
        // The kernel follows the XLA convention: `sign(0) = 0`, not `f64::signum`'s 1.
        (
            "sign",
            (|value: f64| if value == 0.0 { 0.0 } else { value.signum() }) as fn(f64) -> f64,
        ),
    ] {
        let result = match op {
            "neg" => engine.neg_read(TensorRead::from_tensor(&lhs)),
            "abs" => engine.abs_read(TensorRead::from_tensor(&lhs)),
            _ => engine.sign_read(TensorRead::from_tensor(&lhs)),
        }
        .expect("unary");
        let back = download_tensor(engine.runtime(), &result).expect("download");
        let observed = back.as_slice::<f64>().expect("f64 payload");
        let expected: Vec<f64> = lhs_data.iter().map(|value| reference(*value)).collect();
        assert_eq!(observed, &expected[..], "{op}");
    }

    // Binary extrema and clamp with per-element bounds.
    let maximum = engine
        .maximum_read(TensorRead::from_tensor(&lhs), TensorRead::from_tensor(&rhs))
        .expect("maximum");
    let back = download_tensor(engine.runtime(), &maximum).expect("download");
    let observed = back.as_slice::<f64>().expect("f64 payload");
    let expected: Vec<f64> = lhs_data
        .iter()
        .zip(rhs_data.iter())
        .map(|(left, right)| left.max(*right))
        .collect();
    assert_eq!(observed, &expected[..]);

    let lower = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![3, 4], vec![-0.5f64; 12]).expect("lower"),
    )
    .expect("upload");
    let upper = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![3, 4], vec![1.0f64; 12]).expect("upper"),
    )
    .expect("upload");
    let clamped = engine
        .clamp_read(
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&lower),
            TensorRead::from_tensor(&upper),
        )
        .expect("clamp");
    let back = download_tensor(engine.runtime(), &clamped).expect("download");
    let observed = back.as_slice::<f64>().expect("f64 payload");
    let expected: Vec<f64> = lhs_data
        .iter()
        .map(|value| value.clamp(-0.5, 1.0))
        .collect();
    assert_eq!(observed, &expected[..]);

    // Mismatched shapes and the boolean-shaped routes are typed refusals.
    let narrow = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![4], vec![1.0f64; 4]).expect("narrow"),
    )
    .expect("upload");
    assert!(engine
        .add_read(
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&narrow)
        )
        .is_err());
}

#[test]
#[ignore = "requires CUDA"]
fn compare_and_select_round_trip_through_a_boolean_tensor() {
    use tenferro_tensor::{CompareDir, TensorElementwise as _};

    let Some(mut engine) = engine() else { return };
    let lhs_data: Vec<f64> = (0..8).map(|index| index as f64 - 3.5).collect();
    let rhs_data: Vec<f64> = (0..8).map(|index| (index % 3) as f64 - 1.0).collect();
    let lhs = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![8], lhs_data.clone()).expect("lhs"),
    )
    .expect("upload");
    let rhs = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![8], rhs_data.clone()).expect("rhs"),
    )
    .expect("upload");

    let predicate = engine
        .compare_read(
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &CompareDir::Gt,
        )
        .expect("compare");
    assert_eq!(predicate.dtype(), DType::Bool);
    let back = download_tensor(engine.runtime(), &predicate).expect("download");
    let observed = back.as_slice::<bool>().expect("bool payload");
    let expected: Vec<bool> = lhs_data
        .iter()
        .zip(rhs_data.iter())
        .map(|(left, right)| left > right)
        .collect();
    assert_eq!(observed, &expected[..]);

    // Selecting between two branches with that predicate.
    let on_true = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![8], vec![10.0f64; 8]).expect("on_true"),
    )
    .expect("upload");
    let on_false = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![8], vec![-10.0f64; 8]).expect("on_false"),
    )
    .expect("upload");
    let selected = engine
        .select_read(
            TensorRead::from_tensor(&predicate),
            TensorRead::from_tensor(&on_true),
            TensorRead::from_tensor(&on_false),
        )
        .expect("select");
    let back = download_tensor(engine.runtime(), &selected).expect("download");
    let observed = back.as_slice::<f64>().expect("f64 payload");
    let expected: Vec<f64> = lhs_data
        .iter()
        .zip(rhs_data.iter())
        .map(|(left, right)| if left > right { 10.0 } else { -10.0 })
        .collect();
    assert_eq!(observed, &expected[..]);

    // A non-boolean predicate is a typed refusal.
    assert!(engine
        .select_read(
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&on_true),
            TensorRead::from_tensor(&on_false)
        )
        .is_err());
}

#[test]
#[ignore = "requires CUDA"]
fn remainder_and_read_into_match_the_host() {
    use tenferro_tensor::{ElementwiseReadOp, TensorElementwise as _};

    let Some(mut engine) = engine() else { return };

    // Floating remainder through the plain entry.
    let lhs_data: Vec<f64> = (0..8).map(|index| index as f64 * 1.25).collect();
    let rhs_data: Vec<f64> = (0..8)
        .map(|index| 0.5 + (index % 3) as f64 * 0.25)
        .collect();
    let lhs = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![8], lhs_data.clone()).expect("lhs"),
    )
    .expect("upload");
    let rhs = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![8], rhs_data.clone()).expect("rhs"),
    )
    .expect("upload");
    let remainder = engine
        .rem_read(TensorRead::from_tensor(&lhs), TensorRead::from_tensor(&rhs))
        .expect("rem");
    let back = download_tensor(engine.runtime(), &remainder).expect("download");
    let observed = back.as_slice::<f64>().expect("f64 payload");
    let expected: Vec<f64> = lhs_data
        .iter()
        .zip(rhs_data.iter())
        .map(|(left, right)| left - (left / right).trunc() * right)
        .collect();
    for (value, reference) in observed.iter().zip(expected.iter()) {
        assert!((value - reference).abs() < 1e-12, "{value} vs {reference}");
    }

    // Integer remainder, including the checked divide-by-zero flag.
    let ints = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![4], vec![7i32, -7, 7, 7]).expect("ints"),
    )
    .expect("upload");
    let divisors = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![4], vec![2i32, 2, -2, 0]).expect("divisors"),
    )
    .expect("upload");
    // A zero divisor makes the whole call a typed error.
    assert!(engine
        .rem_read(
            TensorRead::from_tensor(&ints),
            TensorRead::from_tensor(&divisors)
        )
        .is_err());

    let safe_divisors = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![4], vec![2i32, 2, -2, 3]).expect("safe"),
    )
    .expect("upload");
    let integer_remainder = engine
        .rem_read(
            TensorRead::from_tensor(&ints),
            TensorRead::from_tensor(&safe_divisors),
        )
        .expect("integer rem");
    let back = download_tensor(engine.runtime(), &integer_remainder).expect("download");
    assert_eq!(back.as_slice::<i32>().expect("i32 payload"), &[1, -1, 1, 1]);

    // The read-into route writes straight into the caller's allocation.
    let mut destination = device(&engine, vec![0.0f64; 8]);
    engine
        .elementwise_read_into(
            ElementwiseReadOp::Add,
            &[TensorRead::from_tensor(&lhs), TensorRead::from_tensor(&rhs)],
            TensorWrite::from_tensor(&mut destination),
        )
        .expect("read into");
    let back = download_tensor(engine.runtime(), &destination).expect("download");
    let observed = back.as_slice::<f64>().expect("f64 payload");
    let expected: Vec<f64> = lhs_data
        .iter()
        .zip(rhs_data.iter())
        .map(|(left, right)| left + right)
        .collect();
    assert_eq!(observed, &expected[..]);

    // Negation through the same route, and the operand-count refusal.
    engine
        .elementwise_read_into(
            ElementwiseReadOp::Negate,
            &[TensorRead::from_tensor(&lhs)],
            TensorWrite::from_tensor(&mut destination),
        )
        .expect("negate into");
    let back = download_tensor(engine.runtime(), &destination).expect("download");
    let observed = back.as_slice::<f64>().expect("f64 payload");
    let expected: Vec<f64> = lhs_data.iter().map(|value| -value).collect();
    assert_eq!(observed, &expected[..]);
    assert!(engine
        .elementwise_read_into(
            ElementwiseReadOp::Add,
            &[TensorRead::from_tensor(&lhs)],
            TensorWrite::from_tensor(&mut destination),
        )
        .is_err());
}

#[test]
#[ignore = "requires CUDA"]
fn a_rank_nine_transpose_uses_the_general_path() {
    let Some(mut engine) = engine() else { return };
    // Nine axes exceed the fast ABI's rank limit, so the view's geometry travels in device
    // memory and the launch goes through the `_any` symbol.
    let shape = vec![2usize, 1, 1, 1, 1, 1, 1, 1, 3];
    let len: usize = shape.iter().product();
    let data: Vec<f64> = (0..len).map(|index| index as f64).collect();
    let host = Tensor::from_vec_col_major(shape.clone(), data.clone()).expect("host tensor");
    let source = upload_tensor(engine.runtime(), &host).expect("upload");

    let perm: Vec<usize> = (0..shape.len()).rev().collect();
    let transposed = engine
        .transpose_read(TensorRead::from_tensor(&source), &perm)
        .expect("rank-9 transpose");
    let expected_shape: Vec<usize> = perm.iter().map(|&axis| shape[axis]).collect();
    assert_eq!(transposed.shape(), &expected_shape[..]);

    let observed = download_tensor(engine.runtime(), &transposed).expect("download");
    let observed = observed.as_slice::<f64>().expect("f64 payload");
    // Column-major: the transposed element at index `i` reads the source element whose axes
    // are the permutation's inverse.
    let mut expected = vec![0.0f64; len];
    for (index, value) in expected.iter_mut().enumerate() {
        // Decode `index` in the transposed shape, then map it back through `perm`.
        let mut remaining = index;
        let mut source_index = 0usize;
        for axis in 0..shape.len() {
            let extent = expected_shape[axis];
            let coordinate = remaining % extent;
            remaining /= extent;
            // `perm[axis]` is the source axis this output axis came from.
            let stride: usize = shape[..perm[axis]].iter().product();
            source_index += coordinate * stride;
        }
        *value = data[source_index];
    }
    assert_eq!(observed, &expected[..]);
}

#[test]
#[ignore = "requires CUDA"]
fn a_rank_nine_reverse_carries_its_negative_strides_and_offset() {
    let Some(mut engine) = engine() else { return };
    // Non-singleton axes on both ends, so a dropped offset or an unnegated stride would show.
    let shape = vec![2usize, 1, 1, 1, 1, 1, 1, 3, 2];
    let len: usize = shape.iter().product();
    let data: Vec<f64> = (0..len).map(|index| index as f64 + 1.0).collect();
    let host = Tensor::from_vec_col_major(shape.clone(), data.clone()).expect("host tensor");
    let source = upload_tensor(engine.runtime(), &host).expect("upload");

    let reversed = engine.reverse(&source, &[0, 7, 8]).expect("rank-9 reverse");
    assert_eq!(reversed.shape(), &shape[..]);
    let observed = download_tensor(engine.runtime(), &reversed).expect("download");
    let observed = observed.as_slice::<f64>().expect("f64 payload");
    let mut expected = vec![0.0f64; len];
    for (index, value) in expected.iter_mut().enumerate() {
        let mut remaining = index;
        let mut source_index = 0usize;
        for axis in 0..shape.len() {
            let extent = shape[axis];
            let mut coordinate = remaining % extent;
            remaining /= extent;
            if matches!(axis, 0 | 7 | 8) {
                coordinate = extent - 1 - coordinate;
            }
            let stride: usize = shape[..axis].iter().product();
            source_index += coordinate * stride;
        }
        *value = data[source_index];
    }
    assert_eq!(observed, &expected[..]);
}

#[test]
#[ignore = "requires CUDA"]
fn a_rank_nine_broadcast_uses_the_general_path() {
    let Some(mut engine) = engine() else { return };
    // A rank-2 source into a rank-9 destination: the destination layout is wider than the
    // by-value ABI form, and one mapped axis (8) is past the fast limit too.
    let source_data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let host = Tensor::from_vec_col_major(vec![3, 2], source_data.clone()).expect("host");
    let source = upload_tensor(engine.runtime(), &host).expect("upload");
    let shape = vec![2usize, 1, 1, 1, 1, 1, 1, 1, 3];
    let dims = vec![8usize, 0];
    let broadcast = engine
        .broadcast_in_dim_read(TensorRead::from_tensor(&source), &shape, &dims)
        .expect("rank-9 broadcast");
    assert_eq!(broadcast.shape(), &shape[..]);

    let observed = download_tensor(engine.runtime(), &broadcast).expect("download");
    let observed = observed.as_slice::<f64>().expect("f64 payload");
    let len: usize = shape.iter().product();
    assert_eq!(observed.len(), len);
    let mut expected = vec![0.0f64; len];
    for (index, value) in expected.iter_mut().enumerate() {
        let mut remaining = index;
        let mut source_index = 0usize;
        for (axis, &extent) in shape.iter().enumerate() {
            let coordinate = remaining % extent;
            remaining /= extent;
            if let Some(source_axis) = dims.iter().position(|&mapped| mapped == axis) {
                // Column-major strides of the `[3, 2]` source.
                source_index += coordinate * if source_axis == 0 { 1 } else { 3 };
            }
        }
        *value = source_data[source_index];
    }
    assert_eq!(observed, &expected[..]);
}

#[test]
#[ignore = "requires CUDA"]
fn a_rank_nine_diagonal_pair_uses_the_general_path() {
    let Some(mut engine) = engine() else { return };
    // Unequal diagonal extents, a non-singleton spectator axis, and the lower axis removed
    // first, so the diagonal's output axis is not the input axis: a shifted axis or a transposed
    // destination would fail.
    let shape = vec![2usize, 3, 1, 1, 1, 1, 1, 1, 4];
    let data: Vec<f64> = (0..24).map(|index| index as f64 + 1.0).collect();
    let host = Tensor::from_vec_col_major(shape.clone(), data.clone()).expect("host");
    let source = upload_tensor(engine.runtime(), &host).expect("upload");
    // Column-major strides of the source: axis 0 is fastest, axis 1 next, axis 8 last.
    let (stride_0, stride_1, stride_8) = (1usize, 2, 6);

    // The diagonal of axes (8, 1): extent min(4, 3) = 3, axis 1 removed, so it lands on output
    // axis 7. An element pairs both axes, hence the stride sum below.
    let extracted = engine
        .extract_diagonal(&source, 8, 1)
        .expect("rank-9 extract diagonal");
    assert_eq!(extracted.shape(), &[2, 1, 1, 1, 1, 1, 1, 3]);
    let diagonal_host = download_tensor(engine.runtime(), &extracted).expect("download");
    let diagonal = diagonal_host.as_slice::<f64>().expect("f64 payload");
    let mut expected = Vec::new();
    for diagonal_index in 0..3 {
        for spectator in 0..2 {
            expected.push(data[spectator * stride_0 + diagonal_index * (stride_1 + stride_8)]);
        }
    }
    assert_eq!(diagonal, expected.as_slice());

    // Embedding the diagonal back at position 1 inserts an axis of extent 3, and every
    // off-diagonal entry stays zero.
    let embedded = engine
        .embed_diagonal(&extracted, 7, 1)
        .expect("rank-9 embed diagonal");
    assert_eq!(embedded.shape(), &[2, 3, 1, 1, 1, 1, 1, 1, 3]);
    let embedded_host = download_tensor(engine.runtime(), &embedded).expect("download");
    let observed = embedded_host.as_slice::<f64>().expect("f64 payload");
    let mut expected = vec![0.0f64; 18];
    // The destination is `[2, 3, 1, .., 3]`: axis 0 is fastest, the inserted axis second, the
    // diagonal axis last.
    for diagonal_index in 0..3 {
        for spectator in 0..2 {
            expected[spectator + diagonal_index * 2 + diagonal_index * 6] =
                data[spectator * stride_0 + diagonal_index * (stride_1 + stride_8)];
        }
    }
    assert_eq!(observed, expected.as_slice());
}

#[test]
#[ignore = "requires CUDA"]
fn a_rank_nine_reduction_uses_the_general_path() {
    let Some(mut engine) = engine() else { return };
    // Nine axes; the first reduction is a single axis and the second a multi-axis sum, so the
    // keepdims passes and the final squeeze both run through the general path.
    let shape = vec![3usize, 1, 1, 1, 1, 1, 1, 2, 4];
    let data: Vec<f64> = (0..24).map(|index| index as f64 * 0.5 - 1.0).collect();
    let host = Tensor::from_vec_col_major(shape.clone(), data.clone()).expect("host");
    let source = upload_tensor(engine.runtime(), &host).expect("upload");
    // Compact column-major strides of `[3, 1, .., 2, 4]`.
    let strides = [1usize, 3, 3, 3, 3, 3, 3, 3, 6];

    let summed = engine
        .reduce_sum_read(TensorRead::from_tensor(&source), &[0])
        .expect("rank-9 reduce");
    // The reduced axes are squeezed off the result, so the keepdims passes are internal.
    assert_eq!(summed.shape(), &[1, 1, 1, 1, 1, 1, 2, 4]);
    let summed_host = download_tensor(engine.runtime(), &summed).expect("download");
    let observed = summed_host.as_slice::<f64>().expect("f64 payload");
    let mut expected = Vec::new();
    for axis8 in 0..4 {
        for axis7 in 0..2 {
            let mut total = 0.0;
            for axis0 in 0..3 {
                total += data[axis0 * strides[0] + axis7 * strides[7] + axis8 * strides[8]];
            }
            expected.push(total);
        }
    }
    assert_eq!(observed, expected.as_slice());

    let squeezed = engine
        .reduce_sum_read(TensorRead::from_tensor(&source), &[0, 7])
        .expect("rank-9 multi-axis reduce");
    assert_eq!(squeezed.shape(), &[1, 1, 1, 1, 1, 1, 4]);
    let squeezed_host = download_tensor(engine.runtime(), &squeezed).expect("download");
    let observed = squeezed_host.as_slice::<f64>().expect("f64 payload");
    let mut expected = Vec::new();
    for axis8 in 0..4 {
        let mut total = 0.0;
        for axis7 in 0..2 {
            for axis0 in 0..3 {
                total += data[axis0 * strides[0] + axis7 * strides[7] + axis8 * strides[8]];
            }
        }
        expected.push(total);
    }
    assert_eq!(observed, expected.as_slice());

    // A product reduction exercises a different fold over the same general path.
    let product = engine
        .reduce_prod_read(TensorRead::from_tensor(&source), &[7])
        .expect("rank-9 product reduce");
    assert_eq!(product.shape(), &[3, 1, 1, 1, 1, 1, 1, 4]);
    let product_host = download_tensor(engine.runtime(), &product).expect("download");
    let observed = product_host.as_slice::<f64>().expect("f64 payload");
    let mut expected = Vec::new();
    for axis8 in 0..4 {
        for axis0 in 0..3 {
            expected.push(
                data[axis0 * strides[0] + axis8 * strides[8]]
                    * data[axis0 * strides[0] + strides[7] + axis8 * strides[8]],
            );
        }
    }
    assert_eq!(observed, expected.as_slice());
}

#[test]
#[ignore = "requires CUDA"]
fn a_sum_of_squares_over_two_axes_squares_once() {
    let Some(mut engine) = engine() else { return };
    // Squaring both passes would return `sum((sum x^2)^2)` instead of `sum(x^2)`.
    let shape = vec![2usize, 1, 1, 1, 1, 1, 1, 1, 2];
    let data = vec![1.0f64, 2.0, 3.0, 4.0];
    let host = Tensor::from_vec_col_major(shape, data.clone()).expect("host");
    let source = upload_tensor(engine.runtime(), &host).expect("upload");
    let squares = engine
        .reduce_sum_squares_read(TensorRead::from_tensor(&source), &[0, 8])
        .expect("sum of squares");
    let squares_host = download_tensor(engine.runtime(), &squares).expect("download");
    let observed = squares_host.as_slice::<f64>().expect("f64 payload");
    let expected: f64 = data.iter().map(|value| value * value).sum();
    assert_eq!(observed, &[expected]);
}

#[test]
#[ignore = "requires CUDA"]
fn a_rank_nine_reduction_over_a_long_axis_picks_a_plane_or_split_strategy() {
    let Some(mut engine) = engine() else { return };
    // Axis lengths past a warp and past the plane threshold make the crate choose a plane or a
    // split reduction, which the short fixtures never exercise.
    for (length, expected_splits) in [(33usize, false), (513, true), (8449, true)] {
        let shape = vec![length, 1, 1, 1, 1, 1, 1, 1, 2];
        let data: Vec<f64> = (0..length * 2)
            .map(|index| (index % 7) as f64 - 3.0)
            .collect();
        let host = Tensor::from_vec_col_major(shape, data.clone()).expect("host");
        let source = upload_tensor(engine.runtime(), &host).expect("upload");
        let summed = engine
            .reduce_sum_read(TensorRead::from_tensor(&source), &[0])
            .expect("rank-9 long-axis reduce");
        let summed_host = download_tensor(engine.runtime(), &summed).expect("download");
        let observed = summed_host.as_slice::<f64>().expect("f64 payload");
        let mut expected = Vec::new();
        for column in 0..2 {
            let total: f64 = (0..length).map(|row| data[row + column * length]).sum();
            expected.push(total);
        }
        assert_eq!(observed, expected.as_slice(), "axis length {length}");
        let _ = expected_splits;
    }
}

#[test]
#[ignore = "requires CUDA"]
fn a_multi_axis_fold_keeps_its_own_operation() {
    let Some(mut engine) = engine() else { return };
    // A product over two axes must stay a product; only a sum of squares changes between passes.
    let host = Tensor::from_vec_col_major(vec![2, 2], vec![1.0f64, 2.0, 3.0, 4.0]).expect("host");
    let source = upload_tensor(engine.runtime(), &host).expect("upload");
    let product = engine
        .reduce_prod_read(TensorRead::from_tensor(&source), &[0, 1])
        .expect("product");
    let product_host = download_tensor(engine.runtime(), &product).expect("download");
    assert_eq!(
        product_host.as_slice::<f64>().expect("f64 payload"),
        &[24.0],
        "1 * 2 * 3 * 4"
    );
    let squares = engine
        .reduce_sum_squares_read(TensorRead::from_tensor(&source), &[0, 1])
        .expect("sum of squares");
    let squares_host = download_tensor(engine.runtime(), &squares).expect("download");
    assert_eq!(
        squares_host.as_slice::<f64>().expect("f64 payload"),
        &[30.0],
        "1 + 4 + 9 + 16"
    );
    let maximum = engine
        .reduce_max_read(TensorRead::from_tensor(&source), &[0, 1])
        .expect("maximum");
    let maximum_host = download_tensor(engine.runtime(), &maximum).expect("download");
    assert_eq!(maximum_host.as_slice::<f64>().expect("f64 payload"), &[4.0]);
}

/// A borrowed view materializes the region it names, and a stride-0 broadcast view repeats its
/// element across every destination position.
///
/// The materializing kernel writes one destination element per logical position, so a region
/// smaller than its allocation must fill exactly `element_count` entries (no overrun into a
/// compact destination), and a view whose strides are all zero must not collapse to its first
/// element.
#[test]
#[ignore = "requires CUDA"]
fn a_borrowed_view_materializes_its_own_region() {
    let Some(mut engine) = engine() else { return };

    // The middle three of a five-element allocation: compact, at a nonzero offset.
    let data = vec![99.0f64, 1.0, 2.0, 3.0, 98.0];
    let wide = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![5], data.clone()).expect("vector"),
    )
    .expect("upload");
    let region = wide
        .as_typed::<f64>()
        .expect("f64 payload")
        .backend_region_view(vec![3], vec![1], 1)
        .expect("region view");
    let materialized = engine
        .to_contiguous_read(TensorRead::from_view(tenferro_tensor::TensorView::F64(
            region,
        )))
        .expect("materialize the region");
    assert_eq!(materialized.shape(), &[3]);
    let back = download_tensor(engine.runtime(), &materialized).expect("download");
    assert_eq!(back.as_slice::<f64>().expect("f64 payload"), &data[1..4]);

    // A rank-0 source broadcast through a metadata view: every axis has stride 0.
    let scalar = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![], vec![2.5f64]).expect("scalar"),
    )
    .expect("upload");
    let broadcast = scalar
        .as_typed::<f64>()
        .expect("f64 payload")
        .backend_region_view(vec![2, 3], vec![0, 0], 0)
        .expect("broadcast view");
    let materialized = engine
        .to_contiguous_read(TensorRead::from_view(tenferro_tensor::TensorView::F64(
            broadcast,
        )))
        .expect("materialize the broadcast");
    let back = download_tensor(engine.runtime(), &materialized).expect("download");
    assert_eq!(
        back.as_slice::<f64>().expect("f64 payload"),
        &[2.5f64; 6][..]
    );
}

// --- Native read entries: the operand is read where it lives (#2223 item 1) ---

use tenferro_tensor::TensorView;

/// A borrowed region of `tensor`'s device allocation, as one read target.
fn f64_region(
    tensor: &Tensor,
    shape: Vec<usize>,
    strides: Vec<isize>,
    offset: isize,
) -> TensorRead<'_> {
    TensorRead::from_view(TensorView::F64(
        tensor
            .as_typed::<f64>()
            .expect("f64 payload")
            .backend_region_view(shape, strides, offset)
            .expect("region view"),
    ))
}

/// The three borrowed-region flavours the diagonal entries have to serve on a `4 x 5`
/// column-major source: a transposed view, an offset view and a negative-stride view.
fn diagonal_view_flavours(source: &Tensor) -> Vec<(&'static str, TensorRead<'_>)> {
    vec![
        ("transposed", f64_region(source, vec![5, 4], vec![4, 1], 0)),
        ("offset", f64_region(source, vec![2, 3], vec![1, 4], 2)),
        ("negative", f64_region(source, vec![4, 2], vec![-1, 4], 3)),
    ]
}

/// Run `run` on the borrowed view and on its materialized compact copy, and require the same
/// shape and values: a read entry must read the region it was handed, not a stand-in.
fn matches_the_compact_operand(
    engine: &mut CudaEngine,
    flavour: &str,
    view: TensorRead<'_>,
    run: impl Fn(&mut CudaEngine, TensorRead<'_>) -> Tensor,
) {
    let observed = run(engine, view.clone());
    let compact = engine
        .to_contiguous_read(view)
        .expect("materialize the view");
    let expected = run(engine, TensorRead::from_tensor(&compact));
    assert_eq!(
        observed.shape(),
        expected.shape(),
        "the {flavour} view produced a different shape"
    );
    assert_eq!(
        read(engine, &observed),
        read(engine, &expected),
        "the {flavour} view read a different region"
    );
}

fn read(engine: &CudaEngine, tensor: &Tensor) -> Vec<f64> {
    let back = download_tensor(engine.runtime(), tensor).expect("download");
    back.as_slice::<f64>().expect("f64 payload").to_vec()
}

#[test]
#[ignore = "requires CUDA"]
fn extract_diagonal_read_follows_a_view_layout() {
    let Some(mut engine) = engine() else { return };
    let data: Vec<f64> = (0..20).map(|index| index as f64).collect();
    let source = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![4, 5], data).expect("matrix"),
    )
    .expect("upload");
    for (flavour, view) in diagonal_view_flavours(&source) {
        for (axis_a, axis_b) in [(0usize, 1usize), (1, 0)] {
            matches_the_compact_operand(&mut engine, flavour, view.clone(), |engine, input| {
                engine
                    .extract_diagonal_read(input, axis_a, axis_b)
                    .expect("extract_diagonal_read")
            });
        }
    }
}

/// The host answer for embedding a diagonal: insert an axis of the `axis_a` extent at
/// `axis_b` and write the input's diagonal positions into the zero-filled result.
fn embed_diagonal_expected(
    input: &[f64],
    shape: &[usize],
    axis_a: usize,
    axis_b: usize,
) -> Vec<f64> {
    let mut out_shape = shape.to_vec();
    out_shape.insert(axis_b, shape[axis_a]);
    let mut strides = vec![1usize; out_shape.len()];
    for axis in 1..out_shape.len() {
        strides[axis] = strides[axis - 1] * out_shape[axis - 1];
    }
    let mut output = vec![0.0; out_shape.iter().product()];
    for (flat, &value) in input.iter().enumerate() {
        let mut remaining = flat;
        let mut output_index = 0;
        for (axis, &extent) in shape.iter().enumerate() {
            let index = remaining % extent;
            remaining /= extent;
            let mapped = axis + if axis >= axis_b { 1 } else { 0 };
            output_index += index * strides[mapped];
            if axis == axis_a {
                output_index += index * strides[axis_b];
            }
        }
        output[output_index] = value;
    }
    output
}

#[test]
#[ignore = "requires CUDA"]
fn embed_diagonal_read_follows_a_view_layout() {
    let Some(mut engine) = engine() else { return };
    let data: Vec<f64> = (0..20).map(|index| index as f64 + 1.0).collect();
    let source = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![4, 5], data).expect("matrix"),
    )
    .expect("upload");
    for (flavour, view) in diagonal_view_flavours(&source) {
        // The expected values come from the view's own materialized data, not from another
        // embed, so a diagonal placed on the wrong pair of axes cannot pass.
        let compact = engine
            .to_contiguous_read(view.clone())
            .expect("materialize the view");
        let input = read(&engine, &compact);
        let shape = compact.shape().to_vec();
        for (axis_a, axis_b) in [(0usize, 1usize), (1, 0), (0, shape.len())] {
            let observed = engine
                .embed_diagonal_read(view.clone(), axis_a, axis_b)
                .expect("embed_diagonal_read");
            let mut expected_shape = shape.clone();
            expected_shape.insert(axis_b, shape[axis_a]);
            assert_eq!(observed.shape(), expected_shape, "the {flavour} view");
            assert_eq!(
                read(&engine, &observed),
                embed_diagonal_expected(&input, &shape, axis_a, axis_b),
                "the {flavour} view"
            );
        }
    }
}

/// Explicit host values for an asymmetric input, both axis orders and insertion past the last
/// input axis.
#[test]
#[ignore = "requires CUDA"]
fn embed_diagonal_read_writes_the_requested_axis_pair() {
    let Some(mut engine) = engine() else { return };
    // Column-major [2, 3]: (row, column) holds `1 + row + 2 * column`.
    let source = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("matrix"),
    )
    .expect("upload");

    let embedded = engine
        .embed_diagonal(&source, 0, 1)
        .expect("embed axis 0 at position 1");
    assert_eq!(embedded.shape(), &[2, 2, 3]);
    assert_eq!(
        read(&engine, &embedded),
        vec![1.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 4.0, 5.0, 0.0, 0.0, 6.0]
    );

    let embedded = engine
        .embed_diagonal(&source, 1, 0)
        .expect("embed axis 1 at position 0");
    assert_eq!(embedded.shape(), &[3, 2, 3]);
    assert_eq!(
        read(&engine, &embedded),
        vec![
            1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0,
            6.0
        ]
    );

    let embedded = engine
        .embed_diagonal(&source, 0, 2)
        .expect("insert at the input's rank");
    assert_eq!(embedded.shape(), &[2, 3, 2]);
    assert_eq!(
        read(&engine, &embedded),
        vec![1.0, 0.0, 3.0, 0.0, 5.0, 0.0, 0.0, 2.0, 0.0, 4.0, 0.0, 6.0]
    );
}

#[test]
#[ignore = "requires CUDA"]
fn diagonal_read_entries_serve_an_empty_view() {
    let Some(mut engine) = engine() else { return };
    let source = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![4, 5], values(20)).expect("matrix"),
    )
    .expect("upload");
    let empty = || f64_region(&source, vec![0, 2], vec![1, 4], 0);

    matches_the_compact_operand(&mut engine, "empty", empty(), |engine, input| {
        engine
            .extract_diagonal_read(input, 0, 1)
            .expect("extract_diagonal_read")
    });
    matches_the_compact_operand(&mut engine, "empty", empty(), |engine, input| {
        engine
            .embed_diagonal_read(input, 0, 1)
            .expect("embed_diagonal_read")
    });
}

#[test]
#[ignore = "requires CUDA"]
fn complex_diagonal_reads_follow_a_view_layout() {
    use tenferro_tensor::{Complex64, TensorStructural as _};

    let Some(mut engine) = engine() else { return };
    let data: Vec<Complex64> = (0..12)
        .map(|index| Complex64::new(index as f64 * 0.5, index as f64))
        .collect();
    let source = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![3, 4], data).expect("matrix"),
    )
    .expect("upload");
    let region = |shape: Vec<usize>, strides: Vec<isize>, offset: isize| {
        TensorRead::from_view(TensorView::C64(
            source
                .as_typed::<Complex64>()
                .expect("complex payload")
                .backend_region_view(shape, strides, offset)
                .expect("region view"),
        ))
    };
    let transpose = || region(vec![4, 3], vec![3, 1], 0);
    let read_complex = |engine: &CudaEngine, tensor: &Tensor| -> Vec<Complex64> {
        let back = download_tensor(engine.runtime(), tensor).expect("download");
        back.as_slice::<Complex64>()
            .expect("complex payload")
            .to_vec()
    };

    let extracted = engine
        .extract_diagonal_read(transpose(), 0, 1)
        .expect("extract_diagonal_read");
    let compact = engine.to_contiguous_read(transpose()).expect("materialize");
    let expected = engine.extract_diagonal(&compact, 0, 1).expect("extract");
    assert_eq!(extracted.shape(), expected.shape());
    assert_eq!(
        read_complex(&engine, &extracted),
        read_complex(&engine, &expected)
    );

    let embedded = engine
        .embed_diagonal_read(transpose(), 0, 1)
        .expect("embed_diagonal_read");
    let expected = engine.embed_diagonal(&compact, 0, 1).expect("embed");
    assert_eq!(embedded.shape(), expected.shape());
    assert_eq!(
        read_complex(&engine, &embedded),
        read_complex(&engine, &expected)
    );
}

// --- Write-side views: the destination region is written where it lives (#2223 item 3) ---

use tenferro_tensor::TensorViewMut;

/// The three destination flavours the write-side tests use: a transposed stride pair, a
/// nonzero offset and a negative stride, all over a `6 x 5` column-major root of thirty
/// sentinel entries. Each covers six entries, so the other twenty-four must survive untouched.
fn write_view_flavours() -> Vec<(&'static str, Vec<usize>, Vec<isize>, isize)> {
    vec![
        ("transposed", vec![2, 3], vec![5, 1], 2),
        ("offset", vec![2, 3], vec![1, 6], 7),
        ("negative", vec![2, 3], vec![-1, 6], 5),
    ]
}

fn sentinel(index: usize) -> f64 {
    100.0 + index as f64
}

/// A `6 x 5` column-major device tensor of sentinels.
fn write_root(engine: &CudaEngine) -> Tensor {
    let data: Vec<f64> = (0..30).map(sentinel).collect();
    upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![6, 5], data).expect("root"),
    )
    .expect("upload")
}

/// The root's linear indices in logical column-major order for one region.
fn region_indices(shape: &[usize], strides: &[isize], offset: isize) -> Vec<usize> {
    let count: usize = shape.iter().product();
    (0..count)
        .map(|flat| {
            let mut remaining = flat;
            let mut index = offset;
            for (&extent, &stride) in shape.iter().zip(strides) {
                let coordinate = remaining % extent;
                remaining /= extent;
                index += coordinate as isize * stride;
            }
            usize::try_from(index).expect("the region starts at a nonnegative element")
        })
        .collect()
}

/// The root a write should have produced: `written` at the region and the sentinel elsewhere.
fn expected_root(region: &[usize], written: &[f64]) -> Vec<f64> {
    assert_eq!(region.len(), written.len());
    let mut root: Vec<f64> = (0..30).map(sentinel).collect();
    for (&index, &value) in region.iter().zip(written) {
        root[index] = value;
    }
    root
}

/// Write through one destination region of `root`, then read the whole root back.
fn write_through_view(
    engine: &mut CudaEngine,
    root: &mut Tensor,
    shape: Vec<usize>,
    strides: Vec<isize>,
    offset: isize,
    write: impl FnOnce(&mut CudaEngine, TensorWrite<'_>) -> tenferro_tensor::Result<()>,
) -> Vec<f64> {
    {
        let region = root
            .as_typed_mut::<f64>()
            .expect("f64 payload")
            .backend_region_view_mut(shape, strides, offset)
            .expect("mutable region view");
        write(engine, TensorWrite::from_view(TensorViewMut::F64(region)))
            .expect("write through the destination view");
    }
    read(engine, root)
}

/// A `[2, 3]` device tensor whose entries are far from the root's sentinels.
fn write_operand(engine: &CudaEngine, offset: f64) -> Tensor {
    let data: Vec<f64> = (0..6).map(|index| offset + index as f64).collect();
    upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![2, 3], data).expect("operand"),
    )
    .expect("upload")
}

#[test]
#[ignore = "requires CUDA"]
fn copy_read_into_writes_a_strided_destination_view() {
    let Some(mut engine) = engine() else { return };
    for (flavour, shape, strides, offset) in write_view_flavours() {
        let mut root = write_root(&engine);
        let source = write_operand(&engine, 1000.0);
        let expected = read(&engine, &source);
        let observed = write_through_view(
            &mut engine,
            &mut root,
            shape.clone(),
            strides.clone(),
            offset,
            |engine, out| engine.copy_read_into(TensorRead::from_tensor(&source), out),
        );
        let region = region_indices(&shape, &strides, offset);
        assert_eq!(
            observed,
            expected_root(&region, &expected),
            "the {flavour} destination"
        );
    }
}

#[test]
#[ignore = "requires CUDA"]
fn copy_read_into_writes_a_compact_offset_destination_from_a_reversed_source() {
    let Some(mut engine) = engine() else { return };
    // A reversed, offset source view: two rows of three sentinels, walking the source's first
    // axis backwards from entry five.
    let shape = vec![2, 3];
    let strides = vec![-1, 6];
    let offset = 5;
    let source = write_root(&engine);
    let source_data = read(&engine, &source);
    let source_region = region_indices(&shape, &strides, offset);
    let expected: Vec<f64> = source_region
        .iter()
        .map(|&index| source_data[index])
        .collect();
    let source_view = TensorView::F64(
        source
            .as_typed::<f64>()
            .expect("f64 payload")
            .backend_region_view(shape.clone(), strides.clone(), offset)
            .expect("source region view"),
    );

    // A destination whose strides are compact for its extents but whose offset is not zero: one
    // contiguous span, which `StridedToStrided` addresses through the region's own layout.
    let mut root = write_root(&engine);
    let observed = write_through_view(
        &mut engine,
        &mut root,
        shape.clone(),
        vec![1, 6],
        6,
        |engine, out| engine.copy_read_into(TensorRead::from_view(source_view), out),
    );
    let destination_region = region_indices(&shape, &[1, 6], 6);
    assert_eq!(
        observed,
        expected_root(&destination_region, &expected),
        "the compact-offset destination"
    );
}

#[test]
#[ignore = "requires CUDA"]
fn copy_read_into_serves_a_flat_and_an_empty_destination_view() {
    let Some(mut engine) = engine() else { return };

    // A flat destination view: offset zero and compact strides, which the exact copy addresses
    // without a layout parameter.
    let source = write_operand(&engine, 1000.0);
    let expected = read(&engine, &source);
    let mut root = write_root(&engine);
    let observed = write_through_view(
        &mut engine,
        &mut root,
        vec![2, 3],
        vec![1, 2],
        0,
        |engine, out| engine.copy_read_into(TensorRead::from_tensor(&source), out),
    );
    assert_eq!(
        observed,
        expected_root(&region_indices(&[2, 3], &[1, 2], 0), &expected),
        "the flat destination"
    );

    // An empty destination view carries no element: both regions were validated, so the copy is
    // a no-op that leaves all thirty sentinels in place.
    let empty = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![0], Vec::<f64>::new()).expect("empty source"),
    )
    .expect("upload the empty source");
    let mut root = write_root(&engine);
    let observed = write_through_view(
        &mut engine,
        &mut root,
        vec![0],
        vec![1],
        0,
        |engine, out| engine.copy_read_into(TensorRead::from_tensor(&empty), out),
    );
    assert_eq!(
        observed,
        (0..30).map(sentinel).collect::<Vec<f64>>(),
        "an empty copy changed the root"
    );
}

#[test]
#[ignore = "requires CUDA"]
fn elementwise_read_into_writes_a_strided_destination_view() {
    use tenferro_tensor::{ElementwiseReadOp, TensorElementwise as _};

    let Some(mut engine) = engine() else { return };
    for (flavour, shape, strides, offset) in write_view_flavours() {
        let mut root = write_root(&engine);
        let lhs = write_operand(&engine, -2.0);
        let rhs = write_operand(&engine, 3.5);
        let lhs_data = read(&engine, &lhs);
        let rhs_data = read(&engine, &rhs);
        let region = region_indices(&shape, &strides, offset);

        // The binary arm: `out = lhs + rhs`.
        let summed: Vec<f64> = lhs_data
            .iter()
            .zip(&rhs_data)
            .map(|(left, right)| left + right)
            .collect();
        let observed = write_through_view(
            &mut engine,
            &mut root,
            shape.clone(),
            strides.clone(),
            offset,
            |engine, out| {
                engine.elementwise_read_into(
                    ElementwiseReadOp::Add,
                    &[TensorRead::from_tensor(&lhs), TensorRead::from_tensor(&rhs)],
                    out,
                )
            },
        );
        assert_eq!(
            observed,
            expected_root(&region, &summed),
            "the {flavour} destination"
        );

        // The unary arm: `out = -lhs`.
        let mut root = write_root(&engine);
        let negated: Vec<f64> = lhs_data.iter().map(|value| -value).collect();
        let observed = write_through_view(
            &mut engine,
            &mut root,
            shape.clone(),
            strides.clone(),
            offset,
            |engine, out| {
                engine.elementwise_read_into(
                    ElementwiseReadOp::Negate,
                    &[TensorRead::from_tensor(&lhs)],
                    out,
                )
            },
        );
        assert_eq!(
            observed,
            expected_root(&region, &negated),
            "the {flavour} destination"
        );
    }
}

#[test]
#[ignore = "requires CUDA"]
fn scale_tensor_write_scales_a_strided_destination_view_in_place() {
    let Some(mut engine) = engine() else { return };
    for (flavour, shape, strides, offset) in write_view_flavours() {
        let mut root = write_root(&engine);
        let region = region_indices(&shape, &strides, offset);
        let scaled: Vec<f64> = region.iter().map(|&index| 2.5 * sentinel(index)).collect();
        let observed = write_through_view(
            &mut engine,
            &mut root,
            shape.clone(),
            strides.clone(),
            offset,
            |engine, out| engine.scale_tensor_write(out, 2.5, "scale_tensor_write test"),
        );
        assert_eq!(
            observed,
            expected_root(&region, &scaled),
            "the {flavour} destination"
        );
    }

    // A factor of one leaves the region untouched, exactly as for an owned tensor.
    let mut root = write_root(&engine);
    let observed = write_through_view(
        &mut engine,
        &mut root,
        vec![2, 3],
        vec![1, 6],
        7,
        |engine, out| engine.scale_tensor_write(out, 1.0, "scale_tensor_write test"),
    );
    assert_eq!(
        observed,
        (0..30).map(sentinel).collect::<Vec<f64>>(),
        "a unit factor changed the root"
    );
}
