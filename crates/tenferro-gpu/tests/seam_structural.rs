#![cfg(feature = "cuda-next")]

//! U5 first slice: the `TensorStructural` surface on the new engine — canonical
//! materialization, a same-placement copy and a dtype cast, through the trait rather than the
//! typed dispatcher (issue #2208 U5).

use tenferro_gpu::cuda_next::{
    download_tensor, gpu_available, upload_tensor, CudaDeviceId, CudaEngine,
};
use tenferro_tensor::{DType, Tensor, TensorRead, TensorStructural, TensorWrite};

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
