#![cfg(feature = "cuda-next")]

//! The dtype policy of the new engine's tensor surface: what the kernel catalogue serves
//! through the shared traits, and what it refuses (issue #2208 U5/U6 dtype scope).

use tenferro_gpu::cuda_next::{
    download_tensor, gpu_available, upload_tensor, CudaDeviceId, CudaEngine,
};
use tenferro_tensor::{
    Complex32, Complex64, DType, Tensor, TensorAnalytic, TensorElementwise, TensorRead,
    TensorStructural,
};

fn engine() -> Option<CudaEngine> {
    if !gpu_available() {
        return None;
    }
    Some(CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("CUDA engine"))
}

fn upload(engine: &CudaEngine, host: Tensor) -> Tensor {
    upload_tensor(engine.runtime(), &host).expect("upload")
}

fn read<T: tenferro_tensor::TensorScalar + Clone>(engine: &CudaEngine, tensor: &Tensor) -> Vec<T> {
    download_tensor(engine.runtime(), tensor)
        .expect("download")
        .as_slice::<T>()
        .expect("typed payload")
        .to_vec()
}

#[test]
fn complex_elementwise_and_the_complex_absolute_value() {
    let Some(mut engine) = engine() else { return };
    let host = Tensor::from_vec_col_major(
        vec![3],
        vec![
            Complex64::new(1.0, -2.0),
            Complex64::new(-3.0, 0.0),
            Complex64::new(0.0, 4.0),
        ],
    )
    .expect("host");
    let source = upload(&engine, host);

    // Conjugation is the catalogue's `Conj`, served for both complex widths.
    let conjugated = engine
        .conj_read(TensorRead::from_tensor(&source))
        .expect("conj");
    assert_eq!(
        read::<Complex64>(&engine, &conjugated),
        vec![
            Complex64::new(1.0, 2.0),
            Complex64::new(-3.0, 0.0),
            Complex64::new(0.0, -4.0),
        ]
    );

    // The absolute value of a complex tensor is real: the one unequal dtype pair.
    let magnitude = engine
        .abs_read(TensorRead::from_tensor(&source))
        .expect("abs");
    assert_eq!(magnitude.dtype(), DType::F64);
    let values = read::<f64>(&engine, &magnitude);
    let expected = [5.0f64.sqrt(), 3.0, 4.0];
    for (observed, expected) in values.iter().zip(expected) {
        assert!(
            (observed - expected).abs() < 1e-12,
            "{observed} != {expected}"
        );
    }

    // C32 travels the same route.
    let host32 = Tensor::from_vec_col_major(
        vec![2],
        vec![Complex32::new(1.0, 1.0), Complex32::new(2.0, -1.0)],
    )
    .expect("host32");
    let source32 = upload(&engine, host32);
    let sum32 = engine
        .add_read(
            TensorRead::from_tensor(&source32),
            TensorRead::from_tensor(&source32),
        )
        .expect("add32");
    assert_eq!(
        read::<Complex32>(&engine, &sum32),
        vec![Complex32::new(2.0, 2.0), Complex32::new(4.0, -2.0)]
    );
}

#[test]
fn integer_and_boolean_dtypes_follow_the_catalogue() {
    let Some(mut engine) = engine() else { return };
    // Integer elementwise arithmetic is served.
    let host = Tensor::from_vec_col_major(vec![3], vec![3i64, -4, 5]).expect("host");
    let source = upload(&engine, host);
    let doubled = engine
        .mul_read(
            TensorRead::from_tensor(&source),
            TensorRead::from_tensor(&source),
        )
        .expect("mul");
    assert_eq!(read::<i64>(&engine, &doubled), vec![9i64, 16, 25]);

    // An analytic operation the catalogue does not serve for integers is refused, and the
    // refusal is the backend not serving the request rather than a runtime failure.
    let square_root = engine.sqrt_read(TensorRead::from_tensor(&source));
    let refused = square_root.expect_err("integer square root");
    assert!(
        matches!(refused, tenferro_tensor::Error::Unsupported { .. }),
        "expected an unsupported-operation refusal, got {refused:?}"
    );
}

#[test]
fn analytic_operations_and_power_follow_the_contract() {
    let Some(mut engine) = engine() else { return };
    let data = vec![0.25f64, 0.5, 1.0, 4.0];
    let source = upload(
        &engine,
        Tensor::from_vec_col_major(vec![4], data.clone()).expect("host"),
    );

    let exponential = engine
        .exp_read(TensorRead::from_tensor(&source))
        .expect("exp");
    let observed = read::<f64>(&engine, &exponential);
    for (observed, value) in observed.iter().zip(&data) {
        assert!((observed - value.exp()).abs() < 1e-12);
    }

    let logarithm = engine
        .log_read(TensorRead::from_tensor(&source))
        .expect("log");
    let observed = read::<f64>(&engine, &logarithm);
    for (observed, value) in observed.iter().zip(&data) {
        assert!((observed - value.ln()).abs() < 1e-12);
    }

    let square_root = engine
        .sqrt_read(TensorRead::from_tensor(&source))
        .expect("sqrt");
    let observed = read::<f64>(&engine, &square_root);
    for (observed, value) in observed.iter().zip(&data) {
        assert!((observed - value.sqrt()).abs() < 1e-12);
    }

    // Float power over two same-shaped tensors, and a rank-0 exponent as the scalar side.
    let exponent = upload(
        &engine,
        Tensor::from_vec_col_major(vec![4], vec![2.0f64; 4]).expect("host"),
    );
    let powered = engine
        .pow_read(
            TensorRead::from_tensor(&source),
            TensorRead::from_tensor(&exponent),
        )
        .expect("pow");
    let observed = read::<f64>(&engine, &powered);
    for (observed, value) in observed.iter().zip(&data) {
        assert!((observed - value.powf(2.0)).abs() < 1e-12);
    }
    let scalar_exponent = upload(
        &engine,
        Tensor::from_vec_col_major(Vec::new(), vec![3.0f64]).expect("host"),
    );
    let cubed = engine
        .pow_read(
            TensorRead::from_tensor(&source),
            TensorRead::from_tensor(&scalar_exponent),
        )
        .expect("scalar pow");
    assert_eq!(cubed.shape(), &[4]);
    let observed = read::<f64>(&engine, &cubed);
    for (observed, value) in observed.iter().zip(&data) {
        assert!((observed - value.powf(3.0)).abs() < 1e-12);
    }

    // Integer power is exact where it fits, `0^0` is one, and a negative exponent is a
    // numerical failure rather than a silent wrap.
    let integers = upload(
        &engine,
        Tensor::from_vec_col_major(vec![3], vec![0i64, 2, -3]).expect("host"),
    );
    let three = upload(
        &engine,
        Tensor::from_vec_col_major(vec![3], vec![0i64, 3, 2]).expect("host"),
    );
    let powered = engine
        .pow_read(
            TensorRead::from_tensor(&integers),
            TensorRead::from_tensor(&three),
        )
        .expect("integer pow");
    assert_eq!(read::<i64>(&engine, &powered), vec![1i64, 8, 9]);

    let negative = upload(
        &engine,
        Tensor::from_vec_col_major(vec![3], vec![-1i64, -1, -1]).expect("host"),
    );
    let refused = engine
        .pow_read(
            TensorRead::from_tensor(&three),
            TensorRead::from_tensor(&negative),
        )
        .expect_err("negative exponent");
    assert!(
        matches!(refused.kind(), tenferro_tensor::ErrorKind::NumericalFailure),
        "expected a numerical failure, got {refused:?}"
    );
}

#[test]
fn a_float_to_integer_cast_is_checked_before_it_truncates() {
    let Some(mut engine) = engine() else { return };
    // In-range values convert.
    let host = Tensor::from_vec_col_major(vec![3], vec![1.75f64, -2.5, 0.0]).expect("host");
    let source = upload(&engine, host);
    let narrowed = engine.cast(&source, DType::I64).expect("cast");
    assert_eq!(read::<i64>(&engine, &narrowed), vec![1i64, -2, 0]);

    // A non-finite or out-of-range value is refused instead of being an undefined C++ cast.
    for bad in [f64::NAN, f64::INFINITY, 1.0e300] {
        let host = Tensor::from_vec_col_major(vec![2], vec![1.0f64, bad]).expect("host");
        let source = upload(&engine, host);
        let refused = engine.cast(&source, DType::I64).expect_err("bad cast");
        assert!(
            matches!(refused, tenferro_tensor::Error::Validation { .. }),
            "expected a validation refusal for {bad}, got {refused:?}"
        );
    }

    // The complex route judges the real component.
    let complex = Tensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(2.0, 5.0), Complex64::new(f64::NAN, 0.0)],
    )
    .expect("host");
    let source = upload(&engine, complex);
    assert!(engine.cast(&source, DType::I32).is_err());
}

#[test]
fn integer_and_complex_division_use_their_own_entries() {
    let Some(mut engine) = engine() else { return };
    // Integer division is the checked one: exact where it divides, and a reported failure where it
    // does not.
    let host = Tensor::from_vec_col_major(vec![3], vec![7i64, -8, 9]).expect("host");
    let left = upload(&engine, host);
    let divisor = upload(
        &engine,
        Tensor::from_vec_col_major(vec![3], vec![2i64, 4, 3]).expect("host"),
    );
    let quotient = engine
        .div_read(
            TensorRead::from_tensor(&left),
            TensorRead::from_tensor(&divisor),
        )
        .expect("integer division");
    assert_eq!(read::<i64>(&engine, &quotient), vec![3i64, -2, 3]);
    let zero = upload(
        &engine,
        Tensor::from_vec_col_major(vec![3], vec![0i64, 1, 1]).expect("host"),
    );
    assert!(
        engine
            .div_read(
                TensorRead::from_tensor(&left),
                TensorRead::from_tensor(&zero)
            )
            .is_err(),
        "a division by zero is reported, not silently zero"
    );

    // Complex division uses the robust entry.
    let numerator = upload(
        &engine,
        Tensor::from_vec_col_major(vec![1], vec![Complex64::new(1.0, 2.0)]).expect("host"),
    );
    let denominator = upload(
        &engine,
        Tensor::from_vec_col_major(vec![1], vec![Complex64::new(3.0, -4.0)]).expect("host"),
    );
    let ratio = engine
        .div_read(
            TensorRead::from_tensor(&numerator),
            TensorRead::from_tensor(&denominator),
        )
        .expect("complex division");
    let observed = read::<Complex64>(&engine, &ratio);
    // (1 + 2i) / (3 - 4i) = (-1 + 2i) / 5
    assert!((observed[0].re + 0.2).abs() < 1e-12, "{:?}", observed[0]);
    assert!((observed[0].im - 0.4).abs() < 1e-12, "{:?}", observed[0]);
}
