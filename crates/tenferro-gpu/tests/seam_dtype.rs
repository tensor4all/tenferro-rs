#![cfg(feature = "cuda-next")]

//! The dtype policy of the new engine's tensor surface: what the kernel catalogue serves
//! through the shared traits, and what it refuses (issue #2208 U5/U6 dtype scope).

use tenferro_gpu::cuda_next::{
    download_tensor, gpu_available, upload_tensor, CudaDeviceId, CudaEngine,
};
use tenferro_tensor::{
    Complex32, Complex64, DType, Tensor, TensorAnalytic, TensorElementwise, TensorRead,
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
