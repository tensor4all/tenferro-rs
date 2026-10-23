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
fn the_unported_families_refuse_instead_of_transferring() {
    let Some(mut engine) = engine() else { return };
    let source = device(&engine, values(4));
    assert!(engine.tril(&source, 0).is_err());
    assert!(engine.triu(&source, 0).is_err());
    assert!(engine.extract_diagonal(&source, 0, 0).is_err());
    assert!(engine.embed_diagonal(&source, 0, 0).is_err());
}
