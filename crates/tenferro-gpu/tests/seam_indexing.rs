#![cfg(feature = "cuda-next")]

//! U5 tail: the `TensorIndexing` surface on the new engine — windowing, padding,
//! concatenation and reversal, through the trait rather than the typed dispatcher
//! (issue #2208 U5).

use tenferro_gpu::cuda_next::{
    download_tensor, gpu_available, upload_tensor, CudaDeviceId, CudaEngine,
};
use tenferro_tensor::{PadConfig, SliceConfig, Tensor, TensorIndexing};

fn engine() -> Option<CudaEngine> {
    if !gpu_available() {
        return None;
    }
    Some(CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("CUDA engine"))
}

/// `rows x cols` column-major values, so an element is `row + col * rows`.
fn matrix(rows: usize, cols: usize) -> Vec<f64> {
    (0..rows * cols).map(|index| index as f64).collect()
}

fn device(engine: &CudaEngine, shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    let host = Tensor::from_vec_col_major(shape, data).expect("host tensor");
    upload_tensor(engine.runtime(), &host).expect("upload")
}

fn read(engine: &CudaEngine, tensor: &Tensor) -> Vec<f64> {
    let back = download_tensor(engine.runtime(), tensor).expect("download");
    back.as_slice::<f64>().expect("f64 payload").to_vec()
}

#[test]
fn a_slice_window_reads_the_strided_region() {
    let Some(mut engine) = engine() else { return };
    let source = device(&engine, vec![4, 5], matrix(4, 5));
    // rows 1..3 and columns 1, 3 — column stride 2.
    let sliced = engine
        .slice(
            &source,
            &SliceConfig {
                starts: vec![1, 1],
                limits: vec![3, 5],
                strides: vec![1, 2],
            },
        )
        .expect("slice");
    assert_eq!(sliced.shape(), &[2, 2]);
    // (row, col) of the source at `(start + index * step)`.
    let expected = vec![5.0, 6.0, 13.0, 14.0];
    assert_eq!(read(&engine, &sliced), expected);

    // A limit past the dimension is refused before any device work.
    assert!(engine
        .slice(
            &source,
            &SliceConfig {
                starts: vec![0, 0],
                limits: vec![5, 5],
                strides: vec![1, 1],
            },
        )
        .is_err());
}

#[test]
fn pad_fills_edges_and_interior_with_zero() {
    let Some(mut engine) = engine() else { return };
    let source = device(&engine, vec![2, 2], matrix(2, 2));
    let padded = engine
        .pad(
            &source,
            &PadConfig {
                edge_padding_low: vec![1, 0],
                edge_padding_high: vec![0, 1],
                interior_padding: vec![1, 1],
            },
        )
        .expect("pad");
    // rows: (2 - 1) * (1 + 1) + 1 + 1 = 4; cols: (2 - 1) * (1 + 1) + 1 + 1 = 4.
    assert_eq!(padded.shape(), &[4, 4]);
    // Source row 0 sits at row 1, row 1 at row 3; the same holds for the columns.
    let expected = vec![
        0.0, 0.0, 0.0, 1.0, // column 0
        0.0, 0.0, 0.0, 0.0, // column 1 is interior padding
        0.0, 2.0, 0.0, 3.0, // column 2
        0.0, 0.0, 0.0, 0.0, // column 3 is edge padding
    ];
    assert_eq!(read(&engine, &padded), expected);

    // A negative interior padding is refused.
    assert!(engine
        .pad(
            &source,
            &PadConfig {
                edge_padding_low: vec![0, 0],
                edge_padding_high: vec![0, 0],
                interior_padding: vec![-1, 0],
            },
        )
        .is_err());
}

#[test]
fn a_dynamic_slice_takes_its_starts_from_the_device() {
    let Some(mut engine) = engine() else { return };
    let source = device(&engine, vec![4, 5], matrix(4, 5));
    let starts_host = Tensor::from_vec_col_major(vec![2], vec![1i64, 2]).expect("starts");
    let starts = upload_tensor(engine.runtime(), &starts_host).expect("upload starts");
    let sliced = engine
        .dynamic_slice(&source, &starts, &[2, 3])
        .expect("dynamic_slice");
    assert_eq!(sliced.shape(), &[2, 3]);
    // Rows 1..3 of columns 2..5.
    let expected = vec![9.0, 10.0, 13.0, 14.0, 17.0, 18.0];
    assert_eq!(read(&engine, &sliced), expected);

    // A window wider than the operand is refused.
    assert!(engine.dynamic_slice(&source, &starts, &[5, 3]).is_err());
}

#[test]
fn concatenate_places_each_input_along_the_axis() {
    let Some(mut engine) = engine() else { return };
    let first = device(&engine, vec![2, 2], vec![0.0, 1.0, 2.0, 3.0]);
    let second = device(&engine, vec![1, 2], vec![4.0, 5.0]);
    let joined = engine
        .concatenate(&[&first, &second], 0)
        .expect("concatenate");
    assert_eq!(joined.shape(), &[3, 2]);
    let expected = vec![0.0, 1.0, 4.0, 2.0, 3.0, 5.0];
    assert_eq!(read(&engine, &joined), expected);

    // Mismatched non-concatenated extents are refused.
    let other = device(&engine, vec![1, 3], vec![0.0; 3]);
    assert!(engine.concatenate(&[&first, &other], 0).is_err());
}

#[test]
fn reverse_negates_only_the_named_axes() {
    let Some(mut engine) = engine() else { return };
    let source = device(&engine, vec![2, 3], matrix(2, 3));
    let reversed = engine.reverse(&source, &[0]).expect("reverse");
    assert_eq!(reversed.shape(), &[2, 3]);
    let expected = vec![1.0, 0.0, 3.0, 2.0, 5.0, 4.0];
    assert_eq!(read(&engine, &reversed), expected);

    // A repeated axis is refused rather than silently negating twice.
    assert!(engine.reverse(&source, &[1, 1]).is_err());
}
