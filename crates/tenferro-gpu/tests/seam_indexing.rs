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

#[test]
fn gather_reads_a_window_per_index_vector() {
    let Some(mut engine) = engine() else { return };
    // Operand [3, 4]: value = row + col * 3.
    let operand = device(&engine, vec![3, 4], matrix(3, 4));
    let indices_host = Tensor::from_vec_col_major(vec![2, 1], vec![0i64, 1]).expect("indices");
    let indices = upload_tensor(engine.runtime(), &indices_host).expect("upload indices");
    let gathered = engine
        .gather(
            &operand,
            &indices,
            &tenferro_tensor::GatherConfig {
                offset_dims: vec![1],
                collapsed_slice_dims: vec![0],
                start_index_map: vec![0],
                index_vector_dim: 1,
                slice_sizes: vec![1, 2],
            },
        )
        .expect("gather");
    // Two batches of a 1x2 window, taken from rows 0 and 1, columns 0..2.
    assert_eq!(gathered.shape(), &[2, 2]);
    assert_eq!(read(&engine, &gathered), vec![0.0, 1.0, 3.0, 4.0]);
}

#[test]
fn scatter_adds_each_update_into_the_operand_copy() {
    let Some(mut engine) = engine() else { return };
    let operand = device(&engine, vec![2, 2], vec![0.0; 4]);
    let indices_host = Tensor::from_vec_col_major(vec![2], vec![1i64, 0]).expect("indices");
    let indices = upload_tensor(engine.runtime(), &indices_host).expect("upload indices");
    let updates_host = Tensor::from_vec_col_major(vec![2], vec![5.0, 7.0]).expect("updates");
    let updates = upload_tensor(engine.runtime(), &updates_host).expect("upload updates");
    let scattered = engine
        .scatter(
            &operand,
            &indices,
            &updates,
            &tenferro_tensor::ScatterConfig {
                update_window_dims: vec![],
                inserted_window_dims: vec![0, 1],
                scatter_dims_to_operand_dims: vec![0],
                index_vector_dim: 1,
            },
        )
        .expect("scatter");
    assert_eq!(scattered.shape(), &[2, 2]);
    // Row 1 column 0 gets 5, row 0 column 0 gets 7.
    assert_eq!(read(&engine, &scattered), vec![7.0, 5.0, 0.0, 0.0]);
    // The operand itself is untouched: scatter returns a new value.
    assert_eq!(read(&engine, &operand), vec![0.0; 4]);
}

#[test]
fn an_empty_window_is_refused_or_produces_an_empty_result() {
    let Some(mut engine) = engine() else { return };
    let source = device(&engine, vec![2, 2], matrix(2, 2));
    // An empty window produces an empty result without launching anything.
    let empty = engine
        .slice(
            &source,
            &SliceConfig {
                starts: vec![1, 1],
                limits: vec![1, 2],
                strides: vec![1, 1],
            },
        )
        .expect("empty slice");
    assert_eq!(empty.shape(), &[0, 1]);
    assert!(read(&engine, &empty).is_empty());

    // Padding an empty axis keeps the zero-extent base, so the extents are low + high.
    let empty_source = device(&engine, vec![0], Vec::new());
    let padded = engine
        .pad(
            &empty_source,
            &PadConfig {
                edge_padding_low: vec![1],
                edge_padding_high: vec![1],
                interior_padding: vec![1],
            },
        )
        .expect("pad an empty axis");
    assert_eq!(padded.shape(), &[2]);
    assert_eq!(read(&engine, &padded), vec![0.0, 0.0]);
}

#[test]
fn mismatched_index_metadata_is_refused_before_any_device_read() {
    let Some(mut engine) = engine() else { return };
    let source = device(&engine, vec![4, 5], matrix(4, 5));
    // One start for a rank-2 operand would make the kernel read past the allocation.
    let one_start = Tensor::from_vec_col_major(vec![1], vec![1i64]).expect("starts");
    let one_start = upload_tensor(engine.runtime(), &one_start).expect("upload starts");
    assert!(engine.dynamic_slice(&source, &one_start, &[2, 3]).is_err());

    // A degenerate gather configuration is refused instead of panicking.
    let scalar_index = Tensor::from_vec_col_major(vec![1], vec![0i64]).expect("index");
    let scalar_index = upload_tensor(engine.runtime(), &scalar_index).expect("upload index");
    assert!(engine
        .gather(
            &device(&engine, vec![1], vec![0.0]),
            &scalar_index,
            &tenferro_tensor::GatherConfig {
                offset_dims: vec![0],
                collapsed_slice_dims: vec![0],
                start_index_map: vec![0],
                index_vector_dim: 1,
                slice_sizes: vec![1],
            },
        )
        .is_err());

    // Scatter refuses an update whose dtype differs from the operand's.
    let indices = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![1], vec![0i64]).expect("indices"),
    )
    .expect("upload indices");
    let wrong_updates = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![1], vec![1i32]).expect("updates"),
    )
    .expect("upload updates");
    assert!(engine
        .scatter(
            &source,
            &indices,
            &wrong_updates,
            &tenferro_tensor::ScatterConfig {
                update_window_dims: vec![],
                inserted_window_dims: vec![0, 1],
                scatter_dims_to_operand_dims: vec![0],
                index_vector_dim: 1,
            },
        )
        .is_err());
}

#[test]
fn float_indices_are_checked_for_exactness() {
    let Some(mut engine) = engine() else { return };
    let source = device(&engine, vec![4, 5], matrix(4, 5));
    let gather_config = tenferro_tensor::GatherConfig {
        offset_dims: vec![1],
        collapsed_slice_dims: vec![0],
        start_index_map: vec![0],
        index_vector_dim: 1,
        slice_sizes: vec![1, 2],
    };

    // An exactly representable float start is served, and the window holds the right values.
    let exact = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![2], vec![1.0f32, 2.0]).expect("starts"),
    )
    .expect("upload starts");
    let sliced = engine
        .dynamic_slice(&source, &exact, &[2, 3])
        .expect("exact float starts");
    assert_eq!(sliced.shape(), &[2, 3]);
    assert_eq!(
        read(&engine, &sliced),
        vec![9.0, 10.0, 13.0, 14.0, 17.0, 18.0]
    );

    // A fractional start is refused, and the error names the offending element.
    let fractional = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![2], vec![1.5f32, 2.0]).expect("starts"),
    )
    .expect("upload starts");
    let refused = engine
        .dynamic_slice(&source, &fractional, &[2, 3])
        .expect_err("fractional start");
    assert!(
        format!("{refused}").contains("index element 0"),
        "the refusal names the offending element: {refused}"
    );

    // A NaN, an infinity and a value past the exact integer range are refused too — the first
    // two for the window starts, the last for an F64 index tensor.
    for bad in [f32::NAN, f32::INFINITY, -f32::INFINITY] {
        let starts = upload_tensor(
            engine.runtime(),
            &Tensor::from_vec_col_major(vec![2], vec![bad, 1.0]).expect("starts"),
        )
        .expect("upload starts");
        assert!(
            engine.dynamic_slice(&source, &starts, &[1, 1]).is_err(),
            "{bad} must be refused"
        );
    }
    let operand = device(&engine, vec![3, 4], matrix(3, 4));
    for bad in [1.5f64, f64::NAN, 1.0e300] {
        let indices = upload_tensor(
            engine.runtime(),
            &Tensor::from_vec_col_major(vec![2, 1], vec![0.0f64, bad]).expect("indices"),
        )
        .expect("upload indices");
        assert!(
            engine.gather(&operand, &indices, &gather_config).is_err(),
            "{bad} must be refused"
        );
    }

    // The same check guards a scatter's index tensor.
    let updates = device(&engine, vec![2], vec![5.0, 7.0]);
    let fractional_indices = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![2], vec![0.5f32, 1.0]).expect("indices"),
    )
    .expect("upload indices");
    assert!(engine
        .scatter(
            &operand,
            &fractional_indices,
            &updates,
            &tenferro_tensor::ScatterConfig {
                update_window_dims: vec![],
                inserted_window_dims: vec![0, 1],
                scatter_dims_to_operand_dims: vec![0],
                index_vector_dim: 1,
            },
        )
        .is_err());
}

#[test]
fn a_rank_nine_concatenation_uses_the_general_path() {
    let Some(mut engine) = engine() else { return };
    // Nine axes exceed the by-value layout form. Concatenating the leading axis interleaves the
    // inputs, so a wrong axis or offset cannot pass as a flat append.
    let first_shape = vec![1usize, 2, 1, 1, 1, 1, 1, 1, 2];
    let second_shape = vec![2usize, 2, 1, 1, 1, 1, 1, 1, 2];
    let first_data: Vec<f64> = (0..4).map(|index| index as f64).collect();
    let second_data: Vec<f64> = (0..8).map(|index| 100.0 + index as f64).collect();
    let first = device(&engine, first_shape.clone(), first_data.clone());
    let second = device(&engine, second_shape, second_data.clone());
    let joined = engine
        .concatenate(&[&first, &second], 0)
        .expect("rank-9 concatenate");
    let mut out_shape = first_shape.clone();
    out_shape[0] = 3;
    assert_eq!(joined.shape(), &out_shape[..]);
    // Column-major: axis 0 is fastest, then axis 1, then axis 8.
    let mut expected = Vec::new();
    for eighth in 0..2 {
        for second_axis in 0..2 {
            // The `[1, 2, .., 2]` source: axis 1 has stride 1, axis 8 has stride 2.
            expected.push(first_data[second_axis + eighth * 2]);
            expected.push(second_data[second_axis * 2 + eighth * 2 * 2]);
            expected.push(second_data[1 + second_axis * 2 + eighth * 2 * 2]);
        }
    }
    assert_eq!(read(&engine, &joined), expected);

    // An empty input is a placement with nothing to copy; it must not break the sequence.
    let empty_shape = vec![0usize, 2, 1, 1, 1, 1, 1, 1, 2];
    let empty = device(&engine, empty_shape, Vec::new());
    let joined = engine
        .concatenate(&[&empty, &first, &second], 0)
        .expect("concatenate with an empty input");
    assert_eq!(joined.shape(), &out_shape[..]);
    assert_eq!(read(&engine, &joined), expected);
}

#[test]
fn a_rank_nine_window_uses_the_general_path() {
    let Some(mut engine) = engine() else { return };
    // Nine axes: the layouts and the per-axis window arrays all travel in device memory.
    let shape = vec![3usize, 1, 1, 1, 1, 1, 1, 1, 4];
    let data: Vec<f64> = (0..12).map(|index| index as f64).collect();
    let source = device(&engine, shape.clone(), data.clone());

    let sliced = engine
        .slice(
            &source,
            &SliceConfig {
                starts: vec![1, 0, 0, 0, 0, 0, 0, 0, 1],
                limits: vec![3, 1, 1, 1, 1, 1, 1, 1, 4],
                strides: vec![1, 1, 1, 1, 1, 1, 1, 1, 2],
            },
        )
        .expect("rank-9 slice");
    assert_eq!(sliced.shape(), &[2, 1, 1, 1, 1, 1, 1, 1, 2]);
    // Rows 1..3 of columns 1 and 3; the source is row + 3 * column.
    assert_eq!(read(&engine, &sliced), vec![4.0, 5.0, 10.0, 11.0]);

    // Interior padding on a non-singleton axis plus edge padding on the trailing axis: the
    // interior array must reach the kernel, and `edge_high` must not be mistaken for it.
    let mut interior = vec![0i64; 9];
    interior[0] = 1;
    let mut high = vec![0i64; 9];
    high[8] = 1;
    let padded = engine
        .pad(
            &source,
            &PadConfig {
                edge_padding_low: vec![0; 9],
                edge_padding_high: high,
                interior_padding: interior,
            },
        )
        .expect("rank-9 pad");
    // Axis 0: (3 - 1) * 2 + 1 = 5; axis 8: 4 + 1 = 5.
    assert_eq!(padded.shape(), &[5, 1, 1, 1, 1, 1, 1, 1, 5]);
    let observed = read(&engine, &padded);
    assert_eq!(observed.len(), 25);
    for column in 0..5 {
        for row in 0..5 {
            // Odd rows are interior padding; the last column is edge padding.
            let expected = if column == 4 || row % 2 == 1 {
                0.0
            } else {
                data[(row / 2) + column * 3]
            };
            assert_eq!(
                observed[row + column * 5],
                expected,
                "row {row} column {column}"
            );
        }
    }
}

#[test]
fn a_rank_nine_dynamic_slice_uses_the_general_path() {
    let Some(mut engine) = engine() else { return };
    let shape = vec![4usize, 1, 1, 1, 1, 1, 1, 1, 5];
    let data: Vec<f64> = (0..20).map(|index| index as f64).collect();
    let source = device(&engine, shape.clone(), data.clone());
    // One start per operand axis; the window is 2 x 1 x ... x 3.
    let starts_host =
        Tensor::from_vec_col_major(vec![9], vec![1i64, 0, 0, 0, 0, 0, 0, 0, 2]).expect("starts");
    let starts = upload_tensor(engine.runtime(), &starts_host).expect("upload starts");
    let mut sizes = vec![1usize; 9];
    sizes[0] = 2;
    sizes[8] = 3;
    let sliced = engine
        .dynamic_slice(&source, &starts, &sizes)
        .expect("rank-9 dynamic slice");
    assert_eq!(sliced.shape(), &sizes[..]);
    // Rows 1..3 of columns 2..5, column-major with axis 8 outermost.
    let mut expected = Vec::new();
    for column in 2..5 {
        for row in 1..3 {
            expected.push(data[row + column * 4]);
        }
    }
    assert_eq!(read(&engine, &sliced), expected);
}
