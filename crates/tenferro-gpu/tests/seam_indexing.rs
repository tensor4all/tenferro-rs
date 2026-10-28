#![cfg(feature = "cuda")]

//! U5 tail: the `TensorIndexing` surface on the new engine — windowing, padding,
//! concatenation and reversal, through the trait rather than the typed dispatcher
//! (issue #2208 U5).

use tenferro_gpu::cuda::{download_tensor, gpu_available, upload_tensor, CudaDeviceId, CudaEngine};
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
#[ignore = "requires CUDA"]
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
#[ignore = "requires CUDA"]
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
#[ignore = "requires CUDA"]
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
#[ignore = "requires CUDA"]
fn a_rank_zero_dynamic_slice_returns_the_scalar_for_owned_and_borrowed_starts() {
    let Some(mut engine) = engine() else { return };
    let scalar = device(&engine, vec![], vec![7.5]);
    // Owned empty starts: an empty allocation has no device address to read.
    let empty = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![0], Vec::<i64>::new()).expect("empty starts"),
    )
    .expect("upload empty starts");
    let owned = engine
        .dynamic_slice(&scalar, &empty, &[])
        .expect("rank-0 dynamic_slice with owned starts");
    assert_eq!(owned.shape(), &[] as &[usize]);
    assert_eq!(read(&engine, &owned), vec![7.5]);

    // Borrowed empty starts at a nonzero offset into a nonempty allocation behave the same.
    let backing = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![2], vec![3i64, 4]).expect("starts backing"),
    )
    .expect("upload starts backing");
    let view = backing
        .as_typed::<i64>()
        .expect("i64 payload")
        .backend_region_view(vec![0], vec![1], 1)
        .expect("empty view");
    let borrowed = engine
        .dynamic_slice_read(
            tenferro_tensor::TensorRead::from_tensor(&scalar),
            tenferro_tensor::TensorRead::from_view(tenferro_tensor::TensorView::I64(view)),
            &[],
        )
        .expect("rank-0 dynamic_slice with borrowed starts");
    assert_eq!(read(&engine, &borrowed), vec![7.5]);
}

#[test]
#[ignore = "requires CUDA"]
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
#[ignore = "requires CUDA"]
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
#[ignore = "requires CUDA"]
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
#[ignore = "requires CUDA"]
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
#[ignore = "requires CUDA"]
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
#[ignore = "requires CUDA"]
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
#[ignore = "requires CUDA"]
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
#[ignore = "requires CUDA"]
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
#[ignore = "requires CUDA"]
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
#[ignore = "requires CUDA"]
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

#[test]
#[ignore = "requires CUDA"]
fn a_rank_nine_gather_and_scatter_use_the_general_path() {
    let Some(mut engine) = engine() else { return };
    // A rank-9 operand whose window rides axis 8 and whose start selects axis 7, so the plan's
    // window, offset, start-map and slice-size arrays are all non-empty and distinct: a swapped
    // pointer order would fail this.
    let operand_shape = vec![3usize, 1, 1, 1, 1, 1, 1, 2, 4];
    let operand_data: Vec<f64> = (0..24).map(|index| index as f64).collect();
    let operand = device(&engine, operand_shape.clone(), operand_data.clone());
    // Column-major strides of the operand: axis 0 fastest, then axis 7, then axis 8.
    let (stride_7, stride_8) = (3usize, 6);
    let indices_host = Tensor::from_vec_col_major(vec![2, 1], vec![0i64, 1]).expect("indices");
    let indices = upload_tensor(engine.runtime(), &indices_host).expect("upload indices");

    let gathered = engine
        .gather(
            &operand,
            &indices,
            &tenferro_tensor::GatherConfig {
                // The window rides operand axis 8 and lands on the second output axis.
                offset_dims: vec![1],
                collapsed_slice_dims: vec![0, 1, 2, 3, 4, 5, 6, 7],
                start_index_map: vec![7],
                index_vector_dim: 1,
                slice_sizes: vec![1, 1, 1, 1, 1, 1, 1, 1, 2],
            },
        )
        .expect("rank-9 gather");
    // One batch per start, each taking the two-element window along axis 8 of its axis-7 row.
    assert_eq!(gathered.shape(), &[2, 2]);
    let mut expected = Vec::new();
    for window in 0..2 {
        for batch in 0..2 {
            expected.push(operand_data[batch * stride_7 + window * stride_8]);
        }
    }
    assert_eq!(read(&engine, &gathered), expected);

    // The additive scatter mirrors it: one update window per start, added into the operand copy.
    let update_shape = vec![2usize, 2];
    let updates_data = vec![1.0f64, 2.0, 3.0, 4.0];
    let updates = device(&engine, update_shape, updates_data.clone());
    let scattered = engine
        .scatter(
            &operand,
            &indices,
            &updates,
            &tenferro_tensor::ScatterConfig {
                update_window_dims: vec![1],
                inserted_window_dims: vec![0, 1, 2, 3, 4, 5, 6, 7],
                scatter_dims_to_operand_dims: vec![7],
                index_vector_dim: 1,
            },
        )
        .expect("rank-9 scatter");
    assert_eq!(scattered.shape(), &operand_shape[..]);
    let mut expected = operand_data;
    for window in 0..2 {
        for batch in 0..2 {
            expected[batch * stride_7 + window * stride_8] += updates_data[batch + window * 2];
        }
    }
    assert_eq!(read(&engine, &scattered), expected);
}

// --- Native read entries: the operand is read where it lives (#2223 item 1) ---

use tenferro_tensor::{TensorRead, TensorStructural, TensorView};

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

/// A borrowed region of an `i64` index tensor, as one read target.
fn i64_region(
    tensor: &Tensor,
    shape: Vec<usize>,
    strides: Vec<isize>,
    offset: isize,
) -> TensorRead<'_> {
    TensorRead::from_view(TensorView::I64(
        tensor
            .as_typed::<i64>()
            .expect("i64 payload")
            .backend_region_view(shape, strides, offset)
            .expect("region view"),
    ))
}

/// The three borrowed-region flavours every read entry has to serve on a `4 x 5`
/// column-major source: a transposed view, an offset view and a negative-stride view.
fn view_flavours(source: &Tensor) -> Vec<(&'static str, TensorRead<'_>)> {
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

#[test]
#[ignore = "requires CUDA"]
fn slice_read_follows_a_view_layout() {
    let Some(mut engine) = engine() else { return };
    let source = device(&engine, vec![4, 5], matrix(4, 5));
    for (flavour, view) in view_flavours(&source) {
        let shape = view.shape().to_vec();
        // Drop the first index on every axis that has one, so the window sits inside the view.
        let starts: Vec<usize> = shape
            .iter()
            .map(|&extent| usize::from(extent > 1))
            .collect();
        let config = SliceConfig {
            starts,
            limits: shape.clone(),
            strides: vec![1; shape.len()],
        };
        matches_the_compact_operand(&mut engine, flavour, view, |engine, input| {
            engine.slice_read(input, &config).expect("slice_read")
        });
    }
}

#[test]
#[ignore = "requires CUDA"]
fn dynamic_slice_read_follows_a_view_layout() {
    let Some(mut engine) = engine() else { return };
    let source = device(&engine, vec![4, 5], matrix(4, 5));
    let all_flavours = view_flavours(&source);
    for (flavour, view) in all_flavours {
        let shape = view.shape().to_vec();
        let starts_host =
            Tensor::from_vec_col_major(vec![shape.len()], vec![0i64; shape.len()]).expect("starts");
        let starts = upload_tensor(engine.runtime(), &starts_host).expect("upload starts");
        let sizes: Vec<usize> = shape.iter().map(|&extent| extent.min(2)).collect();
        matches_the_compact_operand(&mut engine, flavour, view, |engine, input| {
            engine
                .dynamic_slice_read(input, TensorRead::from_tensor(&starts), &sizes)
                .expect("dynamic_slice_read")
        });
    }
}

#[test]
#[ignore = "requires CUDA"]
fn pad_read_follows_a_view_layout() {
    let Some(mut engine) = engine() else { return };
    let source = device(&engine, vec![4, 5], matrix(4, 5));
    let config = PadConfig {
        edge_padding_low: vec![1, 0],
        edge_padding_high: vec![0, 2],
        interior_padding: vec![0, 1],
    };
    for (flavour, view) in view_flavours(&source) {
        matches_the_compact_operand(&mut engine, flavour, view, |engine, input| {
            engine.pad_read(input, &config).expect("pad_read")
        });
    }
}

#[test]
#[ignore = "requires CUDA"]
fn reverse_read_follows_a_view_layout() {
    let Some(mut engine) = engine() else { return };
    let source = device(&engine, vec![4, 5], matrix(4, 5));
    for (flavour, view) in view_flavours(&source) {
        matches_the_compact_operand(&mut engine, flavour, view, |engine, input| {
            engine.reverse_read(input, &[0]).expect("reverse_read")
        });
    }
    // Both axes of a negative-stride region in one call: every stride of the operand layout is
    // negated, so the offset must move to the far end on each of them.
    matches_the_compact_operand(
        &mut engine,
        "negative",
        f64_region(&source, vec![4, 2], vec![-1, 4], 3),
        |engine, input| engine.reverse_read(input, &[0, 1]).expect("reverse_read"),
    );
}

#[test]
#[ignore = "requires CUDA"]
fn gather_read_follows_a_view_layout() {
    let Some(mut engine) = engine() else { return };
    let source = device(&engine, vec![4, 5], matrix(4, 5));
    let indices_host = Tensor::from_vec_col_major(vec![2, 1], vec![0i64, 1]).expect("indices");
    let indices = upload_tensor(engine.runtime(), &indices_host).expect("upload indices");
    let config = tenferro_tensor::GatherConfig {
        offset_dims: vec![0],
        collapsed_slice_dims: vec![1],
        start_index_map: vec![0],
        index_vector_dim: 1,
        slice_sizes: vec![2, 1],
    };
    for (flavour, view) in view_flavours(&source) {
        // Two one-column windows taken at axis-0 starts 0 and 1 of the borrowed region.
        matches_the_compact_operand(&mut engine, flavour, view, |engine, operand| {
            use tenferro_tensor::Engine as _;
            engine
                .gather_read(operand, TensorRead::from_tensor(&indices), &config)
                .expect("gather_read")
        });
    }
}

#[test]
#[ignore = "requires CUDA"]
fn gather_reads_each_per_matrix_permutation_from_an_offset_view() {
    use tenferro_tensor::Engine as _;

    let Some(mut engine) = engine() else { return };
    let operand_values = vec![10.0, 20.0, 30.0, 40.0];
    let operand = device(&engine, vec![4], operand_values.clone());
    // One uploaded index tensor, one permutation per column.
    let permutations = vec![
        0i64, 1, 2, 3, // the identity
        3, 0, 1, 2, 2, 3, 0, 1,
    ];
    let indices = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![4, 3], permutations.clone()).expect("permutations"),
    )
    .expect("upload permutations");
    let config = tenferro_tensor::GatherConfig {
        offset_dims: vec![],
        collapsed_slice_dims: vec![0],
        start_index_map: vec![0],
        index_vector_dim: 1,
        slice_sizes: vec![1],
    };

    for column in 0..3 {
        let view = i64_region(&indices, vec![4], vec![1], (column * 4) as isize);
        let observed = engine
            .gather_read(TensorRead::from_tensor(&operand), view.clone(), &config)
            .expect("gather_read");
        let expected: Vec<f64> = (0..4)
            .map(|element| operand_values[permutations[column * 4 + element] as usize])
            .collect();
        assert_eq!(read(&engine, &observed), expected, "column {column}");

        // The same request against the column's compact copy reads the same values.
        let compact = engine
            .to_contiguous_read(view)
            .expect("materialize the column");
        let reference = engine
            .gather(&operand, &compact, &config)
            .expect("owned gather");
        assert_eq!(read(&engine, &observed), read(&engine, &reference));
    }

    // A non-compact index view is not one contiguous span; it still works through one copy.
    let index_source = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![8], (0..8).map(|index| index as i64).collect())
            .expect("index source"),
    )
    .expect("upload index source");
    let wide_operand = device(&engine, vec![8], (0..8).map(|index| index as f64).collect());
    let observed = engine
        .gather_read(
            TensorRead::from_tensor(&wide_operand),
            i64_region(&index_source, vec![3], vec![2], 1),
            &config,
        )
        .expect("gather_read on a non-compact index view");
    assert_eq!(read(&engine, &observed), vec![1.0, 3.0, 5.0]);
}

#[test]
#[ignore = "requires CUDA"]
fn scatter_read_follows_a_view_layout() {
    let Some(mut engine) = engine() else { return };
    let source = device(&engine, vec![4, 5], matrix(4, 5));
    let indices_host = Tensor::from_vec_col_major(vec![2, 1], vec![0i64, 1]).expect("indices");
    let indices = upload_tensor(engine.runtime(), &indices_host).expect("upload indices");
    // The update window rides a strided region of its own: the odd elements of a four-element
    // allocation, in reverse order.
    let updates_source = device(&engine, vec![4], vec![1.0, 2.0, 3.0, 4.0]);
    let updates = f64_region(&updates_source, vec![2], vec![-2], 3);
    let config = tenferro_tensor::ScatterConfig {
        update_window_dims: vec![],
        inserted_window_dims: vec![0, 1],
        scatter_dims_to_operand_dims: vec![0],
        index_vector_dim: 1,
    };
    for (flavour, view) in view_flavours(&source) {
        // Materialize the operand and the update window independently, then add the updates on
        // the host: the reference cannot reuse the scatter kernel it is checking.
        let operand = engine
            .to_contiguous_read(view.clone())
            .expect("materialize the operand");
        let mut expected = read(&engine, &operand);
        let compact_updates = engine
            .to_contiguous_read(updates.clone())
            .expect("materialize the updates");
        let updates_values = read(&engine, &compact_updates);
        // The indices are 0 and 1 on operand axis 0; every other axis is inserted, so each scalar
        // update lands in the first column of its row.
        expected[0] += updates_values[0];
        expected[1] += updates_values[1];
        let observed = engine
            .scatter_read(
                view,
                TensorRead::from_tensor(&indices),
                updates.clone(),
                &config,
            )
            .expect("scatter_read");
        assert_eq!(observed.shape(), operand.shape(), "the {flavour} view");
        assert_eq!(read(&engine, &observed), expected, "the {flavour} view");
    }
}

#[test]
#[ignore = "requires CUDA"]
fn concatenate_read_follows_a_view_layout() {
    let Some(mut engine) = engine() else { return };
    let transposed_source = device(&engine, vec![3, 2], matrix(3, 2));
    // A real transpose of the compact column-major [3, 2] source: [2, 3] with strides [3, 1].
    let transposed = f64_region(&transposed_source, vec![2, 3], vec![3, 1], 0);
    let offset_source = device(&engine, vec![2, 5], matrix(2, 5));
    let offset = f64_region(&offset_source, vec![2, 3], vec![1, 2], 1);
    let negative_source = device(&engine, vec![2, 3], matrix(2, 3));
    let negative = f64_region(&negative_source, vec![2, 3], vec![-1, 2], 1);

    for axis in [0usize, 1] {
        let inputs = [transposed.clone(), offset.clone(), negative.clone()];
        let observed = engine
            .concatenate_read(&inputs, axis)
            .expect("concatenate_read");
        let mut compact = Vec::with_capacity(inputs.len());
        for input in inputs {
            compact.push(
                engine
                    .to_contiguous_read(input)
                    .expect("materialize the input"),
            );
        }
        let refs: Vec<&Tensor> = compact.iter().collect();
        let expected = engine.concatenate(&refs, axis).expect("concatenate");
        assert_eq!(observed.shape(), expected.shape(), "axis {axis}");
        assert_eq!(
            read(&engine, &observed),
            read(&engine, &expected),
            "axis {axis}"
        );
    }
}

#[test]
#[ignore = "requires CUDA"]
fn read_entries_serve_an_empty_view() {
    let Some(mut engine) = engine() else { return };
    let source = device(&engine, vec![2, 3], matrix(2, 3));
    // A zero-extent region of a non-empty allocation: no element, but a real allocation.
    let empty = || f64_region(&source, vec![0, 3], vec![1, 2], 0);
    // The same with the empty axis last, which is the shape a scatter destination must have: an
    // empty leading axis makes the compact strides of the later axes zero.
    let empty_trailing = || f64_region(&source, vec![3, 0], vec![1, 2], 0);

    matches_the_compact_operand(&mut engine, "empty", empty(), |engine, input| {
        engine
            .slice_read(
                input,
                &SliceConfig {
                    starts: vec![0, 0],
                    limits: vec![0, 3],
                    strides: vec![1, 1],
                },
            )
            .expect("slice_read")
    });
    matches_the_compact_operand(&mut engine, "empty", empty(), |engine, input| {
        engine.reverse_read(input, &[1]).expect("reverse_read")
    });
    matches_the_compact_operand(&mut engine, "empty", empty(), |engine, input| {
        engine
            .pad_read(
                input,
                &PadConfig {
                    edge_padding_low: vec![1, 0],
                    edge_padding_high: vec![0, 1],
                    interior_padding: vec![0, 1],
                },
            )
            .expect("pad_read")
    });

    // Concatenating an empty input with a non-empty one is a plain placement of the latter.
    let kept = device(&engine, vec![2, 3], matrix(2, 3));
    let joined = engine
        .concatenate_read(&[empty(), TensorRead::from_tensor(&kept)], 0)
        .expect("concatenate_read");
    assert_eq!(joined.shape(), &[2, 3]);
    assert_eq!(read(&engine, &joined), read(&engine, &kept));

    // An empty scatter has no update to add, so the operand copy is the answer.
    let no_indices = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![0, 1], Vec::<i64>::new()).expect("indices"),
    )
    .expect("upload indices");
    let no_updates = device(&engine, vec![0], Vec::new());
    let scattered = engine
        .scatter_read(
            empty_trailing(),
            TensorRead::from_tensor(&no_indices),
            TensorRead::from_tensor(&no_updates),
            &tenferro_tensor::ScatterConfig {
                update_window_dims: vec![],
                inserted_window_dims: vec![0, 1],
                scatter_dims_to_operand_dims: vec![0],
                index_vector_dim: 1,
            },
        )
        .expect("scatter_read");
    assert_eq!(scattered.shape(), &[3, 0]);
    assert!(read(&engine, &scattered).is_empty());
}

#[test]
#[ignore = "requires CUDA"]
fn complex_view_reads_match_the_compact_result() {
    use tenferro_tensor::Complex64;

    let Some(mut engine) = engine() else { return };
    let data: Vec<Complex64> = (0..12)
        .map(|index| Complex64::new(index as f64, -(index as f64) * 0.5))
        .collect();
    let source = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![3, 4], data.clone()).expect("matrix"),
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
    let reversed = || region(vec![3, 2], vec![-1, 3], 2);

    let read_complex = |engine: &CudaEngine, tensor: &Tensor| -> Vec<Complex64> {
        let back = download_tensor(engine.runtime(), tensor).expect("download");
        back.as_slice::<Complex64>()
            .expect("complex payload")
            .to_vec()
    };

    // Slice the transpose, and reverse the negative-stride region.
    let config = SliceConfig {
        starts: vec![1, 0],
        limits: vec![4, 3],
        strides: vec![1, 1],
    };
    let sliced = engine.slice_read(transpose(), &config).expect("slice_read");
    let compact = engine.to_contiguous_read(transpose()).expect("materialize");
    let expected = engine.slice(&compact, &config).expect("slice");
    assert_eq!(sliced.shape(), expected.shape());
    assert_eq!(
        read_complex(&engine, &sliced),
        read_complex(&engine, &expected)
    );

    let reversed_out = engine.reverse_read(reversed(), &[1]).expect("reverse_read");
    let compact = engine.to_contiguous_read(reversed()).expect("materialize");
    let expected = engine.reverse(&compact, &[1]).expect("reverse");
    assert_eq!(reversed_out.shape(), expected.shape());
    assert_eq!(
        read_complex(&engine, &reversed_out),
        read_complex(&engine, &expected)
    );
}

/// The logical values a borrowed region names, in column-major order.
fn region_values(data: &[f64], shape: &[usize], strides: &[isize], offset: isize) -> Vec<f64> {
    let count: usize = shape.iter().product();
    let mut values = Vec::with_capacity(count);
    for index in 0..count {
        let mut remaining = index;
        let mut source = offset;
        for (axis, &extent) in shape.iter().enumerate() {
            let coordinate = remaining % extent;
            remaining /= extent;
            source += coordinate as isize * strides[axis];
        }
        values.push(data[source as usize]);
    }
    values
}

fn upload_host(engine: &CudaEngine, host: &Tensor) -> Tensor {
    upload_tensor(engine.runtime(), host).expect("upload")
}

fn cpu_dynamic_update_slice(operand: &Tensor, update: &Tensor, starts: &Tensor) -> Tensor {
    tenferro_cpu::CpuEngine::serial()
        .dynamic_update_slice(operand, update, starts)
        .expect("CPU dynamic_update_slice")
}

/// Every starts index dtype, including negative and too-large values the kernel clamps, agrees
/// with the CPU engine.
#[test]
#[ignore = "requires CUDA"]
fn dynamic_update_slice_matches_the_cpu_engine_for_every_starts_dtype() {
    let Some(mut engine) = engine() else { return };
    let operand_values = matrix(4, 5);
    let update_values = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0];

    macro_rules! starts_case {
        ($ty:ty, $values:expr) => {{
            let operand_host =
                Tensor::from_vec_col_major(vec![4, 5], operand_values.clone()).expect("operand");
            let update_host =
                Tensor::from_vec_col_major(vec![2, 3], update_values.clone()).expect("update");
            let starts_host = Tensor::from_vec_col_major(vec![2], $values).expect("starts");
            let operand = upload_host(&engine, &operand_host);
            let update = upload_host(&engine, &update_host);
            let starts = upload_host(&engine, &starts_host);
            let observed = engine
                .dynamic_update_slice(&operand, &update, &starts)
                .expect("CUDA dynamic_update_slice");
            let expected = cpu_dynamic_update_slice(&operand_host, &update_host, &starts_host);
            assert_eq!(
                read(&engine, &observed),
                expected.as_slice::<f64>().expect("f64 payload"),
                "starts {:?}",
                $values
            );
        }};
    }

    starts_case!(i64, vec![1i64, 2]);
    starts_case!(i64, vec![-2i64, 99]);
    starts_case!(i32, vec![1i32, 2]);
    starts_case!(f32, vec![1.0f32, 2.0]);
    starts_case!(f64, vec![1.0f64, 2.0]);
}

/// The CPU engine's dtype matrix, on compact operands.
#[test]
#[ignore = "requires CUDA"]
fn dynamic_update_slice_matches_the_cpu_engine_for_every_data_dtype() {
    use tenferro_tensor::{Complex32, Complex64};

    let Some(mut engine) = engine() else { return };
    let starts_host = Tensor::from_vec_col_major(vec![2], vec![1i64, 2]).expect("starts");

    macro_rules! dtype_case {
        ($ty:ty, $operand_values:expr, $update_values:expr) => {{
            let operand_host =
                Tensor::from_vec_col_major(vec![4, 5], $operand_values).expect("operand");
            let update_host =
                Tensor::from_vec_col_major(vec![2, 3], $update_values).expect("update");
            let operand = upload_host(&engine, &operand_host);
            let update = upload_host(&engine, &update_host);
            let starts = upload_host(&engine, &starts_host);
            let observed = engine
                .dynamic_update_slice(&operand, &update, &starts)
                .expect("CUDA dynamic_update_slice");
            let expected = cpu_dynamic_update_slice(&operand_host, &update_host, &starts_host);
            let back = download_tensor(engine.runtime(), &observed).expect("download");
            assert_eq!(
                back.as_slice::<$ty>().expect("payload"),
                expected.as_slice::<$ty>().expect("payload")
            );
        }};
    }

    dtype_case!(
        f32,
        (0..20).map(|index| index as f32).collect::<Vec<_>>(),
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    dtype_case!(
        i32,
        (0..20i32).collect::<Vec<_>>(),
        vec![1i32, 2, 3, 4, 5, 6]
    );
    dtype_case!(
        i64,
        (0..20).map(|index| index as i64).collect::<Vec<_>>(),
        vec![1i64, 2, 3, 4, 5, 6]
    );
    dtype_case!(
        bool,
        (0..20).map(|index| index % 2 == 0).collect::<Vec<_>>(),
        vec![true, false, true, false, true, false]
    );
    dtype_case!(
        Complex32,
        (0..20)
            .map(|index| Complex32::new(index as f32, -(index as f32)))
            .collect::<Vec<_>>(),
        vec![
            Complex32::new(1.0, -1.0),
            Complex32::new(2.0, -2.0),
            Complex32::new(3.0, -3.0),
            Complex32::new(4.0, -4.0),
            Complex32::new(5.0, -5.0),
            Complex32::new(6.0, -6.0),
        ]
    );
    dtype_case!(
        Complex64,
        (0..20)
            .map(|index| Complex64::new(index as f64, -(index as f64)))
            .collect::<Vec<_>>(),
        vec![
            Complex64::new(1.0, -1.0),
            Complex64::new(2.0, -2.0),
            Complex64::new(3.0, -3.0),
            Complex64::new(4.0, -4.0),
            Complex64::new(5.0, -5.0),
            Complex64::new(6.0, -6.0),
        ]
    );
}

/// A strided update and an offset starts view are read where they live, and their values agree
/// with the CPU engine on the materialized update.
#[test]
#[ignore = "requires CUDA"]
fn dynamic_update_slice_reads_a_strided_update_and_an_offset_starts_view() {
    let Some(mut engine) = engine() else { return };
    let operand_values = matrix(4, 5);
    let update_source_values = matrix(4, 5);
    let operand_host =
        Tensor::from_vec_col_major(vec![4, 5], operand_values.clone()).expect("operand");
    let update_source = upload_host(
        &engine,
        &Tensor::from_vec_col_major(vec![4, 5], update_source_values.clone())
            .expect("update source"),
    );
    let operand = upload_host(&engine, &operand_host);
    // Rows 1..3 of columns 0..3: a transposed-strided region, not one contiguous span.
    let update_shape = vec![2usize, 3];
    let update_strides = vec![1isize, 4];
    let update_offset = 1isize;
    let update_values = region_values(
        &update_source_values,
        &update_shape,
        &update_strides,
        update_offset,
    );
    let update_view = TensorRead::from_view(TensorView::F64(
        update_source
            .as_typed::<f64>()
            .expect("f64 payload")
            .backend_region_view(update_shape.clone(), update_strides, update_offset)
            .expect("update view"),
    ));

    // The starts tensor holds three per-call vectors; this call reads the middle one.
    let starts_values = vec![0i64, 0, 1, 2, 3, 4];
    let starts_source = upload_host(
        &engine,
        &Tensor::from_vec_col_major(vec![2, 3], starts_values).expect("starts"),
    );
    let starts_shape = vec![2usize];
    let starts_strides = vec![1isize];
    let starts_offset = 2isize;
    let starts = TensorRead::from_view(TensorView::I64(
        starts_source
            .as_typed::<i64>()
            .expect("i64 payload")
            .backend_region_view(starts_shape, starts_strides, starts_offset)
            .expect("starts view"),
    ));

    let observed = engine
        .dynamic_update_slice_read(TensorRead::from_tensor(&operand), update_view, starts)
        .expect("dynamic_update_slice_read");

    let starts_host = Tensor::from_vec_col_major(vec![2], vec![1i64, 2]).expect("starts host");
    let update_host = Tensor::from_vec_col_major(update_shape, update_values).expect("update host");
    let expected = cpu_dynamic_update_slice(&operand_host, &update_host, &starts_host);
    assert_eq!(
        read(&engine, &observed),
        expected.as_slice::<f64>().expect("f64 payload")
    );
}

/// Empty updates, rank 0, and a rank-9 window agree with the CPU engine.
#[test]
#[ignore = "requires CUDA"]
fn dynamic_update_slice_handles_empty_rank_zero_and_rank_nine() {
    let Some(mut engine) = engine() else { return };

    // An empty update leaves the operand copy untouched.
    let operand_host = Tensor::from_vec_col_major(vec![3, 4], matrix(3, 4)).expect("operand");
    let update_host = Tensor::from_vec_col_major(vec![0, 2], Vec::<f64>::new()).expect("update");
    let starts_host = Tensor::from_vec_col_major(vec![2], vec![1i64, 1]).expect("starts");
    let observed = engine
        .dynamic_update_slice(
            &upload_host(&engine, &operand_host),
            &upload_host(&engine, &update_host),
            &upload_host(&engine, &starts_host),
        )
        .expect("empty update");
    let expected = cpu_dynamic_update_slice(&operand_host, &update_host, &starts_host);
    assert_eq!(
        read(&engine, &observed),
        expected.as_slice::<f64>().expect("f64 payload")
    );

    // A rank-0 update is the result, and an empty starts vector is never dereferenced.
    let scalar_operand = Tensor::from_vec_col_major(vec![], vec![7.0f64]).expect("scalar");
    let scalar_update = Tensor::from_vec_col_major(vec![], vec![9.0f64]).expect("update");
    let empty_starts = Tensor::from_vec_col_major(vec![0], Vec::<i64>::new()).expect("starts");
    let observed = engine
        .dynamic_update_slice(
            &upload_host(&engine, &scalar_operand),
            &upload_host(&engine, &scalar_update),
            &upload_host(&engine, &empty_starts),
        )
        .expect("rank-0 update");
    assert_eq!(read(&engine, &observed), vec![9.0]);

    // Nine axes: the general path carries the two layouts in device memory.
    let operand_shape = vec![3usize, 1, 1, 1, 1, 1, 1, 1, 4];
    let update_shape = vec![2usize, 1, 1, 1, 1, 1, 1, 1, 3];
    let operand_host = Tensor::from_vec_col_major(
        operand_shape.clone(),
        (0..12).map(|index| index as f64).collect(),
    )
    .expect("rank-9 operand");
    let update_host = Tensor::from_vec_col_major(
        update_shape.clone(),
        (0..6).map(|index| 100.0 + index as f64).collect(),
    )
    .expect("rank-9 update");
    let starts_host =
        Tensor::from_vec_col_major(vec![9], vec![1i64, 0, 0, 0, 0, 0, 0, 0, 1]).expect("starts");
    let observed = engine
        .dynamic_update_slice(
            &upload_host(&engine, &operand_host),
            &upload_host(&engine, &update_host),
            &upload_host(&engine, &starts_host),
        )
        .expect("rank-9 update");
    let expected = cpu_dynamic_update_slice(&operand_host, &update_host, &starts_host);
    assert_eq!(observed.shape(), &operand_shape[..]);
    assert_eq!(
        read(&engine, &observed),
        expected.as_slice::<f64>().expect("f64 payload")
    );
}
