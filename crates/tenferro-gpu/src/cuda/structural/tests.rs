//! Unit tests for the structural slice's arithmetic helpers and its native read boundary.

use super::super::structural::element_count;
use crate::cuda::{download_tensor, gpu_available, upload_tensor, CudaDeviceId, CudaEngine};
use tenferro_tensor::{
    ElementwiseReadOp, Engine as _, PadConfig, SliceConfig, Tensor, TensorElementwise as _,
    TensorIndexing, TensorRead, TensorStructural, TensorView, TensorViewMut, TensorWrite,
    TypedTensorViewMut,
};

#[test]
fn the_element_count_checks_both_extremes() {
    assert_eq!(element_count(&[]).expect("scalar"), 1);
    assert_eq!(element_count(&[2, 3, 4]).expect("24"), 24);
    assert!(element_count(&[usize::MAX, 2]).is_err());
}

fn engine() -> Option<CudaEngine> {
    // The same guarded probe the integration tests use: with `TENFERRO_REQUIRE_GPU=1` (the GPU
    // lane) a missing device fails here, so an `--ignored` run cannot pass without hardware.
    if !gpu_available() {
        return None;
    }
    Some(CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("CUDA engine"))
}

fn device(engine: &CudaEngine, shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    let host = Tensor::from_vec_col_major(shape, data).expect("host tensor");
    upload_tensor(engine.runtime(), &host).expect("upload")
}

fn read(engine: &CudaEngine, tensor: &Tensor) -> Vec<f64> {
    let back = download_tensor(engine.runtime(), tensor).expect("download");
    back.as_slice::<f64>().expect("f64 payload").to_vec()
}

/// A borrowed region of `tensor`'s device allocation, as one read target.
fn region(
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
fn index_region(
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

/// The `3 x 4` column-major source's transpose: the strided view with offset zero.
fn transposed(source: &Tensor) -> TensorRead<'_> {
    region(source, vec![4, 3], vec![3, 1], 0)
}

/// A `2 x 2` region of the source at a nonzero offset.
fn offset(source: &Tensor) -> TensorRead<'_> {
    region(source, vec![2, 2], vec![1, 3], 1)
}

/// The source's rows in reverse order, two columns wide.
fn negative(source: &Tensor) -> TensorRead<'_> {
    region(source, vec![3, 2], vec![-1, 3], 2)
}

/// A `4 x 3` source's columns in reverse order: the negative-stride flavour at the transpose's
/// own shape, so the two can be concatenated.
fn reversed_columns(source: &Tensor) -> TensorRead<'_> {
    region(source, vec![4, 3], vec![1, -4], 8)
}

/// Every read entry that consumes an operand's own layout, on one borrowed view.
///
/// Before #2223 item 1 each of these resolved the view through `device_read`, so the counter
/// moved once per view operand here (twice for a scatter); the native read entries move it not
/// at all.
#[test]
#[ignore = "requires CUDA"]
fn read_entries_consume_a_strided_view_without_staging() {
    let Some(mut engine) = engine() else { return };
    let source_data: Vec<f64> = (0..12).map(|index| index as f64).collect();
    let source = device(&engine, vec![3, 4], source_data.clone());

    // The non-operand tensors stay on their existing path: compact owned reads.
    let starts = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![2], vec![0i64, 0]).expect("starts"),
    )
    .expect("upload starts");
    let gather_indices = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![2, 1], vec![0i64, 1]).expect("gather indices"),
    )
    .expect("upload gather indices");
    let scatter_indices = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![2, 1], vec![0i64, 1]).expect("scatter indices"),
    )
    .expect("upload scatter indices");
    let updates_source = device(&engine, vec![4], vec![1.0, 2.0, 3.0, 4.0]);
    let updates = TensorRead::from_view(TensorView::F64(
        updates_source
            .as_typed::<f64>()
            .expect("f64 payload")
            .backend_region_view(vec![2], vec![-2], 3)
            .expect("updates view"),
    ));
    let update_slice_starts = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![2], vec![1i64, 1]).expect("update slice starts"),
    )
    .expect("upload update slice starts");

    let before = engine.staged_reads();

    engine
        .slice_read(
            transposed(&source),
            &SliceConfig {
                starts: vec![1, 1],
                limits: vec![4, 3],
                strides: vec![1, 1],
            },
        )
        .expect("slice_read");
    engine
        .dynamic_slice_read(
            transposed(&source),
            TensorRead::from_tensor(&starts),
            &[2, 2],
        )
        .expect("dynamic_slice_read");
    engine
        .pad_read(
            transposed(&source),
            &PadConfig {
                edge_padding_low: vec![1, 0],
                edge_padding_high: vec![0, 1],
                interior_padding: vec![0, 1],
            },
        )
        .expect("pad_read");
    engine
        .reverse_read(negative(&source), &[0])
        .expect("reverse_read");
    let mirror_data: Vec<f64> = (0..12).map(|index| 100.0 + index as f64).collect();
    let mirror = device(&engine, vec![4, 3], mirror_data.clone());
    engine.tril_read(transposed(&source), 0).expect("tril_read");
    engine.triu_read(negative(&source), 1).expect("triu_read");

    // A strided operand with a smaller offset update, so the untouched cells are part of the
    // comparison rather than being overwritten by the window.
    let operand_shape = vec![4usize, 3];
    let update_shape = vec![2usize, 2];
    let operand_values = view_values(&source_data, &operand_shape, &[3, 1], 0);
    let update_values = view_values(&mirror_data, &update_shape, &[1, 3], 1);
    let updated = engine
        .dynamic_update_slice_read(
            transposed(&source),
            region(&mirror, update_shape.clone(), vec![1, 3], 1),
            TensorRead::from_tensor(&update_slice_starts),
        )
        .expect("dynamic_update_slice_read on a strided operand");
    assert_eq!(
        read(&engine, &updated),
        dynamic_update_reference(
            &operand_values,
            &operand_shape,
            &update_values,
            &update_shape,
            &[1, 1],
        ),
        "the strided dynamic update must keep every cell outside its window"
    );

    // The same entry with an offset operand and a strided update.
    let operand_shape = vec![2usize, 2];
    let update_shape = vec![1usize, 2];
    let operand_values = view_values(&source_data, &operand_shape, &[1, 3], 1);
    let update_values = view_values(&mirror_data, &update_shape, &[1, 4], 2);
    let updated = engine
        .dynamic_update_slice_read(
            offset(&source),
            region(&mirror, update_shape.clone(), vec![1, 4], 2),
            TensorRead::from_tensor(&update_slice_starts),
        )
        .expect("dynamic_update_slice_read on an offset operand");
    assert_eq!(
        read(&engine, &updated),
        dynamic_update_reference(
            &operand_values,
            &operand_shape,
            &update_values,
            &update_shape,
            &[1, 1],
        ),
        "the offset dynamic update must keep every cell outside its window"
    );
    engine
        .extract_diagonal_read(offset(&source), 0, 1)
        .expect("extract_diagonal_read");
    engine
        .embed_diagonal_read(offset(&source), 0, 1)
        .expect("embed_diagonal_read");
    engine
        .gather_read(
            transposed(&source),
            TensorRead::from_tensor(&gather_indices),
            &tenferro_tensor::GatherConfig {
                offset_dims: vec![0],
                collapsed_slice_dims: vec![1],
                start_index_map: vec![0],
                index_vector_dim: 1,
                slice_sizes: vec![2, 1],
            },
        )
        .expect("gather_read");
    engine
        .scatter_read(
            transposed(&source),
            TensorRead::from_tensor(&scatter_indices),
            updates.clone(),
            &tenferro_tensor::ScatterConfig {
                update_window_dims: vec![],
                inserted_window_dims: vec![0, 1],
                scatter_dims_to_operand_dims: vec![0],
                index_vector_dim: 1,
            },
        )
        .expect("scatter_read");
    engine
        .concatenate_read(&[transposed(&source), reversed_columns(&mirror)], 1)
        .expect("concatenate_read");

    assert_eq!(
        engine.staged_reads(),
        before,
        "a native read entry staged a borrowed view instead of reading its own layout"
    );
}

/// #1930: one uploaded index tensor holds several per-matrix permutations. Gather, scatter and
/// dynamic slice each read one column as an offset view of that allocation, so none of them
/// stages the index operand.
#[test]
#[ignore = "requires CUDA"]
fn offset_index_views_are_read_in_place() {
    let Some(mut engine) = engine() else { return };
    let operand_values = vec![10.0, 20.0, 30.0, 40.0];
    let operand = device(&engine, vec![4], operand_values.clone());
    let permutations = vec![
        0i64, 1, 2, 3, // the identity
        3, 0, 1, 2, 2, 3, 0, 1,
    ];
    let index_tensor = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![4, 3], permutations.clone()).expect("permutations"),
    )
    .expect("upload permutations");

    let gather_config = tenferro_tensor::GatherConfig {
        offset_dims: vec![],
        collapsed_slice_dims: vec![0],
        start_index_map: vec![0],
        index_vector_dim: 1,
        slice_sizes: vec![1],
    };
    let updates = device(&engine, vec![4], vec![1.0, 2.0, 3.0, 4.0]);
    let scatter_config = tenferro_tensor::ScatterConfig {
        update_window_dims: vec![],
        inserted_window_dims: vec![0],
        scatter_dims_to_operand_dims: vec![0],
        index_vector_dim: 1,
    };

    let before = engine.staged_reads();
    for column in 0..3 {
        let indices = || index_region(&index_tensor, vec![4], vec![1], (column * 4) as isize);

        // A gather whose start list is this column.
        let gathered = engine
            .gather_read(TensorRead::from_tensor(&operand), indices(), &gather_config)
            .expect("gather_read");
        let expected: Vec<f64> = (0..4)
            .map(|element| operand_values[permutations[column * 4 + element] as usize])
            .collect();
        assert_eq!(read(&engine, &gathered), expected, "gather column {column}");

        // A scatter whose start list is the same column.
        let scattered = engine
            .scatter_read(
                TensorRead::from_tensor(&operand),
                indices(),
                TensorRead::from_tensor(&updates),
                &scatter_config,
            )
            .expect("scatter_read");
        let mut expected = operand_values.clone();
        for element in 0..4 {
            expected[permutations[column * 4 + element] as usize] += (element + 1) as f64;
        }
        assert_eq!(
            read(&engine, &scattered),
            expected,
            "scatter column {column}"
        );
    }
    assert_eq!(
        engine.staged_reads(),
        before,
        "an offset index view was staged instead of read in place"
    );

    // Dynamic slice: one starts tensor holds several per-window start vectors.
    let matrix = device(
        &engine,
        vec![4, 5],
        (0..20).map(|index| index as f64).collect(),
    );
    let starts_values = vec![0i64, 0, 1, 2, 2, 1];
    let start_tensor = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![2, 3], starts_values.clone()).expect("starts"),
    )
    .expect("upload starts");
    let before = engine.staged_reads();
    for column in 0..3 {
        let starts = index_region(&start_tensor, vec![2], vec![1], (column * 2) as isize);
        let sliced = engine
            .dynamic_slice_read(TensorRead::from_tensor(&matrix), starts, &[2, 3])
            .expect("dynamic_slice_read");
        let (row, col) = (
            starts_values[column * 2] as usize,
            starts_values[column * 2 + 1] as usize,
        );
        let mut expected = Vec::with_capacity(6);
        for c in 0..3 {
            for r in 0..2 {
                expected.push((row + r + (col + c) * 4) as f64);
            }
        }
        assert_eq!(
            read(&engine, &sliced),
            expected,
            "dynamic_slice column {column}"
        );
    }
    assert_eq!(
        engine.staged_reads(),
        before,
        "an offset starts view was staged instead of read in place"
    );
}

/// A non-compact index view is not one contiguous span, so it keeps exactly one staged copy.
#[test]
#[ignore = "requires CUDA"]
fn a_non_compact_index_view_keeps_one_staging_copy() {
    let Some(mut engine) = engine() else { return };
    let operand = device(&engine, vec![8], (0..8).map(|index| index as f64).collect());
    // The odd elements of an eight-element allocation: shape [3], stride 2, offset 1.
    let index_source = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![8], (0..8).map(|index| index as i64).collect())
            .expect("index source"),
    )
    .expect("upload index source");
    let config = tenferro_tensor::GatherConfig {
        offset_dims: vec![],
        collapsed_slice_dims: vec![0],
        start_index_map: vec![0],
        index_vector_dim: 1,
        slice_sizes: vec![1],
    };
    let before = engine.staged_reads();
    let gathered = engine
        .gather_read(
            TensorRead::from_tensor(&operand),
            index_region(&index_source, vec![3], vec![2], 1),
            &config,
        )
        .expect("gather_read");
    assert_eq!(
        engine.staged_reads(),
        before + 1,
        "a non-compact index view must keep exactly one staging copy"
    );
    assert_eq!(read(&engine, &gathered), vec![1.0, 3.0, 5.0]);
}

/// A float index view is validated through the same rebased pointer its gather reads: the
/// fractional value in the second column is found even though the allocation's base holds an
/// exact integer.
#[test]
#[ignore = "requires CUDA"]
fn a_float_offset_index_view_is_validated_at_its_base() {
    let Some(mut engine) = engine() else { return };
    let operand = device(&engine, vec![4], vec![10.0, 20.0, 30.0, 40.0]);
    let indices = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![4, 2], vec![0.0, 1.0, 2.0, 3.0, 0.5, 1.0, 2.0, 3.0])
            .expect("float indices"),
    )
    .expect("upload float indices");
    let config = tenferro_tensor::GatherConfig {
        offset_dims: vec![],
        collapsed_slice_dims: vec![0],
        start_index_map: vec![0],
        index_vector_dim: 1,
        slice_sizes: vec![1],
    };

    // The exact column is accepted.
    engine
        .gather_read(
            TensorRead::from_tensor(&operand),
            region(&indices, vec![4], vec![1], 0),
            &config,
        )
        .expect("an exact float index column");

    // The fractional column is rejected through the rebased span, not through the base.
    let error = engine
        .gather_read(
            TensorRead::from_tensor(&operand),
            region(&indices, vec![4], vec![1], 4),
            &config,
        )
        .expect_err("a fractional float index must be refused");
    assert!(
        error
            .to_string()
            .contains("not an exactly representable integer"),
        "unexpected error: {error}"
    );
}

/// The counter's negative control: an entry with no stride-consuming kernel, and the explicit
/// canonicalization, both still report the copy they take.
#[test]
#[ignore = "requires CUDA"]
fn a_materialized_view_moves_the_staging_counter() {
    let Some(mut engine) = engine() else { return };
    let source = device(
        &engine,
        vec![3, 4],
        (0..12).map(|index| index as f64).collect(),
    );

    // `tril_read` reads the view through its own layout, so it moves the counter not at all.
    let before = engine.staged_reads();
    engine.tril_read(transposed(&source), 0).expect("tril_read");
    assert_eq!(
        engine.staged_reads(),
        before,
        "the triangular view entry must not stage its operand"
    );

    // `to_contiguous_read` of a view is exactly one materialization.
    let before = engine.staged_reads();
    engine
        .to_contiguous_read(transposed(&source))
        .expect("to_contiguous_read");
    assert_eq!(
        engine.staged_reads(),
        before + 1,
        "canonicalizing a view is one staged read"
    );

    // An owned operand is read in place by every entry, so it moves the counter not at all.
    let before = engine.staged_reads();
    engine
        .slice_read(
            TensorRead::from_tensor(&source),
            &SliceConfig {
                starts: vec![0, 0],
                limits: vec![3, 4],
                strides: vec![1, 1],
            },
        )
        .expect("slice_read");
    assert_eq!(
        engine.staged_reads(),
        before,
        "an owned operand must not be staged"
    );
}

/// The three view flavours resolve to the values the materialized compact copy holds.
#[test]
#[ignore = "requires CUDA"]
fn a_region_view_reads_its_own_elements() {
    let Some(mut engine) = engine() else { return };
    let data: Vec<f64> = (0..12).map(|index| index as f64).collect();
    let source = device(&engine, vec![3, 4], data.clone());

    // Column-major: element `row + column * 3`.
    let at = |row: usize, column: usize| data[row + column * 3];

    let copy = engine
        .to_contiguous_read(transposed(&source))
        .expect("materialize the transpose");
    assert_eq!(
        read(&engine, &copy),
        (0..12)
            .map(|index| at(index / 4, index % 4))
            .collect::<Vec<_>>()
    );

    let copy = engine
        .to_contiguous_read(offset(&source))
        .expect("materialize the offset region");
    assert_eq!(
        read(&engine, &copy),
        vec![at(1, 0), at(2, 0), at(1, 1), at(2, 1)]
    );

    let copy = engine
        .to_contiguous_read(negative(&source))
        .expect("materialize the negative-stride region");
    assert_eq!(
        read(&engine, &copy),
        vec![at(2, 0), at(1, 0), at(0, 0), at(2, 1), at(1, 1), at(0, 1)]
    );
}

/// The logical values a view names, in column-major order, from the source's flat data.
fn view_values(data: &[f64], shape: &[usize], strides: &[isize], offset: isize) -> Vec<f64> {
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

/// The StableHLO dynamic-update result: the operand values with the update window written at
/// the clamped starts.
fn dynamic_update_reference(
    operand: &[f64],
    operand_shape: &[usize],
    update: &[f64],
    update_shape: &[usize],
    starts: &[i64],
) -> Vec<f64> {
    let mut expected = operand.to_vec();
    let origin: Vec<usize> = starts
        .iter()
        .zip(operand_shape)
        .zip(update_shape)
        .map(|((&start, &operand_extent), &update_extent)| {
            start.clamp(0, (operand_extent - update_extent) as i64) as usize
        })
        .collect();
    for (index, &value) in update.iter().enumerate() {
        let mut remaining = index;
        let mut destination = 0usize;
        let mut stride = 1usize;
        for axis in 0..operand_shape.len() {
            let coordinate = remaining % update_shape[axis];
            remaining /= update_shape[axis];
            destination += (origin[axis] + coordinate) * stride;
            stride *= operand_shape[axis];
        }
        expected[destination] = value;
    }
    expected
}

/// The triangle mask of `values` for a rank-2-or-higher shape, batched on every higher axis.
fn triangular_reference(values: &[f64], shape: &[usize], k: i64, lower: bool) -> Vec<f64> {
    let rows = shape[0];
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let row = index % rows;
            let column = (index / rows) % shape[1];
            let difference = column as i64 - row as i64;
            let keeps = if lower {
                difference <= k
            } else {
                difference >= k
            };
            if keeps {
                *value
            } else {
                0.0
            }
        })
        .collect()
}

/// The revision-11 view entries mask transposed, offset, negative-stride and rank-9 regions
/// through their own layouts, with `k` on both ends of the comparison.
#[test]
#[ignore = "requires CUDA"]
fn triangular_reads_mask_borrowed_regions() {
    let Some(mut engine) = engine() else { return };
    let data: Vec<f64> = (0..12).map(|index| index as f64 + 1.0).collect();
    let source = device(&engine, vec![3, 4], data.clone());

    for (flavour, shape, strides, offset) in [
        ("transposed", vec![4, 3], vec![3, 1], 0),
        ("offset", vec![2, 2], vec![1, 3], 1),
        ("compact-offset", vec![3, 2], vec![1, 3], 1),
        ("negative", vec![3, 2], vec![-1, 3], 2),
    ] {
        let logical = view_values(&data, &shape, &strides, offset);
        for k in [0i64, 1, -1] {
            for lower in [true, false] {
                let expected = triangular_reference(&logical, &shape, k, lower);
                let observed = if lower {
                    engine.tril_read(region(&source, shape.clone(), strides.clone(), offset), k)
                } else {
                    engine.triu_read(region(&source, shape.clone(), strides.clone(), offset), k)
                }
                .expect("triangular read");
                assert_eq!(
                    observed.shape(),
                    &shape[..],
                    "{flavour} k={k} lower={lower}"
                );
                assert_eq!(
                    read(&engine, &observed),
                    expected,
                    "{flavour} k={k} lower={lower}"
                );
            }
        }
    }

    // Nine axes, genuinely transposed: axes 0 and 8 exchange the compact strides, so the view
    // is not compact logical order with singleton zero strides.
    let rank_nine_data: Vec<f64> = (0..12).map(|index| index as f64 + 10.0).collect();
    let rank_nine = device(&engine, vec![2, 2, 3], rank_nine_data.clone());
    let shape = vec![2usize, 2, 1, 1, 1, 1, 1, 1, 3];
    let strides = vec![4isize, 2, 2, 2, 2, 2, 2, 2, 1];
    let logical = view_values(&rank_nine_data, &shape, &strides, 0);
    for lower in [true, false] {
        let expected = triangular_reference(&logical, &shape, 0, lower);
        let observed = if lower {
            engine.tril_read(region(&rank_nine, shape.clone(), strides.clone(), 0), 0)
        } else {
            engine.triu_read(region(&rank_nine, shape.clone(), strides.clone(), 0), 0)
        }
        .expect("rank-9 triangular read");
        assert_eq!(read(&engine, &observed), expected, "rank-9 lower={lower}");
    }

    // An empty non-flat rank-9 region has a zero grid; the entry must return the empty output
    // without uploading geometry or launching the zero-grid kernel.
    let empty_shape = vec![2usize, 2, 1, 1, 1, 1, 1, 1, 0];
    let empty = engine
        .tril_read(region(&rank_nine, empty_shape.clone(), strides, 0), 0)
        .expect("empty rank-9 triangular read");
    assert_eq!(empty.shape(), &empty_shape[..]);
    assert!(read(&engine, &empty).is_empty());
}

/// DUS metadata and float-index refusals run before the output allocation or any operand copy.
#[test]
#[ignore = "requires CUDA"]
fn dynamic_update_slice_refuses_bad_metadata_before_allocating_or_staging() {
    let Some(mut engine) = engine() else { return };
    let operand = device(
        &engine,
        vec![4, 5],
        (0..20).map(|index| index as f64).collect(),
    );
    let update = device(
        &engine,
        vec![2, 3],
        (0..6).map(|index| index as f64).collect(),
    );
    let wrong_dtype = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![2, 3], vec![1i32; 6]).expect("wrong dtype"),
    )
    .expect("upload wrong dtype");
    let wrong_rank = device(&engine, vec![6], (0..6).map(|index| index as f64).collect());
    let too_wide = device(
        &engine,
        vec![5, 3],
        (0..15).map(|index| index as f64).collect(),
    );
    let short_starts = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![1], vec![0i64]).expect("short starts"),
    )
    .expect("upload short starts");
    let rank_two_starts = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![1, 2], vec![0i64, 1]).expect("rank-2 starts"),
    )
    .expect("upload rank-2 starts");
    let bool_starts = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![2], vec![true, false]).expect("bool starts"),
    )
    .expect("upload bool starts");

    // A non-compact starts view would stage if the entry resolved it before its metadata checks.
    let starts_source = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![4], vec![0i64, 1, 2, 3]).expect("starts source"),
    )
    .expect("upload starts source");
    let non_compact_starts = TensorRead::from_view(TensorView::I64(
        starts_source
            .as_typed::<i64>()
            .expect("i64 payload")
            .backend_region_view(vec![2], vec![2], 0)
            .expect("strided starts view"),
    ));
    let compact_starts = TensorRead::from_tensor(&short_starts);

    let before_staged = engine.staged_reads();
    let before = engine.runtime().stats();
    for (label, update, starts) in [
        (
            "dtype",
            TensorRead::from_tensor(&wrong_dtype),
            non_compact_starts.clone(),
        ),
        (
            "rank",
            TensorRead::from_tensor(&wrong_rank),
            non_compact_starts.clone(),
        ),
        (
            "extent",
            TensorRead::from_tensor(&too_wide),
            non_compact_starts.clone(),
        ),
        (
            "starts count",
            TensorRead::from_tensor(&update),
            compact_starts.clone(),
        ),
        (
            "starts rank",
            TensorRead::from_tensor(&update),
            TensorRead::from_tensor(&rank_two_starts),
        ),
        (
            "starts dtype",
            TensorRead::from_tensor(&update),
            TensorRead::from_tensor(&bool_starts),
        ),
    ] {
        engine
            .dynamic_update_slice_read(TensorRead::from_tensor(&operand), update, starts)
            .expect_err(label);
    }
    assert_eq!(
        engine.staged_reads(),
        before_staged,
        "metadata refusals must run before a starts view is staged"
    );
    let after = engine.runtime().stats();
    assert_eq!(
        after.live_allocations, before.live_allocations,
        "metadata refusals must run before the output is allocated"
    );
    assert_eq!(
        (after.module_entries, after.artifact_entries),
        (before.module_entries, before.artifact_entries),
        "metadata refusals must run before a launch is prepared"
    );

    // A fractional float start is refused before the update window is written.
    let fractional = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![2], vec![1.5f32, 2.0]).expect("fractional starts"),
    )
    .expect("upload fractional starts");
    let error = engine
        .dynamic_update_slice_read(
            TensorRead::from_tensor(&operand),
            TensorRead::from_tensor(&update),
            TensorRead::from_tensor(&fractional),
        )
        .expect_err("a fractional start must be refused");
    assert!(
        format!("{error}").contains("not an exactly representable integer"),
        "unexpected refusal: {error}"
    );
}

/// Reversing an axis whose extent is at most one is the identity, whatever its stride, and an
/// empty region has no element to reverse at all.
#[test]
#[ignore = "requires CUDA"]
fn reverse_read_keeps_singleton_and_empty_geometry() {
    let Some(mut engine) = engine() else { return };
    let source = device(&engine, vec![4], (0..4).map(|index| index as f64).collect());

    // `isize::MIN` cannot be negated, but a singleton axis is never negated.
    let reversed = engine
        .reverse_read(region(&source, vec![1], vec![isize::MIN], 0), &[0])
        .expect("reverse a singleton with the most negative stride");
    assert_eq!(reversed.shape(), &[1]);
    assert_eq!(read(&engine, &reversed), vec![0.0]);

    // An empty region carries no element, so the reversal arithmetic is skipped entirely.
    let reversed = engine
        .reverse_read(region(&source, vec![0], vec![isize::MIN], 0), &[0])
        .expect("reverse an empty region with the most negative stride");
    assert_eq!(reversed.shape(), &[0]);
    assert!(read(&engine, &reversed).is_empty());
}

/// A refused later input must be found before the first copy is prepared or submitted.
#[test]
#[ignore = "requires CUDA"]
fn concatenate_refuses_a_foreign_later_input_before_preparing_anything() {
    let Some(mut engine) = engine() else { return };
    let local = device(
        &engine,
        vec![2, 2],
        (0..4).map(|index| index as f64).collect(),
    );
    // A second engine owns a different allocation domain.
    let other = CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("second engine");
    let foreign = upload_tensor(
        other.runtime(),
        &Tensor::from_vec_col_major(vec![2, 2], vec![4.0, 5.0, 6.0, 7.0]).expect("host tensor"),
    )
    .expect("foreign upload");

    let before_staged = engine.staged_reads();
    let before = engine.runtime().stats();
    let error = engine
        .concatenate_read(
            &[
                TensorRead::from_tensor(&local),
                TensorRead::from_tensor(&foreign),
            ],
            0,
        )
        .expect_err("a later input of another runtime must be refused");
    assert!(
        format!("{error}").contains("expected a CUDA allocation of this runtime"),
        "unexpected refusal: {error}"
    );
    assert_eq!(
        engine.staged_reads(),
        before_staged,
        "a refused concatenate must not stage an operand"
    );
    let after = engine.runtime().stats();
    assert_eq!(
        after.module_entries, before.module_entries,
        "a refused concatenate must not prepare a dispatcher"
    );
    assert_eq!(
        after.artifact_entries, before.artifact_entries,
        "a refused concatenate must not prepare an artifact"
    );
}

/// A refused elementwise call must be refused before its destination-sized scratch and operands.
///
/// The borrowed-view arm computes into a compact scratch of the destination's own shape. An
/// operand that does not share that shape or dtype used to be found only after the scratch was
/// allocated and every operand staged; the metadata check now runs first, so a large destination
/// view plus a bad operand costs one validation error and nothing else.
#[test]
#[ignore = "requires CUDA"]
fn elementwise_read_into_refuses_a_mismatched_operand_before_staging_the_scratch() {
    let Some(mut engine) = engine() else { return };
    let mut root = device(
        &engine,
        vec![16, 16],
        (0..256).map(|index| index as f64).collect(),
    );
    let operand = device(
        &engine,
        vec![4, 4],
        (0..16).map(|index| index as f64).collect(),
    );
    // A borrowed operand would be materialized by the read boundary, so the staging counter
    // proves the validation ran before any operand copy.
    let borrowed = region(&operand, vec![4, 4], vec![1, 4], 0);
    let before_staged = engine.staged_reads();
    let error = {
        let view = root
            .as_typed_mut::<f64>()
            .expect("f64 payload")
            .backend_region_view_mut(vec![8, 8], vec![2, 32], 0)
            .expect("mutable region view");
        engine
            .elementwise_read_into(
                ElementwiseReadOp::Add,
                &[borrowed.clone(), borrowed],
                TensorWrite::from_view(TensorViewMut::F64(view)),
            )
            .expect_err("a shape mismatch must be refused")
    };
    assert!(
        format!("{error}").contains("every operand must share the destination's shape and dtype"),
        "unexpected refusal: {error}"
    );
    assert_eq!(
        engine.staged_reads(),
        before_staged,
        "a refused elementwise call must not stage an operand"
    );
}

/// A unit factor is not a validation bypass: every factor goes through the write boundary.
#[test]
#[ignore = "requires CUDA"]
fn scale_tensor_write_refuses_a_host_view_whatever_the_factor() {
    let Some(mut engine) = engine() else { return };
    let mut data = [1.0f64, 2.0, 3.0, 4.0];
    for factor in [1.0, 2.5] {
        let view = TypedTensorViewMut::from_slice([2, 2], [1, 2], 0, &mut data).expect("host view");
        let error = engine
            .scale_tensor_write(
                TensorWrite::from_view(TensorViewMut::F64(view)),
                factor,
                "scale_tensor_write test",
            )
            .expect_err("a host view must be refused");
        assert!(
            format!("{error}").contains("a host view would need an implicit transfer"),
            "factor {factor} was not refused by the write boundary: {error}"
        );
    }
    assert_eq!(
        data,
        [1.0, 2.0, 3.0, 4.0],
        "a refused scale changed the host view"
    );
}

/// An owned destination takes the same write boundary as a view, whatever the factor and
/// whatever the tensor's size: the flat kernel's factor-of-one and empty shortcuts must not run
/// before the destination is proved device-resident.
#[test]
#[ignore = "requires CUDA"]
fn scale_tensor_write_refuses_a_host_owned_destination_whatever_the_factor_and_size() {
    let Some(mut engine) = engine() else { return };
    for factor in [1.0, 2.5] {
        let mut host = Tensor::from_vec_col_major(vec![2], vec![1.0f64, 2.0]).expect("host tensor");
        let error = engine
            .scale_tensor_write(
                TensorWrite::from_tensor(&mut host),
                factor,
                "scale_tensor_write test",
            )
            .expect_err("a host tensor destination must be refused");
        assert!(
            format!("{error}").contains("a host tensor would need an implicit transfer"),
            "factor {factor} was not refused by the write boundary: {error}"
        );
        assert_eq!(
            host.as_slice::<f64>().expect("f64 payload"),
            &[1.0, 2.0],
            "a refused scale changed the host tensor"
        );

        // An empty destination is refused too: the write boundary runs before the flat kernel's
        // length shortcut, not after it.
        let mut empty = Tensor::from_vec_col_major(vec![0], Vec::<f64>::new()).expect("empty host");
        let error = engine
            .scale_tensor_write(
                TensorWrite::from_tensor(&mut empty),
                factor,
                "scale_tensor_write test",
            )
            .expect_err("an empty host tensor destination must be refused");
        assert!(
            format!("{error}").contains("a host tensor would need an implicit transfer"),
            "factor {factor} on an empty destination was not refused: {error}"
        );
    }
}

/// The scatter planner's own validation runs before the output allocation and the operand copy:
/// an update axis past the update rank is one error that stages and allocates nothing.
#[test]
#[ignore = "requires CUDA"]
fn scatter_read_refuses_a_bad_update_axis_before_allocating_or_staging() {
    let Some(mut engine) = engine() else { return };
    let operand = device(&engine, vec![2, 2], vec![0.0; 4]);
    let updates = device(&engine, vec![2], vec![5.0, 7.0]);
    // Four index values in the allocation, so the strided two-element view below stays inside it.
    let indices_host = Tensor::from_vec_col_major(vec![4], vec![1i64, 0, 1, 0]).expect("indices");
    let indices = upload_tensor(engine.runtime(), &indices_host).expect("upload indices");
    let config = tenferro_tensor::ScatterConfig {
        // The update rank is one, so axis 99 cannot be an update window axis.
        update_window_dims: vec![99],
        inserted_window_dims: vec![0, 1],
        scatter_dims_to_operand_dims: vec![0],
        index_vector_dim: 1,
    };
    // A strided index view would take one staging copy if the entry resolved it first; the
    // counter proves the configuration validation came before that resolution.
    let view = indices
        .as_typed::<i64>()
        .expect("i64 payload")
        .backend_region_view(vec![2], vec![2], 0)
        .expect("region view");
    let before_staged = engine.staged_reads();
    let before = engine.runtime().stats();
    let error = engine
        .scatter_read(
            TensorRead::from_tensor(&operand),
            TensorRead::from_view(TensorView::I64(view)),
            TensorRead::from_tensor(&updates),
            &config,
        )
        .expect_err("an update axis past the update rank must be refused");
    assert!(
        format!("{error}").contains("axis outside the operand rank"),
        "unexpected refusal: {error}"
    );
    assert_eq!(
        engine.staged_reads(),
        before_staged,
        "a refused scatter must not stage the index operand"
    );
    let after = engine.runtime().stats();
    assert_eq!(
        after.live_allocations, before.live_allocations,
        "a refused scatter must not allocate its output"
    );
    assert_eq!(
        (after.module_entries, after.artifact_entries),
        (before.module_entries, before.artifact_entries),
        "a refused scatter must not prepare a launch"
    );
}

/// The destination-view contraction refuses a dtype cuTENSOR does not serve before staging an
/// operand or allocating the scratch.
#[test]
#[ignore = "requires CUDA"]
fn dot_general_read_into_refuses_an_integer_dtype_before_staging_the_scratch() {
    use tenferro_tensor::{DotGeneralConfig, TensorDot as _};

    let Some(mut engine) = engine() else { return };
    let lhs_host = Tensor::from_vec_col_major(vec![2, 2], vec![1i32, 2, 3, 4]).expect("lhs");
    let rhs_host = Tensor::from_vec_col_major(vec![2, 2], vec![5i32, 6, 7, 8]).expect("rhs");
    let lhs = upload_tensor(engine.runtime(), &lhs_host).expect("upload lhs");
    let rhs = upload_tensor(engine.runtime(), &rhs_host).expect("upload rhs");
    let root_host = Tensor::from_vec_col_major(vec![2, 2], vec![0i32; 4]).expect("destination");
    let mut root = upload_tensor(engine.runtime(), &root_host).expect("upload destination");
    let config = DotGeneralConfig {
        lhs_contracting_dims: [0].into_iter().collect(),
        rhs_contracting_dims: [0].into_iter().collect(),
        lhs_batch_dims: [].into_iter().collect(),
        rhs_batch_dims: [].into_iter().collect(),
    };
    // A transposed lhs view would be staged by the read boundary, so the counter proves the
    // contraction dtype was refused first.
    let borrowed = lhs
        .as_typed::<i32>()
        .expect("i32 payload")
        .backend_region_view(vec![2, 2], vec![2, 1], 0)
        .expect("region view");
    let before_staged = engine.staged_reads();
    let before = engine.runtime().stats();
    let error = {
        let view = root
            .as_typed_mut::<i32>()
            .expect("i32 payload")
            .backend_region_view_mut(vec![2, 2], vec![2, 1], 0)
            .expect("mutable region view");
        engine
            .dot_general_read_into(
                TensorRead::from_view(TensorView::I32(borrowed)),
                TensorRead::from_tensor(&rhs),
                &config,
                TensorWrite::from_view(TensorViewMut::I32(view)),
            )
            .expect_err("an integer contraction is not served")
    };
    assert!(
        format!("{error}").contains("a contraction serves F32, F64, C32 and C64"),
        "unexpected refusal: {error}"
    );
    assert_eq!(
        error.kind(),
        tenferro_tensor::ErrorKind::Unsupported,
        "a contraction dtype cuTENSOR does not serve is an unsupported request"
    );
    assert_eq!(
        engine.staged_reads(),
        before_staged,
        "a refused contraction must not stage an operand"
    );
    let after = engine.runtime().stats();
    assert_eq!(
        after.live_allocations, before.live_allocations,
        "a refused contraction must not allocate the scratch"
    );
}

/// The gather planner's own slice-size check runs before the index operand is staged, the float
/// exactness check is launched or the output is allocated: one refused configuration costs one
/// error and nothing else.
#[test]
#[ignore = "requires CUDA"]
fn gather_read_refuses_an_oversized_slice_before_staging_or_allocating() {
    let Some(mut engine) = engine() else { return };
    let operand = device(&engine, vec![2, 2], vec![0.0; 4]);
    // A float index view that is not one contiguous span: resolving it would stage a copy and
    // then run the exactness check, so the counters prove the plan was validated first.
    let indices_host = Tensor::from_vec_col_major(vec![4], vec![0.0f64; 4]).expect("index values");
    let indices = upload_tensor(engine.runtime(), &indices_host).expect("upload indices");
    let config = tenferro_tensor::GatherConfig {
        // The window axis 0's slice is larger than the operand extent. `gather_shape` derives
        // the output without consulting the operand, so only `plan_gather` rejects this config.
        offset_dims: vec![0, 1],
        collapsed_slice_dims: vec![],
        start_index_map: vec![0],
        index_vector_dim: 1,
        slice_sizes: vec![3, 1],
    };
    let view = indices
        .as_typed::<f64>()
        .expect("f64 payload")
        .backend_region_view(vec![2], vec![2], 0)
        .expect("region view");
    let before_staged = engine.staged_reads();
    let before = engine.runtime().stats();
    let error = engine
        .gather_read(
            TensorRead::from_tensor(&operand),
            TensorRead::from_view(TensorView::F64(view)),
            &config,
        )
        .expect_err("a slice larger than the operand must be refused");
    assert!(
        format!("{error}").contains("operand extents do not match the operation"),
        "unexpected refusal: {error}"
    );
    assert_eq!(
        engine.staged_reads(),
        before_staged,
        "a refused gather must not stage the index operand"
    );
    let after = engine.runtime().stats();
    assert_eq!(
        after.live_allocations, before.live_allocations,
        "a refused gather must not allocate its output"
    );
    assert_eq!(
        (after.module_entries, after.artifact_entries),
        (before.module_entries, before.artifact_entries),
        "a refused gather must not prepare a launch"
    );
}
