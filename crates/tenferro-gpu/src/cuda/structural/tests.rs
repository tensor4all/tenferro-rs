//! Unit tests for the structural slice's arithmetic helpers and its native read boundary.

use super::super::structural::element_count;
use crate::cuda::{download_tensor, gpu_available, upload_tensor, CudaDeviceId, CudaEngine};
use tenferro_tensor::{
    ElementwiseReadOp, Engine as _, PadConfig, SliceConfig, Tensor, TensorElementwise as _,
    TensorIndexing, TensorRead, TensorStructural, TensorView, TensorViewMut, TensorWrite,
    TypedTensorViewMut,
};

/// The counter the absence test reads is live: the zeroing allocator does move it.
///
/// Without this, "the counter did not move" could just mean the counter is dead, and the
/// absence-of-fill assertion would pass on the unfixed build too.
#[test]
fn the_zero_fill_counter_tracks_a_zeroed_allocation() {
    let Some(engine) = engine() else { return };
    let before = engine.runtime().stats().retirement.zero_fills;
    drop(super::alloc_tensor(engine.runtime(), tenferro_tensor::DType::F64, &[4]).expect("alloc"));
    assert!(
        engine.runtime().stats().retirement.zero_fills > before,
        "the zeroing allocator has to be visible in the counter"
    );
}

/// One elementwise launch must overwrite its whole destination, so it must not zero-fill it.
///
/// The counter is the runtime's own (`Stats.retirement.zero_fills`), so this fails on a build that
/// allocates the destination with `alloc_zeroed` and passes once the launch declares it
/// uninitialised (#2529). `add` is the plain arithmetic entry: it takes no status word, so the only
/// zero fill this call could produce is the destination's.
#[test]
fn an_elementwise_launch_does_not_zero_its_destination() {
    use tenferro_runtime::CoreOps;
    let Some(mut engine) = engine() else { return };
    let a = device(
        &engine,
        vec![8],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    );
    let b = device(
        &engine,
        vec![8],
        vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    );
    // Prepare the translation unit first: that is not the launch under test.
    drop(a.add(&b, &mut engine).expect("warm-up add"));
    let before = engine.runtime().stats().retirement.zero_fills;
    let sum = a.add(&b, &mut engine).expect("add");
    assert_eq!(read(&engine, &sum), vec![9.0; 8]);
    assert_eq!(
        engine.runtime().stats().retirement.zero_fills,
        before,
        "the destination of an elementwise launch is fully overwritten, so zeroing it is a wasted pass"
    );
}

/// A destination that reuses pooled storage keeps no byte of its previous contents.
///
/// The operands are uploaded first and only then is a same-sized buffer filled with a signalling
/// NaN and dropped, so the very next allocation of that size — the `neg` destination — is the
/// freed block. A kernel that wrote fewer than `len` elements would leave the pattern in the
/// result, which the zero fill used to mask. The later calls assert the same exact values; their
/// destinations may come from fresh pool memory, so they test the write on their own and not the
/// reuse.
#[test]
fn an_elementwise_result_holds_no_stale_bytes() {
    use tenferro_runtime::CoreOps;
    let Some(mut engine) = engine() else { return };
    let len = 1 << 18;
    let values: Vec<f64> = (0..len).map(|index| index as f64 * 0.5 - 7.0).collect();
    let a = device(&engine, vec![len], values.clone());
    let b = device(&engine, vec![len], values.clone());
    let lower = device(&engine, vec![len], vec![-1.0; len]);
    let upper = device(&engine, vec![len], vec![1.0; len]);
    // A signalling NaN: nothing these operations produce can equal it. Dropped last, so the next
    // same-sized allocation takes its block.
    let poison = f64::from_bits(0x7ff0_0000_dead_beef);
    drop(device(&engine, vec![len], vec![poison; len]));
    let negated = a.neg(&mut engine).expect("neg");
    assert_eq!(
        read(&engine, &negated),
        values.iter().map(|v| -v).collect::<Vec<_>>()
    );
    let summed = a.add(&b, &mut engine).expect("add");
    assert_eq!(
        read(&engine, &summed),
        values.iter().map(|v| 2.0 * v).collect::<Vec<_>>()
    );
    let clamped = a.clamp(&lower, &upper, &mut engine).expect("clamp");
    assert_eq!(
        read(&engine, &clamped),
        values
            .iter()
            .map(|v| v.clamp(-1.0, 1.0))
            .collect::<Vec<_>>()
    );
}

/// The destination is fully written at every length the grid arithmetic can get wrong.
#[test]
fn an_elementwise_launch_covers_every_length() {
    use tenferro_runtime::CoreOps;
    let Some(mut engine) = engine() else { return };
    for len in [0usize, 1, 7, 1_000_003] {
        let values: Vec<f64> = (0..len).map(|index| index as f64 * 0.25 - 3.0).collect();
        let a = device(&engine, vec![len], values.clone());
        let b = device(&engine, vec![len], values.clone());
        let negated = a.neg(&mut engine).expect("neg");
        assert_eq!(
            read(&engine, &negated),
            values.iter().map(|v| -v).collect::<Vec<_>>(),
            "neg at len {len}"
        );
        let summed = a.add(&b, &mut engine).expect("add");
        assert_eq!(
            read(&engine, &summed),
            values.iter().map(|v| 2.0 * v).collect::<Vec<_>>(),
            "add at len {len}"
        );
        // The comparison writes one Bool byte per pair, the only `DType::Bool` destination here.
        let compared = a
            .compare(&b, tenferro_tensor::CompareDir::Le, &mut engine)
            .expect("compare");
        let observed = download_tensor(engine.runtime(), &compared).expect("download");
        assert!(
            observed
                .as_slice::<bool>()
                .expect("bool payload")
                .iter()
                .all(|same| *same),
            "compare at len {len}"
        );
    }
}

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

/// One batched product's host reference, column-major with the batch axis last.
fn host_batched_product(
    a: &[f64],
    b: &[f64],
    batches: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f64> {
    let mut c = vec![0.0f64; batches * m * n];
    for batch in 0..batches {
        for column in 0..n {
            for row in 0..m {
                let mut sum = 0.0;
                for inner in 0..k {
                    sum +=
                        a[row + inner * m + batch * m * k] * b[inner + column * k + batch * k * n];
                }
                c[row + column * m + batch * m * n] = sum;
            }
        }
    }
    c
}

/// A batched product whose operands are the compact spans of their allocations is contracted
/// where the operands lie: neither the copy the read boundary stages for other layouts nor the
/// allocation that copy would need is paid (#2205's batched matmul row: 1024 x 256^3 f64 spent
/// 36 ms of 43 ms staging two 512 MiB operands per call).
#[test]
#[ignore = "requires CUDA"]
fn a_compact_batched_dot_reads_both_operands_in_place() {
    use tenferro_tensor::{DotGeneralConfig, TensorDot as _};

    let Some(mut engine) = engine() else { return };
    let (batches, m, n, k) = (3usize, 5usize, 4usize, 6usize);
    let a: Vec<f64> = (0..batches * m * k)
        .map(|index| (index % 11) as f64 * 0.25 - 1.0)
        .collect();
    let b: Vec<f64> = (0..batches * k * n)
        .map(|index| (index % 7) as f64 - 3.0)
        .collect();
    let lhs = device(&engine, vec![m, k, batches], a.clone());
    let rhs = device(&engine, vec![k, n, batches], b.clone());
    let config = DotGeneralConfig {
        lhs_contracting_dims: [1].into_iter().collect(),
        rhs_contracting_dims: [0].into_iter().collect(),
        lhs_batch_dims: [2].into_iter().collect(),
        rhs_batch_dims: [2].into_iter().collect(),
    };
    let before = engine.staged_reads();
    let product = engine
        .dot_general_read(
            region(
                &lhs,
                vec![m, k, batches],
                vec![1, m as isize, (m * k) as isize],
                0,
            ),
            region(
                &rhs,
                vec![k, n, batches],
                vec![1, k as isize, (k * n) as isize],
                0,
            ),
            &config,
        )
        .expect("dot_general");
    assert_eq!(
        engine.staged_reads(),
        before,
        "a compact operand is read in place, not staged"
    );
    // Every operand entry is a multiple of a quarter and every product a multiple of a sixteenth,
    // so the six-term sums are exact in f64 and the order cuTENSOR accumulates in cannot change the
    // result. Exact equality is then the strongest check available, and unlike a normwise bound it
    // fails on a NaN entry.
    let expected = host_batched_product(&a, &b, batches, m, n, k);
    assert_eq!(read(&engine, &product), expected);
}

/// The same product with a permuted operand keeps the staged copy: a batch axis stored first is
/// not one flat compact span, so the operand stays outside the in-place path.
#[test]
#[ignore = "requires CUDA"]
fn a_permuted_batched_dot_stages_the_permuted_operand() {
    use tenferro_tensor::{DotGeneralConfig, TensorDot as _};

    let Some(mut engine) = engine() else { return };
    let (batches, m, n, k) = (3usize, 5usize, 4usize, 6usize);
    let a: Vec<f64> = (0..batches * m * k)
        .map(|index| (index % 11) as f64 * 0.25 - 1.0)
        .collect();
    let logical_b: Vec<f64> = (0..batches * k * n)
        .map(|index| (index % 7) as f64 - 3.0)
        .collect();
    // The same values stored `(batches, k, n)`, so reading them as `(k, n, batches)` is the
    // permutation `(batches, batches * k, 1)`.
    let mut stored_b = vec![0.0f64; batches * k * n];
    for batch in 0..batches {
        for column in 0..n {
            for inner in 0..k {
                stored_b[batch + batches * inner + batches * k * column] =
                    logical_b[inner + k * column + k * n * batch];
            }
        }
    }
    let lhs = device(&engine, vec![m, k, batches], a.clone());
    let rhs = device(&engine, vec![batches, k, n], stored_b);
    let config = DotGeneralConfig {
        lhs_contracting_dims: [1].into_iter().collect(),
        rhs_contracting_dims: [0].into_iter().collect(),
        lhs_batch_dims: [2].into_iter().collect(),
        rhs_batch_dims: [2].into_iter().collect(),
    };
    let before = engine.staged_reads();
    let product = engine
        .dot_general_read(
            region(
                &lhs,
                vec![m, k, batches],
                vec![1, m as isize, (m * k) as isize],
                0,
            ),
            region(
                &rhs,
                vec![k, n, batches],
                vec![batches as isize, (batches * k) as isize, 1],
                0,
            ),
            &config,
        )
        .expect("dot_general");
    assert_eq!(
        engine.staged_reads(),
        before + 1,
        "a permuted operand is staged once"
    );
    // Exactly representable entries, as in the in-place case above.
    let expected = host_batched_product(&a, &logical_b, batches, m, n, k);
    assert_eq!(read(&engine, &product), expected);
}

/// The compact copy of one host view, in the view's logical order: the reference the generic and
/// the tiled materialization must both reproduce.
fn host_compact(data: &[f64], extents: &[usize], strides: &[isize], offset: isize) -> Vec<f64> {
    let count: usize = extents.iter().product();
    let mut out = Vec::with_capacity(count);
    let mut coords = vec![0usize; extents.len()];
    for _ in 0..count {
        let index: isize = offset
            + coords
                .iter()
                .zip(strides)
                .map(|(&coord, &stride)| coord as isize * stride)
                .sum::<isize>();
        out.push(data[index as usize]);
        for axis in 0..extents.len() {
            coords[axis] += 1;
            if coords[axis] < extents[axis] {
                break;
            }
            coords[axis] = 0;
        }
    }
    out
}

/// The tiled entry is selected for exactly the views whose destination it addresses.
#[test]
fn tiled_transpose_geometry_selects_only_a_compact_pair_swap() {
    let geometry = |extents: &[i64], strides: &[i64], offset: i64| {
        super::tiled_transpose_geometry(extents, strides, offset)
    };

    // A rank-two transpose and the 3-D `[1, 0, 2]` swap of an asymmetric shape: the source's two
    // fastest axes swap and the remaining axes keep one compact run.
    let tiled = geometry(&[4, 3], &[3, 1], 0).expect("rank-two transpose");
    assert_eq!(
        (tiled.dst_fast_extent, tiled.src_fast_extent, tiled.batches),
        (4, 3, 1)
    );
    let tiled = geometry(&[5, 3, 7], &[3, 1, 15], 0).expect("3-D pair swap");
    assert_eq!(
        (tiled.dst_fast_extent, tiled.src_fast_extent, tiled.batches),
        (5, 3, 7)
    );
    assert_eq!(tiled.batch_stride(), 15);

    // The `[2, 0, 1]` rotation of `transpose_3d_1024_1024_512_201`, before its trailing axes are
    // merged: the source's fastest axis is not yet one run with the next one. The selector itself
    // does not merge; `materialize_storage` hands it the coalesced view.
    assert_eq!(geometry(&[512, 1024, 1024], &[1048576, 1, 1024], 0), None);
    // A batch axis that is not the next stride of the source's run.
    assert_eq!(geometry(&[4, 3, 2], &[3, 1, 8], 0), None);
    // Reversed, broadcast and rank-one views, and a negative base offset.
    assert_eq!(geometry(&[4, 3], &[3, -1], 0), None);
    assert_eq!(geometry(&[4, 3], &[3, 0], 0), None);
    assert_eq!(geometry(&[4], &[1], 0), None);
    assert_eq!(geometry(&[4, 3], &[3, 1], -1), None);
    // An empty extent leaves the generic path, which already writes nothing.
    assert_eq!(geometry(&[0, 3], &[3, 1], 0), None);
}

/// The geometry the selector derives addresses the same elements as the view it came from.
#[test]
fn the_tiled_geometry_addresses_every_element_the_view_names() {
    // An asymmetric 3-D pair swap, and a rank-two transpose whose view starts at an offset.
    let cases: [(Vec<usize>, Vec<isize>, isize); 2] = [
        (vec![5, 3, 7], vec![3, 1, 15], 0),
        (vec![4, 3], vec![3, 1], 2),
    ];
    for (extents, strides, offset) in cases {
        let count: usize = extents.iter().product();
        let reach = count + offset as usize;
        let data: Vec<f64> = (0..reach).map(|index| index as f64).collect();
        let extents_i64: Vec<i64> = extents.iter().map(|&extent| extent as i64).collect();
        let strides_i64: Vec<i64> = strides.iter().map(|&stride| stride as i64).collect();
        let geometry = super::tiled_transpose_geometry(&extents_i64, &strides_i64, offset as i64)
            .expect("the tiled geometry");

        // The catalogue's own walk: `dst[b*M*N + i + j*M] = src[o + b*M*N + i*N + j]`.
        let (m, n, batches) = (
            geometry.dst_fast_extent,
            geometry.src_fast_extent,
            geometry.batches,
        );
        let matrix = m * n;
        let mut tiled = vec![f64::NAN; count];
        for batch in 0..batches {
            for i in 0..m {
                for j in 0..n {
                    tiled[(batch * matrix + i + j * m) as usize] =
                        data[(geometry.src_offset + batch * matrix + i * n + j) as usize];
                }
            }
        }
        assert_eq!(
            tiled,
            host_compact(&data, &extents, &strides, offset),
            "tiled geometry {geometry:?}"
        );
    }
}

/// An adjacent pair swap reads through the tiled entry; the values are the view's own.
#[test]
#[ignore = "requires CUDA"]
fn an_adjacent_pair_swap_reads_through_the_tiled_transpose() {
    let Some(mut engine) = engine() else { return };
    // A multi-tile view: `17 x 19` at the default 16-wide tile needs two tile columns on x and
    // two tile rows on y, so a launch that drops `grid.y` leaves the second tile row unwritten
    // and the full comparison below fails.
    let data: Vec<f64> = (0..2261).map(|index| index as f64).collect();
    let source = device(&engine, vec![19, 17, 7], data.clone());

    let before = engine.tiled_transposes();
    let copy = engine
        .to_contiguous_read(region(&source, vec![17, 19, 7], vec![19, 1, 323], 0))
        .expect("materialize the pair swap");
    assert_eq!(
        engine.tiled_transposes(),
        before + 1,
        "an adjacent pair swap takes the tiled transpose"
    );
    assert_eq!(
        read(&engine, &copy),
        host_compact(&data, &[17, 19, 7], &[19, 1, 323], 0)
    );
}

/// A pair swap whose view starts at a nonzero offset reads from the view's base.
#[test]
#[ignore = "requires CUDA"]
fn an_offset_pair_swap_reads_the_view_through_the_tiled_transpose() {
    let Some(mut engine) = engine() else { return };
    // One dummy element ahead of the payload, so the view's base and the allocation's start
    // differ: the launch's `src_offset` argument is what selects the payload, and a launch that
    // passed zero reads every element one place off.
    let data: Vec<f64> = (0..2262).map(|index| index as f64).collect();
    let source = device(&engine, vec![2262], data.clone());

    let before = engine.tiled_transposes();
    let copy = engine
        .to_contiguous_read(region(&source, vec![17, 19, 7], vec![19, 1, 323], 1))
        .expect("materialize the offset pair swap");
    assert_eq!(
        engine.tiled_transposes(),
        before + 1,
        "a pair swap at a nonzero offset takes the tiled transpose"
    );
    assert_eq!(
        read(&engine, &copy),
        host_compact(&data, &[17, 19, 7], &[19, 1, 323], 1)
    );
}

/// A view the tiled entry does not address keeps the generic materialization, with its values.
#[test]
#[ignore = "requires CUDA"]
fn a_three_axis_reversal_stays_on_the_generic_materialization() {
    let Some(mut engine) = engine() else { return };
    let data: Vec<f64> = (0..24).map(|index| index as f64).collect();
    let source = device(&engine, vec![2, 3, 4], data.clone());

    // `[2, 1, 0]` of a compact `2 x 3 x 4`: no two destination axes merge (strides `6, 2, 1`),
    // and the source's fastest axis is the destination's slowest one, so no pair swap exists.
    // (The `[2, 0, 1]` rotation this test used to hold is a pair swap once its trailing axes
    // merge, and now takes the tiled entry.)
    let before = engine.tiled_transposes();
    let copy = engine
        .to_contiguous_read(region(&source, vec![4, 3, 2], vec![6, 2, 1], 0))
        .expect("materialize the reversal");
    assert_eq!(
        engine.tiled_transposes(),
        before,
        "a three-axis reversal keeps the generic path"
    );
    assert_eq!(
        read(&engine, &copy),
        host_compact(&data, &[4, 3, 2], &[6, 2, 1], 0)
    );
}

/// The batched view the device test materializes plans one grid axis per batch, not one batch.
#[test]
fn a_batched_pair_swap_plans_one_grid_axis_per_batch() {
    // The `17 x 19 x 7` view `an_adjacent_pair_swap_reads_through_the_tiled_transpose` materializes:
    // seven independent `17 x 19` transposes, two tiles wide on each of x and y.
    let geometry = super::tiled_transpose_geometry(&[17, 19, 7], &[19, 1, 323], 0)
        .expect("the tiled geometry");
    let (config, plan) =
        super::tiled_transpose_plan(t4a_cuda_kernels::DType::F64, geometry, 2261, 2261)
            .expect("the catalogue plans the launch");
    assert_eq!(
        config,
        t4a_cuda_kernels::TransposeConfig::DEFAULT,
        "a geometry the default tile launches keeps it"
    );

    // Every batch is its own grid block on z; the tile rows are block.y, and the tile itself is
    // the launch's dynamic shared memory. A launch keeping `grid[0]` alone would carry
    // `[2, 1, 1]` here and cover one batch of one tile row from a zero-length shared tile.
    assert_eq!(plan.grid, [2, 2, 7], "tile columns, tile rows, batches");
    assert_eq!(plan.block, [16, 8, 1], "tile lanes, tile row groups");
    assert_eq!(plan.shared_bytes, 16 * 17 * 8, "a 16 x 16 f64 tile, padded");
}

/// Merging the axes a compact destination and its source advance through together.
#[test]
fn coalescing_merges_only_the_axes_the_source_continues() {
    let coalesce = super::coalesce_compact_view;
    // `[2, 0, 1]` of a compact `1024 x 1024 x 512`: the trailing destination axes are one run.
    assert_eq!(
        coalesce(&[512, 1024, 1024], &[1048576, 1, 1024]),
        (vec![512, 1048576], vec![1048576, 1])
    );
    // A compact view is one axis; an axis of extent one is dropped wherever it sits.
    assert_eq!(coalesce(&[4, 3, 2], &[1, 4, 12]), (vec![24], vec![1]));
    assert_eq!(coalesce(&[4, 1, 3], &[3, 99, 1]), (vec![4, 3], vec![3, 1]));
    // No merge: a reversal, a gap, a negative stride and a broadcast axis.
    assert_eq!(
        coalesce(&[4, 3, 2], &[6, 2, 1]),
        (vec![4, 3, 2], vec![6, 2, 1])
    );
    assert_eq!(coalesce(&[4, 3], &[1, 5]), (vec![4, 3], vec![1, 5]));
    assert_eq!(coalesce(&[4, 3], &[-1, 4]), (vec![4, 3], vec![-1, 4]));
    assert_eq!(coalesce(&[4, 3], &[1, 0]), (vec![4, 3], vec![1, 0]));
    // A reversed run merges into one reversed axis.
    assert_eq!(coalesce(&[4, 3], &[-1, -4]), (vec![12], vec![-1]));
    // An empty view and an all-singleton view come back unchanged.
    assert_eq!(coalesce(&[4, 0], &[1, 4]), (vec![4, 0], vec![1, 4]));
    assert_eq!(coalesce(&[1, 1], &[1, 1]), (vec![1, 1], vec![1, 1]));
}

/// The merged view addresses the same source element at every destination position.
#[test]
fn a_coalesced_view_addresses_every_element_the_view_names() {
    let cases: [(Vec<usize>, Vec<isize>, isize); 3] = [
        // `[2, 0, 1]` and `[2, 0, 1, 3]` of compact sources, and a reversed run at an offset.
        (vec![17, 5, 7], vec![35, 1, 5], 0),
        (vec![17, 5, 7, 3], vec![35, 1, 5, 595], 0),
        (vec![4, 1, 3], vec![-1, 7, -4], 11),
    ];
    for (extents, strides, offset) in cases {
        let count: usize = extents.iter().product();
        let reach = (count as isize + offset + 1) as usize;
        let data: Vec<f64> = (0..reach).map(|index| index as f64).collect();
        let extents_i64: Vec<i64> = extents.iter().map(|&extent| extent as i64).collect();
        let strides_i64: Vec<i64> = strides.iter().map(|&stride| stride as i64).collect();
        let (merged_extents, merged_strides) =
            super::coalesce_compact_view(&extents_i64, &strides_i64);
        let merged_extents: Vec<usize> = merged_extents.iter().map(|&e| e as usize).collect();
        let merged_strides: Vec<isize> = merged_strides.iter().map(|&s| s as isize).collect();
        assert_eq!(
            host_compact(&data, &merged_extents, &merged_strides, offset),
            host_compact(&data, &extents, &strides, offset),
            "view {extents:?} / {strides:?} merged to {merged_extents:?} / {merged_strides:?}"
        );
    }
}

/// The order in which a destination region is one compact span at its base.
#[test]
fn a_dense_destination_order_exists_only_for_a_permuted_compact_span() {
    let order = super::dense_destination_order;
    // The inverse-permuted destination of a `[2, 0, 1]` transpose, and of a pair swap.
    assert_eq!(order(&[17, 35, 3], &[35, 1, 595], 0), Some(vec![1, 0, 2]));
    assert_eq!(order(&[4, 3], &[3, 1], 0), Some(vec![1, 0]));
    // A compact destination is its own order; extent-one axes go last.
    assert_eq!(order(&[4, 1, 3], &[1, 9, 4], 0), Some(vec![0, 2, 1]));
    // Refused: an offset, a gap, an overlap, a broadcast, a reversal and an empty extent.
    assert_eq!(order(&[4, 3], &[3, 1], 2), None);
    assert_eq!(order(&[4, 3], &[1, 5], 0), None);
    assert_eq!(order(&[4, 3], &[1, 2], 0), None);
    assert_eq!(order(&[4, 3], &[0, 1], 0), None);
    assert_eq!(order(&[4, 3], &[-1, 4], 0), None);
    assert_eq!(order(&[4, 0], &[1, 4], 0), None);
}

/// Which `gpu/permutation` benchmark views the tiled entry addresses once their axes are merged.
///
/// Each view is the `transpose_read` view of a compact column-major source: destination axis `k`
/// reads source axis `perm[k]` at that axis's compact stride. This is the table the #2528
/// write-up reports; a change in selection shows up here before it shows up in a timing.
#[test]
fn the_benchmark_permutations_the_tiled_transpose_addresses() {
    fn view(shape: &[i64], perm: &[usize]) -> (Vec<i64>, Vec<i64>) {
        let mut compact = vec![1i64; shape.len()];
        for axis in 1..shape.len() {
            compact[axis] = compact[axis - 1] * shape[axis - 1];
        }
        (
            perm.iter().map(|&axis| shape[axis]).collect(),
            perm.iter().map(|&axis| compact[axis]).collect(),
        )
    }
    let tiled = |shape: &[i64], perm: &[usize]| {
        let (extents, strides) = view(shape, perm);
        let (extents, strides) = super::coalesce_compact_view(&extents, &strides);
        super::tiled_transpose_geometry(&extents, &strides, 0)
            .map(|g| (g.dst_fast_extent, g.src_fast_extent, g.batches))
    };
    let reverse = |rank: usize| (0..rank).rev().collect::<Vec<_>>();
    let mut tn_perm: Vec<usize> = vec![0, 1, 2, 3, 22, 4, 23];
    tn_perm.extend(5..22);
    let mut tn_shape = vec![64i64];
    tn_shape.extend(std::iter::repeat_n(2, 23));
    let mut reverse_23 = vec![128i64];
    reverse_23.extend(std::iter::repeat_n(2, 22));
    let mut cyclic: Vec<usize> = (1..18).collect();
    cyclic.push(0);

    assert_eq!(tiled(&[32768, 16384], &[1, 0]), Some((16384, 32768, 1)));
    assert_eq!(
        tiled(&[1024, 1024, 512], &[1, 0, 2]),
        Some((1024, 1024, 512))
    );
    assert_eq!(
        tiled(&[1024, 1024, 512], &[2, 0, 1]),
        Some((512, 1 << 20, 1))
    );
    assert_eq!(
        tiled(&[3; 18], &cyclic),
        Some((3i64.pow(17), 3, 1)),
        "cyclic_18d_3"
    );
    assert_eq!(
        tiled(&[64, 32, 32, 32, 16, 16], &[5, 0, 4, 1, 3, 2]),
        None,
        "rotation_6d"
    );
    assert_eq!(tiled(&reverse_23, &reverse(23)), None, "reverse_23d_128x2");
    assert_eq!(tiled(&[3; 18], &reverse(18)), None, "reverse_18d_3");
    assert_eq!(tiled(&tn_shape, &tn_perm), None, "tn_light_415_24d");
    // The tn_light view keeps its fastest 512-element run: merged, it fits the by-value layout.
    let (extents, strides) = view(&tn_shape, &tn_perm);
    let (extents, _) = super::coalesce_compact_view(&extents, &strides);
    assert!(extents.len() <= t4a_cuda_kernels::MAX_RANK, "{extents:?}");
}

/// A source run the default tile cannot launch plans with the 32-wide tile instead.
#[test]
fn a_source_run_past_the_default_grid_plans_with_the_wide_tile() {
    // 65535 tiles of 16 is the longest source run `grid.y` holds at the default tile.
    let run = 65_535 * 16 + 1;
    let geometry = super::tiled_transpose_geometry(&[3, run], &[run, 1], 0).expect("pair swap");
    let len = 3 * run;
    let (config, plan) =
        super::tiled_transpose_plan(t4a_cuda_kernels::DType::F64, geometry, len, len)
            .expect("the wide tile plans it");
    assert_eq!(config.tile(), 32);
    assert_eq!(plan.grid, [1, 32_768, 1]);
    assert_eq!(plan.block, [32, 8, 1]);
}

/// A rotation whose trailing destination axes merge into one source run reads through the tiled
/// transpose; the values are the view's own.
#[test]
#[ignore = "requires CUDA"]
fn a_rotation_whose_trailing_axes_merge_reads_through_the_tiled_transpose() {
    let Some(mut engine) = engine() else { return };
    // `[2, 0, 1, 3]` of a compact `5 x 7 x 17 x 3`: destination `[17, 5, 7, 3]`, strides
    // `[35, 1, 5, 595]`, which merges to a `17 x 35` pair swap in three batches. Two tile
    // columns on x, three tile rows on y and three batches on z, so a dropped grid axis leaves
    // elements unwritten.
    let data: Vec<f64> = (0..1785).map(|index| index as f64).collect();
    let source = device(&engine, vec![5, 7, 17, 3], data.clone());

    let before = engine.tiled_transposes();
    let copy = engine
        .transpose_read(TensorRead::from_tensor(&source), &[2, 0, 1, 3])
        .expect("transpose");
    assert_eq!(
        engine.tiled_transposes(),
        before + 1,
        "a rotation that merges into a pair swap takes the tiled transpose"
    );
    assert_eq!(
        read(&engine, &copy),
        host_compact(&data, &[17, 5, 7, 3], &[35, 1, 5, 595], 0)
    );
}

/// The same merged rotation read from a view at a nonzero offset reads from the view's base.
#[test]
#[ignore = "requires CUDA"]
fn an_offset_rotation_whose_trailing_axes_merge_reads_through_the_tiled_transpose() {
    let Some(mut engine) = engine() else { return };
    let data: Vec<f64> = (0..1786).map(|index| index as f64).collect();
    let source = device(&engine, vec![1786], data.clone());

    let before = engine.tiled_transposes();
    let copy = engine
        .to_contiguous_read(region(&source, vec![17, 5, 7, 3], vec![35, 1, 5, 595], 1))
        .expect("materialize the offset rotation");
    assert_eq!(
        engine.tiled_transposes(),
        before + 1,
        "an offset rotation that merges into a pair swap takes the tiled transpose"
    );
    assert_eq!(
        read(&engine, &copy),
        host_compact(&data, &[17, 5, 7, 3], &[35, 1, 5, 595], 1)
    );
}

/// A source run longer than the default tile's grid still reads through the tiled transpose.
#[test]
#[ignore = "requires CUDA"]
fn a_source_run_past_the_default_grid_reads_through_the_wide_tile() {
    let Some(mut engine) = engine() else { return };
    let run = 65_535 * 16 + 1;
    let data: Vec<f64> = (0..3 * run).map(|index| index as f64).collect();
    let source = device(&engine, vec![run, 3], data.clone());

    let before = engine.tiled_transposes();
    let copy = engine
        .transpose_read(TensorRead::from_tensor(&source), &[1, 0])
        .expect("transpose");
    assert_eq!(
        engine.tiled_transposes(),
        before + 1,
        "the wide tile serves a run the default grid refuses"
    );
    assert_eq!(
        read(&engine, &copy),
        host_compact(&data, &[3, run], &[run as isize, 1], 0)
    );
}

/// `copy_read_into` writes `source` through the destination region of `root` it names, and
/// returns the whole root afterwards.
fn copy_into_region(
    engine: &mut CudaEngine,
    root: &mut Tensor,
    source: &Tensor,
    shape: Vec<usize>,
    strides: Vec<isize>,
    offset: isize,
) -> Vec<f64> {
    {
        let view = root
            .as_typed_mut::<f64>()
            .expect("f64 payload")
            .backend_region_view_mut(shape, strides, offset)
            .expect("destination region");
        engine
            .copy_read_into(
                TensorRead::from_tensor(source),
                TensorWrite::from_view(TensorViewMut::F64(view)),
            )
            .expect("copy into the destination region");
    }
    read(engine, root)
}

/// `root` with `values` written at the region's positions, in the region's logical order.
fn expected_after_copy(
    root: &[f64],
    shape: &[usize],
    strides: &[isize],
    offset: isize,
    values: &[f64],
) -> Vec<f64> {
    let positions: Vec<f64> = (0..root.len()).map(|index| index as f64).collect();
    let mut expected = root.to_vec();
    for (position, &value) in host_compact(&positions, shape, strides, offset)
        .into_iter()
        .zip(values)
    {
        expected[position as usize] = value;
    }
    expected
}

/// A destination that is a permuted compact span at its base is written through the tiled
/// transpose, and nothing past the span changes.
#[test]
#[ignore = "requires CUDA"]
fn copy_read_into_a_permuted_compact_destination_reads_through_the_tiled_transpose() {
    let Some(mut engine) = engine() else { return };
    let values: Vec<f64> = (0..1785).map(|index| 1000.0 + index as f64).collect();
    let source = device(&engine, vec![17, 35, 3], values.clone());
    // The destination's fastest axis is the source's second one: in the destination's own order
    // the copy reads a `35 x 17` pair swap in three batches. The root holds five sentinels past
    // the 1785-element span.
    let sentinels: Vec<f64> = (0..1790).map(|index| -(index as f64)).collect();
    let mut root = device(&engine, vec![1790], sentinels.clone());
    let (shape, strides) = (vec![17, 35, 3], vec![35, 1, 595]);

    let before = engine.tiled_transposes();
    let observed = copy_into_region(
        &mut engine,
        &mut root,
        &source,
        shape.clone(),
        strides.clone(),
        0,
    );
    assert_eq!(
        engine.tiled_transposes(),
        before + 1,
        "a permuted compact destination takes the tiled transpose"
    );
    assert_eq!(
        observed,
        expected_after_copy(&sentinels, &shape, &strides, 0, &values)
    );
}

/// The same permuted destination at a nonzero offset keeps the strided-to-strided copy.
#[test]
#[ignore = "requires CUDA"]
fn copy_read_into_an_offset_permuted_destination_keeps_the_strided_copy() {
    let Some(mut engine) = engine() else { return };
    let values: Vec<f64> = (0..1785).map(|index| 1000.0 + index as f64).collect();
    let source = device(&engine, vec![17, 35, 3], values.clone());
    let sentinels: Vec<f64> = (0..1790).map(|index| -(index as f64)).collect();
    let mut root = device(&engine, vec![1790], sentinels.clone());
    let (shape, strides) = (vec![17, 35, 3], vec![35, 1, 595]);

    let before = engine.tiled_transposes();
    let observed = copy_into_region(
        &mut engine,
        &mut root,
        &source,
        shape.clone(),
        strides.clone(),
        2,
    );
    assert_eq!(
        engine.tiled_transposes(),
        before,
        "an offset destination keeps the strided copy"
    );
    assert_eq!(
        observed,
        expected_after_copy(&sentinels, &shape, &strides, 2, &values)
    );
}
