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
    let source = device(
        &engine,
        vec![3, 4],
        (0..12).map(|index| index as f64).collect(),
    );

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
    let mirror = device(
        &engine,
        vec![4, 3],
        (0..12).map(|index| 100.0 + index as f64).collect(),
    );
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

    // `tril_read` keeps the catalogue's compact-only kernel in this unit, so it stages.
    let before = engine.staged_reads();
    engine.tril_read(transposed(&source), 0).expect("tril_read");
    assert_eq!(
        engine.staged_reads(),
        before + 1,
        "the compact-only triangular entry must report its staging copy"
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
