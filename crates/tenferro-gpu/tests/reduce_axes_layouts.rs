#![cfg(feature = "cuda")]

//! Axis, multi-axis, strided and norm reductions on the CUDA engine against the CPU engine
//! (#2625).
//!
//! A compact column-major tensor lets one fold pass cover several axes, and a long strided axis
//! coalesces once it is two short ones, so the engine presents the catalogue a shape it derived
//! rather than the caller's. A wrong index in that derivation produces plausible wrong values
//! instead of an error, which is what these matrices are for: every case runs the same input
//! through the CPU engine and the CUDA engine and requires the same shape and the same values at
//! each boundary the presentation changes at.

use tenferro_cpu::CpuEngine;
use tenferro_gpu::cuda::{download_tensor, gpu_available, upload_tensor, CudaDeviceId, CudaEngine};
use tenferro_tensor::{Engine, Tensor, TensorRead, TensorReduction, TensorView};

fn engine() -> Option<CudaEngine> {
    if !gpu_available() {
        return None;
    }
    Some(CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("CUDA engine"))
}

#[derive(Clone, Copy, Debug)]
enum Op {
    Sum,
    Prod,
    Max,
    Min,
    SumSquares,
}

const OPS: [Op; 5] = [Op::Sum, Op::Prod, Op::Max, Op::Min, Op::SumSquares];

fn cpu_reduce(op: Op, source: &Tensor, axes: &[usize]) -> tenferro_tensor::Result<Tensor> {
    let mut engine = CpuEngine::serial();
    let read = TensorRead::from_tensor(source);
    match op {
        Op::Sum => engine.reduce_sum_read(read, axes),
        Op::Prod => engine.reduce_prod_read(read, axes),
        Op::Max => engine.reduce_max_read(read, axes),
        Op::Min => engine.reduce_min_read(read, axes),
        Op::SumSquares => engine.reduce_sum_squares_read(read, axes),
    }
}

fn gpu_reduce(
    engine: &mut CudaEngine,
    op: Op,
    read: TensorRead<'_>,
    axes: &[usize],
) -> tenferro_tensor::Result<Tensor> {
    match op {
        Op::Sum => engine.reduce_sum_read(read, axes),
        Op::Prod => engine.reduce_prod_read(read, axes),
        Op::Max => engine.reduce_max_read(read, axes),
        Op::Min => engine.reduce_min_read(read, axes),
        Op::SumSquares => engine.reduce_sum_squares_read(read, axes),
    }
}

/// Values just under and over one, so a product over a million of them stays `O(1)` and its
/// relative tolerance means something: products of `[0.5, 1.5)` decay to `1e-18`, where an absolute
/// tolerance would accept a result that is identically zero.
fn values(len: usize) -> Vec<f64> {
    (0..len)
        .map(|index| 1.0 + (((index * 7919) % 104729) as f64 / 104729.0 - 0.5) * 0.02)
        .collect()
}

fn tensor(shape: &[usize]) -> Tensor {
    Tensor::from_vec_col_major(shape.to_vec(), values(shape.iter().product())).expect("host tensor")
}

/// Fold `source` on both engines and require the same shape and the same values.
fn assert_matches_cpu(engine: &mut CudaEngine, op: Op, source: &Tensor, axes: &[usize]) {
    let cpu = cpu_reduce(op, source, axes).expect("cpu reduce");
    let hosted = upload(engine, source);
    let device =
        gpu_reduce(engine, op, TensorRead::from_tensor(&hosted), axes).expect("cuda reduce");
    assert_same_fold(engine, op, axes, &device, &cpu, source);
}

/// Require the CUDA result `device` to equal the CPU result `expected` for `op` over `axes`.
fn assert_same_fold(
    engine: &CudaEngine,
    op: Op,
    axes: &[usize],
    device: &Tensor,
    expected: &Tensor,
    source: &Tensor,
) {
    let host = download_tensor(engine.runtime(), device).expect("download");
    assert_eq!(
        host.shape(),
        expected.shape(),
        "{op:?} over {axes:?}: shape"
    );
    assert_eq!(
        host.dtype(),
        expected.dtype(),
        "{op:?} over {axes:?}: dtype"
    );
    let observed = host.as_slice::<f64>().expect("f64 result");
    let expected = expected.as_slice::<f64>().expect("f64 result");
    assert_eq!(
        observed.len(),
        expected.len(),
        "{op:?} over {axes:?}: length"
    );
    match op {
        // An extremum is one of the inputs, whatever the fold order.
        Op::Max | Op::Min => assert_eq!(observed, expected, "{op:?} over {axes:?}"),
        // Every other fold is associated differently from the CPU's serial order, so hold it to
        // the bound any association obeys: `n * u` relative to the sum of magnitudes.
        Op::Sum | Op::Prod | Op::SumSquares => {
            let unit_roundoff = f64::EPSILON / 2.0;
            let folded = source.as_slice::<f64>().expect("f64 payload").len();
            let tolerance = folded as f64 * unit_roundoff;
            for (index, (device_value, cpu_value)) in observed.iter().zip(expected).enumerate() {
                // Relative to the folded value itself: a product of a million terms is not
                // comparable to a unit-magnitude floor.
                let magnitude = cpu_value.abs().max(f64::MIN_POSITIVE);
                assert!(
                    (device_value - cpu_value).abs() <= tolerance * magnitude,
                    "{op:?} over {axes:?} at {index}: cuda {device_value}, cpu {cpu_value}"
                );
            }
        }
    }
}

/// Every shape here sits at a boundary of the presentation the engine derives: one plane wide
/// (32), the split floor (512), a `32 x 32` factorisation, and leading extents of one that make a
/// nonzero axis contiguous, which must not be split.
const SHAPES: [&[usize]; 7] = [
    &[64, 37],
    &[512, 512],
    &[1024, 1024],
    &[1, 1024],
    &[1024, 1],
    &[33, 33],
    &[7, 9, 11],
];

#[test]
#[ignore = "requires CUDA"]
fn both_axes_of_a_column_major_matrix_match_the_cpu() {
    let Some(mut engine) = engine() else { return };
    for shape in SHAPES {
        let source = tensor(shape);
        for axes in [&[0usize][..], &[1][..]] {
            for op in OPS {
                assert_matches_cpu(&mut engine, op, &source, axes);
            }
        }
    }
}

#[test]
#[ignore = "requires CUDA"]
fn every_multi_axis_choice_matches_the_cpu() {
    let Some(mut engine) = engine() else { return };
    // Adjacent reduced axes are one contiguous run (merged), a non-adjacent pair is not, and the
    // requested order and duplicates must not change the answer.
    let matrix = tensor(&[512, 512]);
    for axes in [&[0usize, 1][..], &[1, 0][..]] {
        for op in OPS {
            assert_matches_cpu(&mut engine, op, &matrix, axes);
        }
    }
    // The CPU surface rejects a repeated axis, so the CUDA engine's dedup is its own contract:
    // saying an axis twice must equal saying it once.
    let hosted = upload(&engine, &matrix);
    let repeated = gpu_reduce(
        &mut engine,
        Op::Sum,
        TensorRead::from_tensor(&hosted),
        &[1, 1, 0],
    )
    .expect("repeated axis");
    let once = gpu_reduce(
        &mut engine,
        Op::Sum,
        TensorRead::from_tensor(&hosted),
        &[0, 1],
    )
    .expect("single axis");
    let repeated = download_tensor(engine.runtime(), &repeated).expect("download");
    let once = download_tensor(engine.runtime(), &once).expect("download");
    assert_eq!(
        repeated.as_slice::<f64>().expect("f64 result"),
        once.as_slice::<f64>().expect("f64 result"),
        "a repeated axis must fold the same way as one mention"
    );

    let cube = tensor(&[7, 9, 11]);
    for axes in [
        &[0usize, 1][..],
        &[1, 2][..],
        &[0, 2][..],
        &[2, 0][..],
        &[0, 1, 2][..],
        &[2, 1, 0][..],
    ] {
        for op in OPS {
            assert_matches_cpu(&mut engine, op, &cube, axes);
        }
    }
}

/// A split inserts a shape axis, so every pass planned above it moves up one.
///
/// The extents are pairwise distinct, so folding an axis the caller did not name changes the
/// result's length as well as its values, and a pass list that shifted the wrong way cannot pass.
#[test]
#[ignore = "requires CUDA"]
fn a_split_axis_above_a_lower_pass_keeps_every_index_right() {
    let Some(mut engine) = engine() else { return };
    // Axis 2 splits, axis 1 is kept, axis 0 is folded after it.
    let tower = tensor(&[7, 8, 1024]);
    for axes in [&[0usize, 2][..], &[2, 0][..]] {
        for op in OPS {
            assert_matches_cpu(&mut engine, op, &tower, axes);
        }
    }
    // Two axes above the lower pass, both the kept one and the folded one.
    let wide = tensor(&[7, 8, 9, 1024]);
    for axes in [&[1usize, 3][..], &[3, 1][..], &[0, 1, 3][..]] {
        for op in OPS {
            assert_matches_cpu(&mut engine, op, &wide, axes);
        }
    }
    // A folded axis directly under the split one: the split's outer factor goes between them.
    let stacked = tensor(&[9, 1024]);
    for axes in [&[0usize, 1][..]] {
        for op in OPS {
            assert_matches_cpu(&mut engine, op, &stacked, axes);
        }
    }
    // Two splittable axes, so the second split has to shift the pair the first one planned.
    let separated = tensor(&[2, 512, 3, 512]);
    for axes in [&[1usize, 3][..], &[3, 1][..]] {
        for op in OPS {
            assert_matches_cpu(&mut engine, op, &separated, axes);
        }
    }
}

#[test]
#[ignore = "requires CUDA"]
fn a_strided_view_matches_the_cpu() {
    let Some(mut engine) = engine() else { return };
    // A transposed read of a [64, 37] matrix: the geometry is not square, so folding the wrong
    // axis cannot pass by accident. `device_read` materializes the region, and the contiguous
    // axis of the materialized tensor is the view's second one.
    let source = tensor(&[64, 37]);
    let data = source.as_slice::<f64>().expect("f64 payload");
    // `T[b + a * 37] == A[a + b * 64]`.
    let transposed_values: Vec<f64> = (0..data.len())
        .map(|index| data[index / 37 + index % 37 * 64])
        .collect();
    let transposed_host = Tensor::from_vec_col_major(vec![37, 64], transposed_values)
        .expect("transposed host tensor");
    let hosted = upload(&engine, &source);
    for axes in [&[0usize][..], &[1][..], &[0, 1][..]] {
        for op in OPS {
            let cpu = cpu_reduce(op, &transposed_host, axes).expect("cpu reduce");
            let view = hosted
                .as_typed::<f64>()
                .expect("f64 payload")
                .backend_region_view(vec![37, 64], vec![64, 1], 0)
                .expect("transposed device view");
            let device = gpu_reduce(
                &mut engine,
                op,
                TensorRead::from_view(TensorView::F64(view)),
                axes,
            )
            .expect("cuda reduce");
            assert_same_fold(&engine, op, axes, &device, &cpu, &transposed_host);
        }
    }
    // The same matrix read where it lies must agree with the CPU too.
    for op in OPS {
        assert_matches_cpu(&mut engine, op, &source, &[1]);
    }
}

#[test]
#[ignore = "requires CUDA"]
fn the_squared_norm_of_a_matrix_matches_the_cpu() {
    let Some(mut engine) = engine() else { return };
    for shape in [&[512usize, 512][..], &[1024, 1024][..], &[17, 3, 5][..]] {
        let source = tensor(shape);
        let hosted = upload(&engine, &source);
        let cpu = CpuEngine::serial()
            .norm_squared_read(TensorRead::from_tensor(&source))
            .expect("cpu norm");
        let device = engine
            .norm_squared_read(TensorRead::from_tensor(&hosted))
            .expect("cuda norm");
        let host = download_tensor(engine.runtime(), &device).expect("download");
        assert_eq!(host.shape(), &[] as &[usize], "{shape:?}: shape");
        let observed = host.as_slice::<f64>().expect("f64 norm")[0];
        let expected = cpu.as_slice::<f64>().expect("f64 norm")[0];
        let folded = source.as_slice::<f64>().expect("f64 payload").len() as f64;
        let tolerance = folded * f64::EPSILON / 2.0 * expected.max(1.0);
        assert!(
            (observed - expected).abs() <= tolerance,
            "{shape:?}: cuda {observed}, cpu {expected}"
        );
    }
}

#[test]
#[ignore = "requires CUDA"]
fn a_repeated_fold_returns_the_same_bytes() {
    let Some(mut engine) = engine() else { return };
    // Determinism across calls: the same input gives the same bytes, whatever association the
    // engine's presentation picked.
    let source = upload(&engine, &tensor(&[1024, 1024]));
    for axes in [&[0usize][..], &[1][..], &[0, 1][..]] {
        let first = gpu_reduce(&mut engine, Op::Sum, TensorRead::from_tensor(&source), axes)
            .expect("first fold");
        let first = download_tensor(engine.runtime(), &first).expect("download");
        let second = gpu_reduce(&mut engine, Op::Sum, TensorRead::from_tensor(&source), axes)
            .expect("second fold");
        let second = download_tensor(engine.runtime(), &second).expect("download");
        assert_eq!(
            first.as_slice::<f64>().expect("f64 result"),
            second.as_slice::<f64>().expect("f64 result"),
            "{axes:?}: two identical folds disagreed"
        );
    }
}

fn upload(engine: &CudaEngine, host: &Tensor) -> Tensor {
    upload_tensor(engine.runtime(), host).expect("upload")
}
