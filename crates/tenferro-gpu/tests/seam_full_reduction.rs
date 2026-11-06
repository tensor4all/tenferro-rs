#![cfg(feature = "cuda")]

//! Full and multi-axis reductions on the CUDA engine at every plan boundary (#2406).
//!
//! The reduction writes its squeezed result directly into uninitialised memory and waits once
//! per call, so these tests pin what that must not change: the values at each boundary the
//! planner switches strategy at, the result shape, NaN propagation, that no unwritten byte
//! reaches a result, and that a standalone call waits exactly once.

use tenferro_gpu::cuda::{download_tensor, gpu_available, upload_tensor, CudaDeviceId, CudaEngine};
use tenferro_tensor::{DType, Tensor, TensorRead, TensorReduction};

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
}

const OPS: [Op; 4] = [Op::Sum, Op::Prod, Op::Max, Op::Min];

/// Lengths around each point where the reduction plan changes: the 32-lane plane, the split
/// threshold (`plane_width * 8 * 2`), the 64-partials wide-combine threshold, and the
/// `MAX_SPLITS` cap with a remainder chunk.
const LENGTHS: [usize; 12] = [
    1,
    31,
    32,
    33,
    511,
    512,
    513,
    16_383,
    16_384,
    16_385,
    1 << 20,
    (1 << 20) + 7,
];

fn reduce(engine: &mut CudaEngine, op: Op, source: &Tensor, axes: &[usize]) -> Tensor {
    let read = TensorRead::from_tensor(source);
    match op {
        Op::Sum => engine.reduce_sum_read(read, axes),
        Op::Prod => engine.reduce_prod_read(read, axes),
        Op::Max => engine.reduce_max_read(read, axes),
        Op::Min => engine.reduce_min_read(read, axes),
    }
    .expect("reduce")
}

/// Values whose product stays finite and near one, and whose maximum and minimum are unique
/// and land in the middle of the input rather than at a chunk edge.
fn values(len: usize, op: Op) -> Vec<f64> {
    (0..len)
        .map(|index| {
            let wobble = ((index * 7919) % 1009) as f64 / 1009.0;
            match op {
                Op::Prod => 1.0 + (wobble - 0.5) * 1e-5,
                _ => wobble - 0.5,
            }
        })
        .collect()
}

fn host_reduce(op: Op, data: &[f64]) -> f64 {
    match op {
        Op::Sum => data.iter().sum(),
        Op::Prod => data.iter().product(),
        Op::Max => data.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        Op::Min => data.iter().copied().fold(f64::INFINITY, f64::min),
    }
}

/// Upload `data` as a rank-1 tensor of `dtype` (`F32` or `F64`).
fn upload(engine: &CudaEngine, dtype: DType, data: &[f64]) -> Tensor {
    let host = match dtype {
        DType::F32 => Tensor::from_vec_col_major(
            vec![data.len()],
            data.iter().map(|&value| value as f32).collect::<Vec<_>>(),
        ),
        DType::F64 => Tensor::from_vec_col_major(vec![data.len()], data.to_vec()),
        other => panic!("unsupported test dtype {other:?}"),
    }
    .expect("host tensor");
    upload_tensor(engine.runtime(), &host).expect("upload")
}

/// The scalar of a downloaded rank-0 `F32` or `F64` result, widened to `f64`.
fn scalar(engine: &CudaEngine, result: &Tensor) -> f64 {
    let host = download_tensor(engine.runtime(), result).expect("download");
    match host.dtype() {
        DType::F32 => host.as_slice::<f32>().expect("f32 payload")[0] as f64,
        DType::F64 => host.as_slice::<f64>().expect("f64 payload")[0],
        other => panic!("unexpected result dtype {other:?}"),
    }
}

#[test]
#[ignore = "requires CUDA"]
fn full_reductions_match_the_host_at_every_plan_boundary() {
    let Some(mut engine) = engine() else { return };
    for dtype in [DType::F32, DType::F64] {
        for op in OPS {
            for len in LENGTHS {
                let data = values(len, op);
                // The host reference folds the values the device holds, rounded to `dtype`.
                let stored: Vec<f64> = match dtype {
                    DType::F32 => data.iter().map(|&value| value as f32 as f64).collect(),
                    _ => data.clone(),
                };
                let source = upload(&engine, dtype, &data);
                let result = reduce(&mut engine, op, &source, &[0]);
                assert_eq!(result.shape(), &[] as &[usize], "{op:?} {dtype:?} n={len}");
                assert_eq!(result.dtype(), dtype);
                let observed = scalar(&engine, &result);
                let expected = host_reduce(op, &stored);
                match op {
                    // A maximum or minimum is one of the inputs, whatever the fold order.
                    Op::Max | Op::Min => assert_eq!(observed, expected, "{op:?} {dtype:?} n={len}"),
                    // A sum or product is associated differently from the host's serial fold, so
                    // hold it to the a-priori bound any association obeys: `n * u` relative to
                    // the sum of magnitudes, or to the product.
                    Op::Sum | Op::Prod => {
                        let unit_roundoff = match dtype {
                            DType::F32 => f32::EPSILON as f64 / 2.0,
                            _ => f64::EPSILON / 2.0,
                        };
                        let magnitude = match op {
                            Op::Sum => stored.iter().map(|value| value.abs()).sum::<f64>(),
                            _ => expected.abs(),
                        };
                        let tolerance = len as f64 * unit_roundoff * magnitude;
                        assert!(
                            (observed - expected).abs() <= tolerance,
                            "{op:?} {dtype:?} n={len}: observed {observed}, expected {expected}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires CUDA"]
fn a_nan_anywhere_propagates_through_a_full_max_and_min() {
    let Some(mut engine) = engine() else { return };
    for dtype in [DType::F32, DType::F64] {
        for len in [33, 513, 1 << 20] {
            let mut data = values(len, Op::Max);
            data[len / 2 + 1] = f64::NAN;
            let source = upload(&engine, dtype, &data);
            for op in [Op::Max, Op::Min] {
                let result = reduce(&mut engine, op, &source, &[0]);
                assert!(
                    scalar(&engine, &result).is_nan(),
                    "{op:?} {dtype:?} n={len} drops the NaN"
                );
            }
        }
    }
}

#[test]
#[ignore = "requires CUDA"]
fn unwritten_device_memory_never_reaches_a_result() {
    let Some(mut engine) = engine() else { return };
    // The reduction's destinations come from the stream-ordered pool without a zero fill. Free
    // NaN-filled buffers of the output's and the partials' sizes first, so the pool is likely
    // to hand that memory back: a partial or an output the kernels did not write would then
    // turn the sum of positive values into a NaN. Reuse is the pool's choice, so this test can
    // only fail, never prove, but the full-coverage argument is checked in `reduce_planned`.
    for dtype in [DType::F32, DType::F64] {
        for len in [512, 16_384, 1 << 20] {
            for poison_len in [1usize, 4096, len] {
                let poison = upload(&engine, dtype, &vec![f64::NAN; poison_len]);
                drop(poison);
            }
            let data: Vec<f64> = (0..len).map(|index| (index % 13) as f64 + 1.0).collect();
            let source = upload(&engine, dtype, &data);
            let result = reduce(&mut engine, Op::Sum, &source, &[0]);
            assert_eq!(
                scalar(&engine, &result),
                host_reduce(Op::Sum, &data),
                "{dtype:?} n={len}"
            );
        }
    }
}

#[test]
#[ignore = "requires CUDA"]
fn multi_axis_reductions_squeeze_without_a_copy_and_wait_once() {
    let Some(mut engine) = engine() else { return };
    // A [rows, axis, cols] column-major input whose middle axis is long enough to split.
    let (rows, axis, cols) = (3usize, 1000usize, 5usize);
    let data: Vec<f64> = (0..rows * axis * cols)
        .map(|index| ((index * 31) % 101) as f64 * 0.125)
        .collect();
    let at = |row: usize, middle: usize, col: usize| data[row + rows * (middle + axis * col)];
    let source = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![rows, axis, cols], data.clone()).expect("host tensor"),
    )
    .expect("upload");
    let cases: [(&[usize], &[usize]); 4] = [
        (&[1], &[rows, cols]),
        (&[0, 1], &[cols]),
        (&[1, 2], &[rows]),
        (&[0, 1, 2], &[]),
    ];
    for (axes, shape) in cases {
        let syncs = engine.launch_completion_syncs();
        let result = reduce(&mut engine, Op::Max, &source, axes);
        assert_eq!(
            engine.launch_completion_syncs() - syncs,
            1,
            "a standalone reduction over {axes:?} waits once, whatever its pass count"
        );
        assert_eq!(result.shape(), shape, "axes {axes:?}");
        let back = download_tensor(engine.runtime(), &result).expect("download");
        let observed = back.as_slice::<f64>().expect("f64 payload");
        // Fold the host reference over the same axes, keeping the survivors column-major.
        let mut expected = vec![f64::NEG_INFINITY; observed.len()];
        for col in 0..cols {
            for middle in 0..axis {
                for row in 0..rows {
                    let mut index = 0;
                    let mut scale = 1;
                    for (dim, (coordinate, extent)) in [(row, rows), (middle, axis), (col, cols)]
                        .into_iter()
                        .enumerate()
                    {
                        if !axes.contains(&dim) {
                            index += coordinate * scale;
                            scale *= extent;
                        }
                    }
                    expected[index] = expected[index].max(at(row, middle, col));
                }
            }
        }
        assert_eq!(observed, &expected[..], "axes {axes:?}");
    }
}
