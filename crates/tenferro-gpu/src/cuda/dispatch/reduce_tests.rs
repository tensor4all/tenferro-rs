//! The reduction family — unit and plane strategies, and the split policy with its two passes —
//! planned by the kernel crate and matched against host references (issue #2208 U4).
//!
//! In-crate because the dispatcher is a crate-internal surface.
#![cfg(feature = "cuda")]

//! U4: the reduction family — unit and plane strategies, and the split policy with its
//! two passes — planned by the kernel crate and matched against host references
//! (issue #2208 U4).

use super::Dispatcher;
use crate::cuda::{gpu_available, CudaDeviceId, CudaEngine, CudaRuntime};
use t4a_cuda_kernels::{DType, Entry, Layout, ReduceOp, ReduceStrategy, ReductionPlan};

fn runtime() -> Option<CudaRuntime> {
    if !gpu_available() {
        return None;
    }
    let engine = CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("CUDA engine");
    Some(engine.runtime().clone())
}

fn f64_bytes(values: &[f64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn from_f64_bytes(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(8)
        .map(|word| f64::from_le_bytes(word.try_into().expect("8 bytes")))
        .collect()
}

/// `[rows, axis]` compact input; the keepdims output is `[rows, 1]`.
fn shapes(rows: i64, axis_len: i64) -> (Layout, Layout) {
    (
        Layout::compact(&[rows, axis_len]).expect("input layout"),
        Layout::compact(&[rows, 1]).expect("output layout"),
    )
}

/// Sum over axis 1 of a compact `[rows, axis_len]` input.
fn host_row_sums(data: &[f64], rows: i64, axis_len: i64) -> Vec<f64> {
    (0..rows)
        .map(|row| {
            (0..axis_len)
                .map(|column| data[(row + column * rows) as usize])
                .sum()
        })
        .collect()
}

#[test]
fn unit_and_plane_reductions_match_the_host() {
    let Some(runtime) = runtime() else { return };
    let (rows, axis_len) = (37i64, 24i64);
    let data: Vec<f64> = (0..rows * axis_len)
        .map(|index| index as f64 * 0.5 - 3.0)
        .collect();
    let (in_layout, out_layout) = shapes(rows, axis_len);
    let input = runtime.upload_bytes(&f64_bytes(&data)).expect("input");
    let out = runtime
        .upload_bytes(&vec![0u8; rows as usize * 8])
        .expect("out");
    let mut dispatch = Dispatcher::for_entries(&runtime, &[Entry::Reduce]).expect("prepare");
    let expected = host_row_sums(&data, rows, axis_len);

    // The axis is no longer than a plane, so the unit strategy applies.
    dispatch
        .reduce(
            DType::F64,
            ReduceOp::Sum,
            ReduceStrategy::Unit,
            &out_layout,
            rows,
            &in_layout,
            rows * axis_len,
            1,
            32,
            &out,
            &input,
        )
        .expect("unit reduce");
    assert_eq!(
        from_f64_bytes(&runtime.download_bytes(&out).expect("download")),
        expected
    );

    // The plane strategy reads the axis as a warp plane, so it needs an axis at least
    // one plane wide.
    let (plane_rows, plane_axis) = (37i64, 64i64);
    let plane_data: Vec<f64> = (0..plane_rows * plane_axis)
        .map(|index| index as f64 * 0.25 - 1.0)
        .collect();
    let (plane_in, plane_out_layout) = shapes(plane_rows, plane_axis);
    let plane_input = runtime
        .upload_bytes(&f64_bytes(&plane_data))
        .expect("input");
    let out_plane = runtime
        .upload_bytes(&vec![0u8; plane_rows as usize * 8])
        .expect("out");
    dispatch
        .reduce(
            DType::F64,
            ReduceOp::Sum,
            ReduceStrategy::Plane,
            &plane_out_layout,
            plane_rows,
            &plane_in,
            plane_rows * plane_axis,
            1,
            32,
            &out_plane,
            &plane_input,
        )
        .expect("plane reduce");
    assert_eq!(
        from_f64_bytes(&runtime.download_bytes(&out_plane).expect("download")),
        host_row_sums(&plane_data, plane_rows, plane_axis)
    );
}

#[test]
fn the_split_policy_runs_both_passes() {
    let Some(runtime) = runtime() else { return };
    // A long axis forces the crate's split policy: pass 1 writes partials, pass 2
    // combines them.
    let (rows, axis_len) = (3i64, 4096i64);
    let data: Vec<f64> = (0..rows * axis_len)
        .map(|index| (index % 97) as f64 * 0.25)
        .collect();
    let (in_layout, out_layout) = shapes(rows, axis_len);
    let input = runtime.upload_bytes(&f64_bytes(&data)).expect("input");
    let out = runtime
        .upload_bytes(&vec![0u8; rows as usize * 8])
        .expect("out");
    let partials = runtime
        .upload_bytes(&vec![0u8; rows as usize * 4096 * 8])
        .expect("partials");
    let mut dispatch = Dispatcher::for_entries(&runtime, &[Entry::Reduce, Entry::ReducePartials])
        .expect("prepare");
    let plan = dispatch
        .reduce_auto(
            DType::F64,
            ReduceOp::Sum,
            &out_layout,
            rows,
            &in_layout,
            rows * axis_len,
            1,
            32,
            &out,
            &input,
            &partials,
        )
        .expect("reduce_auto");
    assert!(
        matches!(plan, ReductionPlan::Split { .. }),
        "a 4096-long axis must split under the crate's policy"
    );
    let expected = host_row_sums(&data, rows, axis_len);
    let observed = from_f64_bytes(&runtime.download_bytes(&out).expect("download"));
    // The split sums in a different order, so compare with a relative tolerance.
    for (index, (left, right)) in observed.iter().zip(expected.iter()).enumerate() {
        assert!(
            (left - right).abs() <= 1e-9 * right.abs().max(1.0),
            "row {index}: observed {left}, expected {right}"
        );
    }
}

#[test]
fn a_short_axis_reduces_directly() {
    let Some(runtime) = runtime() else { return };
    let (rows, axis_len) = (5i64, 8i64);
    let data: Vec<f64> = (0..rows * axis_len).map(|index| index as f64).collect();
    let (in_layout, out_layout) = shapes(rows, axis_len);
    let input = runtime.upload_bytes(&f64_bytes(&data)).expect("input");
    let out = runtime
        .upload_bytes(&vec![0u8; rows as usize * 8])
        .expect("out");
    let partials = runtime.upload_bytes(&[0u8; 8]).expect("partials");
    let mut dispatch = Dispatcher::for_entries(&runtime, &[Entry::Reduce]).expect("prepare");
    let plan = dispatch
        .reduce_auto(
            DType::F64,
            ReduceOp::Sum,
            &out_layout,
            rows,
            &in_layout,
            rows * axis_len,
            1,
            32,
            &out,
            &input,
            &partials,
        )
        .expect("reduce_auto");
    assert!(matches!(plan, ReductionPlan::Direct { .. }));
    assert_eq!(
        from_f64_bytes(&runtime.download_bytes(&out).expect("download")),
        host_row_sums(&data, rows, axis_len)
    );
}
