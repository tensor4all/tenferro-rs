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
use t4a_cuda_kernels::{
    plan_reduction, DType, Entry, Layout, ReduceOp, ReduceStrategy, ReductionPlan,
};

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
        .as_chunks::<8>()
        .0
        .iter()
        .map(|word| f64::from_le_bytes(*word))
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

/// Uninitialised device memory of `bytes`, as a reduction destination.
fn uninit(runtime: &CudaRuntime, bytes: usize) -> t4a_cuda_runtime::UninitBuffer {
    runtime.submit().alloc_uninit(bytes, 256).expect("uninit")
}

/// The crate's plan for a sum over axis 1 of a compact `[rows, axis_len]` f64 input.
fn row_sum_plan(rows: i64, axis_len: i64) -> ReductionPlan {
    let (in_layout, out_layout) = shapes(rows, axis_len);
    plan_reduction(
        DType::F64,
        ReduceOp::Sum,
        &out_layout,
        rows,
        &in_layout,
        rows * axis_len,
        1,
        32,
    )
    .expect("plan")
}

/// Bytes of the partials buffer a split `plan` over a `[rows, axis_len]` input needs.
fn partials_bytes(plan: &ReductionPlan, rows: i64, axis_len: i64) -> usize {
    let ReductionPlan::Split { split, .. } = plan else {
        panic!("expected a split plan")
    };
    let (in_layout, _) = shapes(rows, axis_len);
    let partials = t4a_cuda_kernels::reduction_partials_layout(&in_layout, 1, split.splits)
        .expect("partials layout");
    partials.element_count().expect("partials count") as usize * 8
}

#[test]
fn the_split_policy_runs_both_passes() {
    let Some(runtime) = runtime() else { return };
    // A long axis forces the crate's split policy: pass 1 writes partials, pass 2
    // combines them. Both destinations start uninitialised: every element is written.
    let (rows, axis_len) = (3i64, 4096i64);
    let data: Vec<f64> = (0..rows * axis_len)
        .map(|index| (index % 97) as f64 * 0.25)
        .collect();
    let (in_layout, out_layout) = shapes(rows, axis_len);
    let input = runtime.upload_bytes(&f64_bytes(&data)).expect("input");
    let plan = row_sum_plan(rows, axis_len);
    assert!(
        matches!(plan, ReductionPlan::Split { .. }),
        "a 4096-long axis must split under the crate's policy"
    );
    let out = uninit(&runtime, rows as usize * 8);
    let partials = uninit(&runtime, partials_bytes(&plan, rows, axis_len));
    let mut dispatch = Dispatcher::for_entries(&runtime, &[Entry::Reduce, Entry::ReducePartials])
        .expect("prepare");
    dispatch
        .reduce_planned(
            &plan,
            DType::F64,
            ReduceOp::Sum,
            &out_layout,
            &in_layout,
            1,
            &out,
            &input,
            Some(partials),
        )
        .expect("reduce_planned");
    // SAFETY: the combine pass writes each of the `rows` outputs, in stream order before the
    // download below.
    let out = unsafe { out.assume_init() };
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
    let plan = row_sum_plan(rows, axis_len);
    assert!(matches!(plan, ReductionPlan::Direct { .. }));
    let out = uninit(&runtime, rows as usize * 8);
    let mut dispatch = Dispatcher::for_entries(&runtime, &[Entry::Reduce]).expect("prepare");
    dispatch
        .reduce_planned(
            &plan,
            DType::F64,
            ReduceOp::Sum,
            &out_layout,
            &in_layout,
            1,
            &out,
            &input,
            None,
        )
        .expect("reduce_planned");
    // SAFETY: the direct launch writes each of the `rows` outputs before the download.
    let out = unsafe { out.assume_init() };
    assert_eq!(
        from_f64_bytes(&runtime.download_bytes(&out).expect("download")),
        host_row_sums(&data, rows, axis_len)
    );
}

#[test]
fn a_destination_that_the_plan_would_not_cover_is_refused() {
    let Some(runtime) = runtime() else { return };
    let input = runtime
        .upload_bytes(&f64_bytes(&vec![1.0; 3 * 4096]))
        .expect("input");
    let mut dispatch = Dispatcher::for_entries(&runtime, &[Entry::Reduce, Entry::ReducePartials])
        .expect("prepare");
    let refused = |dispatch: &mut Dispatcher,
                   plan: &ReductionPlan,
                   (rows, axis_len): (i64, i64),
                   out: &t4a_cuda_runtime::UninitBuffer,
                   partials: Option<t4a_cuda_runtime::UninitBuffer>| {
        let (in_layout, out_layout) = shapes(rows, axis_len);
        dispatch
            .reduce_planned(
                plan,
                DType::F64,
                ReduceOp::Sum,
                &out_layout,
                &in_layout,
                1,
                out,
                &input,
                partials,
            )
            .is_err()
    };
    // An output longer than the layout would keep bytes no launch writes.
    let direct = row_sum_plan(5, 8);
    assert!(refused(
        &mut dispatch,
        &direct,
        (5, 8),
        &uninit(&runtime, 6 * 8),
        None
    ));
    // A direct plan has no partials to write.
    assert!(refused(
        &mut dispatch,
        &direct,
        (5, 8),
        &uninit(&runtime, 5 * 8),
        Some(uninit(&runtime, 8)),
    ));
    // A split plan needs partials of exactly its layout.
    let split = row_sum_plan(3, 4096);
    let exact = partials_bytes(&split, 3, 4096);
    assert!(refused(
        &mut dispatch,
        &split,
        (3, 4096),
        &uninit(&runtime, 3 * 8),
        None
    ));
    assert!(refused(
        &mut dispatch,
        &split,
        (3, 4096),
        &uninit(&runtime, 3 * 8),
        Some(uninit(&runtime, exact + 8)),
    ));
}
