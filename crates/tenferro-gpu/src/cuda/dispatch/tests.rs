//! The per-family dispatch paths, planned by the kernel crate and matched against host references
//! through the elementwise, comparison, conversion, gather and scatter entries (issue #2208 U2).
//!
//! These are in-crate tests because the dispatcher is a crate-internal surface: its methods take
//! device buffers and lengths the caller must have checked, so it is not part of the public API.
#![cfg(feature = "cuda")]

//! U2: the per-family dispatch paths are planned by the kernel crate and match host
//! references through the elementwise, comparison, conversion, gather and scatter
//! entries (issue #2208 U2).

use super::{Dispatcher, IndexPointer};
use crate::cuda::{gpu_available, CudaDeviceId, CudaEngine, CudaRuntime};
use t4a_cuda_kernels::{
    BinaryOp, CompareOp, DType, Entry, GatherPlan, Layout, ScalarSide, ScatterPlan, UnaryOp,
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

fn values(len: usize) -> Vec<f64> {
    (0..len).map(|index| index as f64 * 0.5 - 3.0).collect()
}

#[test]
fn unary_matches_the_host_reference() {
    let Some(runtime) = runtime() else { return };
    let len = 1024i64;
    let data = values(len as usize);
    let input = runtime.upload_bytes(&f64_bytes(&data)).expect("upload");
    let out = runtime
        .upload_bytes(&vec![0u8; len as usize * 8])
        .expect("out");
    let mut dispatch = Dispatcher::for_entries(&runtime, &[Entry::Unary]).expect("prepare");
    dispatch
        .unary(UnaryOp::Neg, DType::F64, DType::F64, len, &out, &input)
        .expect("launch");
    let observed = from_f64_bytes(&runtime.download_bytes(&out).expect("download"));
    let expected: Vec<f64> = data.iter().map(|value| -value).collect();
    assert_eq!(observed, expected);

    dispatch
        .unary(UnaryOp::Sqrt, DType::F64, DType::F64, len, &out, &input)
        .expect("launch");
    let observed = from_f64_bytes(&runtime.download_bytes(&out).expect("download"));
    for (index, value) in data.iter().enumerate() {
        // A negative input follows IEEE propagation: sqrt(-x) is NaN, never an error.
        let expected = value.max(0.0).sqrt();
        assert_eq!(observed[index].is_nan(), value < &0.0, "element {index}");
        if !observed[index].is_nan() {
            assert_eq!(observed[index], expected, "element {index}");
        }
    }
}

#[test]
fn binary_matches_the_host_reference() {
    let Some(runtime) = runtime() else { return };
    let len = 512i64;
    let lhs_data = values(len as usize);
    let rhs_data: Vec<f64> = (0..len).map(|index| (index as f64 * 0.25) - 1.0).collect();
    let lhs = runtime.upload_bytes(&f64_bytes(&lhs_data)).expect("lhs");
    let rhs = runtime.upload_bytes(&f64_bytes(&rhs_data)).expect("rhs");
    let out = runtime
        .upload_bytes(&vec![0u8; len as usize * 8])
        .expect("out");
    let status = runtime.upload_bytes(&[0u8; 4]).expect("status");
    let mut dispatch = Dispatcher::for_entries(&runtime, &[Entry::Binary]).expect("prepare");
    dispatch
        .binary(
            BinaryOp::Max,
            DType::F64,
            ScalarSide::Both,
            len,
            &out,
            &lhs,
            &rhs,
            &status,
        )
        .expect("launch");
    let observed = from_f64_bytes(&runtime.download_bytes(&out).expect("download"));
    let expected: Vec<f64> = lhs_data
        .iter()
        .zip(rhs_data.iter())
        .map(|(left, right)| left.max(*right))
        .collect();
    assert_eq!(observed, expected);
    assert_eq!(
        runtime.download_bytes(&status).expect("status"),
        vec![0u8; 4]
    );
}

#[test]
fn compare_and_select_match_the_host_reference() {
    let Some(runtime) = runtime() else { return };
    let len = 256i64;
    let lhs_data = values(len as usize);
    let rhs_data: Vec<f64> = (0..len).map(|index| (index as f64 * 0.5) - 2.0).collect();
    let lhs = runtime.upload_bytes(&f64_bytes(&lhs_data)).expect("lhs");
    let rhs = runtime.upload_bytes(&f64_bytes(&rhs_data)).expect("rhs");
    let predicate = runtime
        .upload_bytes(&vec![0u8; len as usize])
        .expect("predicate");
    let mut dispatch =
        Dispatcher::for_entries(&runtime, &[Entry::Compare, Entry::Select]).expect("prepare");
    dispatch
        .compare(CompareOp::Lt, DType::F64, len, &predicate, &lhs, &rhs)
        .expect("compare");
    let observed = runtime.download_bytes(&predicate).expect("download");
    let expected: Vec<u8> = lhs_data
        .iter()
        .zip(rhs_data.iter())
        .map(|(left, right)| u8::from(left < right))
        .collect();
    assert_eq!(
        observed, expected,
        "compare writes one Bool byte per element"
    );

    let on_true = runtime
        .upload_bytes(&f64_bytes(&vec![1.5; len as usize]))
        .expect("true");
    let on_false = runtime
        .upload_bytes(&f64_bytes(&vec![-1.5; len as usize]))
        .expect("false");
    let out = runtime
        .upload_bytes(&vec![0u8; len as usize * 8])
        .expect("out");
    dispatch
        .select(DType::F64, len, &out, &predicate, &on_true, &on_false)
        .expect("select");
    let observed = from_f64_bytes(&runtime.download_bytes(&out).expect("download"));
    for (index, take_true) in expected.iter().enumerate() {
        let expected = if *take_true == 1 { 1.5 } else { -1.5 };
        assert_eq!(observed[index], expected, "element {index}");
    }
}

#[test]
fn clamp_and_convert_match_the_host_reference() {
    let Some(runtime) = runtime() else { return };
    let len = 300i64;
    let data = values(len as usize);
    let input = runtime.upload_bytes(&f64_bytes(&data)).expect("input");
    // The bounds are per element, like the input: the entry indexes them with the same
    // element index, so a scalar bound is a full-length constant array here.
    let lower = runtime
        .upload_bytes(&f64_bytes(&vec![-1.0; len as usize]))
        .expect("lower");
    let upper = runtime
        .upload_bytes(&f64_bytes(&vec![1.0; len as usize]))
        .expect("upper");
    let out = runtime
        .upload_bytes(&vec![0u8; len as usize * 8])
        .expect("out");
    let mut dispatch =
        Dispatcher::for_entries(&runtime, &[Entry::Clamp, Entry::Convert]).expect("prepare");
    dispatch
        .clamp(DType::F64, len, &out, &input, &lower, &upper)
        .expect("clamp");
    let observed = from_f64_bytes(&runtime.download_bytes(&out).expect("download"));
    let expected: Vec<f64> = data.iter().map(|value| value.clamp(-1.0, 1.0)).collect();
    assert_eq!(observed, expected);

    // f64 -> f32 is one of the documented conversion routes.
    let converted = runtime
        .upload_bytes(&vec![0u8; len as usize * 4])
        .expect("f32 out");
    dispatch
        .convert(DType::F32, DType::F64, len, &converted, &input)
        .expect("convert");
    let bytes = runtime.download_bytes(&converted).expect("download");
    let observed: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect();
    let expected: Vec<f32> = data.iter().map(|value| *value as f32).collect();
    assert_eq!(observed, expected);
}

#[test]
fn gather_matches_the_host_reference() {
    let Some(runtime) = runtime() else { return };
    // Operand [3, 3]; gather the window [2, 3] at start 1 along axis 0.
    let operand_values: Vec<f64> = (0..9).map(|index| index as f64 + 0.5).collect();
    let operand_layout = Layout::compact(&[3, 3]).expect("operand layout");
    let out_layout = Layout::compact(&[2]).expect("out layout");
    let index_layout = Layout::compact(&[]).expect("rank-0 index");
    let operand = runtime
        .upload_bytes(&f64_bytes(&operand_values))
        .expect("operand");
    let indices = runtime
        .upload_bytes(&1i32.to_ne_bytes()[..])
        .expect("indices");
    let out = runtime.upload_bytes(&[0u8; 2 * 8]).expect("out");
    let plan = GatherPlan {
        window_dims: vec![0],
        offset_dims: vec![0],
        start_index_map: vec![0],
        slice_sizes: vec![2, 1],
        index_vector_dim: 0,
    };
    let mut dispatch = Dispatcher::for_entries(&runtime, &[Entry::Gather]).expect("prepare");
    dispatch
        .gather(
            DType::F64,
            DType::I32,
            &out_layout,
            2,
            &operand_layout,
            9,
            &index_layout,
            1,
            &plan,
            &out,
            &operand,
            IndexPointer::Direct(&indices),
        )
        .expect("gather");
    let observed = from_f64_bytes(&runtime.download_bytes(&out).expect("download"));
    // The window starts at row 1 and covers rows 1 and 2 of column 0.
    let expected: Vec<f64> = vec![operand_values[1], operand_values[2]];
    assert_eq!(observed, expected);
}

#[test]
fn scatter_add_matches_the_host_reference() {
    let Some(runtime) = runtime() else { return };
    // A [4] destination and two unit updates scattered at starts 1 and 2.
    let out_layout = Layout::compact(&[4]).expect("out layout");
    let index_layout = Layout::compact(&[]).expect("rank-0 index");
    let updates_layout = Layout::compact(&[]).expect("rank-0 updates");
    let out = runtime.upload_bytes(&f64_bytes(&[1.0; 4])).expect("out");
    let first_indices = runtime
        .upload_bytes(&1i32.to_ne_bytes()[..])
        .expect("indices");
    let second_indices = runtime
        .upload_bytes(&2i32.to_ne_bytes()[..])
        .expect("indices");
    let first_update = runtime.upload_bytes(&f64_bytes(&[10.0])).expect("update");
    let second_update = runtime.upload_bytes(&f64_bytes(&[20.0])).expect("update");
    let plan = ScatterPlan {
        window_dims: vec![],
        update_window_dims: vec![],
        scatter_dims: vec![0],
        index_vector_dim: 0,
    };
    let mut dispatch = Dispatcher::for_entries(&runtime, &[Entry::ScatterAdd]).expect("prepare");
    dispatch
        .scatter_add(
            DType::F64,
            DType::I32,
            &out_layout,
            4,
            &index_layout,
            1,
            &updates_layout,
            1,
            &plan,
            &out,
            IndexPointer::Direct(&first_indices),
            &first_update,
        )
        .expect("first scatter");
    dispatch
        .scatter_add(
            DType::F64,
            DType::I32,
            &out_layout,
            4,
            &index_layout,
            1,
            &updates_layout,
            1,
            &plan,
            &out,
            IndexPointer::Direct(&second_indices),
            &second_update,
        )
        .expect("second scatter");
    let observed = from_f64_bytes(&runtime.download_bytes(&out).expect("download"));
    assert_eq!(observed, vec![1.0, 11.0, 21.0, 1.0]);
}
