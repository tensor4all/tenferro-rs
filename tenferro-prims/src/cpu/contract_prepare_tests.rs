use tenferro_device::LogicalMemorySpace;
use tenferro_tensor::{MemoryOrder, Tensor};

use super::*;
use crate::cpu::{tensor_to_view, tensor_to_view_mut};

fn tensor_from_fn<T: Scalar>(dims: &[usize], f: impl Fn(&[usize]) -> T) -> Tensor<T> {
    let n_elements: usize = dims.iter().product();
    let mut data = vec![T::zero(); n_elements];
    let mut idx = vec![0usize; dims.len()];
    let strides = col_major_strides(dims);
    for _ in 0..n_elements {
        let linear: isize = idx
            .iter()
            .zip(strides.iter())
            .map(|(&i, &s)| i as isize * s)
            .sum();
        data[linear as usize] = f(&idx);
        for axis in 0..dims.len() {
            idx[axis] += 1;
            if idx[axis] < dims[axis] {
                break;
            }
            idx[axis] = 0;
        }
    }
    Tensor::from_slice(&data, dims, MemoryOrder::ColumnMajor).unwrap()
}

fn tensor_with_layout<T: Scalar>(
    dims: &[usize],
    strides: &[isize],
    f: impl Fn(&[usize]) -> T,
) -> Tensor<T> {
    let len = if dims.is_empty() {
        1
    } else {
        let max_offset: isize = dims
            .iter()
            .zip(strides.iter())
            .map(|(&dim, &stride)| (dim.saturating_sub(1)) as isize * stride)
            .sum();
        (max_offset as usize) + 1
    };
    let mut data = vec![T::zero(); len];
    let mut idx = vec![0usize; dims.len()];
    let n_elements: usize = dims.iter().product();
    for _ in 0..n_elements {
        let offset: isize = idx
            .iter()
            .zip(strides.iter())
            .map(|(&i, &stride)| i as isize * stride)
            .sum();
        data[offset as usize] = f(&idx);
        for axis in 0..dims.len() {
            idx[axis] += 1;
            if idx[axis] < dims[axis] {
                break;
            }
            idx[axis] = 0;
        }
    }
    Tensor::from_vec(data, dims, strides, 0).unwrap()
}

fn expected_contract_012_23_to_013(a: &Tensor<f64>, b: &Tensor<f64>) -> Tensor<f64> {
    tensor_from_fn(&[a.dims()[0], a.dims()[1], b.dims()[1]], |idx| {
        let mut sum = 0.0;
        for k in 0..a.dims()[2] {
            sum += a.get(&[idx[0], idx[1], k]).copied().unwrap()
                * b.get(&[k, idx[2]]).copied().unwrap();
        }
        sum
    })
}

#[test]
fn temp_view_helpers_permute_and_validate_modes() {
    let mut temp = TempTensor {
        data: vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
        dims: vec![2, 3],
        strides: vec![1, 2],
    };

    let permuted = temp_view_in_modes(&temp, &[0, 1], &[1, 0], "Contract").unwrap();
    assert_eq!(permuted.dims(), &[3, 2]);

    let identity = temp_view_mut_in_modes(&mut temp, &[0, 1], &[0, 1], "Contract").unwrap();
    assert_eq!(identity.dims(), &[2, 3]);

    let err = temp_view_in_modes(&temp, &[0, 1], &[1, 2], "Contract").unwrap_err();
    assert!(
        matches!(err, Error::InvalidArgument(message) if message.contains("unable to reorder"))
    );
}

#[test]
fn inspect_contract_preparation_reports_reorder_failures() {
    let mut ctx = CpuContext::new(1);
    let a = tensor_from_fn(&[2, 3], |idx| 1.0 + (idx[0] * 10 + idx[1]) as f64);
    let b = tensor_from_fn(&[3, 4], |idx| -1.0 + (idx[0] * 10 + idx[1]) as f64);
    let mut out = Tensor::zeros(
        &[2, 4],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let spec = build_contract_gemm_spec(&[0, 1], &[1, 2], &[0, 2]).unwrap();

    let a_view = tensor_to_view(&a).unwrap();
    let b_view = tensor_to_view(&b).unwrap();
    let mut out_view = tensor_to_view_mut(&mut out).unwrap();
    let err = inspect_contract_preparation(
        &mut ctx,
        &[&a_view, &b_view],
        &mut out_view,
        &[0, 9],
        &[1, 2],
        &[0, 2],
        &spec,
    )
    .unwrap_err();

    assert!(matches!(err, Error::InvalidArgument(message) if message.contains("reorder A")));
}

#[test]
fn try_execute_contract_gemm_returns_none_when_no_gemm_spec_exists() {
    let mut ctx = CpuContext::new(1);
    let a = tensor_from_fn(&[2, 2], |idx| 1.0 + (idx[0] * 10 + idx[1]) as f64);
    let b = tensor_from_fn(&[2, 3], |idx| -1.0 + (idx[0] * 10 + idx[1]) as f64);
    let mut out = Tensor::zeros(
        &[3],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let a_view = tensor_to_view(&a).unwrap();
    let b_view = tensor_to_view(&b).unwrap();
    let mut out_view = tensor_to_view_mut(&mut out).unwrap();
    let result = try_execute_contract_gemm(
        &mut ctx,
        1.0,
        &[&a_view, &b_view],
        0.0,
        &mut out_view,
        &[0, 0],
        &[0, 1],
        &[1],
        None,
    )
    .unwrap();

    assert_eq!(result, None);
}

#[test]
fn try_execute_contract_gemm_accumulates_beta_into_temporary_output() {
    let mut ctx = CpuContext::new(1);
    let a = tensor_from_fn(&[3, 2, 4], |idx| {
        1.0 + idx[0] as f64 + 0.1 * idx[1] as f64 + 0.01 * idx[2] as f64
    });
    let b = tensor_from_fn(&[4, 5], |idx| {
        -0.5 + 0.2 * idx[0] as f64 + 0.05 * idx[1] as f64
    });
    let mut out = tensor_with_layout(&[3, 2, 5], &[2, 1, 6], |_| 1.0);

    let a_view = tensor_to_view(&a).unwrap();
    let b_view = tensor_to_view(&b).unwrap();
    let mut out_view = tensor_to_view_mut(&mut out).unwrap();
    let result = try_execute_contract_gemm(
        &mut ctx,
        1.0,
        &[&a_view, &b_view],
        2.0,
        &mut out_view,
        &[0, 1, 2],
        &[2, 3],
        &[0, 1, 3],
        None,
    )
    .unwrap();

    assert_eq!(result, Some(()));

    let expected = expected_contract_012_23_to_013(&a, &b);
    for i in 0..expected.dims()[0] {
        for j in 0..expected.dims()[1] {
            for k in 0..expected.dims()[2] {
                let actual = out.get(&[i, j, k]).copied().unwrap();
                let target = expected.get(&[i, j, k]).copied().unwrap() + 2.0;
                assert!(
                    (actual - target).abs() < 1.0e-10,
                    "mismatch at [{i}, {j}, {k}]: actual={actual} target={target}"
                );
            }
        }
    }
}

#[test]
fn try_execute_contract_gemm_materializes_only_rhs_when_needed() {
    let mut ctx = CpuContext::new(1);
    let a = tensor_from_fn(&[3, 2, 4], |idx| {
        1.0 + idx[0] as f64 + 0.1 * idx[1] as f64 + 0.01 * idx[2] as f64
    });
    let b = tensor_with_layout(&[4, 2, 3], &[1, 4, 12], |idx| {
        -0.5 + 0.2 * idx[0] as f64 + 0.05 * idx[1] as f64 + 0.005 * idx[2] as f64
    });
    let mut out = Tensor::zeros(
        &[3, 2, 2, 3],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let spec = build_contract_gemm_spec(&[0, 1, 2], &[2, 3, 4], &[0, 1, 3, 4]).unwrap();

    let a_view = tensor_to_view(&a).unwrap();
    let b_view = tensor_to_view(&b).unwrap();
    let mut out_view = tensor_to_view_mut(&mut out).unwrap();
    let prep = inspect_contract_preparation(
        &mut ctx,
        &[&a_view, &b_view],
        &mut out_view,
        &[0, 1, 2],
        &[2, 3, 4],
        &[0, 1, 3, 4],
        &spec,
    )
    .unwrap()
    .unwrap();

    assert_eq!(prep.a_strategy, ContractOperandStrategy::Borrowed);
    assert_eq!(prep.b_strategy, ContractOperandStrategy::Materialize);
    assert_eq!(prep.output_strategy, ContractOutputStrategy::Direct);

    let result = try_execute_contract_gemm(
        &mut ctx,
        1.0,
        &[&a_view, &b_view],
        0.0,
        &mut out_view,
        &[0, 1, 2],
        &[2, 3, 4],
        &[0, 1, 3, 4],
        Some(&spec),
    )
    .unwrap();

    assert_eq!(result, Some(()));

    for i in 0..3 {
        for j in 0..2 {
            for l in 0..2 {
                for m in 0..3 {
                    let mut target = 0.0;
                    for k in 0..4 {
                        target += a.get(&[i, j, k]).copied().unwrap()
                            * b.get(&[k, l, m]).copied().unwrap();
                    }
                    let actual = out.get(&[i, j, l, m]).copied().unwrap();
                    assert!(
                        (actual - target).abs() < 1.0e-10,
                        "mismatch at [{i}, {j}, {l}, {m}]: actual={actual} target={target}"
                    );
                }
            }
        }
    }
}
