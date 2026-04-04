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

#[test]
fn col_major_strides_produces_correct_values() {
    assert_eq!(col_major_strides(&[3, 4, 5]), &[1_isize, 3, 12]);
    assert_eq!(col_major_strides(&[1]), &[1_isize]);
    {
        let empty: &[usize] = &[];
        let empty_strides = col_major_strides(empty);
        assert!(empty_strides.is_empty());
    }
}

#[test]
fn element_count_handles_empty_and_nonempty_dims() {
    assert_eq!(element_count(&[]), 1);
    assert_eq!(element_count(&[3]), 3);
    assert_eq!(element_count(&[2, 3, 4]), 24);
}

#[test]
fn perm_for_returns_correct_permutation() {
    assert_eq!(perm_for(&[1, 0], &[0, 1]), Some(vec![1, 0]));
    assert_eq!(perm_for(&[0, 1], &[0, 1]), Some(vec![0, 1]));
    assert_eq!(perm_for(&[0], &[1]), None);
}

#[test]
fn operand_strategy_borrows_when_all_groups_fusible() {
    let dims: Vec<usize> = vec![2, 3, 4, 5, 6, 7];
    let strides = col_major_strides(&dims);
    assert_eq!(
        operand_strategy(&dims, &strides, 2, 2, 2),
        ContractOperandStrategy::Borrowed
    );
}

#[test]
fn operand_strategy_materializes_when_strides_non_contiguous() {
    let dims: Vec<usize> = vec![2, 3, 4, 5, 6, 7];
    let mut strides = col_major_strides(&dims);
    strides[5] = 9999;
    assert_eq!(
        operand_strategy(&dims, &strides, 2, 2, 2),
        ContractOperandStrategy::Materialize
    );
}

#[test]
fn output_strategy_direct_when_fusible() {
    let dims: Vec<usize> = vec![2, 3, 4, 5, 6];
    let strides = col_major_strides(&dims);
    assert_eq!(
        output_strategy(&dims, &strides, 1, 2, 2),
        ContractOutputStrategy::Direct
    );
}

#[test]
fn output_strategy_temporary_when_noncontiguous() {
    let dims: Vec<usize> = vec![2, 3, 4, 5, 6];
    let mut strides = col_major_strides(&dims);
    strides[4] = 9999;
    assert_eq!(
        output_strategy(&dims, &strides, 1, 2, 2),
        ContractOutputStrategy::Temporary
    );
}

#[test]
fn group_is_fusible_with_contiguous_strides() {
    let dims: &[usize] = &[2, 3];
    let strides = col_major_strides(dims);
    assert!(group_is_fusible(dims, &strides));
}

#[test]
fn group_is_fusible_rejects_noncontiguous() {
    assert!(!group_is_fusible(&[2_usize, 3], &[1_isize, 100]));
}

#[test]
fn try_execute_contract_gemm_materializes_lhs_and_uses_temporary_output() {
    let mut ctx = CpuContext::new(1);
    let a = tensor_with_layout(&[3, 2, 4], &[1, 100, 200], |idx| {
        1.0 + idx[0] as f64 + 0.1 * idx[1] as f64 + 0.01 * idx[2] as f64
    });
    let b = tensor_from_fn(&[4, 5], |idx| {
        -0.5 + 0.2 * idx[0] as f64 + 0.05 * idx[1] as f64
    });
    let mut out = tensor_with_layout(&[3, 2, 5], &[2, 1, 100], |_| 0.0);

    let a_view = tensor_to_view(&a).unwrap();
    let b_view = tensor_to_view(&b).unwrap();
    let mut out_view = tensor_to_view_mut(&mut out).unwrap();

    let prep = inspect_contract_preparation(
        &mut ctx,
        &[&a_view, &b_view],
        &mut out_view,
        &[0, 1, 2],
        &[2, 3],
        &[0, 1, 3],
        &build_contract_gemm_spec(&[0, 1, 2], &[2, 3], &[0, 1, 3]).unwrap(),
    )
    .unwrap()
    .unwrap();

    assert_eq!(prep.a_strategy, ContractOperandStrategy::Materialize);
    assert_eq!(prep.b_strategy, ContractOperandStrategy::Borrowed);
    assert_eq!(prep.output_strategy, ContractOutputStrategy::Temporary);

    let result = try_execute_contract_gemm(
        &mut ctx,
        1.0,
        &[&a_view, &b_view],
        0.0,
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
                let target = expected.get(&[i, j, k]).copied().unwrap();
                assert!(
                    (actual - target).abs() < 1.0e-10,
                    "mismatch at [{i}, {j}, {k}]: actual={actual} target={target}"
                );
            }
        }
    }
}

#[test]
fn try_execute_contract_gemm_with_beta_accumulates_into_existing_output() {
    let mut ctx = CpuContext::new(1);
    let a = tensor_from_fn(&[2, 3], |idx| (idx[0] * 3 + idx[1] + 1) as f64);
    let b = tensor_from_fn(&[3, 2], |idx| (idx[0] * 2 + idx[1] + 1) as f64 * 0.5);
    let init_val = 7.0;
    let mut out =
        Tensor::from_slice(&vec![init_val; 4], &[2, 2], MemoryOrder::ColumnMajor).unwrap();

    let a_view = tensor_to_view(&a).unwrap();
    let b_view = tensor_to_view(&b).unwrap();
    let mut out_view = tensor_to_view_mut(&mut out).unwrap();

    let result = try_execute_contract_gemm(
        &mut ctx,
        1.0,
        &[&a_view, &b_view],
        1.0,
        &mut out_view,
        &[0, 1],
        &[1, 2],
        &[0, 2],
        None,
    )
    .unwrap();

    assert_eq!(result, Some(()));

    for i in 0..2 {
        for j in 0..2 {
            let mut expected = init_val;
            for k in 0..3 {
                expected += a.get(&[i, k]).copied().unwrap() * b.get(&[k, j]).copied().unwrap();
            }
            let actual = out.get(&[i, j]).copied().unwrap();
            assert!(
                (actual - expected).abs() < 1.0e-10,
                "mismatch at [{i}, {j}]: actual={actual} expected={expected}"
            );
        }
    }
}

#[test]
fn temp_view_mut_in_modes_permutes_correctly() {
    let mut temp = TempTensor {
        data: vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
        dims: vec![2, 3],
        strides: vec![1, 2],
    };

    let view = temp_view_mut_in_modes(&mut temp, &[0, 1], &[1, 0], "Contract").unwrap();
    assert_eq!(view.dims(), &[3, 2]);
}

#[test]
fn inspect_contract_preparation_rejects_invalid_b_modes() {
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
        &[0, 1],
        &[9, 2],
        &[0, 2],
        &spec,
    )
    .unwrap_err();

    assert!(matches!(err, Error::InvalidArgument(message) if message.contains("reorder B")));
}

#[test]
fn inspect_contract_preparation_rejects_invalid_output_modes() {
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
        &[0, 1],
        &[1, 2],
        &[9, 2],
        &spec,
    )
    .unwrap_err();

    assert!(matches!(err, Error::InvalidArgument(message) if message.contains("reorder output")));
}
