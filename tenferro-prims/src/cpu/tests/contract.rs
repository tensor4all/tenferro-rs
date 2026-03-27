use tenferro_algebra::{Scalar, Standard};
use tenferro_device::LogicalMemorySpace;
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::cpu::contract_prepare::{
    inspect_contract_preparation, ContractOperandStrategy, ContractOutputStrategy,
};
use crate::cpu::{tensor_to_view, tensor_to_view_mut, CpuBackend, CpuContext, CpuPlan};
use crate::{SemiringFastPathDescriptor, TensorSemiringCore, TensorSemiringFastPath};

fn contract_plan<T: Scalar>(
    ctx: &mut CpuContext,
    modes_a: Vec<u32>,
    modes_b: Vec<u32>,
    modes_c: Vec<u32>,
    shapes: &[&[usize]],
) -> CpuPlan<T> {
    <CpuBackend as TensorSemiringFastPath<Standard<T>>>::plan(
        ctx,
        &SemiringFastPathDescriptor::Contract {
            modes_a,
            modes_b,
            modes_c,
        },
        shapes,
    )
    .unwrap()
}

fn execute_contract<T: Scalar>(
    ctx: &mut CpuContext,
    plan: &CpuPlan<T>,
    a: &Tensor<T>,
    b: &Tensor<T>,
    out: &mut Tensor<T>,
) {
    <CpuBackend as TensorSemiringCore<Standard<T>>>::execute(
        ctx,
        plan,
        T::one(),
        &[a, b],
        T::zero(),
        out,
    )
    .unwrap();
}

fn tensor_from_fn<T: Scalar>(dims: &[usize], f: impl Fn(&[usize]) -> T) -> Tensor<T> {
    let n_elements: usize = dims.iter().product();
    let mut data = vec![T::zero(); n_elements];
    let mut idx = vec![0usize; dims.len()];
    let strides = col_major_strides(dims);
    for _ in 0..n_elements {
        let linear: usize = idx.iter().zip(strides.iter()).map(|(&i, &s)| i * s).sum();
        data[linear] = f(&idx);
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
    let len = required_len(dims, strides);
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

fn tensor_zeros<T: Scalar>(dims: &[usize]) -> Tensor<T> {
    Tensor::zeros(
        dims,
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap()
}

fn col_major_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = Vec::with_capacity(dims.len());
    let mut stride = 1usize;
    for &dim in dims {
        strides.push(stride);
        stride *= dim;
    }
    strides
}

fn required_len(dims: &[usize], strides: &[isize]) -> usize {
    if dims.is_empty() {
        return 1;
    }
    let max_offset: isize = dims
        .iter()
        .zip(strides.iter())
        .map(|(&dim, &stride)| (dim.saturating_sub(1)) as isize * stride)
        .sum();
    (max_offset as usize) + 1
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

fn expected_issue_336_final_contract(a: &Tensor<f64>, b: &Tensor<f64>) -> Tensor<f64> {
    tensor_from_fn(&[2, 2, 2, 2, 16, 16], |idx| {
        let mut sum = 0.0;
        for l6 in 0..8 {
            for l0 in 0..2 {
                for l4 in 0..8 {
                    for l5 in 0..2 {
                        sum += a
                            .get(&[l6, idx[5], l0, idx[1], l4, l5, idx[3]])
                            .copied()
                            .unwrap()
                            * b.get(&[l4, idx[4], idx[0], l0, l6, idx[2], l5])
                                .copied()
                                .unwrap();
                    }
                }
            }
        }
        sum
    })
}

#[test]
fn contract_preparation_uses_temp_output_for_non_fusible_output() {
    let mut ctx = CpuContext::new(1);
    let a = tensor_from_fn(&[3, 2, 4], |idx| {
        1.0 + idx[0] as f64 + 0.1 * idx[1] as f64 + 0.01 * idx[2] as f64
    });
    let b = tensor_from_fn(&[4, 5], |idx| {
        -0.5 + 0.2 * idx[0] as f64 + 0.05 * idx[1] as f64
    });
    let mut out = tensor_with_layout(&[3, 2, 5], &[2, 1, 6], |_| 0.0);
    let spec = {
        let CpuPlan::Contract { gemm_spec, .. } = contract_plan::<f64>(
            &mut ctx,
            vec![0, 1, 2],
            vec![2, 3],
            vec![0, 1, 3],
            &[&[3, 2, 4], &[4, 5], &[3, 2, 5]],
        ) else {
            unreachable!()
        };
        gemm_spec.unwrap()
    };

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
        &spec,
    )
    .unwrap()
    .unwrap();

    assert_eq!(prep.a_strategy, ContractOperandStrategy::Borrowed);
    assert_eq!(prep.b_strategy, ContractOperandStrategy::Borrowed);
    assert_eq!(prep.output_strategy, ContractOutputStrategy::Temporary);
}

#[test]
fn contract_preparation_materializes_only_a_when_a_is_non_fusible() {
    let mut ctx = CpuContext::new(1);
    let a = tensor_with_layout(&[3, 2, 4], &[2, 1, 6], |idx| {
        1.0 + idx[0] as f64 + 0.1 * idx[1] as f64 + 0.01 * idx[2] as f64
    });
    let b = tensor_from_fn(&[4, 5], |idx| {
        -0.5 + 0.2 * idx[0] as f64 + 0.05 * idx[1] as f64
    });
    let mut out = tensor_zeros::<f64>(&[3, 2, 5]);
    let spec = {
        let CpuPlan::Contract { gemm_spec, .. } = contract_plan::<f64>(
            &mut ctx,
            vec![0, 1, 2],
            vec![2, 3],
            vec![0, 1, 3],
            &[&[3, 2, 4], &[4, 5], &[3, 2, 5]],
        ) else {
            unreachable!()
        };
        gemm_spec.unwrap()
    };

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
        &spec,
    )
    .unwrap()
    .unwrap();

    assert_eq!(prep.a_strategy, ContractOperandStrategy::Materialize);
    assert_eq!(prep.b_strategy, ContractOperandStrategy::Borrowed);
    assert_eq!(prep.output_strategy, ContractOutputStrategy::Direct);
}

#[test]
fn contract_reuses_temp_pool_for_repeated_prepared_calls() {
    let mut ctx = CpuContext::new(1);
    let a = tensor_with_layout(&[3, 2, 4], &[2, 1, 6], |idx| {
        1.0 + idx[0] as f64 + 0.1 * idx[1] as f64 + 0.01 * idx[2] as f64
    });
    let b = tensor_from_fn(&[4, 5], |idx| {
        -0.5 + 0.2 * idx[0] as f64 + 0.05 * idx[1] as f64
    });
    let plan = contract_plan::<f64>(
        &mut ctx,
        vec![0, 1, 2],
        vec![2, 3],
        vec![0, 1, 3],
        &[&[3, 2, 4], &[4, 5], &[3, 2, 5]],
    );

    let before = ctx.temp_pool_mut().stats();
    let mut out1 = tensor_zeros::<f64>(&[3, 2, 5]);
    execute_contract(&mut ctx, &plan, &a, &b, &mut out1);
    let after_first = ctx.temp_pool_mut().stats();
    let mut out2 = tensor_zeros::<f64>(&[3, 2, 5]);
    execute_contract(&mut ctx, &plan, &a, &b, &mut out2);
    let after_second = ctx.temp_pool_mut().stats();

    assert!(after_first.misses > before.misses);
    assert_eq!(after_second.misses, after_first.misses);
    assert!(after_second.hits > after_first.hits);
}

#[test]
fn prepared_contract_execution_matches_expected_for_non_fusible_output() {
    let mut ctx = CpuContext::new(1);
    let a = tensor_from_fn(&[3, 2, 4], |idx| {
        1.0 + idx[0] as f64 + 0.1 * idx[1] as f64 + 0.01 * idx[2] as f64
    });
    let b = tensor_from_fn(&[4, 5], |idx| {
        -0.5 + 0.2 * idx[0] as f64 + 0.05 * idx[1] as f64
    });
    let expected = expected_contract_012_23_to_013(&a, &b);
    let plan = contract_plan::<f64>(
        &mut ctx,
        vec![0, 1, 2],
        vec![2, 3],
        vec![0, 1, 3],
        &[&[3, 2, 4], &[4, 5], &[3, 2, 5]],
    );
    let mut out = tensor_with_layout(&[3, 2, 5], &[2, 1, 6], |_| 0.0);

    execute_contract(&mut ctx, &plan, &a, &b, &mut out);

    for i in 0..3 {
        for j in 0..2 {
            for k in 0..5 {
                let got = out.get(&[i, j, k]).copied().unwrap();
                let want = expected.get(&[i, j, k]).copied().unwrap();
                assert!(
                    (got - want).abs() < 1e-10,
                    "mismatch at [{i},{j},{k}]: got {got}, want {want}"
                );
            }
        }
    }
}

#[test]
fn prepared_contract_execution_handles_interleaved_output_order() {
    let mut ctx = CpuContext::new(1);
    let a = tensor_from_fn(&[8, 16, 2, 2, 8, 2, 2], |idx| {
        (idx[0] + 3 * idx[1] + 5 * idx[2] + 7 * idx[3] + 11 * idx[4] + 13 * idx[5] + 17 * idx[6])
            as f64
            / 19.0
    });
    let b = tensor_from_fn(&[8, 16, 2, 2, 8, 2, 2], |idx| {
        (2 * idx[0]
            + 5 * idx[1]
            + 7 * idx[2]
            + 11 * idx[3]
            + 13 * idx[4]
            + 17 * idx[5]
            + 19 * idx[6]) as f64
            / 23.0
    });
    let expected = expected_issue_336_final_contract(&a, &b);
    let plan = contract_plan::<f64>(
        &mut ctx,
        vec![6, 13, 0, 9, 4, 5, 11],
        vec![4, 12, 8, 0, 6, 10, 5],
        vec![8, 9, 10, 11, 12, 13],
        &[
            &[8, 16, 2, 2, 8, 2, 2],
            &[8, 16, 2, 2, 8, 2, 2],
            &[2, 2, 2, 2, 16, 16],
        ],
    );
    let mut out = tensor_zeros::<f64>(&[2, 2, 2, 2, 16, 16]);

    execute_contract(&mut ctx, &plan, &a, &b, &mut out);

    for i0 in 0..2 {
        for i1 in 0..2 {
            for i2 in 0..2 {
                for i3 in 0..2 {
                    for i4 in 0..16 {
                        for i5 in 0..16 {
                            let idx = [i0, i1, i2, i3, i4, i5];
                            let got = out.get(&idx).copied().unwrap();
                            let want = expected.get(&idx).copied().unwrap();
                            assert!(
                                (got - want).abs() < 1e-10,
                                "mismatch at {:?}: got {}, want {}",
                                idx,
                                got,
                                want
                            );
                        }
                    }
                }
            }
        }
    }
}
