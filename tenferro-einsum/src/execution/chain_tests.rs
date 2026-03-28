use tenferro_algebra::Standard;
use tenferro_prims::{CpuBackend, CpuContext};
use tenferro_tensor::{MemoryOrder, Tensor};

use super::*;
use crate::execution::pool::BufferPool;
use crate::planning::plan::compile_pairwise_step_plan;
use crate::planning::tree::ContractionTree;
use crate::syntax::subscripts::Subscripts;

fn tensor(data: &[f64], dims: &[usize]) -> Tensor<f64> {
    Tensor::from_slice(data, dims, MemoryOrder::ColumnMajor).unwrap()
}

fn zeros(dims: &[usize]) -> Tensor<f64> {
    Tensor::zeros(
        dims,
        tenferro_device::LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap()
}

fn assert_tensor_close(lhs: &Tensor<f64>, rhs: &Tensor<f64>) {
    assert_eq!(lhs.dims(), rhs.dims(), "dimension mismatch");
    let n: usize = lhs.dims().iter().product();
    let lv = lhs.to_vec();
    let rv = rhs.to_vec();
    for i in 0..n {
        assert!(
            (lv[i] - rv[i]).abs() < 1.0e-10,
            "mismatch at flat index {i}: lhs={} rhs={}",
            lv[i],
            rv[i]
        );
    }
}

#[test]
fn execute_binary_step_wrapper_produces_correct_result() {
    let subs_a = [0_u32, 1_u32];
    let subs_b = [1_u32, 2_u32];
    let subs_c = [0_u32, 2_u32];
    let mut size_dict = std::collections::HashMap::new();
    size_dict.insert(0, 2);
    size_dict.insert(1, 3);
    size_dict.insert(2, 4);
    let plan = compile_pairwise_step_plan(&subs_a, &subs_b, &subs_c, &size_dict).unwrap();

    let a = tensor(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], &[2, 3]);
    let b = tensor(
        &[
            1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0,
        ],
        &[3, 4],
    );
    let mut output = zeros(&[2, 4]);
    let mut ctx = CpuContext::new(1);
    let mut pool = BufferPool::new();

    execute_binary_step::<Standard<f64>, CpuBackend, _>(
        &mut ctx,
        &plan,
        &subs_a,
        &subs_b,
        &subs_c,
        &a,
        &b,
        1.0,
        0.0,
        &mut output,
        &mut pool,
        true,
    )
    .unwrap();

    let expected = crate::api::einsum::<Standard<f64>, CpuBackend>(
        &mut CpuContext::new(1),
        "ij,jk->ik",
        &[&a, &b],
        None,
    )
    .unwrap();
    assert_tensor_close(&output, &expected);
}

#[test]
fn execute_binary_step_wrapper_with_beta_accumulation() {
    let subs_a = [0_u32, 1_u32];
    let subs_b = [1_u32, 2_u32];
    let subs_c = [0_u32, 2_u32];
    let mut size_dict = std::collections::HashMap::new();
    size_dict.insert(0, 2);
    size_dict.insert(1, 2);
    size_dict.insert(2, 2);
    let plan = compile_pairwise_step_plan(&subs_a, &subs_b, &subs_c, &size_dict).unwrap();

    let a = tensor(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let b = tensor(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let mut output =
        Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
    let mut ctx = CpuContext::new(1);
    let mut pool = BufferPool::new();

    execute_binary_step::<Standard<f64>, CpuBackend, _>(
        &mut ctx,
        &plan,
        &subs_a,
        &subs_b,
        &subs_c,
        &a,
        &b,
        2.0,
        1.0,
        &mut output,
        &mut pool,
        true,
    )
    .unwrap();

    assert_tensor_close(&output, &tensor(&[3.0, 2.0, 3.0, 6.0], &[2, 2]));
}

#[test]
fn execute_linear_chain_three_operands_with_pool_return() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);
    let shapes = [&[2, 2][..], &[2, 2][..], &[2, 2][..]];
    let tree = ContractionTree::from_pairs(&subs, &shapes, &[(0, 1), (3, 2)]).unwrap();
    let chain = tree.linear_chain_plan().unwrap();

    let a = tensor(&[1.0, 3.0, 2.0, 4.0], &[2, 2]);
    let b = tensor(&[5.0, 7.0, 6.0, 8.0], &[2, 2]);
    let c = tensor(&[2.0, 0.0, 1.0, 3.0], &[2, 2]);

    let expected = crate::api::einsum::<Standard<f64>, CpuBackend>(
        &mut CpuContext::new(1),
        "ab,bc,cd->ad",
        &[&a, &b, &c],
        None,
    )
    .unwrap();

    let mut output = zeros(expected.dims());
    let mut ctx = CpuContext::new(1);
    let mut pool = BufferPool::new();

    execute_linear_chain_tree::<Standard<f64>, CpuBackend, _>(
        &mut ctx,
        &tree,
        &chain,
        &[&a, &b, &c],
        1.0,
        0.0,
        &mut output,
        &mut pool,
        true,
    )
    .unwrap();

    assert_tensor_close(&output, &expected);
}
