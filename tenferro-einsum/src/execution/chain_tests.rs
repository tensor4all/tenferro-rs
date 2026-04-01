use tenferro_algebra::Standard;
use tenferro_prims::{CpuBackend, CpuContext};
use tenferro_tensor::{MemoryOrder, Tensor};

use super::*;
use crate::execution::pool::BufferPool;
use crate::planning::plan::compile_pairwise_step_plan;
use crate::planning::tree::ContractionTree;
use crate::syntax::subscripts::Subscripts;

fn make_size_dict_for_matmul() -> std::collections::HashMap<u32, usize> {
    let mut sd = std::collections::HashMap::new();
    sd.insert(0, 2);
    sd.insert(1, 3);
    sd.insert(2, 4);
    sd
}

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
fn maybe_plan_ewmul_returns_none_for_non_elementwise_gemm() {
    let subs_a = [0_u32, 1_u32];
    let subs_b = [1_u32, 2_u32];
    let subs_c = [0_u32, 2_u32];
    let size_dict = make_size_dict_for_matmul();
    let plan = compile_pairwise_step_plan(&subs_a, &subs_b, &subs_c, &size_dict).unwrap();

    let mut ctx = CpuContext::new(1);
    let a = tensor(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], &[2, 3]);
    let b = tensor(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[3, 4],
    );
    let c = zeros(&[2, 4]);

    assert!(maybe_plan_ewmul_for_step::<Standard<f64>, CpuBackend>(
        &mut ctx,
        &plan,
        a.dims(),
        b.dims(),
        c.dims(),
    )
    .is_none());
}

#[test]
fn maybe_plan_ewmul_returns_none_for_diagonal_plan() {
    let subs_a = [0_u32, 0_u32];
    let subs_b = [0_u32];
    let subs_c = &[0_u32];
    let mut size_dict = std::collections::HashMap::new();
    size_dict.insert(0, 3);
    let plan = compile_pairwise_step_plan(&subs_a, &subs_b, subs_c, &size_dict).unwrap();

    let mut ctx = CpuContext::new(1);
    let a = tensor(&[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0], &[3, 3]);
    let b = tensor(&[1.0, 2.0, 3.0], &[3]);
    let c = zeros(&[3]);

    assert!(maybe_plan_ewmul_for_step::<Standard<f64>, CpuBackend>(
        &mut ctx,
        &plan,
        a.dims(),
        b.dims(),
        c.dims(),
    )
    .is_none());
}

#[test]
fn maybe_plan_gemm_returns_none_when_skip_and_strict_binary() {
    let subs_a = [0_u32];
    let subs_b = [0_u32];
    let subs_c = &[0_u32];
    let mut size_dict = std::collections::HashMap::new();
    size_dict.insert(0, 3);
    let plan = compile_pairwise_step_plan(&subs_a, &subs_b, subs_c, &size_dict).unwrap();

    if plan.strict_binary.is_some() {
        let mut ctx = CpuContext::new(1);
        assert!(
            maybe_plan_gemm_for_step::<Standard<f64>, CpuBackend>(&mut ctx, &plan, true).is_none()
        );
    }
}

#[test]
fn execute_linear_chain_four_operands_prev_on_left() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2], &[2, 3], &[3, 4]], &[0, 4]);
    let shapes = [&[2, 3][..], &[3, 2][..], &[2, 2][..], &[2, 3][..]];
    let tree = ContractionTree::from_pairs(&subs, &shapes, &[(0, 1), (4, 2), (5, 3)]).unwrap();
    let chain = tree.linear_chain_plan().unwrap();

    let a = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let c = tensor(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let d = tensor(&[2.0, 0.0, 0.0, 2.0, 1.0, 1.0], &[2, 3]);

    let expected = crate::api::einsum::<Standard<f64>, CpuBackend>(
        &mut CpuContext::new(1),
        "ab,bc,cd,de->ae",
        &[&a, &b, &c, &d],
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
        &[&a, &b, &c, &d],
        1.0,
        0.0,
        &mut output,
        &mut pool,
        true,
    )
    .unwrap();

    assert_tensor_close(&output, &expected);
}

#[test]
fn execute_linear_chain_with_beta_accumulation_into_existing_output() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);
    let shapes = [&[2, 2][..], &[2, 2][..], &[2, 2][..]];
    let tree = ContractionTree::from_pairs(&subs, &shapes, &[(0, 1), (3, 2)]).unwrap();
    let chain = tree.linear_chain_plan().unwrap();

    let a = tensor(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let b = tensor(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let c = tensor(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);

    let init_val = 5.0;
    let mut output = Tensor::from_slice(&[init_val; 4], &[2, 2], MemoryOrder::ColumnMajor).unwrap();

    let mut ctx = CpuContext::new(1);
    let mut pool = BufferPool::new();

    execute_linear_chain_tree::<Standard<f64>, CpuBackend, _>(
        &mut ctx,
        &tree,
        &chain,
        &[&a, &b, &c],
        2.0,
        1.0,
        &mut output,
        &mut pool,
        false,
    )
    .unwrap();

    let expected_vals = output.to_vec();
    assert!(
        expected_vals.iter().any(|&v| v != init_val),
        "output should have been modified beyond the initial value"
    );
}

#[test]
fn execute_binary_step_with_plans_no_gemm_plan_uses_pairwise_fallback() {
    let subs_a = [0_u32, 1_u32];
    let subs_b = [1_u32, 2_u32];
    let subs_c = [0_u32, 2_u32];
    let size_dict = make_size_dict_for_matmul();
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

    execute_binary_step_with_plans::<Standard<f64>, CpuBackend, _>(
        &mut ctx,
        &plan,
        None,
        None,
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

fn tensor_row_major(data: &[f64], dims: &[usize]) -> Tensor<f64> {
    Tensor::from_slice(data, dims, MemoryOrder::RowMajor).unwrap()
}

fn tensor_with_strides(
    data: &[f64],
    dims: &[usize],
    strides: &[isize],
    offset: isize,
) -> Tensor<f64> {
    Tensor::from_vec(data.to_vec(), dims, strides, offset).unwrap()
}

#[test]
fn execute_binary_step_with_row_major_inputs() {
    let subs_a = [0_u32, 1_u32];
    let subs_b = [1_u32, 2_u32];
    let subs_c = [0_u32, 2_u32];
    let mut size_dict = std::collections::HashMap::new();
    size_dict.insert(0, 2);
    size_dict.insert(1, 3);
    size_dict.insert(2, 4);
    let plan = compile_pairwise_step_plan(&subs_a, &subs_b, &subs_c, &size_dict).unwrap();

    let a = tensor_row_major(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = tensor_row_major(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
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
fn execute_binary_step_with_mixed_memory_orders() {
    let subs_a = [0_u32, 1_u32];
    let subs_b = [1_u32, 2_u32];
    let subs_c = [0_u32, 2_u32];
    let mut size_dict = std::collections::HashMap::new();
    size_dict.insert(0, 2);
    size_dict.insert(1, 3);
    size_dict.insert(2, 4);
    let plan = compile_pairwise_step_plan(&subs_a, &subs_b, &subs_c, &size_dict).unwrap();

    let a = tensor(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], &[2, 3]);
    let b = tensor_row_major(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
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
fn execute_binary_step_with_noncontiguous_input() {
    let subs_a = [0_u32, 1_u32];
    let subs_b = [1_u32, 2_u32];
    let subs_c = [0_u32, 2_u32];
    let mut size_dict = std::collections::HashMap::new();
    size_dict.insert(0, 2);
    size_dict.insert(1, 3);
    size_dict.insert(2, 4);
    let plan = compile_pairwise_step_plan(&subs_a, &subs_b, &subs_c, &size_dict).unwrap();

    let large_data: Vec<f64> = (0..24).map(|i| (i + 1) as f64).collect();
    let a = tensor_with_strides(&large_data, &[2, 3], &[1, 8], 0);

    let b = tensor(
        &[
            1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0,
        ],
        &[3, 4],
    );

    let a_contiguous = a.clone().into_contiguous(MemoryOrder::ColumnMajor);

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
        &[&a_contiguous, &b],
        None,
    )
    .unwrap();
    assert_tensor_close(&output, &expected);
}

#[test]
fn linear_chain_returns_intermediate_buffers_to_pool() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);
    let shapes = [&[3, 4][..], &[4, 5][..], &[5, 6][..]];
    let tree = ContractionTree::from_pairs(&subs, &shapes, &[(0, 1), (3, 2)]).unwrap();
    let chain = tree.linear_chain_plan().unwrap();

    let a = tensor(
        &(0..12).map(|i| (i + 1) as f64).collect::<Vec<_>>(),
        &[3, 4],
    );
    let b = tensor(
        &(0..20).map(|i| (i + 1) as f64).collect::<Vec<_>>(),
        &[4, 5],
    );
    let c = tensor(
        &(0..30).map(|i| (i + 1) as f64).collect::<Vec<_>>(),
        &[5, 6],
    );

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

#[test]
fn execute_binary_step_with_both_noncontiguous_inputs() {
    let subs_a = [0_u32, 1_u32];
    let subs_b = [1_u32, 2_u32];
    let subs_c = [0_u32, 2_u32];
    let mut size_dict = std::collections::HashMap::new();
    size_dict.insert(0, 2);
    size_dict.insert(1, 2);
    size_dict.insert(2, 2);
    let plan = compile_pairwise_step_plan(&subs_a, &subs_b, &subs_c, &size_dict).unwrap();

    let large_a: Vec<f64> = (0..16).map(|i| (i + 1) as f64).collect();
    let a = tensor_with_strides(&large_a, &[2, 2], &[1, 6], 0);

    let large_b: Vec<f64> = (0..16).map(|i| (i + 1) as f64 * 0.5).collect();
    let b = tensor_with_strides(&large_b, &[2, 2], &[1, 6], 0);

    let a_c = a.clone().into_contiguous(MemoryOrder::ColumnMajor);
    let b_c = b.clone().into_contiguous(MemoryOrder::ColumnMajor);

    let mut output = zeros(&[2, 2]);
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
        &[&a_c, &b_c],
        None,
    )
    .unwrap();
    assert_tensor_close(&output, &expected);
}

#[test]
fn execute_linear_chain_with_row_major_operands() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);
    let shapes = [&[2, 2][..], &[2, 2][..], &[2, 2][..]];
    let tree = ContractionTree::from_pairs(&subs, &shapes, &[(0, 1), (3, 2)]).unwrap();
    let chain = tree.linear_chain_plan().unwrap();

    let a = tensor_row_major(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = tensor_row_major(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
    let c = tensor_row_major(&[2.0, 1.0, 0.0, 3.0], &[2, 2]);

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
