use tenferro_algebra::Standard;
use tenferro_prims::{CpuBackend, CpuContext};
use tenferro_tensor::{MemoryOrder, Tensor};

use super::*;
use crate::execution::chain::{execute_binary_step, execute_binary_step_with_plans};
use crate::execution::execute::execute_tree;
use crate::execution::util::{compute_output_shape, tensor_get, unflatten_index};
use crate::planning::plan::{compile_pairwise_step_plan, GemmPlan, ReducePlan, StepPlan};
use crate::planning::tree::ContractionTree;
use crate::syntax::subscripts::Subscripts;
use crate::tests::semiring_backend::SemiringOnlyCpuBackend;
use tenferro_device::Result;
use tenferro_prims::{
    SemiringCoreDescriptor, SemiringFastPathDescriptor, TensorSemiringCore, TensorSemiringFastPath,
};

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

fn tensor_from_seed(dims: &[usize], offset: usize) -> Tensor<f64> {
    let len: usize = dims.iter().product();
    let data: Vec<f64> = (0..len)
        .map(|i| (((i + offset) * 17 + 3) % 31) as f64 / 31.0 - 0.5)
        .collect();
    Tensor::from_slice(&data, dims, MemoryOrder::ColumnMajor).unwrap()
}

fn assert_tensor_close(lhs: &Tensor<f64>, rhs: &Tensor<f64>) {
    assert_eq!(lhs.dims(), rhs.dims());
    let numel: usize = lhs.dims().iter().product();
    for flat in 0..numel {
        let idx = unflatten_index(flat, lhs.dims());
        let l = tensor_get(lhs, &idx).unwrap();
        let r = tensor_get(rhs, &idx).unwrap();
        assert!(
            (l - r).abs() < 1e-10,
            "mismatch at {:?}: left={} right={}",
            idx,
            l,
            r
        );
    }
}

struct NoFastPathCpuBackend;

impl TensorSemiringCore<Standard<f64>> for NoFastPathCpuBackend {
    type Plan = <CpuBackend as TensorSemiringCore<Standard<f64>>>::Plan;
    type Context = CpuContext;

    fn plan(
        ctx: &mut Self::Context,
        desc: &SemiringCoreDescriptor,
        shapes: &[&[usize]],
    ) -> Result<Self::Plan> {
        <CpuBackend as TensorSemiringCore<Standard<f64>>>::plan(ctx, desc, shapes)
    }

    fn execute(
        ctx: &mut Self::Context,
        plan: &Self::Plan,
        alpha: f64,
        inputs: &[&Tensor<f64>],
        beta: f64,
        output: &mut Tensor<f64>,
    ) -> Result<()> {
        <CpuBackend as TensorSemiringCore<Standard<f64>>>::execute(
            ctx, plan, alpha, inputs, beta, output,
        )
    }
}

impl TensorSemiringFastPath<Standard<f64>> for NoFastPathCpuBackend {
    type Plan = <CpuBackend as TensorSemiringFastPath<Standard<f64>>>::Plan;
    type Context = CpuContext;

    fn plan(
        _ctx: &mut Self::Context,
        _desc: &SemiringFastPathDescriptor,
        _shapes: &[&[usize]],
    ) -> Result<Self::Plan> {
        panic!("NoFastPathCpuBackend must not request fast-path planning")
    }

    fn execute(
        _ctx: &mut Self::Context,
        _plan: &Self::Plan,
        _alpha: f64,
        _inputs: &[&Tensor<f64>],
        _beta: f64,
        _output: &mut Tensor<f64>,
    ) -> Result<()> {
        panic!("NoFastPathCpuBackend must not execute fast paths")
    }

    fn has_fast_path(_desc: SemiringFastPathDescriptor) -> bool {
        false
    }
}

struct NoMakeContiguousCpuBackend;

impl TensorSemiringCore<Standard<f64>> for NoMakeContiguousCpuBackend {
    type Plan = <CpuBackend as TensorSemiringCore<Standard<f64>>>::Plan;
    type Context = CpuContext;

    fn plan(
        ctx: &mut Self::Context,
        desc: &SemiringCoreDescriptor,
        shapes: &[&[usize]],
    ) -> Result<Self::Plan> {
        if matches!(desc, SemiringCoreDescriptor::MakeContiguous) {
            panic!("strict binary direct-output path must not plan MakeContiguous");
        }
        <CpuBackend as TensorSemiringCore<Standard<f64>>>::plan(ctx, desc, shapes)
    }

    fn execute(
        ctx: &mut Self::Context,
        plan: &Self::Plan,
        alpha: f64,
        inputs: &[&Tensor<f64>],
        beta: f64,
        output: &mut Tensor<f64>,
    ) -> Result<()> {
        <CpuBackend as TensorSemiringCore<Standard<f64>>>::execute(
            ctx, plan, alpha, inputs, beta, output,
        )
    }
}

impl TensorSemiringFastPath<Standard<f64>> for NoMakeContiguousCpuBackend {
    type Plan = <CpuBackend as TensorSemiringFastPath<Standard<f64>>>::Plan;
    type Context = CpuContext;

    fn plan(
        _ctx: &mut Self::Context,
        _desc: &SemiringFastPathDescriptor,
        _shapes: &[&[usize]],
    ) -> Result<Self::Plan> {
        panic!("NoMakeContiguousCpuBackend must not request fast-path planning")
    }

    fn execute(
        _ctx: &mut Self::Context,
        _plan: &Self::Plan,
        _alpha: f64,
        _inputs: &[&Tensor<f64>],
        _beta: f64,
        _output: &mut Tensor<f64>,
    ) -> Result<()> {
        panic!("NoMakeContiguousCpuBackend must not execute fast paths")
    }

    fn has_fast_path(_desc: SemiringFastPathDescriptor) -> bool {
        false
    }
}

struct NoDynamicBatchedGemmPlanCpuBackend;

impl TensorSemiringCore<Standard<f64>> for NoDynamicBatchedGemmPlanCpuBackend {
    type Plan = <CpuBackend as TensorSemiringCore<Standard<f64>>>::Plan;
    type Context = CpuContext;

    fn plan(
        ctx: &mut Self::Context,
        desc: &SemiringCoreDescriptor,
        shapes: &[&[usize]],
    ) -> Result<Self::Plan> {
        if matches!(desc, SemiringCoreDescriptor::BatchedGemm { .. }) {
            panic!("strict path must reuse cached BatchedGemm plan");
        }
        <CpuBackend as TensorSemiringCore<Standard<f64>>>::plan(ctx, desc, shapes)
    }

    fn execute(
        ctx: &mut Self::Context,
        plan: &Self::Plan,
        alpha: f64,
        inputs: &[&Tensor<f64>],
        beta: f64,
        output: &mut Tensor<f64>,
    ) -> Result<()> {
        <CpuBackend as TensorSemiringCore<Standard<f64>>>::execute(
            ctx, plan, alpha, inputs, beta, output,
        )
    }
}

impl TensorSemiringFastPath<Standard<f64>> for NoDynamicBatchedGemmPlanCpuBackend {
    type Plan = <CpuBackend as TensorSemiringFastPath<Standard<f64>>>::Plan;
    type Context = CpuContext;

    fn plan(
        _ctx: &mut Self::Context,
        _desc: &SemiringFastPathDescriptor,
        _shapes: &[&[usize]],
    ) -> Result<Self::Plan> {
        panic!("NoDynamicBatchedGemmPlanCpuBackend must not request fast-path planning")
    }

    fn execute(
        _ctx: &mut Self::Context,
        _plan: &Self::Plan,
        _alpha: f64,
        _inputs: &[&Tensor<f64>],
        _beta: f64,
        _output: &mut Tensor<f64>,
    ) -> Result<()> {
        panic!("NoDynamicBatchedGemmPlanCpuBackend must not execute fast paths")
    }

    fn has_fast_path(_desc: SemiringFastPathDescriptor) -> bool {
        false
    }
}

#[derive(Default)]
struct CountingPool {
    inner: BufferPool<f64>,
    returned_caps: Vec<usize>,
}

impl crate::execution::pool::TensorBufferPool<f64> for CountingPool {
    fn take_with_ctx<Ctx: tenferro_prims::TensorTempPoolContext>(
        &mut self,
        _ctx: &mut Ctx,
        len: usize,
    ) -> Vec<f64> {
        self.inner.take(len)
    }

    fn return_buf(&mut self, buf: Vec<f64>) {
        self.returned_caps.push(buf.capacity());
        self.inner.return_buf(buf);
    }
}

#[test]
fn execute_pairwise_dispatches_dynamic_ewmul_after_reduce_b() {
    let mut ctx = CpuContext::new(1);
    let mut pool = BufferPool::new();
    let a = tensor(&[2.0, 3.0], &[2]);
    let b = tensor(&[5.0, 7.0, 11.0, 13.0], &[2, 2]);
    let mut output = zeros(&[2]);

    let plan = StepPlan {
        diag_a: None,
        diag_b: None,
        strict_binary: None,
        gemm: GemmPlan {
            reduce_a: None,
            reduce_b: Some(ReducePlan {
                original_subs: vec![0, 1],
                kept_subs: vec![0],
                out_shape: vec![2],
            }),
            subs_a: vec![0],
            subs_b: vec![0],
            lo_modes: vec![],
            ro_modes: vec![],
            sum_modes: vec![],
            batch_sizes: vec![2],
            m: 1,
            n: 1,
            k: 1,
            target_a: vec![0],
            target_b: vec![0],
            c_gemm_shape: vec![1, 1, 2],
            expanded_shape: vec![2],
            canonical_modes: vec![0],
            needs_final_permute: false,
            a_gemm_shape: vec![1, 1, 2],
            b_gemm_shape: vec![1, 1, 2],
        },
    };

    execute_pairwise_with_plan::<Standard<f64>, CpuBackend, _>(
        &mut ctx,
        &plan,
        None,
        None,
        &[0],
        &[0, 1],
        &[0],
        &a,
        &b,
        1.0,
        0.0,
        &mut output,
        &mut pool,
        false,
    )
    .unwrap();

    let expected = tensor(&[2.0 * (5.0 + 11.0), 3.0 * (7.0 + 13.0)], &[2]);
    assert_tensor_close(&output, &expected);
}

#[test]
fn execute_pairwise_accepts_semiring_only_backend() {
    let mut ctx = CpuContext::new(1);
    let mut pool = BufferPool::new();
    let a = tensor(&[2.0, 3.0], &[2]);
    let b = tensor(&[5.0, 7.0, 11.0, 13.0], &[2, 2]);
    let mut output = zeros(&[2]);

    let plan = StepPlan {
        diag_a: None,
        diag_b: None,
        strict_binary: None,
        gemm: GemmPlan {
            reduce_a: None,
            reduce_b: Some(ReducePlan {
                original_subs: vec![0, 1],
                kept_subs: vec![0],
                out_shape: vec![2],
            }),
            subs_a: vec![0],
            subs_b: vec![0],
            lo_modes: vec![],
            ro_modes: vec![],
            sum_modes: vec![],
            batch_sizes: vec![2],
            m: 1,
            n: 1,
            k: 1,
            target_a: vec![0],
            target_b: vec![0],
            c_gemm_shape: vec![1, 1, 2],
            expanded_shape: vec![2],
            canonical_modes: vec![0],
            needs_final_permute: false,
            a_gemm_shape: vec![1, 1, 2],
            b_gemm_shape: vec![1, 1, 2],
        },
    };

    execute_pairwise_with_plan::<Standard<f64>, SemiringOnlyCpuBackend, _>(
        &mut ctx,
        &plan,
        None,
        None,
        &[0],
        &[0, 1],
        &[0],
        &a,
        &b,
        1.0,
        0.0,
        &mut output,
        &mut pool,
        false,
    )
    .unwrap();

    let expected = tensor(&[2.0 * (5.0 + 11.0), 3.0 * (7.0 + 13.0)], &[2]);
    assert_tensor_close(&output, &expected);
}

#[test]
fn execute_gemm_after_reduce_builds_direct_batched_gemm_plan() {
    let mut ctx = CpuContext::new(1);
    let mut pool = BufferPool::new();
    let a = tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = tensor(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
    let mut output = zeros(&[2, 2]);

    let plan = GemmPlan {
        reduce_a: None,
        reduce_b: None,
        subs_a: vec![0, 1],
        subs_b: vec![1, 2],
        lo_modes: vec![0],
        ro_modes: vec![2],
        sum_modes: vec![1],
        batch_sizes: vec![],
        m: 2,
        n: 2,
        k: 2,
        target_a: vec![0, 1],
        target_b: vec![1, 2],
        c_gemm_shape: vec![2, 2],
        expanded_shape: vec![2, 2],
        canonical_modes: vec![0, 2],
        needs_final_permute: false,
        a_gemm_shape: vec![2, 2],
        b_gemm_shape: vec![2, 2],
    };

    execute_gemm_after_reduce::<Standard<f64>, CpuBackend, _>(
        &mut ctx,
        &plan,
        None,
        &[0, 2],
        &a,
        &b,
        1.0,
        0.0,
        &mut output,
        &mut pool,
        false,
    )
    .unwrap();

    let expected = tensor(&[23.0, 34.0, 31.0, 46.0], &[2, 2]);
    assert_tensor_close(&output, &expected);
}

#[test]
fn execute_gemm_after_reduce_builds_temp_gemm_and_physical_permute() {
    let mut ctx = CpuContext::new(1);
    let mut pool = BufferPool::new();
    let a = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let mut output = zeros(&[2, 2]);

    let plan = GemmPlan {
        reduce_a: None,
        reduce_b: None,
        subs_a: vec![0, 1],
        subs_b: vec![1, 2],
        lo_modes: vec![0],
        ro_modes: vec![2],
        sum_modes: vec![1],
        batch_sizes: vec![],
        m: 2,
        n: 2,
        k: 3,
        target_a: vec![0, 1],
        target_b: vec![1, 2],
        c_gemm_shape: vec![2, 2],
        expanded_shape: vec![2, 2],
        canonical_modes: vec![0, 2],
        needs_final_permute: true,
        a_gemm_shape: vec![2, 3],
        b_gemm_shape: vec![3, 2],
    };

    execute_gemm_after_reduce::<Standard<f64>, CpuBackend, _>(
        &mut ctx,
        &plan,
        None,
        &[2, 0],
        &a,
        &b,
        1.0,
        0.0,
        &mut output,
        &mut pool,
        false,
    )
    .unwrap();

    let expected = tensor(&[22.0, 49.0, 28.0, 64.0], &[2, 2]);
    assert_tensor_close(&output, &expected);
    let reused = pool.take(6);
    assert_eq!(reused.len(), 6);
}

#[test]
fn execute_tree_branching_nary_steps_work_with_cached_gemm_dispatch() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2], &[3, 4], &[4, 5]], &[0, 2, 3, 5]);
    let shapes = [&[2, 2][..], &[2, 2][..], &[2, 2][..], &[2, 2][..]];
    let tree = ContractionTree::from_pairs(&subs, &shapes, &[(0, 1), (2, 3), (4, 5)]).unwrap();
    assert!(
        tree.step_plans[0].strict_binary.is_some(),
        "first branching step should remain strict-binary eligible"
    );

    let a = tensor(&[1.0, 3.0, 2.0, 4.0], &[2, 2]);
    let b = tensor(&[5.0, 7.0, 6.0, 8.0], &[2, 2]);
    let c = tensor(&[2.0, 0.0, 0.0, 3.0], &[2, 2]);
    let d = tensor(&[1.0, 5.0, 4.0, 6.0], &[2, 2]);

    let ab_subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    let cd_subs = Subscripts::new(&[&[3, 4], &[4, 5]], &[3, 5]);
    let out_subs = Subscripts::new(&[&[0, 2], &[3, 5]], &[0, 2, 3, 5]);
    let mut expected_ctx = CpuContext::new(1);
    let ab = crate::einsum_binary_with_subscripts::<Standard<f64>, CpuBackend>(
        &mut expected_ctx,
        &ab_subs,
        &a,
        &b,
        None,
    )
    .unwrap();
    let cd = crate::einsum_binary_with_subscripts::<Standard<f64>, CpuBackend>(
        &mut expected_ctx,
        &cd_subs,
        &c,
        &d,
        None,
    )
    .unwrap();
    let expected = crate::einsum_binary_with_subscripts::<Standard<f64>, CpuBackend>(
        &mut expected_ctx,
        &out_subs,
        &ab,
        &cd,
        None,
    )
    .unwrap();

    let mut output = zeros(&[2, 2, 2, 2]);
    let mut ctx = CpuContext::new(1);
    let mut pool = BufferPool::new();
    execute_tree::<Standard<f64>, NoFastPathCpuBackend, _>(
        &mut ctx,
        &tree,
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
fn execute_binary_step_strict_binary_avoids_make_contiguous_for_fusible_output() {
    let subs_a = [0_u32, 1_u32];
    let subs_b = [1_u32, 2_u32];
    let subs_c = [2_u32, 0_u32];
    let mut size_dict = std::collections::HashMap::new();
    size_dict.insert(0, 2);
    size_dict.insert(1, 3);
    size_dict.insert(2, 4);
    let plan = compile_pairwise_step_plan(&subs_a, &subs_b, &subs_c, &size_dict).unwrap();
    assert!(plan.strict_binary.is_some());

    let a = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = tensor(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[3, 4],
    );
    let expected = tensor(
        &[22.0, 49.0, 76.0, 103.0, 28.0, 64.0, 100.0, 136.0],
        &[4, 2],
    );
    let mut output = zeros(&[4, 2]);
    let mut ctx = CpuContext::new(1);
    let mut pool = BufferPool::new();

    execute_binary_step::<Standard<f64>, NoMakeContiguousCpuBackend, _>(
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

    assert_tensor_close(&output, &expected);
}

#[test]
fn execute_binary_step_with_cached_gemm_plan_skips_dynamic_plan_even_if_strict_exists() {
    let subs_a = [0_u32, 1_u32];
    let subs_b = [1_u32, 2_u32];
    let subs_c = [2_u32, 0_u32];
    let mut size_dict = std::collections::HashMap::new();
    size_dict.insert(0, 2);
    size_dict.insert(1, 3);
    size_dict.insert(2, 4);
    let plan = compile_pairwise_step_plan(&subs_a, &subs_b, &subs_c, &size_dict).unwrap();
    assert!(plan.strict_binary.is_some());

    let desc = SemiringCoreDescriptor::BatchedGemm {
        batch_dims: plan.gemm.batch_sizes.clone(),
        m: plan.gemm.m,
        n: plan.gemm.n,
        k: plan.gemm.k,
    };
    let a_shape = plan.gemm.a_gemm_shape.clone();
    let b_shape = plan.gemm.b_gemm_shape.clone();
    let c_shape = plan.gemm.c_gemm_shape.clone();
    let cached_gemm_plan = <CpuBackend as TensorSemiringCore<Standard<f64>>>::plan(
        &mut CpuContext::new(1),
        &desc,
        &[&a_shape, &b_shape, &c_shape],
    )
    .unwrap();

    let a = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = tensor(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[3, 4],
    );
    let expected = tensor(
        &[22.0, 49.0, 76.0, 103.0, 28.0, 64.0, 100.0, 136.0],
        &[4, 2],
    );
    let mut output = zeros(&[4, 2]);
    let mut ctx = CpuContext::new(1);
    let mut pool = BufferPool::new();

    execute_binary_step_with_plans::<Standard<f64>, NoDynamicBatchedGemmPlanCpuBackend, _>(
        &mut ctx,
        &plan,
        None,
        Some(&cached_gemm_plan),
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

    assert_tensor_close(&output, &expected);
}

#[test]
fn execute_tree_linear_chain_returns_intermediate_buffers_to_pool() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);
    let shapes = [&[2, 2][..], &[2, 2][..], &[2, 2][..]];
    let tree = ContractionTree::from_pairs(&subs, &shapes, &[(0, 1), (2, 3)]).unwrap();
    assert!(tree.linear_chain_plan().is_some());

    let a = tensor(&[1.0, 3.0, 2.0, 4.0], &[2, 2]);
    let b = tensor(&[5.0, 7.0, 6.0, 8.0], &[2, 2]);
    let c = tensor(&[2.0, 0.0, 0.0, 3.0], &[2, 2]);
    let expected = crate::einsum_binary_with_subscripts::<Standard<f64>, CpuBackend>(
        &mut CpuContext::new(1),
        &Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]),
        &a,
        &b,
        None,
    )
    .and_then(|ab| {
        crate::einsum_binary_with_subscripts::<Standard<f64>, CpuBackend>(
            &mut CpuContext::new(1),
            &Subscripts::new(&[&[0, 2], &[2, 3]], &[0, 3]),
            &ab,
            &c,
            None,
        )
    })
    .unwrap();

    let mut output = zeros(&[2, 2]);
    let mut ctx = CpuContext::new(1);
    let mut pool = CountingPool::default();
    execute_tree::<Standard<f64>, NoFastPathCpuBackend, _>(
        &mut ctx,
        &tree,
        &[&a, &b, &c],
        1.0,
        0.0,
        &mut output,
        &mut pool,
        true,
    )
    .unwrap();

    assert_tensor_close(&output, &expected);
    assert!(
        pool.returned_caps.iter().any(|&cap| cap >= 4),
        "linear-chain executor should return the 2x2 intermediate buffer to the pool"
    );
}

#[test]
fn execute_binary_step_handles_issue_336_env6_tree_with_contract_fast_path() {
    let subs = Subscripts::new(
        &[
            &[1, 8, 0, 2],
            &[3, 0, 9, 4],
            &[6, 10, 5, 1],
            &[7, 5, 11, 3],
            &[2, 4, 12],
            &[6, 7, 13],
        ],
        &[8, 9, 10, 11, 12, 13],
    );
    let shapes = [
        &[8, 2, 2, 8][..],
        &[8, 2, 2, 8][..],
        &[8, 2, 2, 8][..],
        &[8, 2, 2, 8][..],
        &[8, 8, 16][..],
        &[8, 8, 16][..],
    ];
    let tree = ContractionTree::optimize(&subs, &shapes).unwrap();
    let inputs = [
        tensor_from_seed(&[8, 2, 2, 8], 101),
        tensor_from_seed(&[8, 2, 2, 8], 102),
        tensor_from_seed(&[8, 2, 2, 8], 103),
        tensor_from_seed(&[8, 2, 2, 8], 104),
        tensor_from_seed(&[8, 8, 16], 105),
        tensor_from_seed(&[8, 8, 16], 106),
    ];

    let mut ctx = CpuContext::new(1);
    let mut pool = BufferPool::new();
    let mut slots: Vec<Option<Tensor<f64>>> = inputs.iter().cloned().map(Some).collect();
    slots.resize_with(inputs.len() + tree.step_count(), || None);

    for step_idx in 0..tree.step_count() {
        let (lhs_idx, rhs_idx) = tree.step_pair(step_idx).unwrap();
        let (subs_a, subs_b, subs_c) = tree.step_subscripts(step_idx).unwrap();
        let plan = compile_pairwise_step_plan(subs_a, subs_b, subs_c, &tree.size_dict).unwrap();
        let out_shape = compute_output_shape(subs_c, &tree.size_dict).unwrap();
        let mut output = zeros(&out_shape);
        let left = slots[lhs_idx].as_ref().unwrap();
        let right = slots[rhs_idx].as_ref().unwrap();

        execute_binary_step::<Standard<f64>, CpuBackend, _>(
            &mut ctx,
            &plan,
            subs_a,
            subs_b,
            subs_c,
            left,
            right,
            1.0,
            0.0,
            &mut output,
            &mut pool,
            false,
        )
        .unwrap_or_else(|err| {
            panic!(
                "issue 336 env6 step {step_idx} failed: subs_a={subs_a:?} subs_b={subs_b:?} subs_c={subs_c:?} lhs_dims={:?} rhs_dims={:?} error={err:?}",
                left.dims(),
                right.dims()
            )
        });

        slots[inputs.len() + step_idx] = Some(output);
    }
}

#[test]
fn execute_tree_handles_issue_336_env6_contract_fast_path() {
    let subs = Subscripts::new(
        &[
            &[1, 8, 0, 2],
            &[3, 0, 9, 4],
            &[6, 10, 5, 1],
            &[7, 5, 11, 3],
            &[2, 4, 12],
            &[6, 7, 13],
        ],
        &[8, 9, 10, 11, 12, 13],
    );
    let shapes = [
        &[8, 2, 2, 8][..],
        &[8, 2, 2, 8][..],
        &[8, 2, 2, 8][..],
        &[8, 2, 2, 8][..],
        &[8, 8, 16][..],
        &[8, 8, 16][..],
    ];
    let tree = ContractionTree::optimize(&subs, &shapes).unwrap();
    let inputs = [
        tensor_from_seed(&[8, 2, 2, 8], 101),
        tensor_from_seed(&[8, 2, 2, 8], 102),
        tensor_from_seed(&[8, 2, 2, 8], 103),
        tensor_from_seed(&[8, 2, 2, 8], 104),
        tensor_from_seed(&[8, 8, 16], 105),
        tensor_from_seed(&[8, 8, 16], 106),
    ];
    let input_refs: Vec<&Tensor<f64>> = inputs.iter().collect();
    let mut fast_ctx = CpuContext::new(1);
    let actual = crate::einsum_with_plan::<Standard<f64>, CpuBackend>(
        &mut fast_ctx,
        &tree,
        &input_refs,
        None,
    )
    .unwrap();

    let mut reference_ctx = CpuContext::new(1);
    let expected = crate::einsum_with_plan::<Standard<f64>, NoFastPathCpuBackend>(
        &mut reference_ctx,
        &tree,
        &input_refs,
        None,
    )
    .unwrap();

    assert_tensor_close(&actual, &expected);
}

use crate::execution::pool::BufferPool;
