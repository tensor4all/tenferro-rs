use std::time::Instant;

use tenferro_algebra::Standard;
#[cfg(feature = "profile-dispatch")]
use tenferro_einsum::print_and_reset_profile;
use tenferro_einsum::{
    einsum_binary_with_subscripts, einsum_with_plan, ContractionTree, Subscripts,
};
use tenferro_prims::{CpuBackend, CpuContext};
use tenferro_tensor::{MemoryOrder, Tensor};

const COL: MemoryOrder = MemoryOrder::ColumnMajor;

fn tensor_from_fn(dims: &[usize], offset: usize) -> Tensor<f64> {
    let len: usize = dims.iter().product();
    let data: Vec<f64> = (0..len)
        .map(|i| (((i + offset) * 17 + 3) % 31) as f64 / 31.0 - 0.5)
        .collect();
    Tensor::from_slice(&data, dims, COL).unwrap()
}

fn print_tree(label: &str, tree: &ContractionTree) {
    println!("{label}: step_count={}", tree.step_count());
    for step_idx in 0..tree.step_count() {
        let (lhs, rhs) = tree.step_pair(step_idx).unwrap();
        let (subs_l, subs_r, subs_out) = tree.step_subscripts(step_idx).unwrap();
        println!(
            "  step {step_idx}: ({lhs}, {rhs})  {:?} x {:?} -> {:?}",
            subs_l, subs_r, subs_out
        );
    }
}

fn replay_tree_with_binary_api(
    ctx: &mut CpuContext,
    tree: &ContractionTree,
    inputs: &[Tensor<f64>],
) -> Tensor<f64> {
    let mut slots: Vec<Option<Tensor<f64>>> = inputs.iter().cloned().map(Some).collect();
    slots.resize_with(inputs.len() + tree.step_count(), || None);

    for step_idx in 0..tree.step_count() {
        let (lhs_idx, rhs_idx) = tree.step_pair(step_idx).unwrap();
        let (subs_l, subs_r, subs_out) = tree.step_subscripts(step_idx).unwrap();
        let step_subs = Subscripts::new(&[subs_l, subs_r], subs_out);
        let out = {
            let lhs = slots[lhs_idx].as_ref().unwrap();
            let rhs = slots[rhs_idx].as_ref().unwrap();
            einsum_binary_with_subscripts::<Standard<f64>, CpuBackend>(
                ctx, &step_subs, lhs, rhs, None,
            )
            .unwrap()
        };
        slots[inputs.len() + step_idx] = Some(out);
    }

    slots[inputs.len() + tree.step_count() - 1].take().unwrap()
}

fn replay_tree_collect_steps(
    ctx: &mut CpuContext,
    tree: &ContractionTree,
    inputs: &[Tensor<f64>],
) -> Vec<(Tensor<f64>, Tensor<f64>, Subscripts, Tensor<f64>)> {
    let mut slots: Vec<Option<Tensor<f64>>> = inputs.iter().cloned().map(Some).collect();
    slots.resize_with(inputs.len() + tree.step_count(), || None);
    let mut records = Vec::with_capacity(tree.step_count());

    for step_idx in 0..tree.step_count() {
        let (lhs_idx, rhs_idx) = tree.step_pair(step_idx).unwrap();
        let (subs_l, subs_r, subs_out) = tree.step_subscripts(step_idx).unwrap();
        let step_subs = Subscripts::new(&[subs_l, subs_r], subs_out);
        let lhs = slots[lhs_idx].as_ref().unwrap().clone();
        let rhs = slots[rhs_idx].as_ref().unwrap().clone();
        let out = einsum_binary_with_subscripts::<Standard<f64>, CpuBackend>(
            ctx, &step_subs, &lhs, &rhs, None,
        )
        .unwrap();
        slots[inputs.len() + step_idx] = Some(out.clone());
        records.push((lhs, rhs, step_subs, out));
    }

    records
}

fn permute_if_needed(tensor: &Tensor<f64>, perm: &[usize]) -> Tensor<f64> {
    if perm.iter().enumerate().all(|(i, &axis)| i == axis) {
        tensor.clone()
    } else {
        tensor.permute(perm).unwrap()
    }
}

fn product(dims: &[usize]) -> usize {
    dims.iter().product::<usize>().max(1)
}

fn bench_prepare_only(
    left: &Tensor<f64>,
    right: &Tensor<f64>,
    subs_l: &[u32],
    subs_r: &[u32],
    iters: usize,
) -> std::time::Duration {
    use std::collections::HashMap;

    let lhs_pos: HashMap<u32, usize> = subs_l
        .iter()
        .copied()
        .enumerate()
        .map(|(i, label)| (label, i))
        .collect();
    let rhs_pos: HashMap<u32, usize> = subs_r
        .iter()
        .copied()
        .enumerate()
        .map(|(i, label)| (label, i))
        .collect();

    let lhs_free: Vec<u32> = subs_l
        .iter()
        .copied()
        .filter(|label| !rhs_pos.contains_key(label))
        .collect();
    let rhs_free: Vec<u32> = subs_r
        .iter()
        .copied()
        .filter(|label| !lhs_pos.contains_key(label))
        .collect();
    let contract: Vec<u32> = subs_l
        .iter()
        .copied()
        .filter(|label| rhs_pos.contains_key(label))
        .collect();

    let lhs_perm: Vec<usize> = lhs_free
        .iter()
        .chain(contract.iter())
        .map(|label| lhs_pos[label])
        .collect();
    let rhs_perm: Vec<usize> = contract
        .iter()
        .chain(rhs_free.iter())
        .map(|label| rhs_pos[label])
        .collect();

    let lhs_free_dims: Vec<usize> = lhs_free
        .iter()
        .map(|label| left.dims()[lhs_pos[label]])
        .collect();
    let rhs_free_dims: Vec<usize> = rhs_free
        .iter()
        .map(|label| right.dims()[rhs_pos[label]])
        .collect();
    let contract_dims: Vec<usize> = contract
        .iter()
        .map(|label| left.dims()[lhs_pos[label]])
        .collect();
    let lhs_matrix_dims = [product(&lhs_free_dims), product(&contract_dims)];
    let rhs_matrix_dims = [product(&contract_dims), product(&rhs_free_dims)];

    let start = Instant::now();
    for _ in 0..iters {
        let _ = permute_if_needed(left, &lhs_perm)
            .contiguous(MemoryOrder::ColumnMajor)
            .reshape(&lhs_matrix_dims)
            .unwrap();
        let _ = permute_if_needed(right, &rhs_perm)
            .contiguous(MemoryOrder::ColumnMajor)
            .reshape(&rhs_matrix_dims)
            .unwrap();
    }
    start.elapsed()
}

fn print_timing(label: &str, elapsed: std::time::Duration, iters: usize) {
    println!(
        "{label:<28} total={:.3}s per_call={:.3}us",
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() * 1e6 / iters as f64
    );
}

#[test]
#[ignore]
fn bench_issue_336_fit_shapes() {
    let subs6 = Subscripts::new(
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
    let shapes6 = [
        &[8, 2, 2, 8][..],
        &[8, 2, 2, 8][..],
        &[8, 2, 2, 8][..],
        &[8, 2, 2, 8][..],
        &[8, 8, 16][..],
        &[8, 8, 16][..],
    ];
    let tree6 = ContractionTree::optimize(&subs6, &shapes6).unwrap();
    print_tree("env6 optimized tree", &tree6);

    let a = tensor_from_fn(&[8, 2, 2, 8], 101);
    let b = tensor_from_fn(&[8, 2, 2, 8], 102);
    let c = tensor_from_fn(&[8, 2, 2, 8], 103);
    let d = tensor_from_fn(&[8, 2, 2, 8], 104);
    let e = tensor_from_fn(&[8, 8, 16], 105);
    let f = tensor_from_fn(&[8, 8, 16], 106);
    let inputs = [a, b, c, d, e, f];
    let input_refs: Vec<&Tensor<f64>> = inputs.iter().collect();

    let iters = 2_000usize;

    let mut ctx = CpuContext::new(1);
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = einsum_with_plan::<Standard<f64>, CpuBackend>(&mut ctx, &tree6, &input_refs, None)
            .unwrap();
    }
    print_timing("env6 generic with plan", t0.elapsed(), iters);
    #[cfg(feature = "profile-dispatch")]
    print_and_reset_profile();

    let mut ctx = CpuContext::new(1);
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = replay_tree_with_binary_api(&mut ctx, &tree6, &inputs);
    }
    print_timing("env6 replay binary api", t0.elapsed(), iters);
    #[cfg(feature = "profile-dispatch")]
    print_and_reset_profile();

    let mut ctx = CpuContext::new(1);
    let step_records = replay_tree_collect_steps(&mut ctx, &tree6, &inputs);
    let per_step_iters = 10_000usize;
    for (step_idx, (lhs, rhs, step_subs, _out)) in step_records.iter().enumerate() {
        let prepare_elapsed = bench_prepare_only(
            lhs,
            rhs,
            &step_subs.inputs[0],
            &step_subs.inputs[1],
            per_step_iters,
        );
        let mut ctx = CpuContext::new(1);
        let t0 = Instant::now();
        for _ in 0..per_step_iters {
            let _ = einsum_binary_with_subscripts::<Standard<f64>, CpuBackend>(
                &mut ctx, step_subs, lhs, rhs, None,
            )
            .unwrap();
        }
        let binary_elapsed = t0.elapsed();
        println!(
            "step {step_idx} prepare_only={:.3}us binary_total={:.3}us lhs_dims={:?} rhs_dims={:?}",
            prepare_elapsed.as_secs_f64() * 1e6 / per_step_iters as f64,
            binary_elapsed.as_secs_f64() * 1e6 / per_step_iters as f64,
            lhs.dims(),
            rhs.dims(),
        );
    }
}
