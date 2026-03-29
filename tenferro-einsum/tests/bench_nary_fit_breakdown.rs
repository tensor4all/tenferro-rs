use std::time::Instant;

use tenferro_algebra::Standard;
use tenferro_einsum::{
    einsum_binary_with_subscripts, einsum_with_path, einsum_with_plan, einsum_with_subscripts,
    ContractionTree, Subscripts,
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

fn print_timing(label: &str, elapsed: std::time::Duration, iters: usize) {
    println!(
        "{label:<24} total={:.3}s per_call={:.3}us",
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() * 1e6 / iters as f64
    );
}

#[test]
#[ignore]
fn bench_nary_fit_breakdown() {
    let env3_subs = Subscripts::new(
        &[&[3, 0, 1], &[0, 4, 2, 5], &[1, 2, 6, 7]],
        &[3, 4, 5, 6, 7],
    );
    let env3_shapes = [&[16, 8, 8][..], &[8, 2, 2, 8][..], &[8, 2, 2, 8][..]];
    let env3_pairs = [(0usize, 1usize), (3usize, 2usize)];
    let env3_tree = ContractionTree::from_pairs(&env3_subs, &env3_shapes, &env3_pairs).unwrap();
    let env3_optimized = ContractionTree::optimize(&env3_subs, &env3_shapes).unwrap();
    let env3_a = tensor_from_fn(&[16, 8, 8], 10);
    let env3_b = tensor_from_fn(&[8, 2, 2, 8], 11);
    let env3_c = tensor_from_fn(&[8, 2, 2, 8], 12);

    let env4_subs = Subscripts::new(
        &[&[6, 1, 0, 2], &[7, 0, 3, 4], &[1, 3, 5, 8], &[2, 4, 5]],
        &[6, 7, 8],
    );
    let env4_shapes = [
        &[8, 2, 2, 8][..],
        &[8, 2, 2, 8][..],
        &[2, 2, 16, 16][..],
        &[8, 8, 16][..],
    ];
    let env4_pairs = [(0usize, 3usize), (4usize, 1usize), (5usize, 2usize)];
    let env4_tree = ContractionTree::from_pairs(&env4_subs, &env4_shapes, &env4_pairs).unwrap();
    let env4_optimized = ContractionTree::optimize(&env4_subs, &env4_shapes).unwrap();
    let env4_a = tensor_from_fn(&[8, 2, 2, 8], 20);
    let env4_b = tensor_from_fn(&[8, 2, 2, 8], 21);
    let env4_c = tensor_from_fn(&[2, 2, 16, 16], 22);
    let env4_d = tensor_from_fn(&[8, 8, 16], 23);

    let env3_iters = 4_000usize;
    println!("env3:");
    println!(
        "optimized pairs: {:?} {:?}",
        env3_optimized.step_pair(0),
        env3_optimized.step_pair(1),
    );
    let t0 = Instant::now();
    for _ in 0..env3_iters {
        let _ = ContractionTree::optimize(&env3_subs, &env3_shapes).unwrap();
    }
    print_timing("optimize only", t0.elapsed(), env3_iters);

    let t0 = Instant::now();
    for _ in 0..env3_iters {
        let _ = ContractionTree::from_pairs(&env3_subs, &env3_shapes, &env3_pairs).unwrap();
    }
    print_timing("from_pairs only", t0.elapsed(), env3_iters);

    let mut ctx = CpuContext::new(1);
    let t0 = Instant::now();
    for _ in 0..env3_iters {
        let _ = einsum_with_plan::<Standard<f64>, CpuBackend>(
            &mut ctx,
            &env3_tree,
            &[&env3_a, &env3_b, &env3_c],
            None,
        )
        .unwrap();
    }
    print_timing("generic with plan", t0.elapsed(), env3_iters);

    let mut ctx = CpuContext::new(1);
    let t0 = Instant::now();
    for _ in 0..env3_iters {
        let _ = einsum_with_path::<Standard<f64>, CpuBackend>(
            &mut ctx,
            &env3_subs,
            &env3_pairs,
            &[&env3_a, &env3_b, &env3_c],
            None,
        )
        .unwrap();
    }
    print_timing("generic with path", t0.elapsed(), env3_iters);

    let mut ctx = CpuContext::new(1);
    let t0 = Instant::now();
    for _ in 0..env3_iters {
        let _ = einsum_with_subscripts::<Standard<f64>, CpuBackend>(
            &mut ctx,
            &env3_subs,
            &[&env3_a, &env3_b, &env3_c],
            None,
        )
        .unwrap();
    }
    print_timing("generic subscripts", t0.elapsed(), env3_iters);

    let env3_ab_subs = Subscripts::new(&[&[3, 0, 1], &[0, 4, 2, 5]], &[3, 1, 4, 2, 5]);
    let env3_out_subs = Subscripts::new(&[&[3, 1, 4, 2, 5], &[1, 2, 6, 7]], &[3, 4, 5, 6, 7]);
    let mut ctx = CpuContext::new(1);
    let t0 = Instant::now();
    for _ in 0..env3_iters {
        let xa = einsum_binary_with_subscripts::<Standard<f64>, CpuBackend>(
            &mut ctx,
            &env3_ab_subs,
            &env3_a,
            &env3_b,
            None,
        )
        .unwrap();
        let _ = einsum_binary_with_subscripts::<Standard<f64>, CpuBackend>(
            &mut ctx,
            &env3_out_subs,
            &xa,
            &env3_c,
            None,
        )
        .unwrap();
    }
    print_timing("pairwise binary", t0.elapsed(), env3_iters);

    let env4_iters = 1_000usize;
    println!("env4:");
    println!(
        "optimized pairs: {:?} {:?} {:?}",
        env4_optimized.step_pair(0),
        env4_optimized.step_pair(1),
        env4_optimized.step_pair(2),
    );
    let t0 = Instant::now();
    for _ in 0..env4_iters {
        let _ = ContractionTree::optimize(&env4_subs, &env4_shapes).unwrap();
    }
    print_timing("optimize only", t0.elapsed(), env4_iters);

    let t0 = Instant::now();
    for _ in 0..env4_iters {
        let _ = ContractionTree::from_pairs(&env4_subs, &env4_shapes, &env4_pairs).unwrap();
    }
    print_timing("from_pairs only", t0.elapsed(), env4_iters);

    let mut ctx = CpuContext::new(1);
    let t0 = Instant::now();
    for _ in 0..env4_iters {
        let _ = einsum_with_plan::<Standard<f64>, CpuBackend>(
            &mut ctx,
            &env4_tree,
            &[&env4_a, &env4_b, &env4_c, &env4_d],
            None,
        )
        .unwrap();
    }
    print_timing("generic with plan", t0.elapsed(), env4_iters);

    let mut ctx = CpuContext::new(1);
    let t0 = Instant::now();
    for _ in 0..env4_iters {
        let _ = einsum_with_path::<Standard<f64>, CpuBackend>(
            &mut ctx,
            &env4_subs,
            &env4_pairs,
            &[&env4_a, &env4_b, &env4_c, &env4_d],
            None,
        )
        .unwrap();
    }
    print_timing("generic with path", t0.elapsed(), env4_iters);

    let mut ctx = CpuContext::new(1);
    let t0 = Instant::now();
    for _ in 0..env4_iters {
        let _ = einsum_with_subscripts::<Standard<f64>, CpuBackend>(
            &mut ctx,
            &env4_subs,
            &[&env4_a, &env4_b, &env4_c, &env4_d],
            None,
        )
        .unwrap();
    }
    print_timing("generic subscripts", t0.elapsed(), env4_iters);

    let env4_ad_subs = Subscripts::new(&[&[6, 1, 0, 2], &[2, 4, 5]], &[6, 1, 0, 4, 5]);
    let env4_adb_subs = Subscripts::new(&[&[6, 1, 0, 4, 5], &[7, 0, 3, 4]], &[6, 1, 5, 7, 3]);
    let env4_out_subs = Subscripts::new(&[&[6, 1, 5, 7, 3], &[1, 3, 5, 8]], &[6, 7, 8]);
    let mut ctx = CpuContext::new(1);
    let t0 = Instant::now();
    for _ in 0..env4_iters {
        let ad = einsum_binary_with_subscripts::<Standard<f64>, CpuBackend>(
            &mut ctx,
            &env4_ad_subs,
            &env4_a,
            &env4_d,
            None,
        )
        .unwrap();
        let adb = einsum_binary_with_subscripts::<Standard<f64>, CpuBackend>(
            &mut ctx,
            &env4_adb_subs,
            &ad,
            &env4_b,
            None,
        )
        .unwrap();
        let _ = einsum_binary_with_subscripts::<Standard<f64>, CpuBackend>(
            &mut ctx,
            &env4_out_subs,
            &adb,
            &env4_c,
            None,
        )
        .unwrap();
    }
    print_timing("pairwise binary", t0.elapsed(), env4_iters);
}
