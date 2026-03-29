use std::time::Instant;

use tenferro_algebra::Standard;
use tenferro_einsum::{
    einsum_binary_with_subscripts, einsum_with_plan, einsum_with_subscripts, ContractionTree,
    Subscripts,
};
use tenferro_prims::{CpuBackend, CpuContext};
use tenferro_tensor::{MemoryOrder, Tensor};

const COL: MemoryOrder = MemoryOrder::ColumnMajor;

fn mat(data: &[f64], dims: &[usize]) -> Tensor<f64> {
    Tensor::from_slice(data, dims, COL).unwrap()
}

fn assert_same_tensor(label: &str, left: &Tensor<f64>, right: &Tensor<f64>) {
    assert_eq!(left.dims(), right.dims(), "{label}: shape mismatch");
    assert_eq!(left.to_vec(), right.to_vec(), "{label}: value mismatch");
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
fn bench_binary_path_breakdown() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    let shapes = [&[16, 16][..], &[16, 16][..]];
    let tree = ContractionTree::from_pairs(&subs, &shapes, &[(0, 1)]).unwrap();

    let a = mat(
        &(0..256).map(|i| i as f64 * 0.25 + 1.0).collect::<Vec<_>>(),
        &[16, 16],
    );
    let b = mat(
        &(0..256).map(|i| i as f64 * 0.5 - 2.0).collect::<Vec<_>>(),
        &[16, 16],
    );

    let iters = 20_000usize;

    let mut ctx = CpuContext::new(1);
    let expected =
        einsum_with_plan::<Standard<f64>, CpuBackend>(&mut ctx, &tree, &[&a, &b], None).unwrap();
    let mut ctx = CpuContext::new(1);
    let generic =
        einsum_with_subscripts::<Standard<f64>, CpuBackend>(&mut ctx, &subs, &[&a, &b], None)
            .unwrap();
    let mut ctx = CpuContext::new(1);
    let binary =
        einsum_binary_with_subscripts::<Standard<f64>, CpuBackend>(&mut ctx, &subs, &a, &b, None)
            .unwrap();
    assert_same_tensor("with_plan vs generic", &expected, &generic);
    assert_same_tensor("with_plan vs binary", &expected, &binary);

    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = ContractionTree::optimize(&subs, &shapes).unwrap();
    }
    let optimize_only = t0.elapsed();

    let mut ctx = CpuContext::new(1);
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = einsum_with_plan::<Standard<f64>, CpuBackend>(&mut ctx, &tree, &[&a, &b], None)
            .unwrap();
    }
    let generic_with_plan = t0.elapsed();

    let mut ctx = CpuContext::new(1);
    let t0 = Instant::now();
    for _ in 0..iters {
        // Keep the generic two-input n-ary entry point separate so the future
        // strict-capable binary API can be compared against it directly.
        let _ =
            einsum_with_subscripts::<Standard<f64>, CpuBackend>(&mut ctx, &subs, &[&a, &b], None)
                .unwrap();
    }
    let generic_binary_api = t0.elapsed();

    let mut ctx = CpuContext::new(1);
    let t0 = Instant::now();
    for _ in 0..iters {
        // Once the binary API routes through strict lowering, this timing becomes the
        // strict-binary measurement without needing benchmark-structure changes.
        let _ = einsum_binary_with_subscripts::<Standard<f64>, CpuBackend>(
            &mut ctx, &subs, &a, &b, None,
        )
        .unwrap();
    }
    let strict_capable_binary_api = t0.elapsed();

    print_timing("optimize only", optimize_only, iters);
    print_timing("generic with plan", generic_with_plan, iters);
    print_timing("generic binary api", generic_binary_api, iters);
    print_timing("binary api (strict)", strict_capable_binary_api, iters);
}
