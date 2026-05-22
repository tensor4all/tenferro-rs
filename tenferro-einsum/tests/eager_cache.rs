use std::num::NonZeroUsize;

use tenferro_einsum::{
    clear_eager_einsum_cache, eager_einsum, eager_einsum_cache_capacity, eager_einsum_cache_stats,
    set_eager_einsum_cache_capacity, DEFAULT_EAGER_EINSUM_CACHE_CAPACITY,
};
use tenferro_tensor::{cpu::CpuBackend, Tensor};

fn run_three_input_einsum(ctx: &mut CpuBackend, mid: usize) {
    let a = Tensor::from_vec_col_major(vec![2, mid], vec![1.0_f64; 2 * mid]);
    let b = Tensor::from_vec_col_major(vec![mid, 3], vec![1.0_f64; mid * 3]);
    let c = Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64; 6]);
    let out = eager_einsum(ctx, &[&a, &b, &c], "ij,jk,kl->il").expect("eager einsum");
    assert_eq!(out.shape(), &[2, 2]);
}

#[test]
fn eager_einsum_cache_is_bounded_and_reports_stats() {
    clear_eager_einsum_cache();
    set_eager_einsum_cache_capacity(NonZeroUsize::new(1).unwrap());
    let mut ctx = CpuBackend::new();

    run_three_input_einsum(&mut ctx, 3);
    run_three_input_einsum(&mut ctx, 4);

    assert_eq!(eager_einsum_cache_capacity().get(), 1);
    let stats = eager_einsum_cache_stats();
    assert_eq!(stats.entries, 1);
    assert!(stats.retained_bytes > 0);

    clear_eager_einsum_cache();
    set_eager_einsum_cache_capacity(
        NonZeroUsize::new(DEFAULT_EAGER_EINSUM_CACHE_CAPACITY).unwrap(),
    );
}
