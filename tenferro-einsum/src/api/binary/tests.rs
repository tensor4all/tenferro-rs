use tenferro_algebra::Standard;
use tenferro_device::LogicalMemorySpace;
use tenferro_prims::{CpuBackend, CpuContext};
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::syntax::subscripts::Subscripts;

use super::{
    binary_contraction_tree, einsum_binary, einsum_binary_into,
    execute_binary_with_subscripts_generic_impl, execute_binary_with_subscripts_impl,
    try_execute_strict_binary_with_subscripts_impl,
};

const COL: MemoryOrder = MemoryOrder::ColumnMajor;
const MEM: LogicalMemorySpace = LogicalMemorySpace::MainMemory;

fn mat(data: &[f64], dims: &[usize]) -> Tensor<f64> {
    Tensor::from_slice(data, dims, COL).unwrap()
}

#[test]
fn binary_matmul_matches_expected_values() {
    let mut ctx = CpuContext::new(1);
    let a = mat(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = mat(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
    let c =
        einsum_binary::<Standard<f64>, CpuBackend>(&mut ctx, "ij,jk->ik", &a, &b, None).unwrap();
    assert_eq!(c.to_vec(), vec![23.0, 34.0, 31.0, 46.0]);
}

#[test]
fn binary_into_accumulates_with_alpha_beta() {
    let mut ctx = CpuContext::new(1);
    let a = mat(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = mat(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
    let mut out = Tensor::<f64>::zeros(&[2, 2], MEM, COL).unwrap();
    einsum_binary_into::<Standard<f64>, CpuBackend>(
        &mut ctx,
        "ij,jk->ik",
        &a,
        &b,
        1.0,
        0.0,
        &mut out,
        None,
    )
    .unwrap();
    einsum_binary_into::<Standard<f64>, CpuBackend>(
        &mut ctx,
        "ij,jk->ik",
        &a,
        &b,
        1.0,
        1.0,
        &mut out,
        None,
    )
    .unwrap();
    assert_eq!(out.to_vec(), vec![46.0, 68.0, 62.0, 92.0]);
}

#[test]
fn binary_rejects_non_binary_notation() {
    let mut ctx = CpuContext::new(1);
    let a = Tensor::<f64>::zeros(&[2, 2], MEM, COL).unwrap();
    let b = Tensor::<f64>::zeros(&[2, 2], MEM, COL).unwrap();
    let result = einsum_binary::<Standard<f64>, CpuBackend>(&mut ctx, "ij,jk,kl->il", &a, &b, None);
    match result {
        Ok(_) => panic!("expected binary notation validation error"),
        Err(err) => {
            assert!(format!("{err}").contains("binary einsum requires exactly 2 inputs"))
        }
    }
}

#[test]
fn binary_tree_is_single_fixed_pair() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    let shapes = [&[3, 4][..], &[4, 5][..]];
    let tree = binary_contraction_tree(&subs, &shapes).unwrap();
    assert_eq!(tree.step_count(), 1);
    assert_eq!(tree.step_pair(0), Some((0, 1)));
}

#[test]
fn strict_binary_matches_generic_for_non_identity_output_permutation() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[2, 0]);
    let a = mat(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = mat(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[3, 4],
    );

    let mut strict_ctx = CpuContext::new(1);
    let strict = try_execute_strict_binary_with_subscripts_impl::<Standard<f64>, CpuBackend>(
        &mut strict_ctx,
        &subs,
        &a,
        &b,
        None,
    )
    .unwrap()
    .expect("strict lowering should handle output permutations");

    let mut generic_ctx = CpuContext::new(1);
    let generic = execute_binary_with_subscripts_generic_impl::<Standard<f64>, CpuBackend>(
        &mut generic_ctx,
        &subs,
        &a,
        &b,
        None,
    )
    .unwrap();

    assert_eq!(strict.dims(), generic.dims());
    assert_eq!(strict.to_vec(), generic.to_vec());
}

#[test]
fn strict_binary_matches_generic_for_zero_contraction_extent() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    let a = mat(&[], &[2, 0]);
    let b = mat(&[], &[0, 4]);

    let mut strict_ctx = CpuContext::new(1);
    let strict = try_execute_strict_binary_with_subscripts_impl::<Standard<f64>, CpuBackend>(
        &mut strict_ctx,
        &subs,
        &a,
        &b,
        None,
    )
    .unwrap();
    assert!(
        strict.is_none(),
        "strict lowering should defer zero-extent cases to the generic path"
    );

    let mut routed_ctx = CpuContext::new(1);
    let routed = execute_binary_with_subscripts_impl::<Standard<f64>, CpuBackend>(
        &mut routed_ctx,
        &subs,
        &a,
        &b,
        None,
    )
    .unwrap();

    let mut generic_ctx = CpuContext::new(1);
    let generic = execute_binary_with_subscripts_generic_impl::<Standard<f64>, CpuBackend>(
        &mut generic_ctx,
        &subs,
        &a,
        &b,
        None,
    )
    .unwrap();

    assert_eq!(routed.dims(), generic.dims());
    assert_eq!(routed.to_vec(), generic.to_vec());
}
