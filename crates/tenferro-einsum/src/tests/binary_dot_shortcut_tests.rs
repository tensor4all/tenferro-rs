//! The two-operand shortcut must give the same bits as the contraction-tree
//! path wherever that path ends in the same `dot_general`, and must decline
//! every case the labelled path handles differently.

use num_complex::Complex64;
use tenferro_cpu::CpuEngine;
use tenferro_tensor::{DType, Tensor, TensorRead, TypedTensor};

use crate::eager::{binary_dot_shortcut_plan, eager_einsum_exec, plan_subscripts};
use crate::{ConcreteEinsumPlan, Subscripts, TensorEinsumExt, TypedTensorEinsumExt};

fn lcg(seed: &mut u64) -> f64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5
}

fn f64_tensor(shape: &[usize], seed: u64) -> Tensor {
    let mut s = seed;
    let n: usize = shape.iter().product();
    Tensor::from_vec_col_major(
        shape.to_vec(),
        (0..n).map(|_| lcg(&mut s)).collect::<Vec<_>>(),
    )
    .unwrap()
}

fn c64_tensor(shape: &[usize], seed: u64) -> Tensor {
    let mut s = seed;
    let n: usize = shape.iter().product();
    let data: Vec<Complex64> = (0..n)
        .map(|_| {
            let re = lcg(&mut s);
            Complex64::new(re, lcg(&mut s))
        })
        .collect();
    Tensor::from_vec_col_major(shape.to_vec(), data).unwrap()
}

fn bits(tensor: &Tensor) -> Vec<u64> {
    match tensor.dtype() {
        DType::F64 => tensor
            .as_slice::<f64>()
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect(),
        DType::C64 => tensor
            .as_slice::<Complex64>()
            .unwrap()
            .iter()
            .flat_map(|z| [z.re.to_bits(), z.im.to_bits()])
            .collect(),
        other => panic!("unexpected dtype {other:?}"),
    }
}

/// The pre-shortcut path: plan a contraction tree and walk it.
fn tree_path(notation: &str, a: &Tensor, b: &Tensor) -> tenferro_tensor::Result<Tensor> {
    let subscripts = Subscripts::parse(notation).unwrap();
    let tree = plan_subscripts(&subscripts, &[a.shape(), b.shape()])?;
    let mut engine = CpuEngine::serial();
    eager_einsum_exec(&mut engine, &[a, b], &tree)
}

fn shortcut_applies(notation: &str, a: &Tensor, b: &Tensor) -> bool {
    let subscripts = Subscripts::parse(notation).unwrap();
    let reads = [TensorRead::from_tensor(a), TensorRead::from_tensor(b)];
    binary_dot_shortcut_plan(&reads, &subscripts).is_some()
}

/// Run every public entry that takes the shortcut and compare against the
/// tree path bit for bit. Returns whether the shortcut was taken.
fn check(notation: &str, a: &Tensor, b: &Tensor, expect_shortcut: bool) {
    assert_eq!(
        shortcut_applies(notation, a, b),
        expect_shortcut,
        "shortcut routing for {notation}"
    );
    let old = tree_path(notation, a, b);
    let mut engine = CpuEngine::serial();
    let new = [a, b].einsum(notation, &mut engine);
    match (&old, &new) {
        (Ok(old), Ok(new)) => {
            assert_eq!(old.shape(), new.shape(), "{notation} shape");
            assert_eq!(bits(old), bits(new), "{notation} bits (einsum)");
            let plan = ConcreteEinsumPlan::prepare([a, b], notation).unwrap();
            let planned = plan.execute([a, b], &mut engine).unwrap();
            assert_eq!(bits(old), bits(&planned), "{notation} bits (plan.execute)");
            let reads = [TensorRead::from_tensor(a), TensorRead::from_tensor(b)];
            let read = plan.execute_read(reads, &mut engine).unwrap();
            assert_eq!(
                bits(old),
                bits(&read),
                "{notation} bits (plan.execute_read)"
            );
        }
        (Err(_), Err(_)) => {}
        _ => panic!("{notation}: tree path {old:?} vs einsum {new:?}"),
    }
}

#[test]
fn shortcut_matches_tree_path_bitwise_f64_and_c64() {
    type Make = fn(&[usize], u64) -> Tensor;
    for make in [f64_tensor as Make, c64_tensor as Make] {
        // rank 2
        check("ij,jk->ik", &make(&[2, 2], 1), &make(&[2, 2], 2), true);
        check("ij,jk->ik", &make(&[8, 141], 3), &make(&[141, 40], 4), true);
        check("ij,kj->ik", &make(&[5, 7], 5), &make(&[6, 7], 6), true);
        // rank 3 (one and two contracted labels, and a batch label)
        check("ijk,kl->ijl", &make(&[2, 3, 4], 7), &make(&[4, 5], 8), true);
        check(
            "ijk,jkl->il",
            &make(&[3, 4, 5], 9),
            &make(&[4, 5, 6], 10),
            true,
        );
        check(
            "ijk,klm->ijlm",
            &make(&[2, 3, 4], 11),
            &make(&[4, 5, 6], 12),
            true,
        );
        check(
            "bij,bjk->ikb",
            &make(&[3, 4, 5], 13),
            &make(&[3, 5, 6], 14),
            true,
        );
        // scalar results
        check("ij,ij->", &make(&[3, 4], 15), &make(&[3, 4], 16), true);
        check("i,i->", &make(&[9], 17), &make(&[9], 18), true);
        // output permutation other than lhs-free, rhs-free, batch: declined
        check("ij,jk->ki", &make(&[3, 4], 19), &make(&[4, 5], 20), false);
        check(
            "bij,bjk->bik",
            &make(&[3, 4, 5], 21),
            &make(&[3, 5, 6], 22),
            false,
        );
        check(
            "ijk,kl->lji",
            &make(&[2, 3, 4], 23),
            &make(&[4, 5], 24),
            false,
        );
        // outer product: no contracted label, declined
        check("i,j->ij", &make(&[3], 25), &make(&[4], 26), false);
        check("ij,kl->ijkl", &make(&[2, 3], 27), &make(&[4, 5], 28), false);
        // label summed in one operand only: declined
        check("ij,jk->k", &make(&[3, 4], 29), &make(&[4, 5], 30), false);
        // repeated labels (diagonals): declined
        check("ii,ij->j", &make(&[3, 3], 31), &make(&[3, 4], 32), false);
        check("ij,jj->i", &make(&[3, 4], 33), &make(&[4, 4], 34), false);
        // size-1 broadcasting of a contracted axis: declined (and the paths agree)
        check("ij,jk->ik", &make(&[3, 1], 35), &make(&[4, 5], 36), false);
    }
}

#[test]
fn shortcut_declines_mixed_dtypes_and_agrees_with_tree_path() {
    let a = f64_tensor(&[3, 4], 41);
    let b = c64_tensor(&[4, 5], 42);
    assert!(!shortcut_applies("ij,jk->ik", &a, &b));
    let old = tree_path("ij,jk->ik", &a, &b);
    let mut engine = CpuEngine::serial();
    let new = [&a, &b].einsum("ij,jk->ik", &mut engine);
    match (old, new) {
        (Ok(old), Ok(new)) => assert_eq!(bits(&old), bits(&new)),
        (Err(_), Err(_)) => {}
        (old, new) => panic!("mixed dtype: {old:?} vs {new:?}"),
    }
}

#[test]
fn shortcut_typed_entry_matches_tree_path_bitwise() {
    let a = f64_tensor(&[6, 9], 51);
    let b = f64_tensor(&[9, 4], 52);
    let ta = TypedTensor::<f64>::from_vec_col_major(
        a.shape().to_vec(),
        a.as_slice::<f64>().unwrap().to_vec(),
    )
    .unwrap();
    let tb = TypedTensor::<f64>::from_vec_col_major(
        b.shape().to_vec(),
        b.as_slice::<f64>().unwrap().to_vec(),
    )
    .unwrap();
    let mut engine = CpuEngine::serial();
    let typed = [&ta, &tb].einsum("ij,jk->ik", &mut engine).unwrap();
    let old = tree_path("ij,jk->ik", &a, &b).unwrap();
    let typed_bits: Vec<u64> = typed
        .host_data()
        .unwrap()
        .iter()
        .map(|x| x.to_bits())
        .collect();
    assert_eq!(typed_bits, bits(&old));

    let plan = ConcreteEinsumPlan::prepare_typed([&ta, &tb], "ij,jk->ik").unwrap();
    let planned = plan.execute_typed([&ta, &tb], &mut engine).unwrap();
    let planned_bits: Vec<u64> = planned
        .host_data()
        .unwrap()
        .iter()
        .map(|x| x.to_bits())
        .collect();
    assert_eq!(planned_bits, bits(&old));
}
