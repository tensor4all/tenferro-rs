use tenferro_algebra::Standard;
use tenferro_device::LogicalMemorySpace;
use tenferro_prims::{CpuBackend, CpuContext};
use tenferro_tensor::{MemoryOrder, Tensor};

use super::{einsum_binary, einsum_binary_into};

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
    let data = c.buffer().as_slice().unwrap();
    assert_eq!(
        data[c.offset() as usize..c.offset() as usize + 4],
        [23.0, 34.0, 31.0, 46.0]
    );
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
    let data = out.buffer().as_slice().unwrap();
    assert_eq!(
        data[out.offset() as usize..out.offset() as usize + 4],
        [46.0, 68.0, 62.0, 92.0]
    );
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
