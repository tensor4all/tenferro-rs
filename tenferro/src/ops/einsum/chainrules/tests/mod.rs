use super::*;
use std::sync::{Arc, Mutex};
use tenferro_device::LogicalMemorySpace;
use tenferro_prims::CpuBackend;
use tenferro_prims::CpuContext;
use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};

fn get(t: &DenseTensor<f64>, idx: &[usize]) -> f64 {
    let data = t.buffer().as_slice().unwrap();
    let pos = t.offset()
        + idx
            .iter()
            .zip(t.strides())
            .map(|(&i, &s)| i as isize * s)
            .sum::<isize>();
    data[pos as usize]
}

#[test]
fn variable_einsum_backward_and_hvp_flow() {
    let runtime_ctx = Arc::new(Mutex::new(CpuContext::new(1)));
    let ad_ctx = context::<f64>();

    let a =
        DenseTensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap();
    let b =
        DenseTensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap();

    let da = DenseTensor::<f64>::ones(
        &[2, 2],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    );

    let a = leaf_in(a, Arc::clone(&ad_ctx), true)
        .unwrap()
        .with_tangent_(da)
        .unwrap();
    let b = leaf_in(b, Arc::clone(&ad_ctx), true).unwrap();

    let y = einsum::<f64, CpuBackend>(runtime_ctx.clone(), "ij,jk->ik", &[&a, &b]).unwrap();
    let loss = einsum::<f64, CpuBackend>(runtime_ctx.clone(), "ij,ij->", &[&y, &y]).unwrap();

    backward(
        &loss,
        BackwardOptions {
            retain_graph: Some(true),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(a.grad().is_some());
    assert!(b.grad().is_some());

    a.zero_grad().unwrap();
    b.zero_grad().unwrap();

    backward_hvp(&loss, BackwardOptions::default()).unwrap();
    assert!(a.grad().is_some());
    assert!(a.hvp().is_some());
}

#[test]
fn grad_tangent_is_side_effect_free_and_zeros_non_grad_leaf() {
    let runtime_ctx = Arc::new(Mutex::new(CpuContext::new(1)));
    let ad_ctx = context::<f64>();

    let a =
        DenseTensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap();
    let b =
        DenseTensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap();

    let a = leaf_in(a, Arc::clone(&ad_ctx), true).unwrap();
    let b = leaf_in(b, Arc::clone(&ad_ctx), false).unwrap();

    assert!(a.requires_grad());
    assert!(!b.requires_grad());
    assert!(b.node_id().is_none());

    let y = einsum::<f64, CpuBackend>(runtime_ctx.clone(), "ij,jk->ik", &[&a, &b]).unwrap();
    let loss = einsum::<f64, CpuBackend>(runtime_ctx.clone(), "ij,ij->", &[&y, &y]).unwrap();

    let grads = grad_tangent(&loss, &[&a, &b], BackwardOptions::default()).unwrap();

    assert_eq!(grads.len(), 2);
    assert!(a.grad().is_none());
    assert!(b.grad().is_none());

    for i in 0..2 {
        for j in 0..2 {
            assert!((get(&grads[0], &[i, j]) - 2.0 * get(a.value(), &[i, j])).abs() < 1e-10);
            assert_eq!(get(&grads[1], &[i, j]), 0.0);
        }
    }
}
