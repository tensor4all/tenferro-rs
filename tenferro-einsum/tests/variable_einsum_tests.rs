use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex};

use chainrules::{AutogradGraph, BackwardOptions, Variable};
use tenferro_algebra::Standard;
use tenferro_device::LogicalMemorySpace;
use tenferro_einsum::{einsum, variable_einsum};
use tenferro_prims::{CpuBackend, CpuContext};
use tenferro_tensor::{MemoryOrder, Tensor};

const COL: MemoryOrder = MemoryOrder::ColumnMajor;
const MEM: LogicalMemorySpace = LogicalMemorySpace::MainMemory;
type S = Standard<f64>;

fn poison_mutex<T>(mutex: &Arc<Mutex<T>>) {
    let mutex = Arc::clone(mutex);
    let _ = catch_unwind(AssertUnwindSafe(move || {
        let _guard = mutex.lock().unwrap();
        panic!("poison backend mutex");
    }));
}

fn get(t: &Tensor<f64>, idx: &[usize]) -> f64 {
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
fn variable_einsum_backward_matmul_loss() {
    let ctx = Arc::new(Mutex::new(CpuContext::new(1)));
    let ad_ctx = AutogradGraph::<Tensor<f64>>::new();

    let a_primal =
        Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], COL).unwrap();
    let b_primal = Tensor::<f64>::from_slice(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[3, 4],
        COL,
    )
    .unwrap();

    let a = Variable::new_in(a_primal.clone(), Arc::clone(&ad_ctx))
        .requires_grad_(true)
        .unwrap();
    let b = Variable::new_in(b_primal.clone(), Arc::clone(&ad_ctx))
        .requires_grad_(true)
        .unwrap();

    let c = variable_einsum::<S, CpuBackend>(ctx.clone(), "ij,jk->ik", &[&a, &b]).unwrap();
    let loss = variable_einsum::<S, CpuBackend>(ctx.clone(), "ij,ij->", &[&c, &c]).unwrap();
    loss.backward(BackwardOptions::default()).unwrap();

    let ga = a.grad().unwrap();
    let gb = b.grad().unwrap();
    assert_eq!(ga.dims(), &[2, 3]);
    assert_eq!(gb.dims(), &[3, 4]);

    let c_val = einsum::<S, CpuBackend>(
        &mut ctx.lock().unwrap(),
        "ij,jk->ik",
        &[&a_primal, &b_primal],
        None,
    )
    .unwrap();
    let two_c = einsum::<S, CpuBackend>(
        &mut ctx.lock().unwrap(),
        "ij,->ij",
        &[
            &c_val,
            &Tensor::<f64>::from_slice(&[2.0], &[], COL).unwrap(),
        ],
        None,
    )
    .unwrap();
    let expected_ga = einsum::<S, CpuBackend>(
        &mut ctx.lock().unwrap(),
        "ik,jk->ij",
        &[&two_c, &b_primal],
        None,
    )
    .unwrap();
    let expected_gb = einsum::<S, CpuBackend>(
        &mut ctx.lock().unwrap(),
        "ij,ik->jk",
        &[&a_primal, &two_c],
        None,
    )
    .unwrap();

    for i in 0..2 {
        for j in 0..3 {
            assert!((get(&ga, &[i, j]) - get(&expected_ga, &[i, j])).abs() < 1e-10);
        }
    }
    for i in 0..3 {
        for j in 0..4 {
            assert!((get(&gb, &[i, j]) - get(&expected_gb, &[i, j])).abs() < 1e-10);
        }
    }
}

#[test]
fn variable_einsum_forward_tangent_propagates() {
    let ctx = Arc::new(Mutex::new(CpuContext::new(1)));

    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], COL).unwrap();
    let da = Tensor::<f64>::ones(&[2, 2], MEM, COL);
    let b = Tensor::<f64>::from_slice(&[2.0, 0.0, 1.0, 3.0], &[2, 2], COL).unwrap();

    let a_var = Variable::new(a.clone()).with_tangent_(da.clone()).unwrap();
    let b_var = Variable::new(b.clone());

    let out =
        variable_einsum::<S, CpuBackend>(ctx.clone(), "ij,jk->ik", &[&a_var, &b_var]).unwrap();
    let tangent = out.tangent().expect("expected tangent");

    let expected =
        einsum::<S, CpuBackend>(&mut ctx.lock().unwrap(), "ij,jk->ik", &[&da, &b], None).unwrap();

    for i in 0..2 {
        for j in 0..2 {
            assert!((get(tangent, &[i, j]) - get(&expected, &[i, j])).abs() < 1e-10);
        }
    }
}

#[test]
fn variable_einsum_hvp_runs_with_tangent_seeded_leaf() {
    let ctx = Arc::new(Mutex::new(CpuContext::new(1)));
    let ad_ctx = AutogradGraph::<Tensor<f64>>::new();

    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], COL).unwrap();
    let b = Tensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], COL).unwrap();
    let da = Tensor::<f64>::ones(&[2, 2], MEM, COL);

    let a_var = Variable::new_in(a, Arc::clone(&ad_ctx))
        .requires_grad_(true)
        .unwrap()
        .with_tangent_(da)
        .unwrap();
    let b_var = Variable::new_in(b, Arc::clone(&ad_ctx))
        .requires_grad_(true)
        .unwrap();

    let out =
        variable_einsum::<S, CpuBackend>(ctx.clone(), "ij,jk->ik", &[&a_var, &b_var]).unwrap();
    let loss = variable_einsum::<S, CpuBackend>(ctx.clone(), "ij,ij->", &[&out, &out]).unwrap();

    loss.backward_hvp(BackwardOptions::default()).unwrap();
    assert!(a_var.grad().is_some());
    assert!(a_var.hvp().is_some());
}

#[test]
fn variable_einsum_rejects_invalid_subscripts() {
    let ctx = Arc::new(Mutex::new(CpuContext::new(1)));
    let a = Variable::new(Tensor::<f64>::ones(&[2, 2], MEM, COL));
    let b = Variable::new(Tensor::<f64>::ones(&[2, 2], MEM, COL));

    let err = variable_einsum::<S, CpuBackend>(ctx, "ij,jk", &[&a, &b])
        .err()
        .unwrap();
    assert!(matches!(err, chainrules::AutodiffError::InvalidArgument(msg) if msg.contains("->")));
}

#[test]
fn variable_einsum_rejects_poisoned_backend_context() {
    let ctx = Arc::new(Mutex::new(CpuContext::new(1)));
    let a = Variable::new(Tensor::<f64>::ones(&[2, 2], MEM, COL));
    let b = Variable::new(Tensor::<f64>::ones(&[2, 2], MEM, COL));

    poison_mutex(&ctx);

    let err = variable_einsum::<S, CpuBackend>(ctx, "ij,jk->ik", &[&a, &b])
        .err()
        .unwrap();
    assert!(
        matches!(err, chainrules::AutodiffError::InvalidArgument(msg) if msg.contains("poisoned"))
    );
}
