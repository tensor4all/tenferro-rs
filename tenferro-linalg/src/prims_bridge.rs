use std::cell::RefCell;

use tenferro_algebra::Standard;
use tenferro_device::{Error, LogicalMemorySpace, Result};
use tenferro_prims::{CpuBackend, CpuContext, SemiringCoreDescriptor, TensorSemiringCore};
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::LinalgScalar;

thread_local! {
    static PRIMS_CTX: RefCell<CpuContext> = RefCell::new(CpuContext::new(1));
}

pub(crate) fn batched_gemm_via_prims<T>(
    a: &[T],
    m: usize,
    k: usize,
    b: &[T],
    n: usize,
) -> Result<Vec<T>>
where
    T: LinalgScalar,
{
    PRIMS_CTX.with(|ctx_cell| {
        let mut ctx = ctx_cell.borrow_mut();
        batched_gemm_with_semiring_core::<T, CpuBackend>(&mut ctx, a, m, k, b, n)
    })
}

pub(crate) fn batched_gemm_with_semiring_core<T, Backend>(
    ctx: &mut <Backend as TensorSemiringCore<Standard<T>>>::Context,
    a: &[T],
    m: usize,
    k: usize,
    b: &[T],
    n: usize,
) -> Result<Vec<T>>
where
    T: LinalgScalar,
    Backend: TensorSemiringCore<Standard<T>>,
{
    let a_shape = [m, k];
    let b_shape = [k, n];
    let c_shape = [m, n];

    let a_strides = [1isize, m as isize];
    let b_strides = [1isize, k as isize];
    let a_tensor = Tensor::from_vec(a.to_vec(), &a_shape, &a_strides, 0)?;
    let b_tensor = Tensor::from_vec(b.to_vec(), &b_shape, &b_strides, 0)?;
    let mut c_tensor = Tensor::zeros(
        &c_shape,
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    );

    let desc = SemiringCoreDescriptor::BatchedGemm {
        batch_dims: vec![],
        m,
        n,
        k,
    };
    let plan = <Backend as TensorSemiringCore<Standard<T>>>::plan(
        ctx,
        &desc,
        &[&a_shape, &b_shape, &c_shape],
    )?;
    <Backend as TensorSemiringCore<Standard<T>>>::execute(
        ctx,
        &plan,
        T::one(),
        &[&a_tensor, &b_tensor],
        T::zero(),
        &mut c_tensor,
    )?;

    c_tensor
        .try_into_data_vec()
        .ok_or_else(|| Error::DeviceError("expected owned CPU output tensor".into()))
}

#[cfg(test)]
mod tests;
