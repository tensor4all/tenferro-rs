use tenferro_cpu::linalg_interop::BufferPool;
use tenferro_tensor::TypedTensor;
use tlinalg_blas::Op;

use super::helpers::{
    batch_element_count, has_zero_dim, matrix_core_and_batch_result, matrix_with_batch_shape,
    pooled_output, provider, tensor_from_vec_with_template, vector_with_batch_shape, LapackLinalg,
};
use crate::cpu::linalg::{pooled_copy, pooled_zeroed};

/// Explicit partial-pivot LU, `P A = L U`, of every matrix of a batch.
pub(crate) fn lu<T: LapackLinalg>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>> {
    const OP: &str = "lu";
    let (m, n, batch_shape) = matrix_core_and_batch_result(input, OP)?;
    let k = m.min(n);
    let batch = batch_element_count(OP, batch_shape)?;
    let p_shape = matrix_with_batch_shape(m, m, batch_shape);
    let l_shape = matrix_with_batch_shape(m, k, batch_shape);
    let u_shape = matrix_with_batch_shape(k, n, batch_shape);
    if has_zero_dim(input.shape()) {
        return Ok(vec![
            tensor_from_vec_with_template(p_shape, Vec::new(), input)?,
            tensor_from_vec_with_template(l_shape, Vec::new(), input)?,
            tensor_from_vec_with_template(u_shape, Vec::new(), input)?,
            tensor_from_vec_with_template(batch_shape.to_vec(), vec![T::unit(); batch], input)?,
        ]);
    }
    // ponytail: these four outputs are still raw pooled vectors, so their storage is freed with
    // the tensor instead of returning to the pool, and a batch that outgrows the pool's leftover
    // small bins re-acquires its capacities on every call. `lu_factor` below takes recycling
    // leases instead. No allocation ceiling covers `lu` yet, so it stays recorded here.
    let mut p = pooled_output::<T>(buffers, OP, "permutation matrix", &p_shape)?;
    let mut l = pooled_output::<T>(buffers, OP, "lower factor", &l_shape)?;
    let mut u = pooled_output::<T>(buffers, OP, "upper factor", &u_shape)?;
    let mut parity = buffers.acquire_with_capacity::<T>(batch);
    let view = input.as_view();
    T::lu(
        buffers,
        super::super::raw_view(OP, &view)?,
        tlinalg_blas::lu::LuOutputs {
            p: &mut p,
            l: &mut l,
            u: &mut u,
            parity: &mut parity,
        },
    )
    .map_err(provider(Op::Lu))?;
    Ok(vec![
        tensor_from_vec_with_template(p_shape, p, input)?,
        tensor_from_vec_with_template(l_shape, l, input)?,
        tensor_from_vec_with_template(u_shape, u, input)?,
        tensor_from_vec_with_template(batch_shape.to_vec(), parity, input)?,
    ])
}

pub(crate) fn lu_factor<T: LapackLinalg>(
    ctx: &tenferro_cpu::CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<(TypedTensor<T>, TypedTensor<i32>, TypedTensor<T>)> {
    let (m, n, batch_shape) = matrix_core_and_batch_result(input, "lu_factor")?;
    let k = m.min(n);
    let batch_total = batch_element_count("lu_factor", batch_shape)?;
    if has_zero_dim(input.shape()) {
        return Ok((
            tensor_from_vec_with_template(input.shape().to_vec(), Vec::new(), input)?,
            tensor_from_vec_with_template(
                vector_with_batch_shape(k, batch_shape),
                Vec::new(),
                input,
            )?,
            tensor_from_vec_with_template(
                batch_shape.to_vec(),
                vec![T::unit(); batch_total],
                input,
            )?,
        ));
    }

    // The three outputs are the provider's in-place destination, so they are taken as pool
    // leases rather than raw pooled vectors: a lease carries the pool recycler, so the buffers
    // come back when the caller drops the tensors and the next call's acquire is a pool hit at
    // any batch size. Raw vectors were dropped with the tensors they became, which left the
    // pool without a buffer for any capacity a batch needs and re-allocated that capacity per
    // call once the batch outgrew the leftover small bins (#2206).
    let pivot_shape = vector_with_batch_shape(k, batch_shape);
    let mut lu_data = pooled_copy::<T>(
        buffers,
        "lu_factor",
        "matrix",
        input.shape(),
        input.host_data()?,
    )?;
    let mut pivot_data = pooled_zeroed::<i32>(buffers, "lu_factor", "pivots", &pivot_shape)?;
    let mut parity_data = pooled_zeroed::<T>(buffers, "lu_factor", "parity", batch_shape)?;
    crate::cpu::tlinalg_blas::factor_batch::<T>(
        ctx,
        Op::LuFactor,
        m,
        n,
        lu_data.host_data_mut()?,
        pivot_data.host_data_mut()?,
        parity_data.host_data_mut()?,
    )?;

    let placement = input.placement();
    lu_data.set_placement(placement.clone());
    pivot_data.set_placement(placement.clone());
    parity_data.set_placement(placement.clone());
    Ok((lu_data, pivot_data, parity_data))
}
