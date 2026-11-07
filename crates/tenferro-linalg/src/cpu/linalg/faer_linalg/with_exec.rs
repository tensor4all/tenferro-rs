//! The typed factorization bodies behind the public borrowed-`Exec` entries (#2462).
//!
//! The public face, its contract and its examples are in `crate::with_exec`; the engine entries
//! call the same functions and only erase the results. Every output, the provider workspace and
//! the transient spectrum of a complex input are leased from the caller's `buffers`.

use tcontract_exec::Exec;
use tenferro_cpu::linalg_interop::BufferPool;
use tenferro_tensor::{TypedTensor, TypedTensorView};
use tlinalg::Op;

use super::{
    checked_product, has_zero_dim, matrix_core_and_batch, matrix_with_batch_shape, pooled_output,
    pooled_workspace, pooled_zeroed, provider, raw_view, square_core_and_batch, tensor_from_lease,
    tensor_from_vec_with_template, vector_with_batch_shape, FaerLinalg,
};

/// `(U, S, Vᴴ)`: `U` and `Vᴴ` in `T`, `S` in its real scalar.
pub(crate) type TypedSvdParts<T> = (
    TypedTensor<T>,
    TypedTensor<<T as FaerLinalg>::RealScalar>,
    TypedTensor<T>,
);

pub(crate) fn svd_typed<T: FaerLinalg>(
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    view: &TypedTensorView<'_, T>,
    full: bool,
) -> tenferro_tensor::Result<TypedSvdParts<T>> {
    let op = if full { "svd_full" } else { "svd" };
    let (m, n, batch_shape) = matrix_core_and_batch(view.shape(), op)?;
    let k = m.min(n);
    let placement = view.placement();
    let (u_cols, v_cols) = if full { (m, n) } else { (k, k) };
    let u_shape = matrix_with_batch_shape(m, u_cols, batch_shape);
    let s_shape = vector_with_batch_shape(k, batch_shape);
    let vt_shape = matrix_with_batch_shape(v_cols, n, batch_shape);
    if has_zero_dim(view.shape()) {
        // The full variant keeps its `m x m` and `n x n` factors even when the other core
        // dimension is zero, so a degenerate factor is not empty: the canonical choice is the
        // identity, which preserves `U Uᴴ = I` and `Vᴴ V = I` and reconstructs the empty input.
        let (u, vt) = if full {
            (
                pooled_identity(buffers, op, "left singular vectors", &u_shape, placement)?,
                pooled_identity(buffers, op, "right singular vectors", &vt_shape, placement)?,
            )
        } else {
            (
                tensor_from_vec_with_template(u_shape, Vec::new(), placement)?,
                tensor_from_vec_with_template(vt_shape, Vec::new(), placement)?,
            )
        };
        let s = tensor_from_vec_with_template(s_shape, Vec::new(), placement)?;
        return Ok((u, s, vt));
    }
    let batch = checked_product(op, "batch shape", batch_shape)?;
    let mut u = pooled_output::<T>(buffers, "svd", "left singular vectors", &u_shape)?;
    // tlinalg 0.10 writes the spectrum in the real scalar (#101), so the lease is the output:
    // no gather of real parts after the call.
    let mut s = pooled_output::<T::RealScalar>(buffers, "svd", "singular values", &s_shape)?;
    let mut vt = pooled_output::<T>(buffers, "svd", "right singular vectors", &vt_shape)?;
    let mut workspace = pooled_workspace::<T>(
        buffers,
        tlinalg::svd::svd_workspace::<T>(m, n, full, batch, exec).map_err(provider(Op::Svd)),
    )?;
    tlinalg::svd::svd_into(
        Op::Svd,
        raw_view(op, view)?,
        full,
        u.as_uninit_slice_mut(),
        s.as_uninit_slice_mut(),
        vt.as_uninit_slice_mut(),
        workspace.bytes_mut(),
        exec,
    )
    .map_err(provider(Op::Svd))?;
    Ok((
        tensor_from_lease(u, placement)?,
        tensor_from_lease(s, placement)?,
        tensor_from_lease(vt, placement)?,
    ))
}

/// A pooled stack of identity matrices of `shape = [dim, dim, b..]`, carrying `placement`.
fn pooled_identity<T: FaerLinalg>(
    buffers: &BufferPool,
    op: &'static str,
    role: &'static str,
    shape: &[usize],
    placement: &tenferro_tensor::Placement,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    let dim = shape[0];
    let mut identity = pooled_zeroed::<T>(buffers, op, role, shape)?;
    let data = identity.host_data_mut()?;
    // `pooled_zeroed` proved the shape product, so `dim * dim` cannot overflow and every index
    // below lies inside its block.
    let per_block = dim * dim;
    if per_block > 0 {
        for block in data.chunks_exact_mut(per_block) {
            for index in 0..dim {
                block[index + index * dim] = T::parity_one();
            }
        }
    }
    identity.set_placement(placement.clone());
    Ok(identity)
}

pub(crate) fn svd_values_typed<T: FaerLinalg>(
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    view: &TypedTensorView<'_, T>,
) -> tenferro_tensor::Result<TypedTensor<T::RealScalar>> {
    const OP: &str = "svd_values";
    let (m, n, batch_shape) = matrix_core_and_batch(view.shape(), OP)?;
    let shape = vector_with_batch_shape(m.min(n), batch_shape);
    if has_zero_dim(view.shape()) {
        return tensor_from_vec_with_template(shape, Vec::new(), view.placement());
    }
    let mut s = pooled_output::<T::RealScalar>(buffers, OP, "singular values", &shape)?;
    let batch = checked_product(OP, "batch shape", batch_shape)?;
    let mut workspace = pooled_workspace::<T>(
        buffers,
        tlinalg::svd::svd_values_workspace::<T>(m, n, batch, exec).map_err(provider(Op::SvdValues)),
    )?;
    T::svd_values_into(
        raw_view(OP, view)?,
        s.as_uninit_slice_mut(),
        workspace.bytes_mut(),
        exec,
    )
    .map_err(provider(Op::SvdValues))?;
    tensor_from_lease(s, view.placement())
}

pub(crate) fn qr_typed<T: FaerLinalg>(
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    view: &TypedTensorView<'_, T>,
) -> tenferro_tensor::Result<(TypedTensor<T>, TypedTensor<T>)> {
    const OP: &str = "qr";
    let (m, n, batch_shape) = matrix_core_and_batch(view.shape(), OP)?;
    let k = m.min(n);
    let placement = view.placement();
    let q_shape = matrix_with_batch_shape(m, k, batch_shape);
    let r_shape = matrix_with_batch_shape(k, n, batch_shape);
    if has_zero_dim(view.shape()) {
        return Ok((
            tensor_from_vec_with_template(q_shape, Vec::new(), placement)?,
            tensor_from_vec_with_template(r_shape, Vec::new(), placement)?,
        ));
    }
    let mut q = pooled_output::<T>(buffers, OP, "Q", &q_shape)?;
    let mut r = pooled_output::<T>(buffers, OP, "R", &r_shape)?;
    let batch = checked_product(OP, "batch shape", batch_shape)?;
    let mut workspace = pooled_workspace::<T>(
        buffers,
        tlinalg::qr::qr_workspace::<T>(m, n, batch, exec).map_err(provider(Op::Qr)),
    )?;
    tlinalg::qr::qr_into(
        Op::Qr,
        raw_view(OP, view)?,
        q.as_uninit_slice_mut(),
        r.as_uninit_slice_mut(),
        workspace.bytes_mut(),
        exec,
    )
    .map_err(provider(Op::Qr))?;
    Ok((
        tensor_from_lease(q, placement)?,
        tensor_from_lease(r, placement)?,
    ))
}

pub(crate) fn eigh_typed<T: FaerLinalg>(
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    view: &TypedTensorView<'_, T>,
) -> tenferro_tensor::Result<(TypedTensor<T::RealScalar>, TypedTensor<T>)> {
    const OP: &str = "eigh";
    let (n, batch_shape) = square_core_and_batch(view.shape(), OP)?;
    let placement = view.placement();
    let values_shape = vector_with_batch_shape(n, batch_shape);
    let vectors_shape = matrix_with_batch_shape(n, n, batch_shape);
    if has_zero_dim(view.shape()) {
        return Ok((
            tensor_from_vec_with_template(values_shape, Vec::new(), placement)?,
            tensor_from_vec_with_template(vectors_shape, Vec::new(), placement)?,
        ));
    }
    let mut values = pooled_output::<T::RealScalar>(buffers, OP, "eigenvalues", &values_shape)?;
    let mut vectors = pooled_output::<T>(buffers, OP, "eigenvectors", &vectors_shape)?;
    let batch = checked_product(OP, "batch shape", batch_shape)?;
    let mut workspace = pooled_workspace::<T>(
        buffers,
        tlinalg::eigh::eigh_workspace::<T>(n, true, batch, exec).map_err(provider(Op::Eigh)),
    )?;
    tlinalg::eigh::eigh_into(
        Op::Eigh,
        raw_view(OP, view)?,
        values.as_uninit_slice_mut(),
        vectors.as_uninit_slice_mut(),
        workspace.bytes_mut(),
        exec,
    )
    .map_err(provider(Op::Eigh))?;
    Ok((
        tensor_from_lease(values, placement)?,
        tensor_from_lease(vectors, placement)?,
    ))
}

pub(crate) fn eigh_values_typed<T: FaerLinalg>(
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    view: &TypedTensorView<'_, T>,
) -> tenferro_tensor::Result<TypedTensor<T::RealScalar>> {
    const OP: &str = "eigh_values";
    let (n, batch_shape) = square_core_and_batch(view.shape(), OP)?;
    let shape = vector_with_batch_shape(n, batch_shape);
    if has_zero_dim(view.shape()) {
        return tensor_from_vec_with_template(shape, Vec::new(), view.placement());
    }
    let mut values = pooled_output::<T::RealScalar>(buffers, OP, "eigenvalues", &shape)?;
    let batch = checked_product(OP, "batch shape", batch_shape)?;
    let mut workspace = pooled_workspace::<T>(
        buffers,
        tlinalg::eigh::eigh_values_workspace::<T>(n, batch, exec).map_err(provider(Op::EighValues)),
    )?;
    T::eigh_values_into(
        raw_view(OP, view)?,
        values.as_uninit_slice_mut(),
        workspace.bytes_mut(),
        exec,
    )
    .map_err(provider(Op::EighValues))?;
    tensor_from_lease(values, view.placement())
}

pub(crate) fn eig_values_typed<T: FaerLinalg>(
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    view: &TypedTensorView<'_, T>,
) -> tenferro_tensor::Result<TypedTensor<T::ComplexScalar>> {
    super::eig_parts(exec, buffers, view, false).map(|(values, _)| values)
}
