// cuSOLVER Xgesvdp integration, written from the NVIDIA API documentation and
// cross-checked against exAClior's issue #1849 harness (MIT-compatible).
// MIT OR Apache-2.0; not derived from NVIDIA's BSD-3 sample implementation.
use super::*;

// Xgesvdp returns V rather than the public SVD tuple's Vᴴ.
type PolarFactors<T> = (
    TypedTensor<T>,
    TypedTensor<<T as LinalgScalar>::Real>,
    TypedTensor<T>,
);

/// Nonempty, validated SVD input. The caller converts V to Vᴴ when requested.
/// No-vector calls leave U/V as scratch, never exposed to the caller.
pub(super) fn svd_polar_typed<T>(
    backend: &mut CudaEngine,
    input: &TypedTensor<T>,
    full: bool,
    vectors: bool,
    op: &'static str,
) -> Result<PolarFactors<T>>
where
    T: LinalgScalar + TensorScalar,
{
    let (m, n) = matrix_dims(op, input.shape())?;
    let k = m.min(n);
    let batch_shape = &input.shape()[2..];
    let batches = batch_count(op, batch_shape)?;
    let u_cols = if full { m } else { k };
    let v_cols = if full { n } else { k };
    let mut u_shape = vec![m, u_cols];
    u_shape.extend_from_slice(batch_shape);
    let mut v_shape = vec![n, v_cols];
    v_shape.extend_from_slice(batch_shape);
    let mut s_shape = vec![k];
    s_shape.extend_from_slice(batch_shape);
    let a_stride = checked_mul_usize(op, "Xgesvdp input stride", m, n)?;
    let u_stride = checked_mul_usize(op, "Xgesvdp U stride", m, u_cols)?;
    let v_stride = checked_mul_usize(op, "Xgesvdp V stride", n, v_cols)?;
    let m_i64 =
        i64::try_from(m).map_err(|_| Error::invalid_argument(op, "m", "dimension exceeds i64"))?;
    let n_i64 =
        i64::try_from(n).map_err(|_| Error::invalid_argument(op, "n", "dimension exceeds i64"))?;
    let jobz = if vectors {
        CusolverEigMode::Vector
    } else {
        CusolverEigMode::NoVector
    };
    let econ = i32::from(!full);

    // Xgesvdp supports both orientations directly; unlike gesvd it needs no
    // wide-matrix adjoint. Its economy factors contain k columns each.
    backend.with_raw(op, |raw| {
        let handles = raw.resource(CudaLinalgHandles::load)?;
        // SAFETY: this stream is used only inside its owning raw session.
        let stream = raw.stream().raw_handle() as usize as CudaStream;
        handles.cusolver().set_stream(stream, op)?;
        let work = raw.alloc_output::<T>(input.shape())?;
        let u = raw.alloc_output::<T>(&u_shape)?;
        let v = raw.alloc_output::<T>(&v_shape)?;
        let s = raw.alloc_output::<<T as LinalgScalar>::Real>(&s_shape)?;
        let info = raw.alloc_output::<i32>(&[batches])?;
        let retained = [
            raw.retain_tensor(input, op)?,
            raw.retain_tensor(&work, op)?,
            raw.retain_tensor(&u, op)?,
            raw.retain_tensor(&v, op)?,
            raw.retain_tensor(&s, op)?,
            raw.retain_tensor(&info, op)?,
        ];
        let src = raw.tensor(input)?;
        let a_ref = raw.tensor_mut(&work)?;
        let u_ref = raw.tensor_mut(&u)?;
        let v_ref = raw.tensor_mut(&v)?;
        let s_ref = raw.tensor_mut(&s)?;
        let info_ref = raw.tensor_mut(&info)?;
        // SAFETY: all spans are validated resident allocations, with exclusive
        // mutable borrows for outputs and the destructive work copy.
        let (a_ptr, u_ptr, v_ptr, s_ptr, info_ptr) = (
            a_ref.raw_ptr(),
            u_ref.raw_ptr(),
            v_ref.raw_ptr(),
            s_ref.raw_ptr(),
            info_ref.raw_ptr(),
        );
        let (device_bytes, host_bytes) = handles.cusolver().gesvdp_buffer_size(
            T::DATA_TYPE,
            jobz,
            econ,
            m_i64,
            n_i64,
            a_ptr.cast_const(),
            s_ptr.cast_const(),
            u_ptr.cast_const(),
            v_ptr.cast_const(),
            op,
        )?;
        let workspace = raw.alloc_bytes(device_bytes.max(1), op)?;
        let mut workspace_ptr = std::ptr::null_mut();
        workspace.with_ptr(|ptr| workspace_ptr = ptr);
        // INVARIANT: the vendor owns scratch initialization. MaybeUninit avoids
        // zero-filling a host buffer that Rust never reads. Each outstanding
        // call gets disjoint host scratch; stream ordering alone only protects
        // reuse of the device workspace.
        let total_host_bytes =
            checked_mul_usize(op, "Xgesvdp host workspace", host_bytes, batches)?;
        let mut host_workspace = Vec::<std::mem::MaybeUninit<u8>>::new();
        host_workspace
            .try_reserve_exact(total_host_bytes)
            .map_err(|error| Error::backend_source(op, error))?;
        let host_ptr = if host_bytes == 0 {
            std::ptr::null_mut()
        } else {
            host_workspace.as_mut_ptr().cast::<c_void>()
        };
        // Heap storage survives even a failed completion barrier. One slot per
        // batch avoids reusing a host output while a prior call is in flight.
        let mut err_sigma = vec![0.0f64; batches];
        let computation = (|| -> Result<()> {
            // SAFETY: source/destination cover the same validated shape and do
            // not alias; retained allocations survive until the barrier below.
            raw.copy_bytes(a_ptr, src.raw_ptr().cast_const(), src.byte_len(), op)?;

            for (batch, sigma) in err_sigma.iter_mut().enumerate() {
                let a_offset = checked_batch_offset(op, "Xgesvdp input offset", batch, a_stride)?;
                let u_offset = checked_batch_offset(op, "Xgesvdp U offset", batch, u_stride)?;
                let v_offset = checked_batch_offset(op, "Xgesvdp V offset", batch, v_stride)?;
                let s_offset = checked_batch_offset(op, "Xgesvdp S offset", batch, k)?;
                let host_offset =
                    checked_batch_offset(op, "Xgesvdp host workspace offset", batch, host_bytes)?;
                // SAFETY: checked strides/offsets cover the allocated batches;
                // workspaces match the query and all host/device buffers remain
                // live through synchronization, including on a vendor error.
                unsafe {
                    handles.cusolver().gesvdp(
                        T::DATA_TYPE,
                        jobz,
                        econ,
                        m_i64,
                        n_i64,
                        batch_ptr::<T>(a_ptr, a_offset),
                        batch_ptr::<<T as LinalgScalar>::Real>(s_ptr, s_offset),
                        batch_ptr::<T>(u_ptr, u_offset),
                        batch_ptr::<T>(v_ptr, v_offset),
                        workspace_ptr,
                        device_bytes,
                        if host_bytes == 0 {
                            host_ptr
                        } else {
                            host_ptr.byte_add(host_offset)
                        },
                        host_bytes,
                        batch_ptr::<i32>(info_ptr, batch).cast(),
                        sigma,
                        op,
                    )?;
                }
            }
            Ok(())
        })();
        if let Err(error) = raw.synchronize() {
            // A failed barrier gives no completion witness. Retain memory rather
            // than permit queued vendor writes into reclaimed allocations.
            std::mem::forget(retained);
            std::mem::forget(workspace);
            std::mem::forget(host_workspace);
            std::mem::forget(err_sigma);
            return Err(error);
        }
        computation?;
        let host_info = raw.download_tensor::<i32>(&info, op)?;
        for &value in host_info.host_data()? {
            check_solver_info(op, "cusolverDnXgesvdp", value)?;
        }
        // `h_err_sigma` is cuSOLVER's absolute bound on the perturbation of the
        // computed singular values. When it is comparable to a matrix's own
        // spectrum that member is not trustworthy; single-precision
        // rank-deficient input makes cuSOLVER report an error as large as the
        // spectrum itself (#2182). Pair each matrix's bound with its own
        // singular values and reject the whole call if any member fails, so a
        // large well-conditioned member cannot mask a bad one.
        // ponytail: downloads the whole spectrum to find each member's max. A
        // device reduction would avoid the copy for large batched `k`, but the
        // SVD already synchronizes here and the copy is small next to it.
        let host_s = raw.download_tensor::<<T as LinalgScalar>::Real>(&s, op)?;
        let host_values = host_s.host_data()?;
        for (batch, &error_bound) in err_sigma.iter().enumerate() {
            let s_max = host_values[batch * k..(batch + 1) * k]
                .iter()
                .fold(0.0_f64, |acc, &value| acc.max(value.into()));
            if !error_bound.is_finite()
                || error_bound > XGESVDP_MAX_RELATIVE_ERROR * s_max.max(f64::MIN_POSITIVE)
            {
                return Err(Error::BackendFailure {
                    op,
                    message: format!(
                        "cusolverDnXgesvdp batch {batch} reported h_err_sigma={error_bound:e} \
                         against a largest singular value of {s_max:e}; the factorization is not \
                         reliable"
                    ),
                });
            }
        }
        // Each member's perturbation is within the bound. This says the
        // singular values are accurate to `XGESVDP_MAX_RELATIVE_ERROR` of the
        // largest one; it does not promise the perturbation is zero, so small
        // singular values (and a pseudoinverse cutoff near them) can still move.
        Ok((u, s, v))
    })
}
