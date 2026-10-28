use super::*;

fn initialize_q_columns_typed<T>(
    backend: &mut CudaEngine,
    rows: usize,
    start: usize,
    end: usize,
    op: &'static str,
) -> Result<TypedTensor<T>>
where
    T: LinalgScalar + TensorScalar,
{
    let width = end
        .checked_sub(start)
        .ok_or_else(|| Error::invalid_argument(op, "range", "start exceeds end"))?;
    let output = backend.alloc_output(T::dtype(), &[rows, width])?;
    if rows != 0 && width != 0 {
        // An identity block whose diagonal starts at row `start`.
        backend.linalg_identity_shifted(&output, start as i64, op)?;
    }
    output
        .into_typed::<T>()
        .map_err(|_| Error::Internal("Q initialisation changed dtype".into()))
}

fn build_explicit_v<T>(
    backend: &mut CudaEngine,
    packed: &TypedTensor<T>,
    columns: usize,
    op: &'static str,
) -> Result<TypedTensor<T>>
where
    T: LinalgScalar + TensorScalar,
{
    let output = backend.alloc_output(T::dtype(), &[packed.shape()[0], columns])?;
    if output.shape().iter().all(|&extent| extent != 0) {
        backend.linalg_householder_explicit_v(packed, &output, op)?;
    }
    output
        .into_typed::<T>()
        .map_err(|_| Error::Internal("explicit Householder vector changed dtype".into()))
}

fn linalg_scalar_bytes<T: LinalgScalar>(value: &T) -> &[u8] {
    // SAFETY: LinalgScalar is sealed in this module to f32/f64/Complex32/
    // Complex64, all of which have initialized byte representations without
    // padding; the returned slice cannot outlive `value`.
    unsafe {
        std::slice::from_raw_parts(
            std::ptr::from_ref(value).cast::<u8>(),
            std::mem::size_of::<T>(),
        )
    }
}

fn apply_householder_reflectors_typed<T>(
    backend: &mut CudaEngine,
    packed: &TypedTensor<T>,
    coeff: &TypedTensor<T>,
    target: &mut TypedTensor<T>,
    adjoint: bool,
    op: &'static str,
) -> Result<()>
where
    T: LinalgScalar + TensorScalar,
{
    let (m, _, k) = compact_qr_state_dims(packed, coeff, op)?;
    ensure_device_resident_typed(op, target)?;
    if target.shape().len() != 2 || target.shape()[0] != m {
        return Err(Error::shape_mismatch(
            op,
            vec![m, target.shape().get(1).copied().unwrap_or(0)],
            target.shape().to_vec(),
        ));
    }
    let columns = target.shape()[1];
    if k == 0 || columns == 0 {
        return Ok(());
    }
    // The explicit reflector vectors are device-local function scratch, not Q.
    // `with_raw` flushes this CubeCL build before the vendor calls below.
    let v = build_explicit_v(backend, packed, k, op)?;
    let m_i32 = as_i32(m, op, "m")?;
    let k_i32 = as_i32(k, op, "k")?;
    let columns_i32 = as_i32(columns, op, "columns")?;
    let m_i64 =
        i64::try_from(m).map_err(|_| Error::invalid_argument(op, "m", "dimension exceeds i64"))?;
    let k_i64 =
        i64::try_from(k).map_err(|_| Error::invalid_argument(op, "k", "dimension exceeds i64"))?;
    let one = T::one();
    let zero = T::zero();
    let minus_one = -one;

    backend.with_raw(op, |raw| {
        let handles = raw.resource(CudaLinalgHandles::load)?;
        // SAFETY: stream handle is valid for this raw-session scope.
        let stream = raw.stream().raw_handle() as usize as CudaStream;
        handles.cusolver().set_stream(stream, op)?;
        handles.cublas().set_stream(stream, op)?;

        let one_device = raw.upload_bytes(linalg_scalar_bytes(&one), op)?;
        let zero_device = raw.upload_bytes(linalg_scalar_bytes(&zero), op)?;
        let minus_one_device = raw.upload_bytes(linalg_scalar_bytes(&minus_one), op)?;
        let mut one_ptr = std::ptr::null_mut::<c_void>();
        let mut zero_ptr = std::ptr::null_mut::<c_void>();
        let mut minus_one_ptr = std::ptr::null_mut::<c_void>();
        one_device.with_ptr(|ptr| one_ptr = ptr);
        zero_device.with_ptr(|ptr| zero_ptr = ptr);
        minus_one_device.with_ptr(|ptr| minus_one_ptr = ptr);

        let t = raw.alloc_output::<T>(&[k, k])?;
        let w = raw.alloc_output::<T>(&[k, columns])?;
        let w2 = raw.alloc_output::<T>(&[k, columns])?;
        let coeff_ref = raw.tensor(coeff)?;
        let target_ref = raw.tensor_mut(target)?;
        let v_ref = raw.tensor(&v)?;
        let t_ref = raw.tensor_mut(&t)?;
        let w_ref = raw.tensor_mut(&w)?;
        let w2_ref = raw.tensor_mut(&w2)?;
        // SAFETY: every handle is a validated live device span in this raw
        // session; pointers are retained only for stream-ordered calls below.
        let (coeff_ptr, target_ptr, v_ptr, t_ptr, w_ptr, w2_ptr) = (
            coeff_ref.raw_ptr().cast_const(),
            target_ref.raw_ptr(),
            v_ref.raw_ptr().cast_const(),
            t_ref.raw_ptr(),
            w_ref.raw_ptr(),
            w2_ref.raw_ptr(),
        );
        let (device_workspace_bytes, host_workspace_bytes) = handles.cusolver().larft_buffer_size(
            T::DATA_TYPE,
            m_i64,
            k_i64,
            v_ptr,
            m_i64,
            coeff_ptr,
            t_ptr,
            k_i64,
            op,
        )?;
        let device_workspace = (device_workspace_bytes > 0)
            .then(|| raw.alloc_bytes(device_workspace_bytes, op))
            .transpose()?;
        let mut device_workspace_ptr = std::ptr::null_mut::<c_void>();
        if let Some(workspace) = &device_workspace {
            workspace.with_ptr(|ptr| device_workspace_ptr = ptr);
        }
        let mut host_workspace = Vec::new();
        host_workspace
            .try_reserve_exact(host_workspace_bytes)
            .map_err(|error| Error::backend_source(op, error))?;
        host_workspace.resize(host_workspace_bytes, 0u8);
        let host_workspace_ptr = if host_workspace.is_empty() {
            std::ptr::null_mut()
        } else {
            host_workspace.as_mut_ptr().cast::<c_void>()
        };
        // SAFETY: V, tau, T, and both workspaces are live and match the sizes
        // accepted by the preceding Xlarft workspace query.
        unsafe {
            handles.cusolver().larft(
                T::DATA_TYPE,
                m_i64,
                k_i64,
                v_ptr,
                m_i64,
                coeff_ptr,
                t_ptr,
                k_i64,
                device_workspace_ptr,
                device_workspace_bytes,
                host_workspace_ptr,
                host_workspace_bytes,
                op,
            )?;

            handles
                .cublas()
                .set_pointer_mode(CublasPointerMode::Device, op)?;
            let computation = (|| -> Result<()> {
                let adjoint_v = match T::DATA_TYPE {
                    CudaDataType::F32 | CudaDataType::F64 => CublasOperation::T,
                    CudaDataType::Complex32 | CudaDataType::Complex64 => CublasOperation::C,
                };
                let apply_t = if adjoint {
                    adjoint_v
                } else {
                    CublasOperation::N
                };
                // SAFETY: dimensions and leading dimensions describe live
                // column-major V, T, W, W2, and target device allocations.
                handles.cublas().gemm(
                    T::DATA_TYPE,
                    adjoint_v,
                    CublasOperation::N,
                    k_i32,
                    columns_i32,
                    m_i32,
                    one_ptr.cast_const(),
                    v_ptr,
                    m_i32,
                    target_ptr.cast_const(),
                    m_i32,
                    zero_ptr.cast_const(),
                    w_ptr,
                    k_i32,
                    op,
                )?;
                handles.cublas().gemm(
                    T::DATA_TYPE,
                    apply_t,
                    CublasOperation::N,
                    k_i32,
                    columns_i32,
                    k_i32,
                    one_ptr.cast_const(),
                    t_ptr.cast_const(),
                    k_i32,
                    w_ptr.cast_const(),
                    k_i32,
                    zero_ptr.cast_const(),
                    w2_ptr,
                    k_i32,
                    op,
                )?;
                handles.cublas().gemm(
                    T::DATA_TYPE,
                    CublasOperation::N,
                    CublasOperation::N,
                    m_i32,
                    columns_i32,
                    k_i32,
                    minus_one_ptr.cast_const(),
                    v_ptr,
                    m_i32,
                    w2_ptr.cast_const(),
                    k_i32,
                    one_ptr.cast_const(),
                    target_ptr,
                    m_i32,
                    op,
                )?;

                Ok(())
            })();
            let reset = handles
                .cublas()
                .set_pointer_mode(CublasPointerMode::Host, op);
            computation?;
            reset
        }
    })
}

fn geqrf_trailing_typed<T>(
    backend: &mut CudaEngine,
    matrix: &mut TypedTensor<T>,
    row_offset: usize,
    op: &'static str,
) -> Result<TypedTensor<T>>
where
    T: LinalgScalar + TensorScalar,
{
    ensure_device_resident_typed(op, matrix)?;
    if matrix.shape().len() != 2 || row_offset > matrix.shape()[0] {
        return Err(Error::invalid_argument(
            op,
            "trailing block",
            "invalid rank or row offset",
        ));
    }
    let rows = matrix.shape()[0] - row_offset;
    let cols = matrix.shape()[1];
    let k = rows.min(cols);
    if rows == 0 || cols == 0 {
        return backend.with_raw(op, |raw| raw.alloc_output::<T>(&[k]));
    }
    let rows_i32 = as_i32(rows, op, "trailing rows")?;
    let cols_i32 = as_i32(cols, op, "trailing columns")?;
    let lda = as_i32(matrix.shape()[0], op, "lda")?;
    backend.with_raw(op, |raw| {
        let handles = raw.resource(CudaLinalgHandles::load)?;
        // SAFETY: stream handle is valid for this raw-session scope.
        let stream = raw.stream().raw_handle() as usize as CudaStream;
        handles.cusolver().set_stream(stream, op)?;
        let matrix_ref = raw.tensor_mut(matrix)?;
        // SAFETY: row_offset is validated inside the first matrix column.
        let matrix_ptr = unsafe { batch_ptr::<T>(matrix_ref.raw_ptr(), row_offset) };
        let lwork = handles.cusolver().geqrf_buffer_size(
            T::DATA_TYPE,
            rows_i32,
            cols_i32,
            matrix_ptr,
            lda,
            op,
        )?;
        let workspace_bytes = usize::try_from(lwork)
            .map_err(|_| Error::invalid_argument(op, "workspace", "negative length"))?
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| Error::invalid_argument(op, "workspace", "byte size overflowed"))?;
        let workspace = raw.alloc_bytes(workspace_bytes, op)?;
        let mut workspace_ptr = std::ptr::null_mut::<c_void>();
        workspace.with_ptr(|ptr| workspace_ptr = ptr);
        let coeff = raw.alloc_output::<T>(&[k])?;
        let info = raw.alloc_output::<i32>(&[1])?;
        let coeff_ref = raw.tensor_mut(&coeff)?;
        let info_ref = raw.tensor_mut(&info)?;
        // SAFETY: matrix/tau/workspace/info pointers are live device spans and
        // dimensions describe the validated trailing submatrix with lda=m.
        unsafe {
            handles.cusolver().geqrf(
                T::DATA_TYPE,
                rows_i32,
                cols_i32,
                matrix_ptr,
                lda,
                coeff_ref.raw_ptr(),
                workspace_ptr,
                lwork,
                info_ref.raw_ptr().cast::<i32>(),
                op,
            )?;
        }
        // ponytail: the doc-hidden compact QR hooks keep the blocking geqrf
        // status read; the public `qr` family skips it.
        let host_info = raw.download_tensor::<i32>(&info, op)?;
        check_solver_info(op, "cusolverDn*geqrf", host_info.host_data()?[0])?;
        Ok(coeff)
    })
}

fn concatenate_compact_typed<T>(
    backend: &mut CudaEngine,
    packed: &TypedTensor<T>,
    block: &TypedTensor<T>,
    coeff: &TypedTensor<T>,
    trailing_coeff: &TypedTensor<T>,
    op: &'static str,
) -> Result<(TypedTensor<T>, TypedTensor<T>)>
where
    T: LinalgScalar + TensorScalar,
{
    let rows = packed.shape()[0];
    let width = packed.shape()[1]
        .checked_add(block.shape()[1])
        .ok_or_else(|| Error::invalid_argument(op, "shape", "column count overflowed"))?;
    let coeff_len = coeff.shape()[0]
        .checked_add(trailing_coeff.shape()[0])
        .ok_or_else(|| Error::invalid_argument(op, "shape", "coefficient count overflowed"))?;
    backend.with_raw(op, |raw| {
        let output = raw.alloc_output::<T>(&[rows, width])?;
        let output_coeff = raw.alloc_output::<T>(&[coeff_len])?;
        let packed_ref = raw.tensor(packed)?;
        let block_ref = raw.tensor(block)?;
        let coeff_ref = raw.tensor(coeff)?;
        let trailing_ref = raw.tensor(trailing_coeff)?;
        let output_ref = raw.tensor_mut(&output)?;
        let output_coeff_ref = raw.tensor_mut(&output_coeff)?;
        // SAFETY: output layouts concatenate two validated compact column-major
        // matrices/vectors, all offsets and byte lengths stay in live spans.
        // SAFETY: both pointers come from device spans of tensors this call owns, and the byte
        // length is the source's own length, so the destination covers it.
        unsafe {
            raw.copy_bytes(
                output_ref.raw_ptr(),
                packed_ref.raw_ptr() as *const _,
                packed_ref.byte_len(),
                op,
            )?
        };
        // SAFETY: the offset is inside the allocation this call owns: the second factor starts
        // exactly after the first, and `block_ref` is no longer than the remaining bytes.
        unsafe {
            raw.copy_bytes(
                output_ref
                    .raw_ptr()
                    .cast::<u8>()
                    .add(packed_ref.byte_len())
                    .cast(),
                block_ref.raw_ptr() as *const _,
                block_ref.byte_len(),
                op,
            )?
        };
        // SAFETY: both pointers come from device spans of tensors this call owns, and the byte
        // length is the source's own length.
        unsafe {
            raw.copy_bytes(
                output_coeff_ref.raw_ptr(),
                coeff_ref.raw_ptr() as *const _,
                coeff_ref.byte_len(),
                op,
            )?
        };
        // SAFETY: the offset is inside the allocation this call owns: the coefficient block
        // starts exactly after the packed factor.
        unsafe {
            raw.copy_bytes(
                output_coeff_ref
                    .raw_ptr()
                    .cast::<u8>()
                    .add(coeff_ref.byte_len())
                    .cast(),
                trailing_ref.raw_ptr() as *const _,
                trailing_ref.byte_len(),
                op,
            )?
        };

        Ok((output, output_coeff))
    })
}

fn gemm_nn_typed<T>(
    backend: &mut CudaEngine,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    op: &'static str,
) -> Result<TypedTensor<T>>
where
    T: LinalgScalar + TensorScalar,
{
    ensure_device_resident_typed(op, lhs)?;
    ensure_device_resident_typed(op, rhs)?;
    if lhs.shape().len() != 2 || rhs.shape().len() != 2 || lhs.shape()[1] != rhs.shape()[0] {
        return Err(Error::invalid_argument(
            op,
            "shape",
            "invalid GEMM operands",
        ));
    }
    let m = lhs.shape()[0];
    let n = rhs.shape()[1];
    let k = lhs.shape()[1];
    if m == 0 || n == 0 || k == 0 {
        return backend.with_raw(op, |raw| raw.alloc_output::<T>(&[m, n]));
    }
    let m_i32 = as_i32(m, op, "gemm m")?;
    let n_i32 = as_i32(n, op, "gemm n")?;
    let k_i32 = as_i32(k, op, "gemm k")?;
    let alpha = T::one();
    let beta = T::zero();
    backend.with_raw(op, |raw| {
        let handles = raw.resource(CudaLinalgHandles::load)?;
        // SAFETY: stream handle is valid for this raw-session scope.
        let stream = raw.stream().raw_handle() as usize as CudaStream;
        handles.cublas().set_stream(stream, op)?;
        handles
            .cublas()
            .set_pointer_mode(CublasPointerMode::Host, op)?;
        let lhs_ref = raw.tensor(lhs)?;
        let rhs_ref = raw.tensor(rhs)?;
        let output = raw.alloc_output::<T>(&[m, n])?;
        let output_ref = raw.tensor_mut(&output)?;
        // SAFETY: all tensors are validated compact column-major device
        // matrices and dimensions/leading dimensions match their shapes.
        unsafe {
            handles.cublas().gemm(
                T::DATA_TYPE,
                CublasOperation::N,
                CublasOperation::N,
                m_i32,
                n_i32,
                k_i32,
                std::ptr::from_ref(&alpha).cast(),
                lhs_ref.raw_ptr() as *const _,
                m_i32,
                rhs_ref.raw_ptr() as *const _,
                k_i32,
                std::ptr::from_ref(&beta).cast(),
                output_ref.raw_ptr(),
                m_i32,
                op,
            )?;
        }
        Ok(output)
    })
}

fn assemble_from_factors_typed<T>(
    backend: &mut CudaEngine,
    packed_q: &TypedTensor<T>,
    coeff_q: &TypedTensor<T>,
    folded_r: &TypedTensor<T>,
    output_cols: usize,
    op: &'static str,
) -> Result<(TypedTensor<T>, TypedTensor<T>)>
where
    T: LinalgScalar + TensorScalar,
{
    let rows = packed_q.shape()[0];
    let factor_width = packed_q.shape()[1];
    let output_k = rows.min(output_cols);
    let dtype = T::dtype();
    let packed = backend.alloc_output(dtype, &[rows, output_cols])?;
    let coeff = backend.alloc_output(dtype, &[output_k])?;
    if !has_zero_dim(packed_q.shape()) {
        backend.linalg_householder_assemble(
            &packed,
            &coeff,
            packed_q,
            coeff_q,
            folded_r,
            factor_width as i64,
            op,
        )?;
    }
    let converted = |tensor: Tensor, what: &str| {
        tensor
            .into_typed::<T>()
            .map_err(|_| Error::Internal(format!("factor assembly changed the {what} dtype")))
    };
    Ok((
        converted(packed, "packed")?,
        converted(coeff, "coefficient")?,
    ))
}

pub(super) fn compact_qr_from_factors_typed<T>(
    backend: &mut CudaEngine,
    q: &TypedTensor<T>,
    r: &TypedTensor<T>,
    op: &'static str,
) -> Result<(TypedTensor<T>, TypedTensor<T>)>
where
    T: LinalgScalar + TensorScalar,
{
    ensure_device_resident_typed(op, q)?;
    ensure_device_resident_typed(op, r)?;
    if q.shape().len() != 2
        || r.shape().len() != 2
        || q.shape()[1] != r.shape()[0]
        || q.shape()[1] > q.shape()[0].min(r.shape()[1])
    {
        return Err(Error::invalid_argument(
            op,
            "shape",
            "incompatible Q/R factors",
        ));
    }
    let (packed_q, coeff_q) = compact_qr_typed(backend, q, op)?;
    let triangular = compact_qr_r_typed(backend, &packed_q, &coeff_q, QrOptions::default(), op)?;
    let folded = gemm_nn_typed(backend, &triangular, r, op)?;
    assemble_from_factors_typed(backend, &packed_q, &coeff_q, &folded, r.shape()[1], op)
}

pub(super) fn compact_qr_append_typed<T>(
    backend: &mut CudaEngine,
    packed: &TypedTensor<T>,
    coeff: &TypedTensor<T>,
    block: &TypedTensor<T>,
    op: &'static str,
) -> Result<(TypedTensor<T>, TypedTensor<T>)>
where
    T: LinalgScalar + TensorScalar,
{
    let (m, _, k) = compact_qr_state_dims(packed, coeff, op)?;
    ensure_device_resident_typed(op, block)?;
    if block.shape().len() != 2 || block.shape()[0] != m {
        return Err(Error::shape_mismatch(
            op,
            vec![m, block.shape().get(1).copied().unwrap_or(0)],
            block.shape().to_vec(),
        ));
    }
    let mut transformed = clone_device_tensor(backend, block, op)?;
    apply_householder_reflectors_typed(backend, packed, coeff, &mut transformed, true, op)?;
    let trailing_coeff = geqrf_trailing_typed(backend, &mut transformed, k, op)?;
    concatenate_compact_typed(backend, packed, &transformed, coeff, &trailing_coeff, op)
}

fn compact_qr_state_dims<T>(
    packed: &TypedTensor<T>,
    coeff: &TypedTensor<T>,
    op: &'static str,
) -> Result<(usize, usize, usize)>
where
    T: LinalgScalar + TensorScalar,
{
    ensure_device_resident_typed(op, packed)?;
    ensure_device_resident_typed(op, coeff)?;
    if packed.shape().len() != 2 {
        return Err(Error::rank_mismatch(op, 2, packed.shape().len()));
    }
    if coeff.shape().len() != 1 {
        return Err(Error::rank_mismatch(op, 1, coeff.shape().len()));
    }
    let m = packed.shape()[0];
    let n = packed.shape()[1];
    let k = m.min(n);
    if coeff.shape()[0] != k {
        return Err(Error::shape_mismatch(op, vec![k], coeff.shape().to_vec()));
    }
    Ok((m, n, k))
}

pub(super) fn compact_qr_q_columns_typed<T>(
    backend: &mut CudaEngine,
    packed: &TypedTensor<T>,
    coeff: &TypedTensor<T>,
    start: usize,
    end: usize,
    options: QrOptions,
    op: &'static str,
) -> Result<TypedTensor<T>>
where
    T: LinalgScalar + TensorScalar,
{
    let (m, _, _k) = compact_qr_state_dims(packed, coeff, op)?;
    // Full-Q width: columns `k..m` span the orthogonal complement of the
    // input's column space, reached by applying the same compact reflectors to
    // the matching identity columns.
    if start > end || end > m {
        return Err(Error::invalid_argument(
            op,
            "range",
            format!("Q column range {start}..{end} exceeds 0..{m}"),
        ));
    }
    let mut q = initialize_q_columns_typed::<T>(backend, m, start, end, op)?;
    apply_householder_reflectors_typed(backend, packed, coeff, &mut q, false, op)?;
    if options.gauge == QrGauge::PositiveDiagonal {
        let mut r = compact_qr_r_typed(backend, packed, coeff, QrOptions::default(), op)?;
        T::apply_positive_qr_gauge(backend, &mut q, &mut r, start, op)?;
    }
    Ok(q)
}

pub(super) fn compact_qr_r_typed<T>(
    backend: &mut CudaEngine,
    packed: &TypedTensor<T>,
    coeff: &TypedTensor<T>,
    options: QrOptions,
    op: &'static str,
) -> Result<TypedTensor<T>>
where
    T: LinalgScalar + TensorScalar,
{
    let (m, n, k) = compact_qr_state_dims(packed, coeff, op)?;
    let input = backend.linalg_slice(
        packed,
        &matrix_slice_config(packed.shape(), k, n),
        "compact QR R",
    )?;
    let mut r = backend
        .linalg_triangular(&input, 0, false, "compact QR R")?
        .into_typed::<T>()
        .map_err(|_| Error::Internal("compact QR R changed dtype".into()))?;
    if options.gauge == QrGauge::PositiveDiagonal {
        // One throwaway column keeps both bindings addressable: the phase kernel writes Q and R in
        // one launch, and only R is returned here.
        let q = backend.alloc_output(T::dtype(), &[m, 1])?;
        let mut q = q
            .into_typed::<T>()
            .map_err(|_| Error::Internal("zero-column Q changed dtype".into()))?;
        T::apply_positive_qr_gauge(backend, &mut q, &mut r, 0, op)?;
    }
    Ok(r)
}

pub(super) fn compact_qr_typed<T>(
    backend: &mut CudaEngine,
    input: &TypedTensor<T>,
    op: &'static str,
) -> Result<(TypedTensor<T>, TypedTensor<T>)>
where
    T: LinalgScalar + TensorScalar,
{
    ensure_device_resident_typed(op, input)?;
    if input.shape().len() != 2 {
        return Err(Error::rank_mismatch(op, 2, input.shape().len()));
    }
    let m = input.shape()[0];
    let n = input.shape()[1];
    let k = m.min(n);
    if has_zero_dim(input.shape()) {
        return backend.with_raw(op, |raw| {
            raw.tensor(input)?;
            Ok((
                raw.alloc_output::<T>(input.shape())?,
                raw.alloc_output::<T>(&[k])?,
            ))
        });
    }

    let m_i32 = as_i32(m, op, "m")?;
    let n_i32 = as_i32(n, op, "n")?;
    let lda = as_i32(m, op, "lda")?;
    backend.with_raw(op, |raw| {
        let handles = raw.resource(CudaLinalgHandles::load)?;
        // SAFETY: the stream handle is valid for this raw-session scope and
        // is bound immediately to the session-owned cuSOLVER handle.
        let stream = raw.stream().raw_handle() as usize as CudaStream;
        handles.cusolver().set_stream(stream, op)?;

        let packed = raw.alloc_output::<T>(input.shape())?;
        {
            let src = raw.tensor(input)?;
            let dst = raw.tensor_mut(&packed)?;
            // SAFETY: source/destination spans are validated on this runtime,
            // destination is exclusive, and the copy is stream ordered.
            unsafe {
                raw.copy_bytes(dst.raw_ptr(), src.raw_ptr() as *const _, src.byte_len(), op)?
            };
        }
        let coeff = raw.alloc_output::<T>(&[k])?;
        let lwork = {
            let packed_ref = raw.tensor(&packed)?;
            handles.cusolver().geqrf_buffer_size(
                T::DATA_TYPE,
                m_i32,
                n_i32,
                // SAFETY: packed_ref is a validated live device matrix.
                packed_ref.raw_ptr(),
                lda,
                op,
            )?
        };
        let workspace_bytes = usize::try_from(lwork)
            .map_err(|_| Error::invalid_argument(op, "workspace", "negative length"))?
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| Error::invalid_argument(op, "workspace", "byte size overflowed"))?;
        let workspace = raw.alloc_bytes(workspace_bytes, op)?;
        let mut workspace_ptr = std::ptr::null_mut::<c_void>();
        workspace.with_ptr(|ptr| workspace_ptr = ptr);
        let info = raw.alloc_output::<i32>(&[1])?;
        let packed_ref = raw.tensor_mut(&packed)?;
        let coeff_ref = raw.tensor_mut(&coeff)?;
        let info_ref = raw.tensor_mut(&info)?;
        // SAFETY: all pointers reference validated live device allocations;
        // dimensions and workspace match the checked rank-2 matrix.
        unsafe {
            handles.cusolver().geqrf(
                T::DATA_TYPE,
                m_i32,
                n_i32,
                packed_ref.raw_ptr(),
                lda,
                coeff_ref.raw_ptr(),
                workspace_ptr,
                lwork,
                info_ref.raw_ptr().cast::<i32>(),
                op,
            )?;
        }
        // ponytail: the doc-hidden compact QR hooks keep the blocking geqrf
        // status read; the public `qr` family skips it.
        let host_info = raw.download_tensor::<i32>(&info, op)?;
        check_solver_info(op, "cusolverDn*geqrf", host_info.host_data()?[0])?;
        Ok((packed, coeff))
    })
}
