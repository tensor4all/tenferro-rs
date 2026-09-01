use super::*;

fn initialize_q_columns_typed<T>(
    backend: &mut CudaExecSession<'_>,
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
    backend.with_cubecl(op, |cubecl| {
        let output = cubecl.alloc_output::<T>(&[rows, width])?;
        if output.n_elements() == 0 {
            return Ok(output);
        }
        let output_arg = cubecl.tensor_binding(&output, op)?;
        let launch_count = cubecl.cube_count_1d(output.n_elements())?;
        // SAFETY: output is a live device tensor and each launched thread
        // writes one selected identity-column element.
        unsafe {
            cubecl_linalg::householder_q_columns_identity::launch_unchecked::<T, CubeclCudaRuntime>(
                cubecl.client(),
                launch_count,
                cubecl.cube_dim_1d(),
                output_arg.into_tensor_arg(),
                start,
            );
        }
        Ok(output)
    })
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
    backend: &mut CudaExecSession<'_>,
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
    ensure_cubecl_resident_typed(op, target)?;
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
    let coeff_adjoint = adjoint
        .then(|| T::conjugate_coeff(backend, coeff, op))
        .transpose()?;
    let apply_coeff = coeff_adjoint.as_ref().unwrap_or(coeff);
    let m_i32 = as_i32(m, op, "m")?;
    let columns_i32 = as_i32(columns, op, "columns")?;
    let one = T::one();
    let zero = T::zero();
    let minus_one = -one;

    backend.with_raw(op, |raw| {
        let handles = raw.resource(CudaLinalgHandles::load)?;
        // SAFETY: stream handle is valid for this raw-session scope.
        let stream = unsafe { raw.stream().raw_handle() } as usize as CudaStream;
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

        let mut v = raw.alloc_output::<T>(&[m])?;
        let mut w = raw.alloc_output::<T>(&[columns])?;
        let packed_ref = raw.tensor(packed)?;
        let coeff_ref = raw.tensor(apply_coeff)?;
        let target_ref = raw.tensor_mut(target)?;
        let v_ref = raw.tensor_mut(&mut v)?;
        let w_ref = raw.tensor_mut(&mut w)?;
        // SAFETY: every handle is a validated live device span in this raw
        // session; pointers are retained only for stream-ordered calls below.
        let (packed_ptr, coeff_ptr, target_ptr, v_ptr, w_ptr) = unsafe {
            (
                packed_ref.raw_ptr().cast_const(),
                coeff_ref.raw_ptr().cast_const(),
                target_ref.raw_ptr(),
                v_ref.raw_ptr(),
                w_ref.raw_ptr(),
            )
        };
        let indices: Box<dyn Iterator<Item = usize>> = if adjoint {
            Box::new(0..k)
        } else {
            Box::new((0..k).rev())
        };
        handles
            .cublas()
            .set_pointer_mode(CublasPointerMode::Device, op)?;
        let computation = (|| -> Result<()> {
            for reflector in indices {
                let len = m - reflector;
                let len_i32 = as_i32(len, op, "reflector length")?;
                let packed_column =
                    checked_mul_usize(op, "reflector packed column offset", reflector, m)?;
                let packed_tail_offset =
                    packed_column.checked_add(reflector + 1).ok_or_else(|| {
                        Error::invalid_argument(op, "reflector offset", "offset overflowed")
                    })?;
                let tail_bytes = checked_mul_usize(
                    op,
                    "reflector tail bytes",
                    len - 1,
                    std::mem::size_of::<T>(),
                )?;
                // SAFETY: checked reflector offsets remain within the packed,
                // coefficient, target, and scratch allocations.
                let (packed_tail, tau, target_sub) = unsafe {
                    (
                        batch_const_ptr::<T>(packed_ptr, packed_tail_offset),
                        batch_const_ptr::<T>(coeff_ptr, reflector),
                        batch_ptr::<T>(target_ptr, reflector),
                    )
                };
                // SAFETY: scalar and tail copies target disjoint live scratch
                // spans and are ordered on the session stream.
                unsafe {
                    raw.copy_bytes(v_ptr, one_ptr.cast_const(), std::mem::size_of::<T>(), op)?;
                    if len > 1 {
                        raw.copy_bytes(batch_ptr::<T>(v_ptr, 1), packed_tail, tail_bytes, op)?;
                    }
                    match T::DATA_TYPE {
                        CudaDataType::F32 | CudaDataType::F64 => {
                            handles.cublas().gemv(
                                T::DATA_TYPE,
                                CublasOperation::T,
                                len_i32,
                                columns_i32,
                                tau,
                                target_sub.cast_const(),
                                m_i32,
                                v_ptr.cast_const(),
                                1,
                                zero_ptr.cast_const(),
                                w_ptr,
                                1,
                                op,
                            )?;
                        }
                        CudaDataType::Complex32 | CudaDataType::Complex64 => {
                            handles.cublas().gemm(
                                T::DATA_TYPE,
                                CublasOperation::C,
                                CublasOperation::N,
                                1,
                                columns_i32,
                                len_i32,
                                tau,
                                v_ptr.cast_const(),
                                len_i32,
                                target_sub.cast_const(),
                                m_i32,
                                zero_ptr.cast_const(),
                                w_ptr,
                                1,
                                op,
                            )?;
                        }
                    }
                    handles.cublas().geru(
                        T::DATA_TYPE,
                        len_i32,
                        columns_i32,
                        minus_one_ptr.cast_const(),
                        v_ptr.cast_const(),
                        1,
                        w_ptr.cast_const(),
                        1,
                        target_sub,
                        m_i32,
                        op,
                    )?;
                }
            }
            Ok(())
        })();
        let reset = handles
            .cublas()
            .set_pointer_mode(CublasPointerMode::Host, op);
        computation?;
        reset
    })
}

fn geqrf_trailing_typed<T>(
    backend: &mut CudaExecSession<'_>,
    matrix: &mut TypedTensor<T>,
    row_offset: usize,
    op: &'static str,
) -> Result<TypedTensor<T>>
where
    T: LinalgScalar + TensorScalar,
{
    ensure_cubecl_resident_typed(op, matrix)?;
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
        let stream = unsafe { raw.stream().raw_handle() } as usize as CudaStream;
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
        let mut coeff = raw.alloc_output::<T>(&[k])?;
        let mut info = raw.alloc_output::<i32>(&[1])?;
        let coeff_ref = raw.tensor_mut(&mut coeff)?;
        let info_ref = raw.tensor_mut(&mut info)?;
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
        let host_info = raw.download_tensor::<i32>(&info, op)?;
        check_solver_info(op, "cusolverDn*geqrf", host_info.host_data()?[0])?;
        Ok(coeff)
    })
}

fn concatenate_compact_typed<T>(
    backend: &mut CudaExecSession<'_>,
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
        let mut output = raw.alloc_output::<T>(&[rows, width])?;
        let mut output_coeff = raw.alloc_output::<T>(&[coeff_len])?;
        let packed_ref = raw.tensor(packed)?;
        let block_ref = raw.tensor(block)?;
        let coeff_ref = raw.tensor(coeff)?;
        let trailing_ref = raw.tensor(trailing_coeff)?;
        let output_ref = raw.tensor_mut(&mut output)?;
        let output_coeff_ref = raw.tensor_mut(&mut output_coeff)?;
        // SAFETY: output layouts concatenate two validated compact column-major
        // matrices/vectors, all offsets and byte lengths stay in live spans.
        unsafe {
            raw.copy_bytes(
                output_ref.raw_ptr(),
                packed_ref.raw_ptr() as *const _,
                packed_ref.byte_len(),
                op,
            )?;
            raw.copy_bytes(
                output_ref
                    .raw_ptr()
                    .cast::<u8>()
                    .add(packed_ref.byte_len())
                    .cast(),
                block_ref.raw_ptr() as *const _,
                block_ref.byte_len(),
                op,
            )?;
            raw.copy_bytes(
                output_coeff_ref.raw_ptr(),
                coeff_ref.raw_ptr() as *const _,
                coeff_ref.byte_len(),
                op,
            )?;
            raw.copy_bytes(
                output_coeff_ref
                    .raw_ptr()
                    .cast::<u8>()
                    .add(coeff_ref.byte_len())
                    .cast(),
                trailing_ref.raw_ptr() as *const _,
                trailing_ref.byte_len(),
                op,
            )?;
        }
        Ok((output, output_coeff))
    })
}

fn gemm_nn_typed<T>(
    backend: &mut CudaExecSession<'_>,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    op: &'static str,
) -> Result<TypedTensor<T>>
where
    T: LinalgScalar + TensorScalar,
{
    ensure_cubecl_resident_typed(op, lhs)?;
    ensure_cubecl_resident_typed(op, rhs)?;
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
        let stream = unsafe { raw.stream().raw_handle() } as usize as CudaStream;
        handles.cublas().set_stream(stream, op)?;
        handles
            .cublas()
            .set_pointer_mode(CublasPointerMode::Host, op)?;
        let lhs_ref = raw.tensor(lhs)?;
        let rhs_ref = raw.tensor(rhs)?;
        let mut output = raw.alloc_output::<T>(&[m, n])?;
        let output_ref = raw.tensor_mut(&mut output)?;
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
    backend: &mut CudaExecSession<'_>,
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
    backend.with_cubecl(op, |cubecl| {
        let packed = cubecl.alloc_output::<T>(&[rows, output_cols])?;
        let coeff = cubecl.alloc_output::<T>(&[output_k])?;
        let launch_len = packed.n_elements().max(coeff.n_elements());
        if launch_len == 0 {
            return Ok((packed, coeff));
        }
        let packed_arg = cubecl.tensor_binding(&packed, op)?;
        let coeff_arg = cubecl.tensor_binding(&coeff, op)?;
        let packed_q_arg = cubecl.tensor_binding(packed_q, op)?;
        let coeff_q_arg = cubecl.tensor_binding(coeff_q, op)?;
        let folded_arg = cubecl.tensor_binding(folded_r, op)?;
        let launch_count = cubecl.cube_count_1d(launch_len)?;
        // SAFETY: bindings describe live device tensors and the launch covers
        // both packed and coefficient outputs.
        unsafe {
            cubecl_linalg::householder_from_factors_assemble::launch_unchecked::<
                T,
                CubeclCudaRuntime,
            >(
                cubecl.client(),
                launch_count,
                cubecl.cube_dim_1d(),
                packed_arg.into_tensor_arg(),
                coeff_arg.into_tensor_arg(),
                packed_q_arg.into_tensor_arg(),
                coeff_q_arg.into_tensor_arg(),
                folded_arg.into_tensor_arg(),
                factor_width,
            );
        }
        Ok((packed, coeff))
    })
}

pub(super) fn compact_qr_from_factors_typed<T>(
    backend: &mut CudaExecSession<'_>,
    q: &TypedTensor<T>,
    r: &TypedTensor<T>,
    op: &'static str,
) -> Result<(TypedTensor<T>, TypedTensor<T>)>
where
    T: LinalgScalar + TensorScalar,
{
    ensure_cubecl_resident_typed(op, q)?;
    ensure_cubecl_resident_typed(op, r)?;
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
    backend: &mut CudaExecSession<'_>,
    packed: &TypedTensor<T>,
    coeff: &TypedTensor<T>,
    block: &TypedTensor<T>,
    op: &'static str,
) -> Result<(TypedTensor<T>, TypedTensor<T>)>
where
    T: LinalgScalar + TensorScalar,
{
    let (m, _, k) = compact_qr_state_dims(packed, coeff, op)?;
    ensure_cubecl_resident_typed(op, block)?;
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
    ensure_cubecl_resident_typed(op, packed)?;
    ensure_cubecl_resident_typed(op, coeff)?;
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
    backend: &mut CudaExecSession<'_>,
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
    let (m, _, k) = compact_qr_state_dims(packed, coeff, op)?;
    if start > end || end > k {
        return Err(Error::invalid_argument(
            op,
            "range",
            format!("Q column range {start}..{end} exceeds 0..{k}"),
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
    backend: &mut CudaExecSession<'_>,
    packed: &TypedTensor<T>,
    coeff: &TypedTensor<T>,
    options: QrOptions,
    op: &'static str,
) -> Result<TypedTensor<T>>
where
    T: LinalgScalar + TensorScalar,
{
    let (m, n, k) = compact_qr_state_dims(packed, coeff, op)?;
    let input = backend.slice_typed(packed, &matrix_slice_config(packed.shape(), k, n))?;
    let mut r = backend.triu_typed(&input, 0)?;
    if options.gauge == QrGauge::PositiveDiagonal {
        let mut q = backend.with_cubecl(op, |cubecl| cubecl.alloc_output::<T>(&[m, 0]))?;
        T::apply_positive_qr_gauge(backend, &mut q, &mut r, 0, op)?;
    }
    Ok(r)
}

pub(super) fn compact_qr_typed<T>(
    backend: &mut CudaExecSession<'_>,
    input: &TypedTensor<T>,
    op: &'static str,
) -> Result<(TypedTensor<T>, TypedTensor<T>)>
where
    T: LinalgScalar + TensorScalar,
{
    ensure_cubecl_resident_typed(op, input)?;
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
        let stream = unsafe { raw.stream().raw_handle() } as usize as CudaStream;
        handles.cusolver().set_stream(stream, op)?;

        let mut packed = raw.alloc_output::<T>(input.shape())?;
        {
            let src = raw.tensor(input)?;
            let dst = raw.tensor_mut(&mut packed)?;
            // SAFETY: source/destination spans are validated on this runtime,
            // destination is exclusive, and the copy is stream ordered.
            unsafe {
                raw.copy_bytes(dst.raw_ptr(), src.raw_ptr() as *const _, src.byte_len(), op)?;
            }
        }
        let mut coeff = raw.alloc_output::<T>(&[k])?;
        let lwork = {
            let packed_ref = raw.tensor(&packed)?;
            handles.cusolver().geqrf_buffer_size(
                T::DATA_TYPE,
                m_i32,
                n_i32,
                // SAFETY: packed_ref is a validated live device matrix.
                unsafe { packed_ref.raw_ptr() },
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
        let mut info = raw.alloc_output::<i32>(&[1])?;
        let packed_ref = raw.tensor_mut(&mut packed)?;
        let coeff_ref = raw.tensor_mut(&mut coeff)?;
        let info_ref = raw.tensor_mut(&mut info)?;
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
        let host_info = raw.download_tensor::<i32>(&info, op)?;
        check_solver_info(op, "cusolverDn*geqrf", host_info.host_data()?[0])?;
        Ok((packed, coeff))
    })
}
