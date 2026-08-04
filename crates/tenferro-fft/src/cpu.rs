use std::mem::MaybeUninit;

use num_complex::Complex;
use num_traits::{Float, FromPrimitive, Zero};
use tenferro_cpu::CpuExecSession;
use tenferro_tensor::{
    AllocationDomainId, DType, DeviceKind, HostAccessError, MemoryKind, Placement,
    SharedTensorAllocationDomain, StorageBuffer, Tensor, TensorRead, TensorScalar, TensorView,
    TypedTensor,
};

use crate::backend::FftExecutionCache;
use crate::cache::{cached_fft_plan, CachedFftPlanScalar, ExtensionFftPlanCache, FftPlanProvider};
use crate::{
    expected_dtype_description, fft_op_name, output_shape_c2c, output_shape_c2r, output_shape_r2c,
    transform_len, validate_c2r_spectrum_len, FftBackend, FftNorm, FftOperation, FftPlanSpec,
};

impl FftBackend for CpuExecSession<'_> {
    fn validate_fft_read_input(
        &self,
        op: &'static str,
        input: &TensorRead<'_>,
    ) -> tenferro_tensor::Result<()> {
        validate_host_fft_read_input(op, input)
    }

    fn execute_fft(
        &mut self,
        input: &Tensor,
        spec: &FftPlanSpec,
        mut cache: FftExecutionCache<'_>,
    ) -> tenferro_tensor::Result<Tensor> {
        validate_spec_input(input, spec)?;
        let mut plans = ExtensionFftPlanCache::new(cache.store_mut());
        let allocation_domain = self.shared_allocation_domain();
        match allocation_domain.as_deref() {
            Some(domain) => execute_managed_fft_with_plans(input, spec, domain, &mut plans),
            None => {
                validate_host_fft_input(fft_op_name(spec.operation()), input)?;
                execute_fft_with_plans(input, spec, &mut plans)
            }
        }
    }
}

fn validate_spec_input(input: &Tensor, spec: &FftPlanSpec) -> tenferro_tensor::Result<()> {
    if input.dtype() != spec.input_dtype() {
        return Err(tenferro_tensor::Error::dtype_mismatch(
            fft_op_name(spec.operation()),
            spec.input_dtype(),
            input.dtype(),
        ));
    }
    if input.shape() != spec.input_shape() {
        return Err(tenferro_tensor::Error::invalid_argument(
            fft_op_name(spec.operation()),
            "input shape",
            format!(
                "validated FFT spec shape {:?} does not match execution input shape {:?}",
                spec.input_shape(),
                input.shape()
            ),
        ));
    }
    if !spec.requires_compact_column_major() {
        return Err(tenferro_tensor::Error::unsupported(
            fft_op_name(spec.operation()),
            "CpuBackend FFT requires compact column-major input",
        ));
    }
    Ok(())
}

pub(crate) fn execute_fft_with_plans(
    input: &Tensor,
    spec: &FftPlanSpec,
    plans: &mut (impl FftPlanProvider + ?Sized),
) -> tenferro_tensor::Result<Tensor> {
    let operation = spec.operation();
    let axis = spec.normalized_axis();
    let n = spec.requested_len();
    let norm = spec.norm();

    let output = match (operation, input) {
        (FftOperation::C2cForward, Tensor::C64(input))
        | (FftOperation::C2cInverse, Tensor::C64(input)) => {
            Tensor::C64(TypedTensor::from_vec_col_major(
                output_shape_c2c(input.shape(), axis, n)?,
                execute_c2c(input, axis, n, operation.is_forward(), norm, plans)?,
            )?)
        }
        (FftOperation::C2cForward, Tensor::C32(input))
        | (FftOperation::C2cInverse, Tensor::C32(input)) => {
            Tensor::C32(TypedTensor::from_vec_col_major(
                output_shape_c2c(input.shape(), axis, n)?,
                execute_c2c(input, axis, n, operation.is_forward(), norm, plans)?,
            )?)
        }
        (FftOperation::R2cFull, Tensor::F64(input))
        | (FftOperation::R2cOnesided, Tensor::F64(input)) => {
            Tensor::C64(TypedTensor::from_vec_col_major(
                output_shape_r2c(input.shape(), axis, n, operation.is_onesided())?,
                execute_r2c(input, axis, n, operation.is_onesided(), norm, plans)?,
            )?)
        }
        (FftOperation::R2cFull, Tensor::F32(input))
        | (FftOperation::R2cOnesided, Tensor::F32(input)) => {
            Tensor::C32(TypedTensor::from_vec_col_major(
                output_shape_r2c(input.shape(), axis, n, operation.is_onesided())?,
                execute_r2c(input, axis, n, operation.is_onesided(), norm, plans)?,
            )?)
        }
        (FftOperation::C2r, Tensor::C64(input)) => Tensor::F64(TypedTensor::from_vec_col_major(
            output_shape_c2r(input.shape(), axis, n)?,
            execute_c2r(input, axis, n, norm, plans)?,
        )?),
        (FftOperation::C2r, Tensor::C32(input)) => Tensor::F32(TypedTensor::from_vec_col_major(
            output_shape_c2r(input.shape(), axis, n)?,
            execute_c2r(input, axis, n, norm, plans)?,
        )?),
        (operation, other) => {
            return Err(crate::tensor_unsupported_dtype(
                fft_op_name(operation),
                other.dtype(),
                expected_dtype_description(operation),
            ));
        }
    };
    Ok(output)
}

fn execute_managed_fft_with_plans(
    input: &Tensor,
    spec: &FftPlanSpec,
    domain: &dyn SharedTensorAllocationDomain,
    plans: &mut (impl FftPlanProvider + ?Sized),
) -> tenferro_tensor::Result<Tensor> {
    let operation = spec.operation();
    let op = fft_op_name(operation);
    let axis = spec.normalized_axis();
    let n = spec.requested_len();
    let norm = spec.norm();

    match (operation, input) {
        (FftOperation::C2cForward, Tensor::C64(input))
        | (FftOperation::C2cInverse, Tensor::C64(input)) => {
            let shape = output_shape_c2c(input.shape(), axis, n)?;
            let values = with_managed_read(input, domain.id(), op, |input_data| {
                execute_c2c_data(
                    input.shape(),
                    input_data,
                    axis,
                    n,
                    operation.is_forward(),
                    norm,
                    plans,
                )
            })?;
            let output = domain.allocate(DType::C64, &shape)?;
            write_managed_output_c64(output, domain.id(), op, &values)
        }
        (FftOperation::C2cForward, Tensor::C32(input))
        | (FftOperation::C2cInverse, Tensor::C32(input)) => {
            let shape = output_shape_c2c(input.shape(), axis, n)?;
            let values = with_managed_read(input, domain.id(), op, |input_data| {
                execute_c2c_data(
                    input.shape(),
                    input_data,
                    axis,
                    n,
                    operation.is_forward(),
                    norm,
                    plans,
                )
            })?;
            let output = domain.allocate(DType::C32, &shape)?;
            write_managed_output_c32(output, domain.id(), op, &values)
        }
        (FftOperation::R2cFull, Tensor::F64(input))
        | (FftOperation::R2cOnesided, Tensor::F64(input)) => {
            let shape = output_shape_r2c(input.shape(), axis, n, operation.is_onesided())?;
            let values = with_managed_read(input, domain.id(), op, |input_data| {
                execute_r2c_data(
                    input.shape(),
                    input_data,
                    axis,
                    n,
                    operation.is_onesided(),
                    norm,
                    plans,
                )
            })?;
            let output = domain.allocate(DType::C64, &shape)?;
            write_managed_output_c64(output, domain.id(), op, &values)
        }
        (FftOperation::R2cFull, Tensor::F32(input))
        | (FftOperation::R2cOnesided, Tensor::F32(input)) => {
            let shape = output_shape_r2c(input.shape(), axis, n, operation.is_onesided())?;
            let values = with_managed_read(input, domain.id(), op, |input_data| {
                execute_r2c_data(
                    input.shape(),
                    input_data,
                    axis,
                    n,
                    operation.is_onesided(),
                    norm,
                    plans,
                )
            })?;
            let output = domain.allocate(DType::C32, &shape)?;
            write_managed_output_c32(output, domain.id(), op, &values)
        }
        (FftOperation::C2r, Tensor::C64(input)) => {
            let shape = output_shape_c2r(input.shape(), axis, n)?;
            let values = with_managed_read(input, domain.id(), op, |input_data| {
                execute_c2r_data(input.shape(), input_data, axis, n, norm, plans)
            })?;
            let output = domain.allocate(DType::F64, &shape)?;
            write_managed_output_f64(output, domain.id(), op, &values)
        }
        (FftOperation::C2r, Tensor::C32(input)) => {
            let shape = output_shape_c2r(input.shape(), axis, n)?;
            let values = with_managed_read(input, domain.id(), op, |input_data| {
                execute_c2r_data(input.shape(), input_data, axis, n, norm, plans)
            })?;
            let output = domain.allocate(DType::F32, &shape)?;
            write_managed_output_f32(output, domain.id(), op, &values)
        }
        (operation, other) => Err(crate::tensor_unsupported_dtype(
            op,
            other.dtype(),
            expected_dtype_description(operation),
        )),
    }
}

fn with_managed_read<T, R>(
    input: &TypedTensor<T>,
    expected_domain: AllocationDomainId,
    op: &'static str,
    execute: impl FnOnce(&[T]) -> tenferro_tensor::Result<R>,
) -> tenferro_tensor::Result<R>
where
    T: Send + Sync + 'static,
{
    let StorageBuffer::Backend(buffer) = input.buffer() else {
        return Err(tenferro_tensor::Error::host_access(
            op,
            HostAccessError::Unsupported { backend: "host" },
        ));
    };
    match buffer.allocation_domain() {
        Some(actual) if actual == expected_domain => {}
        Some(actual) => {
            return Err(tenferro_tensor::Error::host_access(
                op,
                HostAccessError::ForeignDomain {
                    expected: expected_domain,
                    actual,
                },
            ));
        }
        None => {
            return Err(tenferro_tensor::Error::host_access(
                op,
                HostAccessError::Unsupported {
                    backend: buffer.backend_family(),
                },
            ));
        }
    }
    if input.placement().memory_kind != MemoryKind::Managed {
        return Err(tenferro_tensor::Error::host_access(
            op,
            HostAccessError::Unsupported {
                backend: buffer.backend_family(),
            },
        ));
    }
    let read = buffer
        .map_read()
        .map_err(|source| tenferro_tensor::Error::host_access(op, source))?;
    execute(&read)
}

fn write_managed_output<T>(
    output: &mut TypedTensor<T>,
    expected_domain: AllocationDomainId,
    op: &'static str,
    values: &[T],
) -> tenferro_tensor::Result<()>
where
    T: Send + Sync + 'static,
{
    if output.allocation_domain() != Some(expected_domain) {
        return Err(tenferro_tensor::Error::runtime_state(
            op,
            "shared allocation owner returned an output outside its domain",
        ));
    }
    if output.placement().memory_kind != MemoryKind::Managed {
        return Err(tenferro_tensor::Error::runtime_state(
            op,
            "shared allocation owner returned a non-managed output",
        ));
    }
    let Some(buffer) = output.backend_buffer_mut() else {
        return Err(tenferro_tensor::Error::runtime_state(
            op,
            "shared allocation owner returned a host output",
        ));
    };
    let mut write = buffer
        .map_write()
        .map_err(|source| tenferro_tensor::Error::host_access(op, source))?;
    write
        .copy_from_slice(values)
        .map_err(|source| tenferro_tensor::Error::host_access(op, source))
}

macro_rules! write_managed_output {
    ($name:ident, $variant:ident, $scalar:ty) => {
        fn $name(
            output: Tensor,
            expected_domain: AllocationDomainId,
            op: &'static str,
            values: &[$scalar],
        ) -> tenferro_tensor::Result<Tensor> {
            let Tensor::$variant(mut output) = output else {
                return Err(tenferro_tensor::Error::runtime_state(
                    op,
                    concat!(
                        "shared allocation owner returned a non-",
                        stringify!($variant),
                        " output"
                    ),
                ));
            };
            write_managed_output(&mut output, expected_domain, op, values)?;
            Ok(Tensor::$variant(output))
        }
    };
}

write_managed_output!(write_managed_output_f32, F32, f32);
write_managed_output!(write_managed_output_f64, F64, f64);
write_managed_output!(write_managed_output_c32, C32, num_complex::Complex32);
write_managed_output!(write_managed_output_c64, C64, num_complex::Complex64);

pub(crate) fn validate_host_fft_input(
    op: &'static str,
    input: &Tensor,
) -> tenferro_tensor::Result<()> {
    validate_host_fft_placement(op, input.placement(), input.is_backend_buffer())
}

pub(crate) fn validate_host_fft_read_input(
    op: &'static str,
    input: &TensorRead<'_>,
) -> tenferro_tensor::Result<()> {
    match input {
        TensorRead::Tensor(tensor) => validate_host_fft_input(op, tensor),
        TensorRead::View(view) => validate_host_fft_view_input(op, view),
    }
}

fn validate_host_fft_view_input(
    op: &'static str,
    view: &TensorView<'_>,
) -> tenferro_tensor::Result<()> {
    match view {
        TensorView::F32(view) => {
            validate_host_fft_placement(op, view.placement(), view.backend_buffer().is_some())
        }
        TensorView::F64(view) => {
            validate_host_fft_placement(op, view.placement(), view.backend_buffer().is_some())
        }
        TensorView::I32(view) => {
            validate_host_fft_placement(op, view.placement(), view.backend_buffer().is_some())
        }
        TensorView::I64(view) => {
            validate_host_fft_placement(op, view.placement(), view.backend_buffer().is_some())
        }
        TensorView::Bool(view) => {
            validate_host_fft_placement(op, view.placement(), view.backend_buffer().is_some())
        }
        TensorView::C32(view) => {
            validate_host_fft_placement(op, view.placement(), view.backend_buffer().is_some())
        }
        TensorView::C64(view) => {
            validate_host_fft_placement(op, view.placement(), view.backend_buffer().is_some())
        }
    }
}

fn validate_host_fft_placement(
    op: &'static str,
    placement: &Placement,
    is_backend_buffer: bool,
) -> tenferro_tensor::Result<()> {
    let is_device = matches!(placement.memory_kind, MemoryKind::Device);
    if !is_device && !is_backend_buffer {
        return Ok(());
    }

    let location = match placement.device.as_ref().map(|device| &device.kind) {
        Some(DeviceKind::Gpu(kind)) => format!("GPU backend {kind:?}"),
        Some(kind) => format!("device kind {kind:?}"),
        None if is_device => "device tensor without device metadata".to_string(),
        None => "backend buffer".to_string(),
    };
    Err(tenferro_tensor::Error::unsupported(
        op,
        format!(
            "tenferro-fft CpuBackend supports host tensors only; unsupported {location} input; \
             download the tensor to CPU before FFT"
        ),
    ))
}

fn execute_c2c<T>(
    input: &TypedTensor<Complex<T>>,
    axis: usize,
    n: Option<usize>,
    forward: bool,
    norm: FftNorm,
    plans: &mut (impl FftPlanProvider + ?Sized),
) -> tenferro_tensor::Result<Vec<Complex<T>>>
where
    T: CachedFftPlanScalar + TensorScalar,
    Complex<T>: TensorScalar,
{
    let input_data = input.host_data()?;
    execute_c2c_data(input.shape(), input_data, axis, n, forward, norm, plans)
}

#[allow(clippy::too_many_arguments)]
fn execute_c2c_data<T>(
    in_shape: &[usize],
    input_data: &[Complex<T>],
    axis: usize,
    n: Option<usize>,
    forward: bool,
    norm: FftNorm,
    plans: &mut (impl FftPlanProvider + ?Sized),
) -> tenferro_tensor::Result<Vec<Complex<T>>>
where
    T: CachedFftPlanScalar + TensorScalar,
{
    let fft_len = transform_len(in_shape, axis, n)?;
    let out_shape = output_shape_c2c(in_shape, axis, n)?;
    let out_axis_len = out_shape[axis];
    let output_len = checked_shape_product("fft", "output", &out_shape)?;
    let mut output = uninit_output_vec(output_len);
    let fft_plan = cached_fft_plan::<T, _>(plans, fft_len, forward);
    let scale: T = scale_for(norm, forward, fft_len)?;
    let mut lane = vec![Complex::zero(); fft_len];

    for_axis_lane(in_shape, axis, out_axis_len, |lane_ctx| {
        // INVARIANT: zero-fill is transform padding semantics when the input
        // lane is shorter than `fft_len`; it is not redundant initialization.
        lane.fill(Complex::zero());
        let copy_len = lane_ctx.in_axis_len.min(fft_len);
        for (slot, offset) in lane
            .iter_mut()
            .take(copy_len)
            .zip(lane_ctx.input_offsets(copy_len))
        {
            *slot = input_data[offset];
        }
        fft_plan.process(&mut lane);
        if scale != T::one() {
            for value in &mut lane {
                *value = *value * scale;
            }
        }
        for (value, offset) in lane
            .iter()
            .take(out_axis_len)
            .copied()
            .zip(lane_ctx.output_offsets(out_axis_len))
        {
            output[offset].write(value);
        }
        Ok(())
    })?;

    // SAFETY: `for_axis_lane` covers every element in the compact column-major
    // output exactly once, and each lane writes all `out_axis_len` positions.
    Ok(unsafe { assume_init_output_vec(output) })
}

fn execute_r2c<T>(
    input: &TypedTensor<T>,
    axis: usize,
    n: Option<usize>,
    onesided: bool,
    norm: FftNorm,
    plans: &mut (impl FftPlanProvider + ?Sized),
) -> tenferro_tensor::Result<Vec<Complex<T>>>
where
    T: CachedFftPlanScalar + TensorScalar,
    Complex<T>: TensorScalar,
{
    let input_data = input.host_data()?;
    execute_r2c_data(input.shape(), input_data, axis, n, onesided, norm, plans)
}

#[allow(clippy::too_many_arguments)]
fn execute_r2c_data<T>(
    in_shape: &[usize],
    input_data: &[T],
    axis: usize,
    n: Option<usize>,
    onesided: bool,
    norm: FftNorm,
    plans: &mut (impl FftPlanProvider + ?Sized),
) -> tenferro_tensor::Result<Vec<Complex<T>>>
where
    T: CachedFftPlanScalar,
{
    let fft_len = transform_len(in_shape, axis, n)?;
    let out_shape = output_shape_r2c(in_shape, axis, n, onesided)?;
    let out_axis_len = out_shape[axis];
    let output_len = checked_shape_product("rfft", "output", &out_shape)?;
    let mut output = uninit_output_vec(output_len);
    let fft_plan = cached_fft_plan::<T, _>(plans, fft_len, true);
    let scale: T = scale_for(norm, true, fft_len)?;
    let mut lane = vec![Complex::zero(); fft_len];

    for_axis_lane(in_shape, axis, out_axis_len, |lane_ctx| {
        // INVARIANT: zero-fill is rfft padding semantics when the real input
        // lane is shorter than `fft_len`; later writes cover only `copy_len`.
        lane.fill(Complex::zero());
        let copy_len = lane_ctx.in_axis_len.min(fft_len);
        for (slot, offset) in lane
            .iter_mut()
            .take(copy_len)
            .zip(lane_ctx.input_offsets(copy_len))
        {
            *slot = Complex::new(input_data[offset], T::zero());
        }
        fft_plan.process(&mut lane);
        if scale != T::one() {
            for value in &mut lane {
                *value = *value * scale;
            }
        }
        for (value, offset) in lane
            .iter()
            .take(out_axis_len)
            .copied()
            .zip(lane_ctx.output_offsets(out_axis_len))
        {
            output[offset].write(value);
        }
        Ok(())
    })?;

    // SAFETY: `for_axis_lane` covers every element in the compact column-major
    // output exactly once, and each lane writes all `out_axis_len` positions.
    Ok(unsafe { assume_init_output_vec(output) })
}

fn execute_c2r<T>(
    input: &TypedTensor<Complex<T>>,
    axis: usize,
    n: Option<usize>,
    norm: FftNorm,
    plans: &mut (impl FftPlanProvider + ?Sized),
) -> tenferro_tensor::Result<Vec<T>>
where
    T: CachedFftPlanScalar + TensorScalar,
    Complex<T>: TensorScalar,
{
    let input_data = input.host_data()?;
    execute_c2r_data(input.shape(), input_data, axis, n, norm, plans)
}

fn execute_c2r_data<T>(
    in_shape: &[usize],
    input_data: &[Complex<T>],
    axis: usize,
    n: Option<usize>,
    norm: FftNorm,
    plans: &mut (impl FftPlanProvider + ?Sized),
) -> tenferro_tensor::Result<Vec<T>>
where
    T: CachedFftPlanScalar,
{
    let out_shape = output_shape_c2r(in_shape, axis, n)?;
    let out_axis_len = out_shape[axis];
    let expected_half = validate_c2r_spectrum_len(in_shape[axis], out_axis_len)?;
    let output_len = checked_shape_product("irfft", "output", &out_shape)?;
    let mut output = uninit_output_vec(output_len);
    let fft_plan = cached_fft_plan::<T, _>(plans, out_axis_len, false);
    let scale: T = scale_for(norm, false, out_axis_len)?;
    let mut lane = vec![Complex::zero(); out_axis_len];

    for_axis_lane(in_shape, axis, out_axis_len, |lane_ctx| {
        // INVARIANT: zero-fill clears the inverse lane before writing the
        // one-sided spectrum and mirrored tail for this lane.
        lane.fill(Complex::zero());
        for (slot, offset) in lane
            .iter_mut()
            .take(expected_half)
            .zip(lane_ctx.input_offsets(expected_half))
        {
            *slot = input_data[offset];
        }
        for k in expected_half..out_axis_len {
            let mirror = out_axis_len - k;
            if mirror < lane.len() {
                lane[k] = lane[mirror].conj();
            }
        }
        fft_plan.process(&mut lane);
        for (value, offset) in lane
            .iter()
            .take(out_axis_len)
            .zip(lane_ctx.output_offsets(out_axis_len))
        {
            output[offset].write(value.re * scale);
        }
        Ok(())
    })?;

    // SAFETY: `for_axis_lane` covers every element in the compact column-major
    // output exactly once, and each lane writes all `out_axis_len` positions.
    Ok(unsafe { assume_init_output_vec(output) })
}

fn scale_for<T>(norm: FftNorm, forward: bool, n: usize) -> tenferro_tensor::Result<T>
where
    T: Float + FromPrimitive,
{
    let len = T::from_usize(n).ok_or_else(|| {
        tenferro_tensor::Error::invalid_argument(
            "tenferro_fft::scale_for",
            "FFT length",
            format!("{n} cannot be represented as scalar"),
        )
    })?;
    Ok(match (norm, forward) {
        (FftNorm::Backward, true) | (FftNorm::Forward, false) => T::one(),
        (FftNorm::Backward, false) | (FftNorm::Forward, true) => T::one() / len,
        (FftNorm::Ortho, _) => T::one() / len.sqrt(),
    })
}

fn uninit_output_vec<T>(len: usize) -> Vec<MaybeUninit<T>> {
    let mut output = Vec::with_capacity(len);
    // SAFETY: Uninitialized bytes are valid for `MaybeUninit<T>` slots. The
    // slots are converted to `T` only after all output positions are written.
    unsafe { output.set_len(len) };
    output
}

unsafe fn assume_init_output_vec<T>(mut output: Vec<MaybeUninit<T>>) -> Vec<T> {
    let len = output.len();
    let capacity = output.capacity();
    let ptr = output.as_mut_ptr().cast::<T>();
    std::mem::forget(output);
    // SAFETY: `MaybeUninit<T>` has the same layout as `T`; the caller
    // guarantees every slot has been initialized exactly once.
    unsafe { Vec::from_raw_parts(ptr, len, capacity) }
}

#[derive(Clone, Copy)]
pub(crate) struct LaneContext {
    input_base: usize,
    output_base: usize,
    axis_stride: usize,
    in_axis_len: usize,
    out_axis_len: usize,
}

impl LaneContext {
    fn input_offsets(self, count: usize) -> impl Iterator<Item = usize> {
        debug_assert!(count <= self.in_axis_len);
        lane_offsets(self.input_base, self.axis_stride, count)
    }

    fn output_offsets(self, count: usize) -> impl Iterator<Item = usize> {
        debug_assert!(count <= self.out_axis_len);
        lane_offsets(self.output_base, self.axis_stride, count)
    }
}

fn lane_offsets(base: usize, stride: usize, count: usize) -> impl Iterator<Item = usize> {
    // INVARIANT: `for_axis_lane` checks input/output lane coverage before it
    // constructs any `LaneContext`, so every `base + k * stride` for
    // `k < count` stays within the compact column-major buffer.
    (0..count).map(move |k| base + k * stride)
}

pub(crate) fn for_axis_lane(
    in_shape: &[usize],
    axis: usize,
    out_axis_len: usize,
    mut f: impl FnMut(LaneContext) -> tenferro_tensor::Result<()>,
) -> tenferro_tensor::Result<()> {
    let in_axis_len = in_shape[axis];
    let axis_stride = checked_shape_product("fft", "axis stride", &in_shape[..axis])?;
    let outer = checked_shape_product("fft", "outer lane count", &in_shape[axis + 1..])?;
    let in_block = checked_mul("fft", "input lane block", axis_stride, in_axis_len)?;
    let out_block = checked_mul("fft", "output lane block", axis_stride, out_axis_len)?;
    let _input_len = checked_mul("fft", "input lane coverage", outer, in_block)?;
    let _output_len = checked_mul("fft", "output lane coverage", outer, out_block)?;

    // INVARIANT: lanes are processed sequentially so one scratch lane can be
    // reused while writing into a single `MaybeUninit` output buffer. Parallel
    // lane execution needs disjoint output splitting plus per-worker scratch.
    for outer_idx in 0..outer {
        let in_outer_base = checked_mul("fft", "input outer base", outer_idx, in_block)?;
        let out_outer_base = checked_mul("fft", "output outer base", outer_idx, out_block)?;
        for inner in 0..axis_stride {
            let input_base = checked_add("fft", "input lane base", in_outer_base, inner)?;
            let output_base = checked_add("fft", "output lane base", out_outer_base, inner)?;
            f(LaneContext {
                input_base,
                output_base,
                axis_stride,
                in_axis_len,
                out_axis_len,
            })?;
        }
    }
    Ok(())
}

pub(crate) fn checked_shape_product(
    op: &'static str,
    role: &'static str,
    shape: &[usize],
) -> tenferro_tensor::Result<usize> {
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| {
            tenferro_tensor::Error::invalid_argument(
                op,
                "shape product",
                format!("{role} shape product overflows usize"),
            )
        })
}

fn checked_mul(
    op: &'static str,
    role: &'static str,
    lhs: usize,
    rhs: usize,
) -> tenferro_tensor::Result<usize> {
    lhs.checked_mul(rhs).ok_or_else(|| {
        tenferro_tensor::Error::invalid_argument(
            op,
            "arithmetic",
            format!("{role} overflows usize"),
        )
    })
}

fn checked_add(
    op: &'static str,
    role: &'static str,
    lhs: usize,
    rhs: usize,
) -> tenferro_tensor::Result<usize> {
    lhs.checked_add(rhs).ok_or_else(|| {
        tenferro_tensor::Error::invalid_argument(
            op,
            "arithmetic",
            format!("{role} overflows usize"),
        )
    })
}

#[cfg(test)]
#[path = "cpu/managed_tests.rs"]
mod managed_tests;
