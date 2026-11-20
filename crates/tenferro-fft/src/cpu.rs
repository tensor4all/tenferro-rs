/// The Rust scalar type behind a preset variant name a macro received.
macro_rules! preset_scalar {
    (F32) => {
        f32
    };
    (F64) => {
        f64
    };
    (I32) => {
        i32
    };
    (I64) => {
        i64
    };
    (Bool) => {
        bool
    };
    (C32) => {
        num_complex::Complex32
    };
    (C64) => {
        num_complex::Complex64
    };
}
mod lanes;

use crate::backend::FftExecutionCache;
use crate::cache::{CachedFftPlanScalar, ExtensionFftPlanCache, FftPlanProvider};
use crate::{
    expected_dtype_description, fft_op_name, output_shape_c2c, output_shape_c2r, output_shape_r2c,
    transform_len, validate_c2r_spectrum_len, FftBackend, FftNorm, FftOperation, FftPlanSpec,
};
use num_complex::Complex;
use num_traits::{Float, FromPrimitive};
use std::mem::MaybeUninit;
use tenferro_cpu::linalg_interop::{BufferPool, PoolScalar, PooledUninitOutput};
use tenferro_cpu::CpuEngine;
use tenferro_tensor::{
    AllocationDomainId, DeviceKind, HostAccessError, MemoryKind, Placement,
    SharedTensorAllocationDomain, Tensor, TensorRead, TensorScalar, TensorStructural, TensorView,
    TensorViewMut, TensorWrite, TypedTensor,
};

impl FftBackend for CpuEngine {
    fn validate_fft_read_input(
        &self,
        op: &'static str,
        input: &TensorRead<'_>,
    ) -> tenferro_tensor::Result<()> {
        validate_host_fft_read_input(op, input)
    }

    fn execute_fft_read(
        &mut self,
        input: TensorRead<'_>,
        spec: &FftPlanSpec,
        mut cache: FftExecutionCache<'_>,
    ) -> tenferro_tensor::Result<Tensor> {
        let view = match input {
            TensorRead::Tensor(input) => return self.execute_fft(input, spec, cache),
            TensorRead::View(view) => view,
        };
        validate_host_fft_read_input(
            fft_op_name(spec.operation()),
            &TensorRead::View(view.clone()),
        )?;
        validate_spec_metadata(view.dtype(), view.shape(), spec)?;
        if !view.is_col_major_contiguous()? {
            let owned = self.to_contiguous_read(TensorRead::View(view))?;
            return self.execute_fft(&owned, spec, cache);
        }
        let mut plans = ExtensionFftPlanCache::for_engine(cache.store_mut(), self)?;
        self.with_linalg_pool(|exec, buffers| {
            macro_rules! transform {
                ($input:expr, $kind:ident, $variant:ident, $project:expr) => {
                    pooled_transform(
                        lanes::Input::$kind($input.as_slice()?),
                        $input.shape(),
                        spec,
                        &mut plans,
                        buffers,
                        $project,
                        exec.token(),
                    )
                    .map(Tensor::from_typed::<preset_scalar!($variant)>)
                };
            }
            match (spec.operation(), view) {
                (FftOperation::C2cForward | FftOperation::C2cInverse, TensorView::C64(x)) => {
                    transform!(x, Complex, C64, |v| v)
                }
                (FftOperation::C2cForward | FftOperation::C2cInverse, TensorView::C32(x)) => {
                    transform!(x, Complex, C32, |v| v)
                }
                (FftOperation::R2cFull | FftOperation::R2cOnesided, TensorView::F64(x)) => {
                    transform!(x, Real, C64, |v| v)
                }
                (FftOperation::R2cFull | FftOperation::R2cOnesided, TensorView::F32(x)) => {
                    transform!(x, Real, C32, |v| v)
                }
                (FftOperation::C2r, TensorView::C64(x)) => {
                    transform!(x, Complex, F64, |v: Complex<f64>| v.re)
                }
                (FftOperation::C2r, TensorView::C32(x)) => {
                    transform!(x, Complex, F32, |v: Complex<f32>| v.re)
                }
                (operation, other) => Err(crate::tensor_unsupported_dtype(
                    fft_op_name(operation),
                    other.dtype(),
                    expected_dtype_description(operation),
                )),
            }
        })
    }

    fn execute_fft_read_into(
        &mut self,
        input: TensorRead<'_>,
        spec: &FftPlanSpec,
        mut cache: FftExecutionCache<'_>,
        dest: TensorWrite<'_>,
    ) -> tenferro_tensor::Result<()> {
        validate_spec_metadata(input.dtype(), input.shape(), spec)?;
        validate_host_fft_read_input(fft_op_name(spec.operation()), &input)?;
        validate_into_destination(spec, &dest)?;
        // INVARIANT: owned Tensor storage is compact column-major by construction;
        // only a borrowed view needs layout inspection (owned inspection builds metadata).
        if matches!(&input, TensorRead::View(_)) && !input.is_col_major_contiguous()? {
            // shortcut: arbitrary-strided reads retain same-placement gathering;
            // use direct strided lanes when the FFT SPI accepts their layout.
            let owned = self.to_contiguous_read(input)?;
            return self.execute_fft_read_into(TensorRead::from_tensor(&owned), spec, cache, dest);
        }
        let mut plans = ExtensionFftPlanCache::for_engine(cache.store_mut(), self)?;
        self.with_linalg_pool(|exec, buffers| {
            macro_rules! write {
                ($input_type:ty, $kind:ident, $output_type:ty, $variant:ident, $project:expr) => {{
                    let mut execute = |values: &mut [$output_type]| {
                        // SAFETY: an initialized scalar span may be borrowed as
                        // MaybeUninit; disjoint lanes overwrite every output slot.
                        let output = unsafe {
                            std::slice::from_raw_parts_mut(values.as_mut_ptr().cast(), values.len())
                        };
                        let read = match &input {
                            TensorRead::Tensor(tensor) => tensor.as_slice::<$input_type>()?,
                            TensorRead::View(view) => view.as_slice::<$input_type>()?,
                        };
                        transform_into(
                            lanes::Input::$kind(read),
                            input.shape(),
                            spec,
                            &mut plans,
                            buffers,
                            output,
                            $project,
                            exec.token(),
                        )
                    };
                    match dest {
                        TensorWrite::Tensor(output) => {
                            execute(output.as_slice_mut::<$output_type>()?)
                        }
                        TensorWrite::View(TensorViewMut::$variant(mut output)) => {
                            let offset = usize::try_from(output.offset()).map_err(|_| {
                                tenferro_tensor::Error::invalid_argument(
                                    "fft",
                                    "destination offset",
                                    "must be nonnegative",
                                )
                            })?;
                            let len = output.n_elements();
                            // INVARIANT: compactness plus the mutable view's checked
                            // reachable range bounds this complete offset span.
                            execute(&mut output.host_storage_mut()?[offset..offset + len])
                        }
                        _ => Err(tenferro_tensor::Error::unsupported(
                            "fft",
                            "destination dtype does not match FFT output",
                        )),
                    }
                }};
            }
            match (spec.operation(), input.dtype()) {
                (
                    FftOperation::C2cForward | FftOperation::C2cInverse,
                    tenferro_tensor::DType::C64,
                ) => write!(Complex<f64>, Complex, Complex<f64>, C64, |v| v),
                (
                    FftOperation::C2cForward | FftOperation::C2cInverse,
                    tenferro_tensor::DType::C32,
                ) => write!(Complex<f32>, Complex, Complex<f32>, C32, |v| v),
                (
                    FftOperation::R2cFull | FftOperation::R2cOnesided,
                    tenferro_tensor::DType::F64,
                ) => write!(f64, Real, Complex<f64>, C64, |v| v),
                (
                    FftOperation::R2cFull | FftOperation::R2cOnesided,
                    tenferro_tensor::DType::F32,
                ) => write!(f32, Real, Complex<f32>, C32, |v| v),
                (FftOperation::C2r, tenferro_tensor::DType::C64) => {
                    write!(Complex<f64>, Complex, f64, F64, |v: Complex<f64>| v.re)
                }
                (FftOperation::C2r, tenferro_tensor::DType::C32) => {
                    write!(Complex<f32>, Complex, f32, F32, |v: Complex<f32>| v.re)
                }
                (operation, dtype) => Err(crate::tensor_unsupported_dtype(
                    fft_op_name(operation),
                    dtype,
                    expected_dtype_description(operation),
                )),
            }
        })
    }

    fn execute_fft(
        &mut self,
        input: &Tensor,
        spec: &FftPlanSpec,
        mut cache: FftExecutionCache<'_>,
    ) -> tenferro_tensor::Result<Tensor> {
        validate_spec_input(input, spec)?;
        let mut plans = ExtensionFftPlanCache::for_engine(cache.store_mut(), self)?;
        let domain = self.shared_allocation_domain().cloned();
        let managed = if input.is_backend_buffer() {
            domain.as_deref()
        } else {
            None
        };
        if managed.is_none() {
            validate_host_fft_input(fft_op_name(spec.operation()), input)?;
        }
        self.with_linalg_pool(|exec, buffers| {
            {
                execute_fft_with_plans(input, spec, &mut plans, buffers, managed, exec.token())
            }
        })
    }
}

/// The typed tensor behind an FFT operand.
///
/// The tables that call this match on the payload's dtype first, so the refusal is unreachable; it
/// names the operation rather than a dtype description, which the wildcard arms still report.
fn fft_operand<'a, T: tenferro_tensor::TensorScalar>(
    input: &'a Tensor,
    op: &'static str,
) -> tenferro_tensor::Result<&'a tenferro_tensor::TypedTensor<T>> {
    input
        .as_typed::<T>()
        .ok_or_else(|| crate::tensor_unsupported_dtype(op, input.dtype(), "a supported dtype"))
}

#[cfg(feature = "autodiff")]
pub(crate) fn execute_in_place(
    session: &mut CpuEngine,
    input: &mut Tensor,
    spec: &FftPlanSpec,
    mut cache: FftExecutionCache<'_>,
) -> tenferro_tensor::Result<()> {
    validate_spec_input(input, spec)?;
    validate_host_fft_input("fft_in_place", input)?;
    if !matches!(
        spec.operation(),
        FftOperation::C2cForward | FftOperation::C2cInverse
    ) || spec
        .requested_len()
        .is_some_and(|n| n != input.shape()[spec.normalized_axis()])
    {
        return Err(tenferro_tensor::Error::unsupported(
            "fft_in_place",
            "in-place FFT requires shape-preserving complex input",
        ));
    }
    let mut plans = ExtensionFftPlanCache::for_engine(cache.store_mut(), session)?;
    session.with_linalg_pool(|exec, buffers| match input.dtype() {
        tenferro_tensor::DType::C64 => in_place_typed(
            in_place_operand::<f64>(&mut *input)?,
            spec,
            &mut plans,
            buffers,
            exec.token(),
        ),
        tenferro_tensor::DType::C32 => in_place_typed(
            in_place_operand::<f32>(&mut *input)?,
            spec,
            &mut plans,
            buffers,
            exec.token(),
        ),
        other => Err(crate::tensor_unsupported_dtype(
            "fft_in_place",
            other,
            "C32 or C64",
        )),
    })
}

/// The typed tensor behind an in-place FFT operand, which the plan writes through.
///
/// The arms that call this match on the payload's dtype first, so the refusal reports a dtype the plan
/// cannot write rather than a caller mistake.
#[cfg(feature = "autodiff")]
fn in_place_operand<T: tenferro_tensor::TensorScalar>(
    input: &mut Tensor,
) -> tenferro_tensor::Result<&mut TypedTensor<Complex<T>>>
where
    Complex<T>: tenferro_tensor::TensorScalar,
{
    let dtype = input.dtype();
    input
        .as_typed_mut::<Complex<T>>()
        .ok_or_else(|| crate::tensor_unsupported_dtype("fft_in_place", dtype, "C32 or C64"))
}

#[cfg(feature = "autodiff")]
fn in_place_typed<T: CachedFftPlanScalar + TensorScalar>(
    input: &mut TypedTensor<Complex<T>>,
    spec: &FftPlanSpec,
    plans: &mut (impl FftPlanProvider + ?Sized),
    buffers: &mut BufferPool,
    exec: tenferro_cpu::threadteam::Exec<'_>,
) -> tenferro_tensor::Result<()>
where
    Complex<T>: TensorScalar + PoolScalar,
{
    let shape = input.shape().to_vec();
    let axis = spec.normalized_axis();
    let len = shape[axis];
    // Use the descriptor-bounded write guard, not the whole root allocation:
    // compact slices may have a nonzero offset or a smaller logical extent.
    input.with_host_write(|values| {
        // SAFETY: this is an exclusive borrow of initialized Complex<T> storage.
        // Shape-preserving c2c with identity projection writes only valid Complex<T>
        // values; on error or unwind unwritten values also remain initialized.
        unsafe {
            let output = std::slice::from_raw_parts_mut(
                values.as_mut_ptr().cast::<MaybeUninit<Complex<T>>>(),
                values.len(),
            );
            lanes::execute::<T, Complex<T>>(
                None,
                &shape,
                axis,
                len,
                len,
                spec.operation(),
                spec.norm(),
                plans,
                buffers,
                output,
                |v| v,
                exec,
            )
        }
    })?
}

fn validate_spec_input(input: &Tensor, spec: &FftPlanSpec) -> tenferro_tensor::Result<()> {
    validate_spec_metadata(input.dtype(), input.shape(), spec)
}

fn validate_spec_metadata(
    dtype: tenferro_tensor::DType,
    shape: &[usize],
    spec: &FftPlanSpec,
) -> tenferro_tensor::Result<()> {
    if dtype != spec.input_dtype() {
        return Err(tenferro_tensor::Error::dtype_mismatch(
            fft_op_name(spec.operation()),
            spec.input_dtype(),
            dtype,
        ));
    }
    if shape != spec.input_shape() {
        return Err(tenferro_tensor::Error::invalid_argument(
            fft_op_name(spec.operation()),
            "input shape",
            format!(
                "validated FFT spec shape {:?} does not match execution input shape {:?}",
                spec.input_shape(),
                shape
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

fn execute_fft_with_plans(
    input: &Tensor,
    spec: &FftPlanSpec,
    plans: &mut (impl FftPlanProvider + ?Sized),
    buffers: &mut BufferPool,
    domain: Option<&dyn SharedTensorAllocationDomain>,
    exec: tenferro_cpu::threadteam::Exec<'_>,
) -> tenferro_tensor::Result<Tensor> {
    macro_rules! transform {
        ($input:expr, $kind:ident, $variant:ident, $project:expr) => {
            transform(
                $input,
                spec,
                plans,
                buffers,
                domain,
                |values| lanes::Input::$kind(values),
                $project,
                exec,
            )
            .map(Tensor::from_typed::<preset_scalar!($variant)>)
        };
    }
    match (spec.operation(), input.dtype()) {
        (FftOperation::C2cForward | FftOperation::C2cInverse, tenferro_tensor::DType::C64) => {
            let x = fft_operand::<Complex<f64>>(input, fft_op_name(spec.operation()))?;
            transform!(x, Complex, C64, |v| v)
        }
        (FftOperation::C2cForward | FftOperation::C2cInverse, tenferro_tensor::DType::C32) => {
            let x = fft_operand::<Complex<f32>>(input, fft_op_name(spec.operation()))?;
            transform!(x, Complex, C32, |v| v)
        }
        (FftOperation::R2cFull | FftOperation::R2cOnesided, tenferro_tensor::DType::F64) => {
            let x = fft_operand::<f64>(input, fft_op_name(spec.operation()))?;
            transform!(x, Real, C64, |v| v)
        }
        (FftOperation::R2cFull | FftOperation::R2cOnesided, tenferro_tensor::DType::F32) => {
            let x = fft_operand::<f32>(input, fft_op_name(spec.operation()))?;
            transform!(x, Real, C32, |v| v)
        }
        (FftOperation::C2r, tenferro_tensor::DType::C64) => {
            let x = fft_operand::<Complex<f64>>(input, fft_op_name(spec.operation()))?;
            transform!(x, Complex, F64, |v: Complex<f64>| v.re)
        }
        (FftOperation::C2r, tenferro_tensor::DType::C32) => {
            let x = fft_operand::<Complex<f32>>(input, fft_op_name(spec.operation()))?;
            transform!(x, Complex, F32, |v: Complex<f32>| v.re)
        }
        (operation, other) => Err(crate::tensor_unsupported_dtype(
            fft_op_name(operation),
            other,
            expected_dtype_description(operation),
        )),
    }
}

fn output_shape(in_shape: &[usize], spec: &FftPlanSpec) -> tenferro_tensor::Result<Vec<usize>> {
    let axis = spec.normalized_axis();
    let n = spec.requested_len();
    match spec.operation() {
        FftOperation::C2cForward | FftOperation::C2cInverse => output_shape_c2c(in_shape, axis, n),
        FftOperation::R2cFull | FftOperation::R2cOnesided => {
            output_shape_r2c(in_shape, axis, n, spec.operation().is_onesided())
        }
        FftOperation::C2r => output_shape_c2r(in_shape, axis, n),
    }
}

// INVARIANT: this scalar-dispatched helper carries the validated spec and the
// existing execution resources; it does not define a second public descriptor.
#[allow(clippy::too_many_arguments)]
fn transform<I: TensorScalar, T: CachedFftPlanScalar + PoolScalar, O: PoolScalar>(
    input: &TypedTensor<I>,
    spec: &FftPlanSpec,
    plans: &mut (impl FftPlanProvider + ?Sized),
    buffers: &mut BufferPool,
    domain: Option<&dyn SharedTensorAllocationDomain>,
    wrap: for<'a> fn(&'a [I]) -> lanes::Input<'a, T>,
    project: impl Fn(Complex<T>) -> O + Sync,
    exec: tenferro_cpu::threadteam::Exec<'_>,
) -> tenferro_tensor::Result<TypedTensor<O>>
where
    Complex<T>: PoolScalar,
{
    let produce = |read: &[I]| {
        pooled_transform(
            wrap(read),
            input.shape(),
            spec,
            plans,
            buffers,
            &project,
            exec,
        )
    };
    if let Some(domain) = domain {
        let op = fft_op_name(spec.operation());
        // Validate the input domain before allocating or mapping the output.
        with_managed_read(input, domain.id(), op, |read| {
            let output = domain.allocate(O::dtype(), &output_shape(input.shape(), spec)?)?;
            let mut output = O::into_typed(output).map_err(|_| {
                tenferro_tensor::Error::runtime_state(
                    op,
                    "shared allocation owner returned an output with the wrong dtype",
                )
            })?;
            if output.allocation_domain() != Some(domain.id()) {
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
            // INVARIANT: BackendStorage::map_write exposes only a full-buffer
            // copy callback, not a writable span. Preserve that explicit provider
            // boundary; the host staging allocation is recycled, not recreated.
            let staging = produce(read)?;
            if let Some(buffer) = output.backend_buffer_mut() {
                buffer
                    .map_write()
                    .map_err(|source| tenferro_tensor::Error::host_access(op, source))?
                    .copy_from_slice(staging.host_data()?)
                    .map_err(|source| tenferro_tensor::Error::host_access(op, source))?;
            } else {
                output.with_host_write(|write| {
                    if write.len() != staging.host_data()?.len() {
                        return Err(tenferro_tensor::Error::runtime_state(
                            op,
                            "shared allocation owner returned an output with the wrong length",
                        ));
                    }
                    write.copy_from_slice(staging.host_data()?);
                    Ok(())
                })??;
            }
            Ok(output)
        })
    } else {
        produce(input.host_data()?)
    }
}

fn pooled_transform<T: CachedFftPlanScalar + PoolScalar, O: PoolScalar>(
    input: lanes::Input<'_, T>,
    in_shape: &[usize],
    spec: &FftPlanSpec,
    plans: &mut (impl FftPlanProvider + ?Sized),
    buffers: &mut BufferPool,
    project: impl Fn(Complex<T>) -> O + Sync,
    exec: tenferro_cpu::threadteam::Exec<'_>,
) -> tenferro_tensor::Result<TypedTensor<O>>
where
    Complex<T>: PoolScalar,
{
    let shape = output_shape(in_shape, spec)?;
    let mut output = PooledUninitOutput::<O>::new(buffers, shape)?;
    transform_into(
        input,
        in_shape,
        spec,
        plans,
        buffers,
        output.as_uninit_slice_mut(),
        project,
        exec,
    )?;
    // SAFETY: successful execution initializes the complete output and joins writers.
    unsafe { output.assume_init_recycled() }
}

fn validate_into_destination(
    spec: &FftPlanSpec,
    dest: &TensorWrite<'_>,
) -> tenferro_tensor::Result<()> {
    let op = fft_op_name(spec.operation());
    let expected_dtype = match spec.operation() {
        FftOperation::C2r => match spec.input_dtype() {
            tenferro_tensor::DType::C32 => tenferro_tensor::DType::F32,
            _ => tenferro_tensor::DType::F64,
        },
        _ => match spec.input_dtype() {
            tenferro_tensor::DType::F32 | tenferro_tensor::DType::C32 => {
                tenferro_tensor::DType::C32
            }
            _ => tenferro_tensor::DType::C64,
        },
    };
    if dest.dtype() != expected_dtype {
        return Err(tenferro_tensor::Error::dtype_mismatch(
            op,
            expected_dtype,
            dest.dtype(),
        ));
    }
    let axis = spec.normalized_axis();
    let fft_len = transform_len_for_spec(spec)?;
    let axis_len = if spec.operation() == FftOperation::R2cOnesided {
        fft_len / 2 + 1
    } else {
        fft_len
    };
    if dest.shape().len() != spec.input_shape().len()
        || dest.shape().iter().enumerate().any(|(i, &n)| {
            n != if i == axis {
                axis_len
            } else {
                spec.input_shape()[i]
            }
        })
    {
        return Err(tenferro_tensor::Error::invalid_argument(
            op,
            "destination shape",
            "does not match the FFT output shape",
        ));
    }
    // INVARIANT: an owned Tensor is compact column-major by construction;
    // inspecting only views avoids constructing synthetic high-rank metadata.
    if matches!(dest, TensorWrite::View(_)) && !dest.is_col_major_contiguous()? {
        return Err(tenferro_tensor::Error::unsupported(
            op,
            "CPU FFT destination must be compact column-major",
        ));
    }
    match dest {
        TensorWrite::Tensor(tensor) => validate_host_fft_input(op, tensor),
        TensorWrite::View(view) => {
            macro_rules! validate {
                ($v:expr) => {
                    validate_host_fft_placement(op, $v.placement(), $v.backend_buffer().is_some())
                };
            }
            match view {
                TensorViewMut::F32(v) => validate!(v),
                TensorViewMut::F64(v) => validate!(v),
                TensorViewMut::I32(v) => validate!(v),
                TensorViewMut::I64(v) => validate!(v),
                TensorViewMut::Bool(v) => validate!(v),
                TensorViewMut::C32(v) => validate!(v),
                TensorViewMut::C64(v) => validate!(v),
            }
        }
    }
}

fn transform_len_for_spec(spec: &FftPlanSpec) -> tenferro_tensor::Result<usize> {
    if spec.operation() == FftOperation::C2r {
        let input_len = spec.input_shape()[spec.normalized_axis()];
        let len = match spec.requested_len() {
            Some(len) => len,
            None => crate::default_c2r_output_len(input_len)?,
        };
        if len == 0 {
            return Err(tenferro_tensor::Error::invalid_argument(
                "irfft",
                "output length",
                "must be positive",
            ));
        }
        validate_c2r_spectrum_len(input_len, len)?;
        Ok(len)
    } else {
        transform_len(
            spec.input_shape(),
            spec.normalized_axis(),
            spec.requested_len(),
        )
    }
}

#[allow(clippy::too_many_arguments)]
// INVARIANT: shared numerical execution accepts validated metadata and borrowed resources.
fn transform_into<T: CachedFftPlanScalar + PoolScalar, O: Send>(
    input: lanes::Input<'_, T>,
    in_shape: &[usize],
    spec: &FftPlanSpec,
    plans: &mut (impl FftPlanProvider + ?Sized),
    buffers: &mut BufferPool,
    output: &mut [MaybeUninit<O>],
    project: impl Fn(Complex<T>) -> O + Sync,
    exec: tenferro_cpu::threadteam::Exec<'_>,
) -> tenferro_tensor::Result<()>
where
    Complex<T>: PoolScalar,
{
    let axis = spec.normalized_axis();
    let fft_len = transform_len_for_spec(spec)?;
    let out_axis_len = if spec.operation() == FftOperation::R2cOnesided {
        fft_len / 2 + 1
    } else {
        fft_len
    };
    // SAFETY: input and writable output are disjoint borrows, and lane execution
    // validates complete coverage before any writes.
    unsafe {
        if matches!(
            spec.operation(),
            FftOperation::R2cFull | FftOperation::R2cOnesided | FftOperation::C2r
        ) {
            lanes::execute_real(
                Some(input),
                in_shape,
                axis,
                fft_len,
                out_axis_len,
                spec.operation(),
                spec.norm(),
                plans,
                buffers,
                output,
                project,
                exec,
            )
        } else {
            lanes::execute(
                Some(input),
                in_shape,
                axis,
                fft_len,
                out_axis_len,
                spec.operation(),
                spec.norm(),
                plans,
                buffers,
                output,
                project,
                exec,
            )
        }
    }
}

fn with_managed_read<T: TensorScalar, R>(
    input: &TypedTensor<T>,
    expected_domain: AllocationDomainId,
    op: &'static str,
    execute: impl FnOnce(&[T]) -> tenferro_tensor::Result<R>,
) -> tenferro_tensor::Result<R> {
    if input.placement().memory_kind != MemoryKind::Managed {
        return Err(tenferro_tensor::Error::host_access(
            op,
            HostAccessError::Unsupported {
                backend: if matches!(
                    input.placement().memory_kind,
                    MemoryKind::PinnedHost | MemoryKind::UnpinnedHost
                ) {
                    "host"
                } else {
                    "backend"
                },
            },
        ));
    }
    match input.allocation_domain() {
        Some(actual) if actual == expected_domain => {}
        Some(actual) => {
            return Err(tenferro_tensor::Error::host_access(
                op,
                HostAccessError::ForeignDomain {
                    expected: expected_domain,
                    actual,
                },
            ))
        }
        None => {
            return Err(tenferro_tensor::Error::host_access(
                op,
                HostAccessError::Unsupported { backend: "backend" },
            ))
        }
    }
    if let Some(buffer) = input.backend_buffer() {
        let read = buffer
            .map_read()
            .map_err(|source| tenferro_tensor::Error::host_access(op, source))?;
        execute(&read)
    } else {
        input.with_host_read(execute)?
    }
}

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
        TensorRead::View(view) => {
            macro_rules! validate {
                ($v:expr) => {
                    validate_host_fft_placement(op, $v.placement(), $v.backend_buffer().is_some())
                };
            }
            match view {
                TensorView::F32(v) => validate!(v),
                TensorView::F64(v) => validate!(v),
                TensorView::I32(v) => validate!(v),
                TensorView::I64(v) => validate!(v),
                TensorView::Bool(v) => validate!(v),
                TensorView::C32(v) => validate!(v),
                TensorView::C64(v) => validate!(v),
            }
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
    Err(tenferro_tensor::Error::unsupported(op, format!(
        "tenferro-fft CpuBackend supports host tensors only; unsupported {location} input; download the tensor to CPU before FFT")))
}

fn scale_for<T: Float + FromPrimitive>(
    norm: FftNorm,
    forward: bool,
    n: usize,
) -> tenferro_tensor::Result<T> {
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

#[derive(Debug)]
pub(crate) struct LaneLayout {
    stride: usize,
    in_block: usize,
    out_block: usize,
    lanes: usize,
    input_len: usize,
    output_len: usize,
}
impl LaneLayout {
    pub(crate) fn new(
        in_shape: &[usize],
        axis: usize,
        out_axis_len: usize,
    ) -> tenferro_tensor::Result<Self> {
        let stride = checked_shape_product("fft", "axis stride", &in_shape[..axis])?;
        let outer = checked_shape_product("fft", "outer lanes", &in_shape[axis + 1..])?;
        let in_block = checked_mul("fft", "input block", stride, in_shape[axis])?;
        let out_block = checked_mul("fft", "output block", stride, out_axis_len)?;
        Ok(Self {
            stride,
            in_block,
            out_block,
            lanes: checked_mul("fft", "lane count", outer, stride)?,
            input_len: checked_mul("fft", "input coverage", outer, in_block)?,
            output_len: checked_mul("fft", "output coverage", outer, out_block)?,
        })
    }
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

#[cfg(test)]
#[path = "cpu/managed_tests.rs"]
mod managed_tests;
