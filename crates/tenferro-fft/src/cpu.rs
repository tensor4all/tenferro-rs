mod lanes;

use crate::backend::FftExecutionCache;
use crate::cache::{CachedFftPlanScalar, ExtensionFftPlanCache, FftPlanProvider};
use crate::{
    expected_dtype_description, fft_op_name, output_shape_c2c, output_shape_c2r, output_shape_r2c,
    transform_len, validate_c2r_spectrum_len, FftBackend, FftNorm, FftOperation, FftPlanSpec,
};
use num_complex::Complex;
use num_traits::{Float, FromPrimitive};
#[cfg(feature = "autodiff")]
use std::mem::MaybeUninit;
use tenferro_cpu::linalg_interop::{BufferPool, PoolScalar, PooledUninitOutput};
use tenferro_cpu::CpuExecSession;
use tenferro_tensor::{
    AllocationDomainId, DeviceKind, HostAccessError, MemoryKind, Placement,
    SharedTensorAllocationDomain, Tensor, TensorRead, TensorScalar, TensorStructural, TensorView,
    TypedTensor,
};

impl FftBackend for CpuExecSession<'_> {
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
        let mut plans = ExtensionFftPlanCache::new(cache.store_mut());
        self.with_linalg_pool(|context, buffers| {
            macro_rules! transform {
                ($input:expr, $kind:ident, $variant:ident, $project:expr) => {
                    pooled_transform(
                        lanes::Input::$kind($input.as_slice()?),
                        $input.shape(),
                        spec,
                        &mut plans,
                        buffers,
                        context.native_thread_count(),
                        $project,
                    )
                    .map(Tensor::$variant)
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

    fn execute_fft(
        &mut self,
        input: &Tensor,
        spec: &FftPlanSpec,
        mut cache: FftExecutionCache<'_>,
    ) -> tenferro_tensor::Result<Tensor> {
        validate_spec_input(input, spec)?;
        let mut plans = ExtensionFftPlanCache::new(cache.store_mut());
        let domain = self.shared_allocation_domain();
        let managed = if input.is_backend_buffer() {
            domain.as_deref()
        } else {
            None
        };
        if managed.is_none() {
            validate_host_fft_input(fft_op_name(spec.operation()), input)?;
        }
        self.with_linalg_pool(|context, buffers| {
            execute_fft_with_plans(
                input,
                spec,
                &mut plans,
                buffers,
                managed,
                context.native_thread_count(),
            )
        })
    }
}

#[cfg(feature = "autodiff")]
pub(crate) fn execute_in_place(
    session: &mut CpuExecSession<'_>,
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
    let mut plans = ExtensionFftPlanCache::new(cache.store_mut());
    session.with_linalg_pool(|context, _| match input {
        Tensor::C64(x) => in_place_typed(x, spec, &mut plans, context.native_thread_count()),
        Tensor::C32(x) => in_place_typed(x, spec, &mut plans, context.native_thread_count()),
        other => Err(crate::tensor_unsupported_dtype(
            "fft_in_place",
            other.dtype(),
            "C32 or C64",
        )),
    })
}

#[cfg(feature = "autodiff")]
fn in_place_typed<T: CachedFftPlanScalar + TensorScalar>(
    input: &mut TypedTensor<Complex<T>>,
    spec: &FftPlanSpec,
    plans: &mut (impl FftPlanProvider + ?Sized),
    threads: usize,
) -> tenferro_tensor::Result<()>
where
    Complex<T>: TensorScalar,
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
                output,
                threads,
                |v| v,
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
    threads: usize,
) -> tenferro_tensor::Result<Tensor> {
    macro_rules! transform {
        ($input:expr, $kind:ident, $variant:ident, $project:expr) => {
            transform(
                $input,
                spec,
                plans,
                buffers,
                domain,
                threads,
                |values| lanes::Input::$kind(values),
                $project,
            )
            .map(Tensor::$variant)
        };
    }
    match (spec.operation(), input) {
        (FftOperation::C2cForward | FftOperation::C2cInverse, Tensor::C64(x)) => {
            transform!(x, Complex, C64, |v| v)
        }
        (FftOperation::C2cForward | FftOperation::C2cInverse, Tensor::C32(x)) => {
            transform!(x, Complex, C32, |v| v)
        }
        (FftOperation::R2cFull | FftOperation::R2cOnesided, Tensor::F64(x)) => {
            transform!(x, Real, C64, |v| v)
        }
        (FftOperation::R2cFull | FftOperation::R2cOnesided, Tensor::F32(x)) => {
            transform!(x, Real, C32, |v| v)
        }
        (FftOperation::C2r, Tensor::C64(x)) => transform!(x, Complex, F64, |v: Complex<f64>| v.re),
        (FftOperation::C2r, Tensor::C32(x)) => transform!(x, Complex, F32, |v: Complex<f32>| v.re),
        (operation, other) => Err(crate::tensor_unsupported_dtype(
            fft_op_name(operation),
            other.dtype(),
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
fn transform<I: TensorScalar, T: CachedFftPlanScalar, O: PoolScalar>(
    input: &TypedTensor<I>,
    spec: &FftPlanSpec,
    plans: &mut (impl FftPlanProvider + ?Sized),
    buffers: &mut BufferPool,
    domain: Option<&dyn SharedTensorAllocationDomain>,
    threads: usize,
    wrap: for<'a> fn(&'a [I]) -> lanes::Input<'a, T>,
    project: impl Fn(Complex<T>) -> O + Sync,
) -> tenferro_tensor::Result<TypedTensor<O>> {
    let mut produce = |read: &[I]| {
        pooled_transform(
            wrap(read),
            input.shape(),
            spec,
            plans,
            buffers,
            threads,
            &project,
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

fn pooled_transform<T: CachedFftPlanScalar, O: PoolScalar>(
    input: lanes::Input<'_, T>,
    in_shape: &[usize],
    spec: &FftPlanSpec,
    plans: &mut (impl FftPlanProvider + ?Sized),
    buffers: &mut BufferPool,
    threads: usize,
    project: impl Fn(Complex<T>) -> O + Sync,
) -> tenferro_tensor::Result<TypedTensor<O>> {
    let shape = output_shape(in_shape, spec)?;
    let axis = spec.normalized_axis();
    let out_axis_len = shape[axis];
    let fft_len = if spec.operation() == FftOperation::C2r {
        validate_c2r_spectrum_len(in_shape[axis], out_axis_len)?;
        out_axis_len
    } else {
        transform_len(in_shape, axis, spec.requested_len())?
    };
    let mut output = PooledUninitOutput::<O>::new(buffers, shape)?;
    // SAFETY: Some(input) selects the out-of-place path with borrowed reads and
    // a disjoint fresh destination. Successful execution joins all writers and
    // initializes every element before the owning handoff.
    unsafe {
        lanes::execute(
            Some(input),
            in_shape,
            axis,
            fft_len,
            out_axis_len,
            spec.operation(),
            spec.norm(),
            plans,
            output.as_uninit_slice_mut(),
            threads,
            project,
        )?;
        output.assume_init_recycled()
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
