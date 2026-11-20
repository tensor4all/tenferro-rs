//! CPU FFT lane execution over disjoint output index sets.
#[cfg(test)]
mod tests;
use super::{scale_for, LaneLayout};
use crate::cache::{
    cached_fft_plan, cached_real_plan, CachedFftPlanScalar, FftPlanProvider, RealPlan,
};
use crate::{FftNorm, FftOperation};
use num_complex::Complex;
use num_traits::Zero;
use std::{marker::PhantomData, mem::MaybeUninit};
use tenferro_cpu::linalg_interop::{BufferPool, PoolScalar};
use tenferro_cpu::threadteam::Exec;

/// Disjoint per-job lane buffers, shared by the forked jobs.
///
/// `Exec::fork` runs job `i` exactly once, so job `i` is the only code that
/// touches element `i`; that is what makes the unchecked per-job access sound.
struct JobSlots<T>(*mut T);
// SAFETY: each job index runs exactly once under `Exec::fork`, so jobs access
// distinct elements; the fork joins before the owning slice is used again.
unsafe impl<T: Send> Send for JobSlots<T> {}
// SAFETY: as above, shared access is only used for the disjoint elements.
unsafe impl<T: Send> Sync for JobSlots<T> {}
impl<T> JobSlots<T> {
    /// # Safety
    /// The requested range must be in bounds and exclusive to this job.
    #[allow(clippy::mut_from_ref)]
    // INVARIANT: fork assigns each disjoint range to exactly one job.
    unsafe fn slice(&self, start: usize, len: usize) -> &mut [T] {
        // SAFETY: the caller proves range bounds and exclusive access.
        unsafe { std::slice::from_raw_parts_mut(self.0.add(start), len) }
    }
}

pub(super) enum Input<'a, T> {
    Complex(&'a [Complex<T>]),
    Real(&'a [T]),
}

struct Output<'a, T> {
    pointer: *mut MaybeUninit<T>,
    _borrow: PhantomData<&'a mut [MaybeUninit<T>]>,
}
// SAFETY: execute partitions the complete lane domain among jobs. Distinct lanes
// write disjoint indices, and the scoped parallel iterator joins before return.
unsafe impl<T: Send> Send for Output<'_, T> {}
// SAFETY: shared access is used only for the disjoint writes proven above.
unsafe impl<T: Send> Sync for Output<'_, T> {}
impl<T> Output<'_, T> {
    unsafe fn read_complex<U>(&self, index: usize) -> Complex<U> {
        // SAFETY: the in-place caller proves T=Complex<U>, initialized storage,
        // and that this bounded index belongs exclusively to the current lane.
        unsafe { self.pointer.cast::<Complex<U>>().add(index).read() }
    }

    unsafe fn write(&self, index: usize, value: T) {
        // SAFETY: the caller proves index is in the output span and uniquely
        // assigned to its lane for the duration of this scoped execution.
        unsafe {
            self.pointer.add(index).write(MaybeUninit::new(value));
        }
    }
}

// FFT work includes gather/scatter and the transform, unlike elementwise maps.
// Below this grain spawning jobs costs more than the representative small FFTs.
const MIN_PARALLEL_ELEMENTS: usize = 32 * 1024;

fn planner_is_threaded(_plans: &mut (impl FftPlanProvider + ?Sized)) -> bool {
    #[cfg(feature = "fftw")]
    {
        _plans
            .fftw()
            .is_some_and(|(provider, _)| provider.threaded())
    }
    #[cfg(not(feature = "fftw"))]
    false
}

#[allow(clippy::too_many_arguments)]
// INVARIANT: these arguments are validated FFT metadata and the two borrowed
// buffers; no public configurable operation descriptor is duplicated here.
/// # Safety
/// For absent input (in-place), O must be Complex<T>, the output must initially contain valid
/// complex values, and the operation must be shape-preserving c2c with identity
/// projection. Out-of-place variants have no additional unsafe preconditions.
pub(super) unsafe fn execute<T, O>(
    input: Option<Input<'_, T>>,
    in_shape: &[usize],
    axis: usize,
    fft_len: usize,
    out_axis_len: usize,
    operation: FftOperation,
    norm: FftNorm,
    plans: &mut (impl FftPlanProvider + ?Sized),
    buffers: &mut BufferPool,
    output: &mut [MaybeUninit<O>],
    project: impl Fn(Complex<T>) -> O + Sync,
    exec: Exec<'_>,
) -> tenferro_tensor::Result<()>
where
    T: CachedFftPlanScalar,
    O: Send,
    Complex<T>: PoolScalar,
{
    let LaneLayout {
        stride,
        in_block,
        out_block,
        lanes,
        input_len,
        output_len,
    } = LaneLayout::new(in_shape, axis, out_axis_len)?;
    if fft_len == 0 || out_axis_len > fft_len {
        return Err(tenferro_tensor::Error::invalid_argument(
            "fft",
            "transform lengths",
            "FFT length must be positive and cover the output lane",
        ));
    }
    let in_len = in_shape[axis];
    if operation == FftOperation::C2r {
        // Mirror reads must stay inside the gathered initialized spectrum.
        crate::validate_c2r_spectrum_len(in_len, fft_len)?;
    }

    let input_actual = match &input {
        Some(Input::Complex(x)) => x.len(),
        Some(Input::Real(x)) => x.len(),
        None => output.len(),
    };
    if input_actual != input_len || output.len() != output_len {
        return Err(tenferro_tensor::Error::invalid_argument(
            "fft",
            "buffer length",
            "FFT buffers do not match validated lane coverage",
        ));
    }
    if lanes == 0 {
        return Ok(());
    }
    let threaded = planner_is_threaded(plans);
    #[cfg(feature = "fftw")]
    let fftw = plans
        .fftw()
        .map(|(provider, cache)| provider.plan::<T>(cache, fft_len, operation))
        .transpose()?;
    #[cfg(feature = "fftw")]
    let plan = fftw
        .is_none()
        .then(|| cached_fft_plan::<T, _>(plans, fft_len, operation.is_forward()));
    #[cfg(not(feature = "fftw"))]
    let plan = Some(cached_fft_plan::<T, _>(
        plans,
        fft_len,
        operation.is_forward(),
    ));
    let scale = scale_for::<T>(norm, operation.is_forward(), fft_len)?;
    let jobs = if threaded || output_len < MIN_PARALLEL_ELEMENTS {
        1
    } else {
        exec.budget().get().min(lanes)
    };
    let per_job = lanes.div_ceil(jobs);
    let out = Output {
        pointer: output.as_mut_ptr(),
        _borrow: PhantomData,
    };
    let scratch_len = plan
        .as_ref()
        .map_or(0, |plan| plan.get_inplace_scratch_len());
    let job_len = fft_len.checked_add(scratch_len).ok_or_else(|| {
        tenferro_tensor::Error::invalid_argument("fft", "scratch length", "overflows usize")
    })?;
    let total_len = super::checked_mul("fft", "parallel scratch", jobs, job_len)?;
    // Both lane and vendor scratch storage are reused as one slab, avoiding
    // per-job metadata allocations even when the engine changes job count.
    let mut storage = buffers.acquire_with_capacity::<Complex<T>>(total_len);
    // SAFETY: capacity covers total_len; MaybeUninit exposes no uninitialized
    // Complex values, and the owning Vec stays length zero until recycled.
    let uninit = unsafe {
        std::slice::from_raw_parts_mut(
            storage.as_mut_ptr().cast::<MaybeUninit<Complex<T>>>(),
            total_len,
        )
    };
    let run = |job: usize, storage: &mut [MaybeUninit<Complex<T>>]| {
        let (lane_storage, scratch_storage) = storage.split_at_mut(fft_len);
        scratch_storage.fill(MaybeUninit::new(Complex::zero()));
        // SAFETY: every vendor scratch element was initialized immediately above.
        let scratch = unsafe {
            std::slice::from_raw_parts_mut(
                scratch_storage.as_mut_ptr().cast::<Complex<T>>(),
                scratch_len,
            )
        };
        let start = job * per_job;
        let end = start.saturating_add(per_job).min(lanes);
        if start >= end {
            return;
        }
        // INVARIANT: nonzero lanes proves stride > 0. Advance within each
        // checked block rather than divide every lane index by runtime stride.
        let mut outer_index = start / stride;
        let mut inner = start % stride;
        let mut input_base = outer_index * in_block + inner;
        let mut output_base = outer_index * out_block + inner;
        let copy_len = in_len.min(fft_len);
        for _ in start..end {
            match &input {
                None => {
                    for (slot, offset) in lane_storage
                        .iter_mut()
                        .zip((input_base..).step_by(stride))
                        .take(copy_len)
                    {
                        // SAFETY: the in-place caller proves O=Complex<T> and
                        // initialized input. Jobs read/write only their own lane;
                        // gather completes before that lane is overwritten.
                        slot.write(unsafe { out.read_complex::<T>(offset) });
                    }
                }
                Some(Input::Complex(values)) => {
                    for (slot, offset) in lane_storage
                        .iter_mut()
                        .zip((input_base..).step_by(stride))
                        .take(copy_len)
                    {
                        slot.write(values[offset]);
                    }
                }
                Some(Input::Real(values)) => {
                    for (slot, offset) in lane_storage
                        .iter_mut()
                        .zip((input_base..).step_by(stride))
                        .take(copy_len)
                    {
                        slot.write(Complex::new(values[offset], T::zero()));
                    }
                }
            }
            if operation == FftOperation::C2r {
                for k in copy_len..fft_len {
                    // SAFETY: the spectrum prefix is gathered above; later
                    // mirrored indices refer only to that initialized prefix.
                    let mirrored = unsafe { lane_storage[fft_len - k].assume_init() }.conj();
                    lane_storage[k].write(mirrored);
                }
            } else {
                // INVARIANT: only missing input positions are padding; the
                // gathered slots were written above, including real-input
                // imaginary parts.
                lane_storage[copy_len..].fill(MaybeUninit::new(Complex::zero()));
            }
            // SAFETY: gather plus padding/mirroring initialized the entire lane.
            let lane = unsafe {
                std::slice::from_raw_parts_mut(
                    lane_storage.as_mut_ptr().cast::<Complex<T>>(),
                    fft_len,
                )
            };
            if let Some(plan) = &plan {
                plan.process_with_scratch(lane, scratch);
            }
            #[cfg(feature = "fftw")]
            if let Some(plan) = &fftw {
                plan.complex(lane);
            }
            // Fold the normalization into the scatter. This is the same single
            // multiply per emitted value as the removed separate full-lane pass,
            // so the numerical result is unchanged; `scale == 1` keeps the
            // identity fast path free of arithmetic in the strided scatter.
            for (value, offset) in lane
                .iter()
                .take(out_axis_len)
                .zip((output_base..).step_by(stride))
            {
                let value = if scale == T::one() {
                    *value
                } else {
                    *value * scale
                };
                // SAFETY: each lane owns indices base + k*stride, k<out_axis_len.
                // Unique (outer_index,inner) pairs exactly tile checked output_len.
                unsafe {
                    out.write(offset, project(value));
                }
            }
            inner += 1;
            if inner == stride {
                inner = 0;
                outer_index += 1;
                // INVARIANT: advancing after the last lane reaches at most
                // the checked input/output length, never a dereferenced offset.
                input_base = outer_index * in_block;
                output_base = outer_index * out_block;
            } else {
                input_base += 1;
                output_base += 1;
            }
        }
    };
    if jobs == 1 {
        run(0, uninit);
    } else {
        let slots = JobSlots(uninit.as_mut_ptr());
        exec.fork(jobs, |job, _child| {
            // SAFETY: job < jobs, and the checked slab has jobs disjoint job_len ranges.
            run(job, unsafe { slots.slice(job * job_len, job_len) });
        });
    }
    Complex::<T>::pool_release(buffers, storage);
    Ok(())
}

/// Execute a real-to-complex or complex-to-real transform with a `realfft` plan.
///
/// The input lanes gather the real (R2C) or one-sided spectrum (C2R) values; the
/// plan produces the spectrum (R2C) or the real output (C2R). The normalization
/// is folded into the scatter, like the complex path.
#[allow(clippy::too_many_arguments)]
// INVARIANT: these arguments are validated FFT metadata and the two borrowed
// buffers; no public configurable operation descriptor is duplicated here.
/// # Safety
/// The caller proves the input and output lengths match the validated lane
/// coverage and that real transforms are out of place (`input` is `Some`).
pub(super) unsafe fn execute_real<T, O>(
    input: Option<Input<'_, T>>,
    in_shape: &[usize],
    axis: usize,
    fft_len: usize,
    out_axis_len: usize,
    operation: FftOperation,
    norm: FftNorm,
    plans: &mut (impl FftPlanProvider + ?Sized),
    buffers: &mut BufferPool,
    output: &mut [MaybeUninit<O>],
    project: impl Fn(Complex<T>) -> O + Sync,
    exec: Exec<'_>,
) -> tenferro_tensor::Result<()>
where
    T: CachedFftPlanScalar + PoolScalar,
    O: Send,
    Complex<T>: PoolScalar,
{
    let LaneLayout {
        stride,
        in_block,
        out_block,
        lanes,
        input_len,
        output_len,
    } = LaneLayout::new(in_shape, axis, out_axis_len)?;
    let in_len = in_shape[axis];
    let input_actual = match &input {
        Some(Input::Complex(values)) => values.len(),
        Some(Input::Real(values)) => values.len(),
        None => 0,
    };
    if input_actual != input_len || output.len() != output_len {
        return Err(tenferro_tensor::Error::invalid_argument(
            "fft",
            "buffer length",
            "FFT buffers do not match validated lane coverage",
        ));
    }
    if lanes == 0 {
        return Ok(());
    }
    let threaded = planner_is_threaded(plans);
    #[cfg(feature = "fftw")]
    let fftw = plans
        .fftw()
        .map(|(provider, cache)| provider.plan::<T>(cache, fft_len, operation))
        .transpose()?;
    #[cfg(feature = "fftw")]
    let plan = fftw
        .is_none()
        .then(|| cached_real_plan::<T, _>(plans, fft_len, operation.is_forward()));
    #[cfg(not(feature = "fftw"))]
    let plan = Some(cached_real_plan::<T, _>(
        plans,
        fft_len,
        operation.is_forward(),
    ));
    let scale = scale_for::<T>(norm, operation.is_forward(), fft_len)?;
    let jobs = if threaded || output_len < MIN_PARALLEL_ELEMENTS {
        1
    } else {
        exec.budget().get().min(lanes)
    };
    let per_job = lanes.div_ceil(jobs);
    let out = Output {
        pointer: output.as_mut_ptr(),
        _borrow: PhantomData,
    };
    let real_len = fft_len;
    let spectrum_len = fft_len / 2 + 1;
    let even_length = fft_len.is_multiple_of(2);
    let mirror_end = if even_length {
        spectrum_len - 1
    } else {
        spectrum_len
    };
    // `R2cFull` asks for the full spectrum, so the lane holds room for the
    // conjugate mirror as well as the plan's non-negative half.
    let lane_len = out_axis_len.max(spectrum_len);
    let complex_len = if operation == FftOperation::C2r {
        spectrum_len
    } else {
        lane_len
    };
    let scratch_len = match &plan {
        Some(RealPlan::Forward(plan)) => plan.get_scratch_len(),
        Some(RealPlan::Inverse(plan)) => plan.get_scratch_len(),
        None => 0,
    };
    let complex_job_len = complex_len.checked_add(scratch_len).ok_or_else(|| {
        tenferro_tensor::Error::invalid_argument("fft", "scratch length", "overflows usize")
    })?;
    let real_total = super::checked_mul("fft", "parallel real lanes", jobs, real_len)?;
    let complex_total = super::checked_mul("fft", "parallel complex lanes", jobs, complex_job_len)?;
    // INVARIANT: realfft's safe API requires initialized mutable output and
    // scratch slices. Initialize each pooled slab once; per-lane gathering
    // below only zeroes semantic padding, never the overwritten output.
    let mut real_lanes = buffers.acquire_zeroed::<T>(real_total);
    let mut complex_lanes = buffers.acquire_zeroed::<Complex<T>>(complex_total);
    let run = |job: usize,
               real_lane: &mut [T],
               complex_storage: &mut [Complex<T>]|
     -> tenferro_tensor::Result<()> {
        let (complex_lane, scratch) = complex_storage.split_at_mut(complex_len);
        let start = job * per_job;
        let end = start.saturating_add(per_job).min(lanes);
        if start >= end {
            return Ok(());
        }
        let mut outer_index = start / stride;
        let mut inner = start % stride;
        let mut input_base = outer_index * in_block + inner;
        let mut output_base = outer_index * out_block + inner;
        let copy_len = in_len.min(real_len);
        for _ in start..end {
            match (&input, operation) {
                (Some(Input::Real(values)), FftOperation::R2cFull | FftOperation::R2cOnesided) => {
                    // Only missing input samples are semantic padding. Every
                    // spectrum value is overwritten by realfft or mirroring.
                    real_lane[copy_len..].fill(T::zero());
                    for (slot, offset) in real_lane
                        .iter_mut()
                        .zip((input_base..).step_by(stride))
                        .take(copy_len)
                    {
                        *slot = values[offset];
                    }
                    if let Some(RealPlan::Forward(plan)) = &plan {
                        plan.process_with_scratch(
                            real_lane,
                            complex_lane[..spectrum_len].as_mut(),
                            scratch,
                        )
                        .map_err(|error| {
                            tenferro_tensor::Error::runtime_state(
                                "fft",
                                format!("real-to-complex plan rejected the lane buffers: {error}"),
                            )
                        })?;
                    }
                    #[cfg(feature = "fftw")]
                    if let Some(plan) = &fftw {
                        plan.real(real_lane, &mut complex_lane[..spectrum_len]);
                    }
                    // Fill the negative-frequency half only when the caller
                    // asked for the full spectrum; the one-sided output already
                    // holds every requested bin.
                    if out_axis_len > spectrum_len {
                        for k in 1..mirror_end {
                            let mirrored = complex_lane[k].conj();
                            complex_lane[out_axis_len - k] = mirrored;
                        }
                    }
                    for (value, offset) in complex_lane
                        .iter()
                        .take(out_axis_len)
                        .zip((output_base..).step_by(stride))
                    {
                        let value = if scale == T::one() {
                            *value
                        } else {
                            *value * scale
                        };
                        // SAFETY: each lane owns indices base + k*stride,
                        // k<out_axis_len; the jobs tile the checked output.
                        unsafe {
                            out.write(offset, project(value));
                        }
                    }
                }
                (Some(Input::Complex(values)), FftOperation::C2r) => {
                    complex_lane[copy_len.min(spectrum_len)..].fill(Complex::zero());
                    for (slot, offset) in complex_lane
                        .iter_mut()
                        .zip((input_base..).step_by(stride))
                        .take(copy_len.min(spectrum_len))
                    {
                        *slot = values[offset];
                    }
                    // A real inverse transform treats the one-sided spectrum as
                    // Hermitian: DC is real for every length. A Nyquist bin
                    // exists only for even lengths; the final odd-length bin
                    // keeps its imaginary component.
                    if spectrum_len > 0 {
                        complex_lane[0] = Complex::new(complex_lane[0].re, T::zero());
                        if even_length {
                            let last = spectrum_len - 1;
                            complex_lane[last] = Complex::new(complex_lane[last].re, T::zero());
                        }
                    }
                    if let Some(RealPlan::Inverse(plan)) = &plan {
                        plan.process_with_scratch(complex_lane, real_lane, scratch)
                            .map_err(|error| {
                                tenferro_tensor::Error::runtime_state(
                                    "fft",
                                    format!(
                                        "complex-to-real plan rejected the lane buffers: {error}"
                                    ),
                                )
                            })?;
                    }
                    #[cfg(feature = "fftw")]
                    if let Some(plan) = &fftw {
                        plan.real(real_lane, complex_lane);
                    }
                    for (value, offset) in real_lane
                        .iter()
                        .take(out_axis_len)
                        .zip((output_base..).step_by(stride))
                    {
                        let value = if scale == T::one() {
                            *value
                        } else {
                            *value * scale
                        };
                        // SAFETY: each lane owns indices base + k*stride,
                        // k<out_axis_len; the jobs tile the checked output.
                        unsafe {
                            out.write(offset, project(Complex::new(value, T::zero())));
                        }
                    }
                }
                _ => {
                    return Ok(());
                }
            }
            inner += 1;
            if inner == stride {
                inner = 0;
                outer_index += 1;
                input_base = outer_index * in_block;
                output_base = outer_index * out_block;
            } else {
                input_base += 1;
                output_base += 1;
            }
        }
        Ok(())
    };
    let result = if jobs == 1 {
        run(0, &mut real_lanes, &mut complex_lanes)
    } else {
        let real_slots = JobSlots(real_lanes.as_mut_ptr());
        let complex_slots = JobSlots(complex_lanes.as_mut_ptr());
        let first_error = std::sync::Mutex::new(None);
        exec.fork(jobs, |job, _child| {
            // SAFETY: each job owns one checked, disjoint range of each slab.
            let outcome = run(
                job,
                unsafe { real_slots.slice(job * real_len, real_len) },
                unsafe { complex_slots.slice(job * complex_job_len, complex_job_len) },
            );
            if let Err(error) = outcome {
                first_error
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .get_or_insert(error);
            }
        });
        match first_error
            .into_inner()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
        {
            Some(error) => Err(error),
            None => Ok(()),
        }
    };
    T::pool_release(buffers, real_lanes);
    Complex::<T>::pool_release(buffers, complex_lanes);
    result
}
