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
use rayon::prelude::*;
use std::{marker::PhantomData, mem::MaybeUninit};
use tenferro_cpu::linalg_interop::{BufferPool, PoolScalar};

/// Worker budget for one lane execution, and the engine pool to run fan-out on.
///
/// The lane count is decided on the calling thread first; only a fan-out of
/// more than one job enters `pool`, so small transforms never leave the caller
/// (#2206).
#[derive(Clone, Copy)]
pub(super) struct LaneThreads<'p> {
    count: usize,
    pool: Option<&'p rayon::ThreadPool>,
}

impl<'p> LaneThreads<'p> {
    pub(super) fn new(count: usize, pool: Option<&'p rayon::ThreadPool>) -> Self {
        Self { count, pool }
    }

    pub(super) fn from_context(context: &tenferro_cpu::CpuExecutionContext<'p>) -> Self {
        Self::new(context.native_thread_count(), context.rayon_pool())
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
    threads: LaneThreads<'_>,
    project: impl Fn(Complex<T>) -> O + Sync,
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
    let in_len = in_shape[axis];
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
    let plan = cached_fft_plan::<T, _>(plans, fft_len, operation.is_forward());
    let scale = scale_for::<T>(norm, operation.is_forward(), fft_len)?;
    let jobs = if output_len < MIN_PARALLEL_ELEMENTS {
        1
    } else {
        threads.count.max(1).min(lanes)
    };
    let per_job = lanes.div_ceil(jobs);
    let out = Output {
        pointer: output.as_mut_ptr(),
        _borrow: PhantomData,
    };
    // INVARIANT: each job writes every element of its own lane before
    // `plan.process_with_scratch` reads it -- the gather fills `copy_len` slots
    // and the padding or conjugate-mirror fill covers the rest -- so a lane
    // buffer may be taken from the pool with uninitialized capacity. Taking
    // capacity instead of a zero-filled full-length vector removes the
    // per-call lane allocation and its zero fill.
    let mut lane_buffers: Vec<Vec<Complex<T>>> = Vec::with_capacity(jobs);
    for _ in 0..jobs {
        lane_buffers.push(buffers.acquire_with_capacity::<Complex<T>>(fft_len));
    }
    let run = |job: usize, lane: &mut Vec<Complex<T>>| {
        let start = job * per_job;
        let end = start.saturating_add(per_job).min(lanes);
        if start >= end {
            return;
        }
        let mut scratch = vec![Complex::<T>::zero(); plan.get_inplace_scratch_len()];
        // INVARIANT: nonzero lanes proves stride > 0. Advance within each
        // checked block rather than divide every lane index by runtime stride.
        let mut outer_index = start / stride;
        let mut inner = start % stride;
        let mut input_base = outer_index * in_block + inner;
        let mut output_base = outer_index * out_block + inner;
        let copy_len = in_len.min(fft_len);
        for _ in start..end {
            lane.clear();
            match &input {
                None => {
                    for offset in (input_base..).step_by(stride).take(copy_len) {
                        // SAFETY: the in-place caller proves O=Complex<T> and
                        // initialized input. Jobs read/write only their own lane;
                        // gather completes before that lane is overwritten.
                        lane.push(unsafe { out.read_complex::<T>(offset) });
                    }
                }
                Some(Input::Complex(values)) => {
                    for offset in (input_base..).step_by(stride).take(copy_len) {
                        lane.push(values[offset]);
                    }
                }
                Some(Input::Real(values)) => {
                    for offset in (input_base..).step_by(stride).take(copy_len) {
                        lane.push(Complex::new(values[offset], T::zero()));
                    }
                }
            }
            if operation == FftOperation::C2r {
                for k in copy_len..fft_len {
                    let mirrored = lane[fft_len - k].conj();
                    lane.push(mirrored);
                }
            } else {
                // INVARIANT: only missing input positions are padding; the
                // gathered slots were written above, including real-input
                // imaginary parts.
                lane.resize(fft_len, Complex::zero());
            }
            plan.process_with_scratch(lane.as_mut_slice(), &mut scratch);
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
        run(0, &mut lane_buffers[0]);
    } else {
        let mut fan_out = || {
            (0..jobs)
                .into_par_iter()
                .zip(lane_buffers.par_iter_mut())
                .for_each(|(job, lane)| run(job, lane));
        };
        match threads.pool {
            Some(pool) if pool.current_thread_index().is_none() => pool.install(fan_out),
            _ => fan_out(),
        }
    }
    for lane in lane_buffers {
        Complex::<T>::pool_release(buffers, lane);
    }
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
    _buffers: &mut BufferPool,
    output: &mut [MaybeUninit<O>],
    threads: LaneThreads<'_>,
    project: impl Fn(Complex<T>) -> O + Sync,
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
    let plan = cached_real_plan::<T, _>(plans, fft_len, operation.is_forward());
    let scale = scale_for::<T>(norm, operation.is_forward(), fft_len)?;
    let jobs = if output_len < MIN_PARALLEL_ELEMENTS {
        1
    } else {
        threads.count.max(1).min(lanes)
    };
    let per_job = lanes.div_ceil(jobs);
    let out = Output {
        pointer: output.as_mut_ptr(),
        _borrow: PhantomData,
    };
    let real_len = fft_len;
    let spectrum_len = fft_len / 2 + 1;
    // `R2cFull` asks for the full spectrum, so the lane holds room for the
    // conjugate mirror as well as the plan's non-negative half.
    let lane_len = out_axis_len.max(spectrum_len);
    let complex_len = if operation == FftOperation::C2r {
        spectrum_len
    } else {
        lane_len
    };
    let mut real_lanes: Vec<Vec<T>> = (0..jobs).map(|_| vec![T::zero(); real_len]).collect();
    let mut complex_lanes: Vec<Vec<Complex<T>>> = (0..jobs)
        .map(|_| vec![Complex::zero(); complex_len])
        .collect();
    let run = |job: usize,
               real_lane: &mut Vec<T>,
               complex_lane: &mut Vec<Complex<T>>|
     -> tenferro_tensor::Result<()> {
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
            for value in real_lane.iter_mut() {
                *value = T::zero();
            }
            for value in complex_lane.iter_mut() {
                *value = Complex::zero();
            }
            match (&input, operation) {
                (Some(Input::Real(values)), FftOperation::R2cFull | FftOperation::R2cOnesided) => {
                    for (slot, offset) in real_lane
                        .iter_mut()
                        .zip((input_base..).step_by(stride))
                        .take(copy_len)
                    {
                        *slot = values[offset];
                    }
                    if let RealPlan::Forward(plan) = &plan {
                        plan.process(
                            real_lane.as_mut_slice(),
                            complex_lane[..spectrum_len].as_mut(),
                        )
                        .map_err(|_| {
                            tenferro_tensor::Error::runtime_state(
                                "fft",
                                "real-to-complex plan rejected the lane buffers",
                            )
                        })?;
                    }
                    // Fill the negative-frequency half only when the caller
                    // asked for the full spectrum; the one-sided output already
                    // holds every requested bin.
                    if out_axis_len > spectrum_len {
                        for k in 1..spectrum_len.saturating_sub(1) {
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
                    for (slot, offset) in complex_lane
                        .iter_mut()
                        .zip((input_base..).step_by(stride))
                        .take(copy_len.min(spectrum_len))
                    {
                        *slot = values[offset];
                    }
                    if let RealPlan::Inverse(plan) = &plan {
                        plan.process(complex_lane.as_mut_slice(), real_lane.as_mut_slice())
                            .map_err(|_| {
                                tenferro_tensor::Error::runtime_state(
                                    "fft",
                                    "complex-to-real plan rejected the lane buffers",
                                )
                            })?;
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
    if jobs == 1 {
        run(0, &mut real_lanes[0], &mut complex_lanes[0])
    } else {
        let mut fan_out = || {
            real_lanes
                .par_iter_mut()
                .zip(complex_lanes.par_iter_mut())
                .enumerate()
                .try_for_each(|(job, (real_lane, complex_lane))| run(job, real_lane, complex_lane))
        };
        match threads.pool {
            Some(pool) if pool.current_thread_index().is_none() => pool.install(fan_out),
            _ => fan_out(),
        }
    }
}
