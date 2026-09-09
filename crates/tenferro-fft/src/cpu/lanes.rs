//! CPU FFT lane execution over disjoint output index sets.
#[cfg(test)]
mod tests;
use super::{scale_for, LaneLayout};
use crate::cache::{cached_fft_plan, CachedFftPlanScalar, FftPlanProvider};
use crate::{FftNorm, FftOperation};
use num_complex::Complex;
use num_traits::Zero;
use rayon::prelude::*;
use std::{marker::PhantomData, mem::MaybeUninit};

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
    output: &mut [MaybeUninit<O>],
    threads: usize,
    project: impl Fn(Complex<T>) -> O + Sync,
) -> tenferro_tensor::Result<()>
where
    T: CachedFftPlanScalar,
    O: Send,
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
        threads.max(1).min(lanes)
    };
    let per_job = lanes.div_ceil(jobs);
    let out = Output {
        pointer: output.as_mut_ptr(),
        _borrow: PhantomData,
    };
    let run = |job: usize| {
        let start = job * per_job;
        let end = start.saturating_add(per_job).min(lanes);
        if start >= end {
            return;
        }
        let mut lane = vec![Complex::<T>::zero(); fft_len];
        let mut scratch = vec![Complex::<T>::zero(); plan.get_inplace_scratch_len()];
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
                    for (slot, offset) in lane
                        .iter_mut()
                        .take(copy_len)
                        .zip((input_base..).step_by(stride))
                    {
                        // SAFETY: the in-place caller proves O=Complex<T> and
                        // initialized input. Jobs read/write only their own lane;
                        // gather completes before that lane is overwritten.
                        *slot = unsafe { out.read_complex::<T>(offset) };
                    }
                }
                Some(Input::Complex(values)) => {
                    for (slot, offset) in lane
                        .iter_mut()
                        .take(copy_len)
                        .zip((input_base..).step_by(stride))
                    {
                        *slot = values[offset];
                    }
                }
                Some(Input::Real(values)) => {
                    for (slot, offset) in lane
                        .iter_mut()
                        .take(copy_len)
                        .zip((input_base..).step_by(stride))
                    {
                        *slot = Complex::new(values[offset], T::zero());
                    }
                }
            }
            if operation == FftOperation::C2r {
                for k in copy_len..fft_len {
                    lane[k] = lane[fft_len - k].conj();
                }
            } else {
                // INVARIANT: only missing input positions are padding; all other
                // slots were overwritten above, including real-input imaginary parts.
                lane[copy_len..].fill(Complex::zero());
            }
            plan.process_with_scratch(&mut lane, &mut scratch);
            // Preserve the existing identity-normalization fast path. Scaling
            // contiguous scratch also avoids arithmetic in the strided scatter.
            if scale != T::one() {
                for value in &mut lane {
                    *value = *value * scale;
                }
            }
            for (value, offset) in lane
                .iter()
                .take(out_axis_len)
                .zip((output_base..).step_by(stride))
            {
                // SAFETY: each lane owns indices base + k*stride, k<out_axis_len.
                // Unique (outer_index,inner) pairs exactly tile checked output_len.
                unsafe {
                    out.write(offset, project(*value));
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
        run(0);
    } else {
        (0..jobs).into_par_iter().for_each(run);
    }
    Ok(())
}
