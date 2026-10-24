//! Parallel chunking helpers shared by the CPU kernels.
//!
//! The kernels split their output into cache-sized groups, then hand several
//! groups to each worker. Small problems launch fewer workers so the
//! synchronization cost does not dominate, mirroring PyTorch's `parallel_for`
//! grain selection.
//!
//! The fan-out itself belongs to the engine's pool: raw Rayon iterators outside
//! a pool run on the process-global one, which a serial engine must never start
//! and a pooled engine must never reach through (#2206, #2273).

use std::mem::MaybeUninit;

use rayon::prelude::*;

use crate::provider::CpuExecutionContext;

/// Bytes in one cache-sized group of slices.
pub(crate) const GROUP_BYTES: usize = 32 * 1024;

/// Minimum amount of work (in bytes) that justifies one worker.
pub(crate) const MIN_BYTES_PER_TASK: usize = 32 * 1024;

/// Worker budget for one kernel fan-out, and the pool to run it on.
///
/// The task split is decided on the calling thread from the budget; only a
/// fan-out of more than one task enters the pool, so a serial context stays on
/// the calling thread.
#[derive(Clone, Copy)]
pub(crate) struct FanOut<'p> {
    workers: usize,
    pool: Option<&'p rayon::ThreadPool>,
}

impl<'p> FanOut<'p> {
    /// The fan-out of `context`: its engine pool bounded by its thread budget,
    /// or the calling thread.
    pub(crate) fn from_context(context: &CpuExecutionContext<'p>) -> Self {
        Self {
            workers: context.native_thread_count(),
            pool: context.rayon_pool(),
        }
    }

    /// One worker on the calling thread.
    #[cfg(test)]
    pub(crate) const SERIAL: Self = Self {
        workers: 1,
        pool: None,
    };

    /// Number of elements in one parallel task.
    ///
    /// The task is a whole number of `unit`-sized chunks. Its count is capped by
    /// the smaller of the worker budget and the work divided by
    /// `MIN_BYTES_PER_TASK`, so a large pool on a small tensor does not pay the
    /// launch cost.
    pub(crate) fn task_len(self, n: usize, unit: usize, element_size: usize) -> usize {
        let units = n.div_ceil(unit);
        let workers = (n * element_size / MIN_BYTES_PER_TASK)
            .max(1)
            .min(self.workers.max(1));
        let units_per_task = units.div_ceil(workers).max(1);
        (unit * units_per_task).min(n).max(1)
    }

    /// Run `kernel` on the disjoint `task`-sized chunks of `dst` and `src`.
    pub(crate) fn run<T, F>(self, dst: &mut [MaybeUninit<T>], src: &[T], task: usize, kernel: F)
    where
        T: Send + Sync,
        F: Fn(&mut [MaybeUninit<T>], &[T]) + Sync + Send,
    {
        match self.pool {
            Some(pool) if self.workers > 1 => pool.install(|| {
                dst.par_chunks_mut(task)
                    .zip(src.par_chunks(task))
                    .for_each(|(out, input)| kernel(out, input));
            }),
            _ => {
                for (out, input) in dst.chunks_mut(task).zip(src.chunks(task)) {
                    kernel(out, input);
                }
            }
        }
    }
}

/// Number of elements in one cache-sized group.
///
/// A group never exceeds the whole tensor and always contains at least one
/// slice, so a single slice longer than `GROUP_BYTES` stays intact.
pub(crate) fn group_len(n: usize, axis_len: usize, element_size: usize) -> usize {
    let slices = (GROUP_BYTES / (axis_len * element_size)).max(1);
    (axis_len * slices).min(n).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_is_cache_sized_and_covers_at_least_one_slice() {
        assert_eq!(group_len(32_768, 64, 4), 8_192);
        assert_eq!(group_len(2_097_152, 512, 4), 8_192);
        assert_eq!(group_len(64, 64, 4), 64);
        assert_eq!(group_len(1_000_000, 1_000_000, 8), 1_000_000);
    }

    #[test]
    fn task_caps_workers_for_small_problems() {
        let four = FanOut {
            workers: 4,
            pool: None,
        };
        // 128 KB on a four-worker budget: 32 KB per worker keeps all four busy.
        assert_eq!(four.task_len(32_768, 8_192, 4), 8_192);
        // A one-worker budget uses one task.
        assert_eq!(FanOut::SERIAL.task_len(2_097_152, 4_096, 4), 2_097_152);
        let task = four.task_len(2_097_152, 4_096, 4);
        assert!((4_096..=2_097_152).contains(&task) && task.is_multiple_of(4_096));
        assert_eq!(four.task_len(16, 16, 4), 16);
    }

    #[test]
    fn serial_fan_out_covers_every_element_on_the_calling_thread() {
        let caller = std::thread::current().id();
        let src = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut dst = [MaybeUninit::<f32>::uninit(); 5];
        FanOut::SERIAL.run(&mut dst, &src, 2, |out, input| {
            assert_eq!(std::thread::current().id(), caller);
            assert_eq!(out.len(), input.len());
            for (slot, &value) in out.iter_mut().zip(input) {
                slot.write(value);
            }
        });
        // SAFETY: the closure above wrote every element.
        let dst: &[f32] = unsafe { std::slice::from_raw_parts(dst.as_ptr().cast(), dst.len()) };
        assert_eq!(dst, src);
    }
}
