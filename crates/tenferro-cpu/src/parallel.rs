//! Parallel chunking helpers shared by the CPU kernels.
//!
//! The kernels split their output into cache-sized tasks and hand each worker
//! one of them. Small problems launch fewer workers so the synchronization cost
//! does not dominate, mirroring PyTorch's `parallel_for` grain selection.
//!
//! The fan-out itself belongs to the engine's pool: a raw Rayon iterator outside
//! a pool runs on the process-global one, which a serial engine must never start
//! and a pooled engine must never reach through (#2206, #2273).

use std::mem::MaybeUninit;

use crate::provider::CpuExecutionContext;

/// Minimum amount of work (in bytes) that justifies one worker.
pub(crate) const MIN_BYTES_PER_TASK: usize = 32 * 1024;

/// Worker budget for one kernel fan-out.
///
/// The task split is decided on the calling thread from the budget; only a
/// fan-out of more than one task enters the pool, so a serial context stays on
/// the calling thread. Where the pool is comes from the engine's target-pool
/// scope, which strided's splitter reads when it takes its one entry.
#[derive(Clone, Copy)]
pub(crate) struct FanOut {
    workers: usize,
}

impl FanOut {
    /// The fan-out of `context`: its thread budget, or one without a pool.
    pub(crate) fn from_context(context: &CpuExecutionContext<'_>) -> Self {
        Self {
            workers: context.native_thread_count(),
        }
    }

    /// One worker on the calling thread.
    #[cfg(test)]
    pub(crate) const SERIAL: Self = Self { workers: 1 };

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
    ///
    /// The fan-out goes through strided's policy-aware splitter: it clamps the
    /// width to the execution policy, enters the engine's target pool once, and
    /// clears strided's placement scopes while it waits, so a job that runs on
    /// the waiting thread cannot inherit this operation's placement.
    pub(crate) fn run<T, F>(self, dst: &mut [MaybeUninit<T>], src: &[T], task: usize, kernel: F)
    where
        T: Send + Sync,
        F: Fn(&mut [MaybeUninit<T>], &[T]) + Sync + Send,
    {
        let task = task.max(1);
        let len = dst.len().min(src.len());
        let tasks = len.div_ceil(task);
        // A single task has nothing to fan out: stay on the calling thread even
        // when the engine has a pool.
        if self.workers <= 1 || tasks <= 1 {
            for start in (0..len).step_by(task) {
                let end = (start + task).min(len);
                kernel(&mut dst[start..end], &src[start..end]);
            }
            return;
        }
        let dst_ptr = strided_basic::execution::SendPtr(dst.as_mut_ptr());
        let map = |range: core::ops::Range<usize>| {
            for index in range {
                let start = index * task;
                let end = (start + task).min(len);
                // SAFETY: the task spans partition `0..len`, so this closure
                // writes only its own output elements and reads the matching
                // input elements; the splitter hands each span to one worker.
                unsafe {
                    let out =
                        core::slice::from_raw_parts_mut(dst_ptr.as_ptr().add(start), end - start);
                    kernel(out, &src[start..end]);
                }
            }
        };
        strided_basic::execution::parallel_map_reduce(0..tasks, self.workers, &map, &|(), ()| ());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_caps_workers_for_small_problems() {
        let four = FanOut { workers: 4 };
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
