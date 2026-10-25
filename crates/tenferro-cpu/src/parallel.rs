//! Parallel chunking helper shared by the CPU kernels.
//!
//! The kernels split their output into cache-sized tasks and hand each worker
//! one of them. Small problems launch fewer workers so the synchronization cost
//! does not dominate, mirroring PyTorch's `parallel_for` grain selection.
//!
//! The fan-out is the placement token's: [`Exec::for_ranges`] splits the work
//! over at most the token's budget and runs inline, without any pool call, for a
//! serial token or a single task. A raw Rayon iterator outside a pool runs on
//! the process-global one, which a serial engine must never start and a pooled
//! engine must never reach through (#2206, #2273).

use std::mem::MaybeUninit;
use std::num::NonZeroUsize;

use strided_basic::execution::SendPtr;
use strided_exec::Exec;

/// Minimum amount of work (in bytes) that justifies one worker.
pub(crate) const MIN_BYTES_PER_TASK: usize = 32 * 1024;

/// Run `kernel` on disjoint chunks of `dst` and `src`.
///
/// A chunk is a whole number of `unit`-sized pieces, except for the last one.
/// The pieces are divided over at most the budget of `exec`, and each task
/// receives at least [`MIN_BYTES_PER_TASK`] bytes of work, so a large pool on a
/// small tensor does not pay the launch cost. A serial token, or a tensor that
/// is one task's worth, runs `kernel` once on the calling thread.
pub(crate) fn run_chunks<T, F>(
    dst: &mut [MaybeUninit<T>],
    src: &[T],
    unit: usize,
    kernel: F,
    exec: Exec<'_>,
) where
    T: Send + Sync,
    F: Fn(&mut [MaybeUninit<T>], &[T]) + Sync,
{
    let len = dst.len().min(src.len());
    let unit = unit.max(1);
    let units = len.div_ceil(unit);
    let unit_bytes = unit.saturating_mul(size_of::<T>().max(1));
    let grain =
        NonZeroUsize::new((MIN_BYTES_PER_TASK / unit_bytes).max(1)).unwrap_or(NonZeroUsize::MIN);
    let dst_ptr = SendPtr(dst.as_mut_ptr());
    exec.for_ranges(units, grain, |range| {
        let start = range.start * unit;
        let end = (range.end * unit).min(len);
        // SAFETY: `for_ranges` hands out disjoint ranges that partition
        // `0..units`, so the element spans `start..end` partition `0..len`: this
        // closure writes only its own output elements and reads the matching
        // input elements.
        unsafe {
            let out = core::slice::from_raw_parts_mut(dst_ptr.as_ptr().add(start), end - start);
            kernel(out, &src[start..end]);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_all(out: &mut [MaybeUninit<f32>], input: &[f32]) {
        assert_eq!(out.len(), input.len());
        for (slot, &value) in out.iter_mut().zip(input) {
            slot.write(value);
        }
    }

    #[test]
    fn serial_fan_out_covers_every_element_on_the_calling_thread() {
        let caller = std::thread::current().id();
        let src = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut dst = [MaybeUninit::<f32>::uninit(); 5];
        run_chunks(
            &mut dst,
            &src,
            2,
            |out, input| {
                assert_eq!(std::thread::current().id(), caller);
                write_all(out, input);
            },
            Exec::serial(),
        );
        // SAFETY: the closure above wrote every element.
        let dst: &[f32] = unsafe { std::slice::from_raw_parts(dst.as_ptr().cast(), dst.len()) };
        assert_eq!(dst, src);
    }

    #[test]
    fn small_problem_on_a_wide_token_is_one_task_without_a_pool_entry() {
        let pool = strided_exec::Pool::builder().threads(4).build().unwrap();
        let exec = Exec::new(&pool, 4).unwrap();
        #[cfg(feature = "ledger")]
        let root = exec.ledger_root().expect("a token from `new` is a root");
        let caller = std::thread::current().id();
        let src = vec![1.0f32; 1024];
        let mut dst = vec![MaybeUninit::<f32>::uninit(); 1024];
        run_chunks(
            &mut dst,
            &src,
            256,
            |out, input| {
                assert_eq!(std::thread::current().id(), caller);
                write_all(out, input);
            },
            exec,
        );
        #[cfg(feature = "ledger")]
        {
            assert_eq!(strided_exec::ledger::installs(root), 0);
            assert_eq!(strided_exec::ledger::max_live(root), 1);
        }
    }

    #[test]
    fn large_problem_covers_every_element_within_the_budget() {
        let pool = strided_exec::Pool::builder().threads(4).build().unwrap();
        let exec = Exec::new(&pool, 3).unwrap();
        #[cfg(feature = "ledger")]
        let root = exec.ledger_root().expect("a token from `new` is a root");
        // 1 MiB of f32: several tasks' worth, with a ragged tail.
        let n = 262_144 + 37;
        let src: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut dst = vec![MaybeUninit::<f32>::uninit(); n];
        run_chunks(&mut dst, &src, 2048, write_all, exec);
        // SAFETY: every chunk was written.
        let dst: &[f32] = unsafe { std::slice::from_raw_parts(dst.as_ptr().cast(), n) };
        assert_eq!(dst, &src[..]);
        #[cfg(feature = "ledger")]
        assert!(strided_exec::ledger::max_live(root) <= 3);
    }
}
