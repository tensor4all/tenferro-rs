//! Parallel chunking helpers shared by the CPU kernels.
//!
//! The kernels split their output into cache-sized groups, then hand several
//! groups to each worker. Small problems launch fewer workers so the
//! synchronization cost does not dominate, mirroring PyTorch's `parallel_for`
//! grain selection.

/// Bytes in one cache-sized group of slices.
pub(crate) const GROUP_BYTES: usize = 32 * 1024;

/// Minimum amount of work (in bytes) that justifies one worker.
pub(crate) const MIN_BYTES_PER_TASK: usize = 32 * 1024;

/// Number of elements in one cache-sized group.
///
/// A group never exceeds the whole tensor and always contains at least one
/// slice, so a single slice longer than `GROUP_BYTES` stays intact.
pub(crate) fn group_len(n: usize, axis_len: usize, element_size: usize) -> usize {
    let slices = (GROUP_BYTES / (axis_len * element_size)).max(1);
    (axis_len * slices).min(n).max(1)
}

/// Number of elements in one parallel task.
///
/// The task is a whole number of groups. Its count is capped by the smaller of
/// the current pool size and the work divided by `min_bytes_per_task`, so a
/// large pool on a small tensor does not pay the launch cost.
pub(crate) fn task_len(
    n: usize,
    group: usize,
    element_size: usize,
    min_bytes_per_task: usize,
) -> usize {
    let groups = n.div_ceil(group);
    let workers = (n * element_size / min_bytes_per_task)
        .max(1)
        .min(rayon::current_num_threads().max(1));
    let groups_per_task = groups.div_ceil(workers).max(1);
    (group * groups_per_task).min(n).max(1)
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
        // 128 KB on a four-worker pool: 32 KB per worker keeps all four busy.
        assert_eq!(task_len(32_768, 8_192, 4, MIN_BYTES_PER_TASK), 8_192);
        // A one-worker pool uses one task.
        // (The test runner's pool size is not fixed, so only the bounds are
        // asserted for the other cases.)
        let task = task_len(2_097_152, 4_096, 4, MIN_BYTES_PER_TASK);
        assert!((4_096..=2_097_152).contains(&task) && task.is_multiple_of(4_096));
        assert_eq!(task_len(16, 16, 4, MIN_BYTES_PER_TASK), 16);
    }
}
