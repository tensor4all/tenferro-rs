use std::num::NonZeroUsize;

/// Return a best-effort CPU count available to the current process.
///
/// This first tries an OS-standard process-affinity query when supported, then
/// falls back to `std::thread::available_parallelism()`, and finally to `1`.
///
/// # Examples
///
/// ```
/// let available = tenferro_tensor::cpu::available_parallelism();
/// assert!(available >= 1);
/// ```
pub fn available_parallelism() -> usize {
    process_cpu_affinity_count()
        .or_else(standard_available_parallelism)
        .unwrap_or(1)
}

/// Return the current process affinity mask size when the platform exposes a
/// standard affinity API.
///
/// Platforms without an affinity query return `None`.
///
/// # Examples
///
/// ```
/// let count = tenferro_tensor::cpu::process_cpu_affinity_count();
/// if let Some(count) = count {
///     assert!(count >= 1);
/// }
/// ```
pub fn process_cpu_affinity_count() -> Option<usize> {
    platform_process_cpu_affinity_count()
}

pub(crate) fn standard_available_parallelism() -> Option<usize> {
    std::thread::available_parallelism()
        .ok()
        .map(NonZeroUsize::get)
}

#[cfg(any(target_os = "linux", target_os = "android", test))]
fn count_affinity_mask_bits(mask: &[u8]) -> Option<usize> {
    let count = mask.iter().map(|byte| byte.count_ones() as usize).sum();
    (count > 0).then_some(count)
}

#[cfg(any(target_os = "linux", target_os = "android"))]
const LINUX_EINVAL: i32 = 22;

#[cfg(any(target_os = "linux", target_os = "android"))]
fn linux_next_affinity_mask_bytes(mask_bytes: usize, errno: Option<i32>) -> Option<usize> {
    (errno == Some(LINUX_EINVAL))
        .then(|| mask_bytes.checked_mul(2))
        .flatten()
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn platform_process_cpu_affinity_count() -> Option<usize> {
    unsafe extern "C" {
        fn sched_getaffinity(pid: i32, cpusetsize: usize, mask: *mut core::ffi::c_void) -> i32;
    }

    const INITIAL_MASK_BYTES: usize = 128;

    let mut mask_bytes = INITIAL_MASK_BYTES;
    loop {
        let mut mask = vec![0u8; mask_bytes];
        let rc = unsafe {
            sched_getaffinity(0, mask_bytes, mask.as_mut_ptr().cast::<core::ffi::c_void>())
        };
        if rc == 0 {
            return count_affinity_mask_bits(&mask);
        }

        mask_bytes = linux_next_affinity_mask_bytes(
            mask_bytes,
            std::io::Error::last_os_error().raw_os_error(),
        )?;
    }
}

#[cfg(target_os = "windows")]
fn platform_process_cpu_affinity_count() -> Option<usize> {
    type Handle = *mut core::ffi::c_void;
    type DwordPtr = usize;
    type Word = u16;

    unsafe extern "system" {
        fn GetCurrentProcess() -> Handle;
        fn GetProcessAffinityMask(
            process: Handle,
            process_affinity_mask: *mut DwordPtr,
            system_affinity_mask: *mut DwordPtr,
        ) -> i32;
        fn GetActiveProcessorGroupCount() -> Word;
        fn GetActiveProcessorCount(group_number: Word) -> u32;
        fn GetProcessGroupAffinity(
            process: Handle,
            group_count: *mut Word,
            group_array: *mut Word,
        ) -> i32;
    }

    let process = unsafe { GetCurrentProcess() };
    let system_group_count = unsafe { GetActiveProcessorGroupCount() };

    if system_group_count <= 1 {
        let mut process_mask = 0usize;
        let mut system_mask = 0usize;
        let ok = unsafe {
            GetProcessAffinityMask(
                process,
                std::ptr::addr_of_mut!(process_mask),
                std::ptr::addr_of_mut!(system_mask),
            )
        };
        if ok != 0 {
            let count = process_mask.count_ones() as usize;
            return (count > 0).then_some(count);
        }
        let count = unsafe { GetActiveProcessorCount(0) } as usize;
        return (count > 0).then_some(count);
    }

    let mut group_count: Word = 0;
    let ok = unsafe {
        GetProcessGroupAffinity(
            process,
            std::ptr::addr_of_mut!(group_count),
            std::ptr::null_mut(),
        )
    };
    if ok != 0 || group_count == 0 {
        let count = unsafe { GetActiveProcessorCount(u16::MAX) } as usize;
        return (count > 0).then_some(count);
    }

    let mut groups = vec![0u16; group_count as usize];
    let ok = unsafe {
        GetProcessGroupAffinity(
            process,
            std::ptr::addr_of_mut!(group_count),
            groups.as_mut_ptr(),
        )
    };
    if ok == 0 || group_count == 0 {
        let count = unsafe { GetActiveProcessorCount(u16::MAX) } as usize;
        return (count > 0).then_some(count);
    }

    if group_count == 1 {
        let mut process_mask = 0usize;
        let mut system_mask = 0usize;
        let ok = unsafe {
            GetProcessAffinityMask(
                process,
                std::ptr::addr_of_mut!(process_mask),
                std::ptr::addr_of_mut!(system_mask),
            )
        };
        if ok != 0 {
            let count = process_mask.count_ones() as usize;
            return (count > 0).then_some(count);
        }
    }

    let count = groups
        .into_iter()
        .map(|group| unsafe { GetActiveProcessorCount(group) } as usize)
        .sum();
    (count > 0).then_some(count)
}

#[cfg(not(any(target_os = "linux", target_os = "android", target_os = "windows")))]
fn platform_process_cpu_affinity_count() -> Option<usize> {
    None
}

#[cfg(test)]
mod tests;
