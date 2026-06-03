use super::*;

#[test]
fn count_affinity_mask_bits_returns_none_for_zero_masks() {
    assert_eq!(count_affinity_mask_bits(&[]), None);
    assert_eq!(count_affinity_mask_bits(&[0, 0, 0]), None);
}

#[test]
fn count_affinity_mask_bits_sums_all_set_bits() {
    assert_eq!(count_affinity_mask_bits(&[0b0000_0001]), Some(1));
    assert_eq!(
        count_affinity_mask_bits(&[0b1010_0001, 0b1111_0000]),
        Some(7)
    );
}

#[cfg(any(target_os = "linux", target_os = "android"))]
#[test]
fn linux_next_affinity_mask_bytes_retries_only_on_einval() {
    assert_eq!(
        linux_next_affinity_mask_bytes(128, Some(LINUX_EINVAL)),
        Some(256)
    );
    assert_eq!(linux_next_affinity_mask_bytes(128, Some(1)), None);
    assert_eq!(linux_next_affinity_mask_bytes(128, None), None);
    assert_eq!(
        linux_next_affinity_mask_bytes(usize::MAX / 2 + 1, Some(LINUX_EINVAL)),
        None
    );
}

#[test]
fn available_parallelism_helpers_report_positive_counts() {
    assert!(available_parallelism() >= 1);
    if let Some(count) = standard_available_parallelism() {
        assert!(count >= 1);
    }
    if let Some(count) = process_cpu_affinity_count() {
        assert!(count >= 1);
    }
}
