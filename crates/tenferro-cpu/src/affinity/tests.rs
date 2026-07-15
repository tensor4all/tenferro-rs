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

#[test]
fn affinity_mask_preserves_sparse_logical_cpu_ids() {
    let cpus = cpu_set_from_affinity_mask(&[0b1000_0010, 0b0000_0101]).unwrap();
    assert_eq!(cpus.as_usize_vec(), vec![1, 7, 8, 10]);
}

#[test]
fn affinity_mask_rejects_extreme_cpu_ids_before_allocation() {
    let cpus = CpuSet::new([CpuId::new(usize::MAX)]).unwrap();
    let error = build_affinity_mask(&cpus).unwrap_err();
    assert!(error.contains("exceeds supported affinity mask"));
}

#[test]
fn affinity_mask_builds_sparse_logical_cpu_ids() {
    let cpus = CpuSet::new([CpuId::new(1), CpuId::new(7), CpuId::new(10)]).unwrap();
    let mask = build_affinity_mask(&cpus).unwrap();

    assert_eq!(mask.len(), 128);
    assert_eq!(mask[0], 0b1000_0010);
    assert_eq!(mask[1], 0b0000_0100);
    assert!(mask[2..].iter().all(|&byte| byte == 0));
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
#[test]
fn system_thread_affinity_reports_unsupported_platform() {
    let error = SystemThreadAffinity.pin_current(CpuId::new(0)).unwrap_err();

    assert_eq!(
        error,
        "setting thread affinity is unsupported on this platform"
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
    if let Some(cpus) = process_cpu_affinity() {
        assert!(!cpus.is_empty());
        assert_eq!(process_cpu_affinity_count(), Some(cpus.len()));
    }
}

#[test]
fn windows_group_affinity_probe_documents_expected_failure_path() {
    let source = include_str!("../affinity.rs");
    assert!(
        source.contains("expected failure path")
            && source.contains("required processor-group")
            && source.contains("count. That probe"),
        "Windows GetProcessGroupAffinity null-array probe must document that failure with a count is expected"
    );
}
