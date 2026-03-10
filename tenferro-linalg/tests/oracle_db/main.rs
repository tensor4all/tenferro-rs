mod replay;

#[test]
fn oracle_db_replay_against_tensor_ad_oracles() {
    let summary = replay::run_database_replay();

    assert_eq!(
        summary.validated_records, 348,
        "unexpected replay summary: validated={}, expected_error={:?}, failures={:?}",
        summary.validated_records, summary.expected_error_case_ids, summary.failures
    );
    assert_eq!(
        summary.expected_error_case_ids,
        vec![
            "eigh_c128_gauge_ill_defined_001".to_string(),
            "svd_c128_gauge_ill_defined_001".to_string(),
        ]
    );
    assert!(
        summary.failures.is_empty(),
        "oracle replay failures: {:?}",
        summary.failures
    );
}
